import logging
import os
import warnings
from pathlib import Path
from typing import Union, Any, Optional, List

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm_sound import TqdmSound

from configs import get_global_config, iter_processing_configs
from file_utils import parse_rttm, filter_files_by_stems, dict_to_csv, audacity_writer
from logger import global_logger
from system_config import is_cuda


class Transcriber:
    """
    Handles ASR transcription for WAV+RTTM pairs, manages model and processor.

    NOTE:
    Downstream consumers should use `transcription/` and `transcription_labeled/` as canonical text sources.
    RTTM boundaries can diverge over time from transcript row boundaries.
    """

    MODEL_ID = "Systran/faster-distil-whisper-large-v3"

    def __init__(self):
        self._init_libs()
        self.logger = global_logger("transcription")
        self.global_config = get_global_config()
        self.transcribe_beam_size = self._env_int("INTENSION_TRANSCRIBE_BEAM_SIZE", 1)
        self.transcribe_best_of = self._env_int("INTENSION_TRANSCRIBE_BEST_OF", 1)
        self.transcribe_batch_size = self._env_int("INTENSION_TRANSCRIBE_BATCH_SIZE", 16) # depends on your vram size, not much benefit past 16 for most cases
        self.use_batched_transcribe = self._env_bool("INTENSION_TRANSCRIBE_BATCHED", True) 
        self.model = self._load_model()
        self.batched_model = (
            BatchedInferencePipeline(model=self.model) if self.use_batched_transcribe else None
        )

    def _init_libs(self):
        """
        Handle all logger/warning suppression.
        """
        logging.getLogger("faster_whisper").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message=".*The input name `inputs` is deprecated.*",
            category=FutureWarning,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            return default
        return max(1, parsed)

    def _load_model(self):
        force_cpu = os.environ.get("INTENSION_FORCE_CPU", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        safe_gpu_mode = os.environ.get("INTENSION_GPU_SAFE_MODE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        use_cuda = is_cuda and not force_cpu
        compute_type = "int8_float16" if (use_cuda and safe_gpu_mode) else ("float16" if use_cuda else "float32")
        device = "cuda" if use_cuda else "cpu"
        return WhisperModel(
            self.MODEL_ID,
            device=device,
            compute_type=compute_type,
        )

    @staticmethod
    def make_url(cfg, filename: str, start: int) -> str:
        config_type = type(cfg).__name__
        start = int(start)

        metadata = cfg.get_metadata_by_filename(filename)
        if not metadata:
            return ""
        video_id = metadata.get("video_id")
        if not video_id:
            return ""

        url = None
        if config_type == 'YouTubeConfig':
            url = f'https://www.youtube.com/watch?v={video_id}&t={start}'
        elif config_type == 'TwitchConfig':
            h, rem = divmod(start, 3600)
            m, s = divmod(rem, 60)
            timestamp = f"{h:02d}h{m:02d}m{s:02d}s"

            url = f'https://www.twitch.tv/videos/{video_id}?t={timestamp}'

        return url

    def _transcribe_single_segment(self, segment_audio: np.ndarray) -> str:
        result_segments, _ = self.model.transcribe(
            segment_audio,
            language="en",
            vad_filter=False,
            without_timestamps=True,
            beam_size=self.transcribe_beam_size,
            best_of=self.transcribe_best_of,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in result_segments).strip()

    def _transcribe_batch_from_clips(
            self,
            audio_data: np.ndarray,
            sample_rate: int,
            segment_batch: List[dict],
    ) -> List[str]:
        if self.batched_model is None:
            raise RuntimeError("Batched transcription is disabled.")

        clip_timestamps = []
        for seg in segment_batch:
            start_sample = int(seg.get("start_time") * sample_rate)
            end_sample = int(seg.get("end_time") * sample_rate)
            if end_sample <= start_sample:
                end_sample = start_sample + 1
            clip_timestamps.append({"start": start_sample, "end": end_sample})

        result_segments, _ = self.batched_model.transcribe(
            audio_data,
            language="en",
            vad_filter=False,
            without_timestamps=True,
            clip_timestamps=clip_timestamps,
            batch_size=self.transcribe_batch_size,
            beam_size=self.transcribe_beam_size,
            best_of=self.transcribe_best_of,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        decoded = [s.text.strip() for s in result_segments]
        if len(decoded) != len(segment_batch):
            raise RuntimeError(
                f"Batch decode mismatch: expected {len(segment_batch)} segments, got {len(decoded)}."
            )
        return decoded

    @staticmethod
    def _merge_adjacent_same_speaker_rows(transcript_rows: List[dict]) -> List[dict]:
        if len(transcript_rows) < 2:
            return transcript_rows

        merged_rows = [dict(transcript_rows[0])]
        for current in transcript_rows[1:]:
            previous = merged_rows[-1]
            if current.get("speaker") == previous.get("speaker"):
                previous_text = (previous.get("text") or "").strip()
                current_text = (current.get("text") or "").strip()
                previous["text"] = f"{previous_text} {current_text}".strip()
                previous["end_time"] = current.get("end_time")
                previous["duration"] = previous.get("end_time") - previous.get("start_time")
                continue

            merged_rows.append(dict(current))

        return merged_rows

    def segment_and_transcribe(
            self,
            rttm_file: Path,
            wav_file: Path,
            transcript_file: Path,
            audacity_file: Path,
            cfg: Any = None,
            progress: Optional[Any] = None,
            segment_desc: Optional[str] = None,
    ) -> None:
        """
        Perform ASR on segments defined in an RTTM file and write transcript CSV and Audacity label file.

        NOTE:
        This step materializes the canonical transcript rows in `transcription/`.
        Do not assume RTTM rows remain a stable 1:1 pairing with transcript rows later.
        """
        pair_existed_at_start = transcript_file.exists() and audacity_file.exists()
        segments = parse_rttm(rttm_file)
        if not segments:
            self.logger.info(f"No segments found in RTTM: {rttm_file}")
            transcript_file.write_text("", encoding="utf-8")
            audacity_file.write_text("", encoding="utf-8")
            return

        valid_segments = []
        for seg in segments:
            start = seg.get("start_time")
            end = seg.get("end_time")
            if (end - start) < 0.5:
                continue
            valid_segments.append(seg)

        transcript_rows = []
        filename = wav_file.stem

        # Load audio once into memory to avoid re-opening the file per segment
        audio_data, sample_rate = sf.read(wav_file, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        if self.batched_model is None:
            segment_iter = valid_segments
            if progress is not None and valid_segments:
                segment_iter = progress.progress_bar(
                    valid_segments,
                    total=len(valid_segments),
                    desc=segment_desc or f"Segments - {filename}",
                    unit="segment",
                    leave=False,
                    ten_percent_ticks=True,
                )
            for seg in segment_iter:
                start = seg.get("start_time")
                end = seg.get("end_time")

                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                text = self._transcribe_single_segment(segment_audio)

                url = ''
                if cfg is not None:
                    url = self.make_url(cfg, filename, start)

                transcript_rows.append({
                    "speaker": seg.get("speaker"),
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "text": text,
                    "timestamp_url": url
                })
        else:
            batch_offsets = range(0, len(valid_segments), self.transcribe_batch_size)
            batch_iter = batch_offsets
            if progress is not None and valid_segments:
                total_batches = len(batch_offsets)
                batch_iter = progress.progress_bar(
                    batch_offsets,
                    total=total_batches,
                    desc=f"{segment_desc or f'Segments - {filename}'} (batched)",
                    unit="batch",
                    leave=False,
                    ten_percent_ticks=True,
                )

            for i in batch_iter:
                segment_batch = valid_segments[i:i + self.transcribe_batch_size]
                try:
                    batch_texts = self._transcribe_batch_from_clips(audio_data, sample_rate, segment_batch)
                except Exception as e:
                    self.logger.warning(
                        f"Batched decode failed for {filename} [{i}:{i + len(segment_batch)}], "
                        f"falling back to per-segment decode: {e}"
                    )
                    batch_texts = []
                    for seg in segment_batch:
                        start_sample = int(seg.get("start_time") * sample_rate)
                        end_sample = int(seg.get("end_time") * sample_rate)
                        segment_audio = audio_data[start_sample:end_sample]
                        batch_texts.append(self._transcribe_single_segment(segment_audio))

                for seg, text in zip(segment_batch, batch_texts):
                    start = seg.get("start_time")
                    end = seg.get("end_time")

                    url = ''
                    if cfg is not None:
                        url = self.make_url(cfg, filename, start)

                    transcript_rows.append({
                        "speaker": seg.get("speaker"),
                        "start_time": start,
                        "end_time": end,
                        "duration": end - start,
                        "text": text,
                        "timestamp_url": url
                    })

        if not pair_existed_at_start:
            original_count = len(transcript_rows)
            transcript_rows = self._merge_adjacent_same_speaker_rows(transcript_rows)
            merged_count = len(transcript_rows)
            if merged_count < original_count:
                self.logger.info(
                    f"Merged adjacent same-speaker transcript rows for {filename}: "
                    f"{original_count} -> {merged_count}"
                )

        audacity_rows = [
            {
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "text": row.get("text", ""),
            }
            for row in transcript_rows
        ]

        dict_to_csv(transcript_file, transcript_rows)
        audacity_writer(audacity_file, audacity_rows)

    def transcribe_from_rttm(
            self,
            wav_file: Union[Path, str],
            rttm_file: Union[Path, str] = None,
    ) -> None:
        """
        Given a WAV file and RTTM file, transcribe each segment and write transcript and Audacity label outputs.
        """
        wav_file = Path(wav_file)
        if rttm_file is not None:
            rttm_file = Path(rttm_file)
        else:
            rttm_file = wav_file.with_suffix('.rttm')

        stem = wav_file.stem
        out_dir = wav_file.parent
        transcript_file = out_dir / f"{stem}_transcript.csv"
        audacity_file = out_dir / f"{stem}_transcript_audacity.txt"

        self.segment_and_transcribe(rttm_file, wav_file, transcript_file, audacity_file)

    def process_file(self, file: Path, cfg: Any, progress: Optional[Any] = None) -> str:
        """
        Transcribe a single WAV file given its RTTM segment file.
        """
        channel_dir = file.parents[1]
        transcript_dir = channel_dir / "transcription"
        audacity_dir = channel_dir / "transcription_audacity"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        audacity_dir.mkdir(parents=True, exist_ok=True)

        wav_file = channel_dir / "wav" / file.with_suffix(".wav").name
        rttm_file = channel_dir / "rttm" / file.with_suffix(".rttm").name
        transcript_file = transcript_dir / file.with_suffix(".csv").name
        audacity_file = audacity_dir / file.with_suffix(".txt").name

        if not rttm_file.exists():
            return f"Skipped (no RTTM): {file.name}"

        if transcript_file.exists() and audacity_file.exists():
            return f"Skipped (exists): {file.name}"

        cfg_id = cfg.channel_name_or_term
        cfg_name = cfg.name

        try:
            segment_desc = f"{cfg_name} ({cfg_id}) Segments - {file.stem}"
            self.segment_and_transcribe(
                rttm_file,
                wav_file,
                transcript_file,
                audacity_file,
                cfg,
                progress=progress,
                segment_desc=segment_desc,
            )
            self.logger.info(f"Transcribed: {file.name}")
            return f"Processed: {cfg_name} ({cfg_id}) - {file.name}"

        except Exception as e:
            self.logger.error(f"Error at {file.name}: {e}")
            return f"{cfg_name} ({cfg_id}) - Error: {file.name} - {e}"

    def batch(self):
        """
        Iterate over all channel configs, batch-transcribe all WAV+RTTM pairs for all DownloadConfig types,
        and write transcript/Audacity label files.
        """
        progress = TqdmSound(
            activity_mute_seconds=0,
            dynamic_settings_file=str(self.global_config.project_root / "confs" / "sound.json")
        )

        for cfg in iter_processing_configs(include_manual=True):
            wav_dir = cfg.output_path / "wav"
            if not wav_dir.exists():
                continue

            channel_data_dir = cfg.output_path
            if not channel_data_dir:
                continue

            rttm_dir = channel_data_dir / 'rttm'
            transcription_dir = channel_data_dir / 'transcription'
            transcription_audacity_dir = channel_data_dir / 'transcription_audacity'

            rttm_files = filter_files_by_stems(
                rttm_dir, 'rttm', [transcription_dir, transcription_audacity_dir]
            )

            cfg_id = cfg.channel_name_or_term
            cfg_name = cfg.name

            if not rttm_files:
                self.logger.info(f"No files to transcribe for: {cfg_name} ({cfg_id})")
                print(f"No files to transcribe for: {cfg_name} ({cfg_id})")
                continue

            bar = progress.progress_bar(
                rttm_files,
                total=len(rttm_files),
                desc=f"{cfg_name} ({cfg_id}): Transcribing",
                unit="file",
                leave=True,
                ten_percent_ticks=True,
            )

            for file in bar:
                bar.set_description(f"{cfg_name} ({cfg_id}) Transcribing - {file.name}")
                self.process_file(file, cfg, progress=progress)


def main():
    transcriber = Transcriber()
    transcriber.batch()


if __name__ == "__main__":
    main()
