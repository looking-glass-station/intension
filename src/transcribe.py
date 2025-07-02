import concurrent.futures
import contextlib
import io
import logging
import warnings
from pathlib import Path
from typing import Union, Any

import torch
import torchaudio
from tqdm_sound import TqdmSound
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import logging as hf_logging

from configs import get_configs, get_global_config
from file_utils import parse_rttm, filter_files_by_stems, dict_to_csv, audacity_writer
from logger import global_logger
from system_config import device, torch_dtype, is_cuda, max_workers


class Transcriber:
    """
    Handles ASR transcription for WAV+RTTM pairs, manages model and processor.
    """

    MODEL_ID = "distil-whisper/distil-large-v3"

    def __init__(self):
        self._init_libs()
        self.logger = global_logger("transcription")
        self.global_config = get_global_config()
        self.model = self._load_model()
        self.processor = self._load_processor()
        self.pipeline = self._build_pipeline()

    def _init_libs(self):
        """
        Handle all logger/warning suppression and deferred imports.
        """
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)

        # FutureWarning suppressions
        warnings.filterwarnings(
            "ignore",
            message=".*The input name `inputs` is deprecated.*",
            category=FutureWarning,
        )

        # Defer heavy imports to here and hide their chatter
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    def _load_model(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        if is_cuda:
            model = model.half()
        return model

    def _load_processor(self):
        return AutoProcessor.from_pretrained(self.MODEL_ID)

    def _build_pipeline(self):
        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            batch_size=32,
        )

    @staticmethod
    def make_url(cfg, filename: str, start: int) -> str:
        config_type = type(cfg).__name__
        start = int(start)
        video_id = cfg.get_metadata_by_filename(filename)['video_id']

        url = None
        if config_type == 'YouTubeConfig':
            url = f'https://www.youtube.com/watch?v={video_id}&t={start}'
        elif config_type == 'TwitchConfig':
            h, rem = divmod(start, 3600)
            m, s = divmod(rem, 60)
            timestamp = f"{h:02d}h{m:02d}m{s:02d}s"

            url = f'https://www.twitch.tv/videos/{video_id}?t={timestamp}'

        return url

    def segment_and_transcribe(
            self,
            rttm_file: Path,
            wav_file: Path,
            transcript_file: Path,
            audacity_file: Path,
            cfg: Any
    ) -> None:
        """
        Perform ASR on segments defined in an RTTM file and write transcript CSV and Audacity label file.
        """
        segments = parse_rttm(rttm_file)
        if not segments:
            self.logger.info(f"No segments found in RTTM: {rttm_file}")
            transcript_file.write_text("", encoding="utf-8")
            audacity_file.write_text("", encoding="utf-8")
            return

        waveform, sr = torchaudio.load(wav_file)
        audio_items = []
        for seg in segments:
            start = seg.get("start_time")
            end = seg.get("end_time")
            if (end - start) < 0.5:
                continue

            start_sample = int(start * sr)
            end_sample = int(end * sr)
            audio_seg = waveform[:, start_sample:end_sample].squeeze().numpy()
            audio_items.append((seg, audio_seg))

        workers = max_workers(multiplier=1)

        def _transcribe(item):
            seg, audio = item
            with torch.no_grad():
                result = self.pipeline([audio])[0]
            return seg, result.get("text", "").strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_transcribe, audio_items))

        transcript_rows = []
        audacity_rows = []
        for seg, text in results:
            start = seg.get("start_time")
            end = seg.get("end_time")

            filename = wav_file.stem
            url = self.make_url(cfg, filename, start)

            transcript_rows.append({
                "speaker": seg.get("speaker"),
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "text": text,
                "timestamp_url": url

            })
            audacity_rows.append({
                "start_time": start,
                "end_time": end,
                "text": text,
            })

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

    def process_file(self, file: Path, cfg: Any) -> str:
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
            self.segment_and_transcribe(rttm_file, wav_file, transcript_file, audacity_file, cfg)
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

        for channel_cfg in get_configs():
            for config_list in vars(channel_cfg.download_configs).values():
                if not isinstance(config_list, list) or not config_list:
                    continue

                for cfg in config_list:
                    wav_dir = cfg.output_path / "wav"
                    if not wav_dir.exists():
                        continue

                    channel_data_dir = cfg.output_path
                    if not channel_data_dir:
                        continue

                    rttm_dir = channel_data_dir / 'rttm'
                    transcription_dir = channel_data_dir / 'transcription'
                    transcription_audacity_dir = channel_data_dir / 'transcription_audacity'

                    if transcription_dir.exists() and transcription_audacity_dir.exists():
                        continue

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
                        self.process_file(file, cfg)


def main():
    transcriber = Transcriber()
    transcriber.batch()


if __name__ == "__main__":
    main()
