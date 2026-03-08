import contextlib
import io
import inspect
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import torch
from tqdm_sound import TqdmSound
from whisperx.diarize import DiarizationPipeline
from pyannote.audio import Pipeline

import system_config
from configs import get_global_config, iter_processing_configs
from file_utils import filter_files_by_stems, audacity_writer
from logger import global_logger


class Diarizer:
    """
    Handles diarization for single files or in batch, manages model internally.
    """

    def __init__(self):
        self.logger = global_logger("diarization")
        self.global_config = get_global_config()
        self.merge_short_segments = self._env_bool("INTENSION_MERGE_SHORT_DIARIZATION_SEGMENTS", True)
        self.short_segment_seconds = self._env_float("INTENSION_SHORT_DIARIZATION_SEGMENT_SECONDS", 1.0)
        self.merge_gap_seconds = self._env_float("INTENSION_DIARIZATION_MERGE_GAP_SECONDS", 0.5)
        self.model = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = self._load_model()

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
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            parsed = float(raw)
        except ValueError:
            return default
        return max(0.0, parsed)

    def _merge_short_adjacent_segments(self, segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not self.merge_short_segments or len(segments) < 2:
            return segments

        ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
        merged = [ordered[0].copy()]

        for current in ordered[1:]:
            previous = merged[-1]
            previous_duration = previous["end"] - previous["start"]
            current_duration = current["end"] - current["start"]
            gap = current["start"] - previous["end"]
            is_short_pair = (
                previous_duration <= self.short_segment_seconds
                or current_duration <= self.short_segment_seconds
            )

            if (
                current["label"] == previous["label"]
                and gap <= self.merge_gap_seconds
                and is_short_pair
            ):
                previous["end"] = max(previous["end"], current["end"])
                continue

            merged.append(current.copy())

        return merged

    def _write_rttm_and_audacity(
            self,
            rttm_path: Path,
            audacity_path: Path,
            wav_name: str,
            segments: List[Dict[str, float]],
    ) -> None:
        escaped_name = wav_name.replace(" ", "_")
        rttm_lines = [
            f"SPEAKER {escaped_name} 1 {seg['start']:.3f} {seg['end'] - seg['start']:.3f} <NA> <NA> {seg['label']} <NA> <NA>\n"
            for seg in segments
        ]
        audacity_labels = [
            {
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["label"],
            }
            for seg in segments
        ]
        rttm_path.write_text("".join(rttm_lines), encoding="utf-8")
        audacity_writer(audacity_path, audacity_labels)

    def _load_model(self):
        from whisperx.diarize import DiarizationPipeline

        try:
            # Compatibility shim for newer huggingface_hub versions where
            # hf_hub_download no longer accepts use_auth_token.
            from pyannote.audio.core import pipeline as pyannote_pipeline_module

            hf_download = getattr(pyannote_pipeline_module, "hf_hub_download", None)
            if hf_download is not None:
                params = inspect.signature(hf_download).parameters
                if "use_auth_token" not in params:
                    original_hf_download = hf_download

                    def _hf_hub_download_compat(*args, use_auth_token=None, **kwargs):
                        if use_auth_token is not None and "token" not in kwargs:
                            kwargs["token"] = use_auth_token
                        return original_hf_download(*args, **kwargs)

                    pyannote_pipeline_module.hf_hub_download = _hf_hub_download_compat

            # PyTorch 2.6 defaults to weights_only=True and blocks some globals.
            # Allow trusted pyannote checkpoint classes used during load.
            from pyannote.audio.core.task import Problem, Resolution, Specifications
            torch.serialization.add_safe_globals(
                [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
            )
            diarizer = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=self.global_config.hf_token,
                device=system_config.device
            )
            if diarizer.model is None:
                raise RuntimeError("DiarizationPipeline.model is None — likely due to missing HF token access.")
            return diarizer
        except Exception as e:
            self.logger.error(f"Failed to load WhisperX diarization pipeline: {e}")
            raise


    def diarize_file(self, wav_file: Path, output_dir: Path = None, host_count: Optional[int] = None):
        """
        Diarizes a wav file and writes RTTM and Audacity label files.
        """
        if isinstance(wav_file, str):
            wav_file = Path(wav_file)
        if output_dir is None:
            output_dir = wav_file.parent

        rttm_path = output_dir / "rttm" / f"{wav_file.stem}.rttm"
        audacity_path = output_dir / "audacity_labels" / f"{wav_file.stem}.txt"

        # Ensure output dirs exist
        rttm_path.parent.mkdir(parents=True, exist_ok=True)
        audacity_path.parent.mkdir(parents=True, exist_ok=True)

        if rttm_path.exists() and audacity_path.exists():
            self.logger.info(f"Diarizing skipped: {rttm_path.name}")
            return

        # dont do this, too many hosts use random clips
        #max_speakers = host_count + 2 if host_count is not None else None
        #diarization_kwargs = {"max_speakers": max_speakers} if max_speakers is not None else {}
        self._ensure_model()
        diarization_kwargs={}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = self.model(str(wav_file), **diarization_kwargs)

        diarize_df = (
            result.get("diarization") if isinstance(result, dict) and "diarization" in result else result
        )
        segments = [
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "label": row.get("speaker") or row.get("label"),
            }
            for _, row in diarize_df.iterrows()
        ]
        raw_segment_count = len(segments)
        segments = self._merge_short_adjacent_segments(segments)
        merged_segment_count = len(segments)
        if merged_segment_count < raw_segment_count:
            self.logger.info(
                f"Merged adjacent short diarization segments for {wav_file.name}: "
                f"{raw_segment_count} -> {merged_segment_count} "
                f"(short<={self.short_segment_seconds:.2f}s, gap<={self.merge_gap_seconds:.2f}s)"
            )

        self._write_rttm_and_audacity(rttm_path, audacity_path, wav_file.name, segments)
        self.logger.info(f"Diarized: {wav_file.name}")

    def batch(self):
        """
        Runs diarization on all files from configs.
        """
        progress = TqdmSound(
            activity_mute_seconds=0,
            dynamic_settings_file=str(self.global_config.project_root / "confs" / "sound.json")
        )

        for cfg in iter_processing_configs(include_manual=True):
            cfg_id = cfg.channel_name_or_term
            cfg_name = cfg.name
            channel_data_dir = cfg.output_path

            if not channel_data_dir:
                continue

            wav_dir = channel_data_dir / "wav"
            rttm_dir = channel_data_dir / 'rttm'
            audacity_dir = channel_data_dir / 'audacity_labels'

            rttm_dir.mkdir(parents=True, exist_ok=True)
            audacity_dir.mkdir(parents=True, exist_ok=True)

            if not wav_dir.exists():
                continue

            wav_files = filter_files_by_stems(wav_dir, 'wav', [rttm_dir, audacity_dir])

            if not wav_files:
                self.logger.info(f"No files to diarize for: {cfg_name} ({cfg_id})")
                print(f"No files to diarize for: {cfg_name} ({cfg_id})")
                continue

            bar = progress.progress_bar(
                wav_files,
                total=len(wav_files),
                desc=f"{cfg_name} ({cfg_id}): Diarizing",
                unit="file",
                leave=True,
                ten_percent_ticks=True,
            )

            for wav_file in wav_files:
                bar.set_description(f"{cfg_name} ({cfg_id}): {wav_file.stem}")
                self.diarize_file(wav_file, channel_data_dir, host_count=len(cfg.hosts))
                bar.update(1)

            bar.close()


def main():
    diarizer = Diarizer()
    diarizer.batch()


if __name__ == "__main__":
    main()
