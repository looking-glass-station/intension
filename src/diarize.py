import contextlib
import io
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

import torch
from tqdm_sound import TqdmSound
from whisperx.diarize import DiarizationPipeline
from pyannote.audio import Pipeline

import system_config
from configs import get_global_config, iter_processing_configs, resolve_diarization_backend
from file_utils import filter_files_by_stems, audacity_writer
from logger import global_logger

# Isolated NeMo env for the Sortformer streaming backend (see tools/nemo_diarize/
# and benchmarks/diarization/NOTES.md). NeMo force-upgrades torch/transformers, so
# it can't share this env - it runs as a subprocess.
NEMO_DIR = Path(__file__).resolve().parent.parent / "tools" / "nemo_diarize"
NEMO_SCRIPT = NEMO_DIR / "nemo_diarize.py"
NEMO_MODEL = NEMO_DIR / "models" / "diar_streaming_sortformer_4spk-v2.1.nemo"
# Tuned config from Phase 2: high-latency streaming preset + CallHome post-processing.
NEMO_SORTFORMER_ARGS = ["--mode", "sortformer-streaming", "--postprocessing", "callhome"]


def _nemo_python() -> Optional[Path]:
    py = NEMO_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return py if py.exists() else None


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
        self.default_backend = resolve_diarization_backend()

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


    @staticmethod
    def _parse_rttm_file(rttm_path: Path) -> List[Dict[str, float]]:
        """
        Parse an RTTM into start/end/label dicts. Parsed from the right because
        NeMo writes unescaped spaces into the file id.
        """
        segments: List[Dict[str, float]] = []
        for line in rttm_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 10 and p[0] == "SPEAKER":
                start, dur, label = float(p[-7]), float(p[-6]), p[-3]
                segments.append({"start": start, "end": start + dur, "label": label})
        return segments

    def _finalize_segments(
            self,
            wav_file: Path,
            rttm_path: Path,
            audacity_path: Path,
            segments: List[Dict[str, float]],
    ) -> None:
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

    def _sortformer_available(self) -> bool:
        py = _nemo_python()
        if py is None or not NEMO_SCRIPT.exists():
            return False
        return NEMO_MODEL.exists()

    def _resolve_backend(self, backend: Optional[str]) -> str:
        backend = (backend or self.default_backend or "pyannote").lower()
        if backend == "sortformer" and not self._sortformer_available():
            self.logger.warning(
                "diarization_backend=sortformer but the NeMo env/model is missing "
                f"({NEMO_DIR}); falling back to pyannote."
            )
            return "pyannote"
        return backend

    def _run_nemo_sortformer(self, wav_files: List[Path], staging_dir: Path) -> Dict[str, Path]:
        """
        Run the Sortformer streaming worker over wav_files in one subprocess.
        Returns a {stem: staged_rttm_path} map for files it produced.
        """
        staging_dir.mkdir(parents=True, exist_ok=True)
        summary_path = staging_dir / "_summary.json"
        cmd = [
            str(_nemo_python()), str(NEMO_SCRIPT),
            *NEMO_SORTFORMER_ARGS,
            "--out-dir", str(staging_dir),
            "--json-out", str(summary_path),
            "--device", "auto",
            "--audio", *[str(p) for p in wav_files],
        ]
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        self.logger.info(
            f"Running NeMo Sortformer worker over {len(wav_files)} file(s) "
            f"(one process; NeMo import ~2 min)"
        )
        proc = subprocess.run(cmd, env=env)
        produced: Dict[str, Path] = {}
        for wav in wav_files:
            staged = staging_dir / f"{wav.stem}.rttm"
            if staged.exists():
                produced[wav.stem] = staged
            else:
                self.logger.error(
                    f"NeMo Sortformer produced no RTTM for {wav.name} (rc={proc.returncode})"
                )
        return produced

    # Cap wavs per NeMo subprocess: keeps the argv well under the Windows command
    # line limit; the ~2 min model load is re-paid per chunk (rare batch is >50).
    NEMO_BATCH_CHUNK = 50

    def _diarize_sortformer_batch(self, jobs: List[tuple]) -> List[Path]:
        """
        jobs: list of (wav_file, output_dir). Runs the NeMo subprocess over the
        group (chunked), then distributes + finalizes each RTTM. Returns wavs left
        undone (worker failure) so the caller can fall back to pyannote.
        """
        undone: List[Path] = []
        for i in range(0, len(jobs), self.NEMO_BATCH_CHUNK):
            chunk = jobs[i:i + self.NEMO_BATCH_CHUNK]
            with tempfile.TemporaryDirectory(prefix="intension_sortformer_") as tmp:
                staging = Path(tmp)
                produced = self._run_nemo_sortformer([w for w, _ in chunk], staging)
                for wav_file, output_dir in chunk:
                    staged = produced.get(wav_file.stem)
                    if staged is None:
                        undone.append(wav_file)
                        continue
                    rttm_path = output_dir / "rttm" / f"{wav_file.stem}.rttm"
                    audacity_path = output_dir / "audacity_labels" / f"{wav_file.stem}.txt"
                    rttm_path.parent.mkdir(parents=True, exist_ok=True)
                    audacity_path.parent.mkdir(parents=True, exist_ok=True)
                    self._finalize_segments(
                        wav_file, rttm_path, audacity_path, self._parse_rttm_file(staged)
                    )
        return undone

    def diarize_file(self, wav_file: Path, output_dir: Path = None, host_count: Optional[int] = None,
                     backend: Optional[str] = None):
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

        resolved = self._resolve_backend(backend)
        if resolved == "sortformer":
            undone = self._diarize_sortformer_batch([(wav_file, output_dir)])
            if not undone:
                return
            self.logger.warning(f"Sortformer failed for {wav_file.name}; using pyannote")

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
        self._finalize_segments(wav_file, rttm_path, audacity_path, segments)

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

            backend = self._resolve_backend(getattr(cfg, "diarization_backend", None))

            bar = progress.progress_bar(
                wav_files,
                total=len(wav_files),
                desc=f"{cfg_name} ({cfg_id}): Diarizing ({backend})",
                unit="file",
                leave=True,
                ten_percent_ticks=True,
            )

            if backend == "sortformer":
                # One NeMo subprocess for the whole config (import alone costs ~2 min).
                jobs = [(wav_file, channel_data_dir) for wav_file in wav_files]
                undone = self._diarize_sortformer_batch(jobs)
                bar.update(len(wav_files) - len(undone))
                for wav_file in undone:
                    bar.set_description(f"{cfg_name} ({cfg_id}): {wav_file.stem} (pyannote)")
                    self.diarize_file(wav_file, channel_data_dir,
                                      host_count=len(cfg.hosts), backend="pyannote")
                    bar.update(1)
            else:
                for wav_file in wav_files:
                    bar.set_description(f"{cfg_name} ({cfg_id}): {wav_file.stem}")
                    self.diarize_file(wav_file, channel_data_dir,
                                      host_count=len(cfg.hosts), backend=backend)
                    bar.update(1)

            bar.close()


def main():
    diarizer = Diarizer()
    diarizer.batch()


if __name__ == "__main__":
    main()
