import contextlib
import io
import sys
from pathlib import Path

import torch
from tqdm_sound import TqdmSound
from whisperx.diarize import DiarizationPipeline

import system_config
from configs import get_global_config, get_configs
from file_utils import filter_files_by_stems, audacity_writer
from logger import global_logger


class Diarizer:
    """
    Handles diarization for single files or in batch, manages model internally.
    """

    def __init__(self):
        self.logger = global_logger("diarization")
        self.global_config = get_global_config()
        self.model = self._load_model()

    def _load_model(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = DiarizationPipeline(
                use_auth_token=self.global_config.hf_token,
                device=system_config.device
            )
            try:
                model.model = model.model.half()
            except Exception:
                pass
            if sys.platform != "win32":
                try:
                    model.model = torch.compile(model.model, mode="reduce-overhead")
                except Exception:
                    pass
        return model

    def diarize_file(self, wav_file: Path, output_dir: Path = None):
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

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = self.model(str(wav_file))

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
        escaped_name = wav_file.name.replace(" ", "_")
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
        self.logger.info(f"Diarized: {wav_file.name}")

    def batch(self):
        """
        Runs diarization on all files from configs.
        """
        progress = TqdmSound(
            activity_mute_seconds=0,
            dynamic_settings_file=str(self.global_config.project_root / "confs" / "sound.json")
        )

        for channel_config in get_configs():
            for config_list in vars(channel_config.download_configs).values():
                if not isinstance(config_list, list) or not config_list:
                    continue

                for cfg in config_list:
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
                        self.diarize_file(wav_file, channel_data_dir)
                        bar.update(1)

                    bar.close()


def main():
    diarizer = Diarizer()
    diarizer.batch()


if __name__ == "__main__":
    main()
