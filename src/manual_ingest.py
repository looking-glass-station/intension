from pathlib import Path

from configs import get_manual_configs, get_global_config
from file_utils import get_audio_files
from logger import global_logger
from to_wav import convert_to_wav
from tqdm_sound import TqdmSound


def main():
    logger = global_logger("manual_ingest")
    global_config = get_global_config()
    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )

    manual_configs = get_manual_configs()
    if not manual_configs:
        print("[*] Manual ingest: no configs found.")
        return

    for cfg in manual_configs:
        base = cfg.output_path
        if base is None:
            continue

        raw_dir = base / "raw"
        wav_dir = base / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            logger.info(f"No manual raw dir for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        audio_files = get_audio_files(raw_dir)
        if not audio_files:
            logger.info(f"No manual raw audio for: {cfg.name} ({cfg.channel_name_or_term})")
            continue

        bar = progress.progress_bar(
            audio_files,
            total=len(audio_files),
            desc=f"{cfg.name} ({cfg.channel_name_or_term}) - manual ingest",
            unit="file",
            leave=True,
            ten_percent_ticks=True,
        )

        for src in bar:
            out = wav_dir / f"{src.stem}.wav"
            if out.exists():
                continue
            try:
                convert_to_wav(src, out)
                logger.info(f"Converted manual file to wav: {src.name}")
            except Exception as e:
                logger.error(f"Manual ingest failed for {src}: {e}")


if __name__ == "__main__":
    main()
