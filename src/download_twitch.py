import contextlib
import io
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from tqdm_sound import TqdmSound
from yt_dlp import YoutubeDL

from configs import get_global_config, get_twitch_configs
from get_twitch import parse_all_raw_metadata
from logger import global_logger
from system_config import max_workers
from to_wav import convert_to_wav
from videos import video_is_static_from_info

logger = global_logger("download_twitch")


def download_video(
        entry: Dict[str, Any],
        config: Any,
        download_dir: Path,
        bar: Optional[TqdmSound.progress_bar] = None,
        base_desc: str = ""
) -> None:
    """
    Download a Twitch video or audio, skip static videos if requested, convert to WAV, and handle errors.

    Args:
        entry (Dict[str, Any]): Metadata row with 'url', 'video_id', and 'sanitize_title'.
        config (Any): TwitchConfig instance including audio_only, get_static_videos, and output_path.
        download_dir (Path): Directory path where video/audio files are saved.
        bar (Optional): TqdmSound progress bar instance for updating status.
        base_desc (str): Base description for progress bar status.

    Raises:
        Exception: Propagates download or conversion errors after logging.
    """
    global_conf = get_global_config()
    sanitize_title = entry.get("sanitize_title")
    file_format = "m4a" if config.audio_only else "mp4"
    output_file = download_dir / f"{sanitize_title}.{file_format}"
    wav_file = config.output_path / "wav" / f"{sanitize_title}.wav"

    # Skip static videos unless requested
    if not output_file.exists() and not config.get_static_videos:
        buf = io.StringIO()
        ydl = YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True})
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                info = ydl.extract_info(entry["url"], download=False)
            if video_is_static_from_info(info):
                logger.info(f"Skipping static content: {sanitize_title}")
                return

        except Exception:
            pass

    # Download video if not already present
    if not output_file.exists():
        slug = entry["url"].split("/")[-1] or ""
        cmd = [
            "twitch-dl", "download", slug,
            "--output", str(output_file),
            "--format", file_format,
            "--quality", "source",
            "--max-workers", str(max_workers(absolute_count=4)),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        summary_printed = False
        for line in proc.stdout:
            if line.startswith("Downloaded "):
                if bar:
                    # truncate long titles for display in console
                    title = sanitize_title if len(sanitize_title) <= 40 else sanitize_title[:40] + '...'
                    bar.set_description(f"{base_desc} - {title} - ({line.strip()})")

                summary_printed = True

        proc.wait()

        if proc.returncode != 0:
            # Check for not found errors
            out_lower = (proc.stdout.read() if proc.stdout else "").lower()
            if "404" in out_lower or "not found" in out_lower:
                logger.error(f"Video slug '{slug}' not found: skipping")
                return

            logger.error(f"twitch-dl CLI error for {sanitize_title}: exit code {proc.returncode}")
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        if not summary_printed:
            logger.warning(f"No download summary for slug '{slug}' (stdout was empty)")

    # Convert to WAV if not already present
    if not wav_file.exists():
        buf = io.StringIO()
        try:
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                convert_to_wav(output_file, wav_file)

            logger.info(f"Converted to wav: {wav_file}")

        except Exception:
            logger.error(f"Error converting {output_file}: {buf.getvalue().strip()}")
            raise

    # Optionally cleanup raw file
    if not global_conf.keep_raw_downloads:
        output_file.unlink(missing_ok=True)

    logger.info(f"Completed: {output_file}")


def main():
    """
    Download and process Twitch videos/clips for each configured channel.
    Updates progress bar, skips processed items, checks disk space, and retries on rate limit.
    """
    parsed_entries = parse_all_raw_metadata(write_files=False)
    configs = get_twitch_configs()
    global_conf = get_global_config()

    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_conf.project_root / "confs" / "sound.json")
    )

    while True:
        rate_limit = False
        for cfg in configs:
            entries = [e for e in parsed_entries if e.get("config_name") == cfg.channel_name_or_term]
            file_format = "m4a" if cfg.audio_only else "mp4"
            download_dir = cfg.input_path / "downloads"
            wav_dir = cfg.output_path / "wav"
            download_dir.mkdir(parents=True, exist_ok=True)
            wav_dir.mkdir(parents=True, exist_ok=True)

            # Check disk space
            if shutil.disk_usage(str(download_dir)).free < global_conf.min_free_disk_space_gb * 1024 ** 3:
                logger.error("Low disk space, aborting.")
                sys.exit(1)

            downloaded = {p.stem for p in download_dir.glob(f"*.{file_format}")}
            downloaded |= {p.stem for p in wav_dir.glob("*.wav")}
            to_download = [e for e in entries if e.get("sanitize_title") not in downloaded]

            if not to_download:
                logger.info(f"Nothing new for {cfg.channel_name_or_term}")
                continue

            base_desc = f"{cfg.name} ({cfg.channel_name_or_term}) twitch"
            bar = progress.progress_bar(
                range(len(to_download)),
                total=len(to_download),
                desc=base_desc,
                unit="item",
                leave=True,
                position=0,
                ten_percent_ticks=True,
            )

            for i, entry in enumerate(to_download):
                try:
                    download_video(entry, cfg, download_dir, bar, base_desc)
                except Exception as ex:
                    if "rate limit" in str(ex).lower():
                        logger.error("Rate limited; sleeping 10m")
                        time.sleep(600)
                        rate_limit = True
                        break
                finally:
                    bar.update(1)

            bar.close()
            if rate_limit:
                break

        if not rate_limit:
            break


if __name__ == "__main__":
    main()
