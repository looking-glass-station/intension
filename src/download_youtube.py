import concurrent.futures
import contextlib
import io
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

from pathvalidate import sanitize_filename
from tqdm_sound import TqdmSound
from yt_dlp import YoutubeDL

from configs import get_global_config, get_configs
from file_utils import csv_to_dict
from logger import global_logger
from system_config import max_workers
from to_wav import convert_to_wav
from videos import video_is_static_from_info

logger = global_logger('download_youtube')


def download_video(entry: Dict[str, Any], config: Any, download_dir: Path) -> None:
    """
    Download a YouTube video (or audio-only), optionally skip static videos, convert to WAV, and log errors.

    Args:
        entry: Metadata row with 'url', 'video_id', and 'sanitize_title'.
        config: YouTubeConfig instance (audio_only, get_static_videos, output_path, etc).
        download_dir: Path to directory for saving video/audio files.

    Raises:
        Exception: Propagates download or conversion errors after logging stderr output.
    """
    global_conf = get_global_config()
    browser = global_conf.youtube.cookies_from
    raw_title = entry['sanitize_title']
    safe_title = sanitize_filename(raw_title)

    ydl_opts = {
        "outtmpl": str(download_dir / f"{safe_title}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": (browser,),
        "logger": logging.getLogger("yt_dlp_silent"),
        "progress_hooks": [],
    }

    if config.audio_only:
        file_format = "m4a"
        ydl_opts["format"] = "bestaudio[ext=m4a]/bestaudio"
    else:
        file_format = "mp4"
        ydl_opts["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"

    download_file = download_dir / f"{safe_title}.{file_format}"
    wav_file = config.output_path / "wav" / f"{safe_title}.wav"

    # Download if neither raw nor WAV exists
    if not download_file.exists() and not wav_file.exists():
        with YoutubeDL(ydl_opts) as ydl:
            if not config.get_static_videos:
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        info = ydl.extract_info(entry["url"], download=False)
                except Exception:
                    logger.error(f"yt-dlp extract_info error for {raw_title}: {buf.getvalue().strip()}")
                    raise
                if video_is_static_from_info(info):
                    logger.info(f"Skipping static video: {entry['title']}")
                    return

            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    ydl.download([f"https://www.youtube.com/watch?v={entry['video_id']}"])
            except Exception:
                logger.error(f"yt-dlp download error for {raw_title}: {buf.getvalue().strip()}")
                raise

    # Convert to WAV if not already done
    if not wav_file.exists():
        buf = io.StringIO()
        try:
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                convert_to_wav(download_file, wav_file)
            logger.info(f"Converted to wav: {wav_file}")
        except Exception:
            err_output = buf.getvalue().strip()
            logger.error(f"Error converting {download_file}: {err_output or Exception}")
            raise

    # Remove original file if global config says so
    if not global_conf.keep_raw_downloads:
        os.unlink(download_file)

    logger.info(f"Completed video: {download_file}")


def download_and_convert_by_id(video_str: str, download_dir: Union[str, Path]) -> Path:
    """
    Download a YouTube video by ID and convert it to WAV format.

    Args:
        video_str: The YouTube video ID (or URL).
        download_dir: Directory where video and WAV files will be stored.

    Returns:
        Path to generated WAV file.

    Raises:
        Exception: If download or conversion fails.
    """
    url = 'https://www.youtube.com/watch?v=' + video_str.split('v=', 1)[1].split('&', 1)[0] \
        if 'v=' in video_str else video_str

    if isinstance(download_dir, str):
        download_dir = Path(download_dir)

    # Extract video info to get the title
    ydl_info_opts = {"quiet": True, "no_warnings": True}
    with YoutubeDL(ydl_info_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get('title', video_str)
    safe_title = sanitize_filename(title)

    # Download audio-only format
    download_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(download_dir / f"{safe_title}.%(ext)s")
    ydl_download_opts = {
        "outtmpl": outtmpl,
        "format": "bestaudio[ext=m4a]/bestaudio",
        "quiet": True,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_download_opts) as ydl:
        ydl.download([url])

    # Convert to WAV
    source_file = download_dir / f"{safe_title}.m4a"
    wav_file = download_dir / f"{safe_title}.wav"
    convert_to_wav(source_file, wav_file)

    return wav_file


def main():
    """
    Iterate over parsed YouTube metadata, download new videos (with retry on rate limits), and convert to WAV.
    """
    global_config = get_global_config()

    # TqdmSound progress bar as in get_youtube.py
    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )

    for channel_config in get_configs():
        for config_list in vars(channel_config.download_configs).values():
            if not isinstance(config_list, list) or not config_list:
                continue

            for cfg in config_list:
                download_dir = cfg.input_path / "downloads"
                wav_dir = cfg.output_path / "wav"

                download_dir.mkdir(parents=True, exist_ok=True)
                wav_dir.mkdir(parents=True, exist_ok=True)

                free_bytes = shutil.disk_usage(str(download_dir)).free
                if free_bytes < global_config.min_free_disk_space_gb * 1024 ** 3:
                    logger.error(f"<{global_config.min_free_disk_space_gb}GB left, aborting downloads")
                    sys.exit(1)

                (download_dir / 'BACKUP.ignore').touch()
                (wav_dir / 'BACKUP.ignore').touch()

                downloaded = {p.stem for p in download_dir.glob("*.mp4")} | {p.stem for p in
                                                                             download_dir.glob("*.m4a")}
                wav_files = {p.stem for p in wav_dir.glob("*.wav")}

                metadata = cfg.output_path / 'metadata' / 'metadata_parsed.csv'
                if not metadata.exists():
                    continue

                to_process = [
                    r for r in sorted(csv_to_dict(metadata),
                                      key=lambda r: datetime.fromisoformat(r.get("date_uploaded", "1970-01-01")),
                                      reverse=True)
                    if r['sanitize_title'] not in (downloaded | wav_files)
                ]

                if not to_process:
                    logger.info(f"Nothing to process for {cfg.channel_name_or_term}")

                    continue

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers(absolute_count=1)) as executor:
                    futures = {
                        executor.submit(download_video, row, cfg, download_dir): row
                        for row in to_process
                    }

                    bar = progress.progress_bar(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=f"{cfg.name} ({cfg.channel_name_or_term}): ",
                        unit="video",
                        leave=True,
                        ten_percent_ticks=True,
                    )
                    for future in bar:
                        row = futures[future]
                        title = row.get('sanitize_title', '<unknown>')
                        bar.set_description(f"{cfg.name} ({cfg.channel_name_or_term}): {title}")

                        try:
                            future.result()
                        except Exception as e:
                            if "rate-limited by YouTube" in str(e):
                                logger.error("YouTube rate limit reached; sleeping for 80m")
                                time.sleep(80 * 60)
                                break
                            else:
                                logger.error(f"Error {title}: {e}")


if __name__ == "__main__":
    main()
