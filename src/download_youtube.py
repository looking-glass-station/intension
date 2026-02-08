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
JS_RUNTIME_DEFAULT = Path(r"C:\Program Files\nodejs\node.exe")
WINDOWS_USER_DENO = Path.home() / ".deno" / "bin" / "deno.exe"
EJS_ERROR_MARKERS = (
    "challenge solving failed",
    "only images are available",
    "challenge solver script distribution",
)
FORMAT_UNAVAILABLE_MARKERS = (
    "requested format is not available",
    "only images are available",
)


def ensure_js_runtime() -> str:
    """
    Ensure YT_DLP_JS is set to a valid JS runtime so EJS challenges can be solved.
    """
    js_env = os.environ.get("YT_DLP_JS")
    if js_env:
        js_path = Path(js_env)
        if js_path.exists():
            return str(js_path)
        logger.error(f"YT_DLP_JS is set but missing: {js_env}")
        raise RuntimeError(f"YT_DLP_JS path not found: {js_env}")

    deno_path = find_deno_runtime()
    if deno_path:
        os.environ["YT_DLP_JS"] = deno_path
        logger.info(f"YT_DLP_JS not set; defaulting to deno at {deno_path}")
        return deno_path

    if JS_RUNTIME_DEFAULT.exists():
        os.environ["YT_DLP_JS"] = str(JS_RUNTIME_DEFAULT)
        logger.info(f"YT_DLP_JS not set; defaulting to node at {JS_RUNTIME_DEFAULT}")
        return str(JS_RUNTIME_DEFAULT)

    raise RuntimeError(
        "YT_DLP_JS is not set and default Node runtime was not found. "
        "Install Node or set YT_DLP_JS to a valid JS interpreter."
    )


def build_js_runtime_config(js_runtime: str) -> Dict[str, Dict[str, str]]:
    """
    Build yt-dlp js_runtimes config dict from a runtime path.
    """
    runtime_path = Path(js_runtime)
    name = runtime_path.stem.lower()
    if "deno" in name:
        return {"deno": {"path": js_runtime}}
    if "node" in name:
        return {"node": {"path": js_runtime}}
    if "bun" in name:
        return {"bun": {"path": js_runtime}}
    if "qjs" in name or "quickjs" in name:
        return {"quickjs": {"path": js_runtime}}
    return {}


def find_deno_runtime() -> str:
    """
    Resolve a deno runtime path using env vars, common install locations, then project venv.
    """
    env_candidates = [
        os.environ.get("DENO_EXE"),
        os.environ.get("DENO"),
        os.environ.get("DENO_PATH"),
    ]
    for candidate in env_candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return str(path)

    deno_path = shutil.which("deno")
    if deno_path:
        return deno_path

    if WINDOWS_USER_DENO.exists():
        return str(WINDOWS_USER_DENO)

    project_root = Path(__file__).parent.parent
    venv_deno = project_root / ".venv" / "Scripts" / "deno.exe"
    if venv_deno.exists():
        return str(venv_deno)

    return ""


def is_ejs_error(msg: str) -> bool:
    msg = (msg or "").lower()
    return any(marker in msg for marker in EJS_ERROR_MARKERS)


def is_format_unavailable(msg: str) -> bool:
    msg = (msg or "").lower()
    return any(marker in msg for marker in FORMAT_UNAVAILABLE_MARKERS)


def parse_cookies_from(raw: str):
    """
    Parse cookies_from config. Supports "browser" or "browser:profile".
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if ":" in text:
        browser, profile = text.split(":", 1)
        browser = browser.strip()
        profile = profile.strip()
        if browser and profile:
            return (browser, profile)
    return (text,)


def select_best_audio_format_id(info: Dict[str, Any]) -> str:
    """
    Pick the highest bitrate audio-only format id.
    """
    formats = info.get("formats") or []
    audio_formats = [
        f for f in formats
        if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
    ]
    if not audio_formats:
        return ""

    def score(fmt: Dict[str, Any]) -> float:
        abr = fmt.get("abr") or 0
        tbr = fmt.get("tbr") or 0
        return float(abr or tbr or 0)

    best = max(audio_formats, key=score)
    return str(best.get("format_id") or "")


def find_downloaded_file(download_dir: Path, safe_title: str) -> Path:
    """
    Locate the most likely downloaded media file for a given title.
    """
    candidates = []
    for p in download_dir.glob(f"{safe_title}.*"):
        if p.suffix in {".part", ".ytdl", ".txt", ".json", ".description"}:
            continue
        if p.suffix == ".wav":
            continue
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"Downloaded file not found for {safe_title} in {download_dir}")
    preferred_exts = [".mp4", ".m4a", ".webm", ".mkv", ".mp3", ".opus"]
    for ext in preferred_exts:
        for p in candidates:
            if p.suffix.lower() == ext:
                return p
    return candidates[0]


def has_raw_download(download_dir: Path, safe_title: str) -> bool:
    """
    Check if any non-wav download exists for a given title.
    """
    for p in download_dir.glob(f"{safe_title}.*"):
        if p.suffix in {".part", ".ytdl", ".txt", ".json", ".description"}:
            continue
        if p.suffix == ".wav":
            continue
        return True
    return False


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
    cookies_spec = parse_cookies_from(global_conf.youtube.cookies_from or "")
    raw_title = entry['sanitize_title']
    safe_title = sanitize_filename(raw_title)
    js_runtime = ensure_js_runtime()
    url = entry.get("url") or f"https://www.youtube.com/watch?v={entry['video_id']}"
    watch_url = f"https://www.youtube.com/watch?v={entry['video_id']}"
    js_runtime_cfg = build_js_runtime_config(js_runtime)

    ydl_opts = {
        "outtmpl": str(download_dir / f"{safe_title}.%(ext)s"),
        "noplaylist": True,
        "quiet": False,  # keep stdout/stderr so EJS warnings are visible in logs
        "no_warnings": False,
        "logger": logging.getLogger("yt_dlp_silent"),
        "progress_hooks": [],
        "js_runtimes": js_runtime_cfg,
        "remote_components": ["ejs:github"],
        # Uncomment for diagnosing download issues:
        # "verbose": True,
    }
    if cookies_spec:
        # Enable authenticated/age-restricted downloads when cookies are available.
        ydl_opts["cookiesfrombrowser"] = cookies_spec

    if config.audio_only:
        file_format = "m4a"
        ydl_opts["format"] = "bestaudio[ext=m4a]/bestaudio"
    else:
        file_format = "mp4"
        ydl_opts["format"] = (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/"
            "bestvideo[ext=webm]+bestaudio[ext=webm]/webm/"
            "best"
        )

    download_file = download_dir / f"{safe_title}.{file_format}"
    wav_file = config.output_path / "wav" / f"{safe_title}.wav"

    def perform_download(opts: Dict[str, Any], target_url: str):
        buf = io.StringIO()
        try:
            with YoutubeDL(opts) as ydl, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ydl.download([target_url])
            return True, buf.getvalue().strip(), None
        except Exception as exc:
            return False, buf.getvalue().strip(), exc

    # Download if neither raw nor WAV exists
    if not has_raw_download(download_dir, safe_title) and not wav_file.exists():
        if not config.get_static_videos:
            buf = io.StringIO()
            try:
                with YoutubeDL(ydl_opts) as ydl, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    info = ydl.extract_info(url, download=False)
                if video_is_static_from_info(info):
                    logger.info(f"Skipping static video: {entry['title']}")
                    return
            except Exception:
                logger.error(f"yt-dlp extract_info error for {raw_title}: {buf.getvalue().strip()}")
                raise

        ok, output, err = perform_download(ydl_opts, watch_url)
        if not ok and is_ejs_error(output):
            if cookies_spec:
                logger.warning(
                    f"EJS error detected for {raw_title}; cookies are enabled, skipping android/ios fallback. "
                    "Check JS runtime or update yt-dlp."
                )
            else:
                alt_opts = dict(ydl_opts)
                alt_opts["extractor_args"] = {"youtube": {"player_client": ["android"]}}
                alt_opts.pop("cookiesfrombrowser", None)  # android client cannot use cookies
                logger.warning(f"EJS error detected for {raw_title}; retrying with android client")
                ok, output, err = perform_download(alt_opts, watch_url)
                if ok:
                    logger.info(f"Retried with android client: {raw_title}")
                if not ok and is_ejs_error(output):
                    # Second fallback: iOS client without cookies
                    alt_opts = dict(ydl_opts)
                    alt_opts["extractor_args"] = {"youtube": {"player_client": ["ios"]}}
                    alt_opts.pop("cookiesfrombrowser", None)
                    logger.warning(f"EJS error persists for {raw_title}; retrying with ios client")
                    ok, output, err = perform_download(alt_opts, watch_url)
                    if ok:
                        logger.info(f"Retried with ios client: {raw_title}")

        if not ok and is_format_unavailable(output):
            # As a last resort, fall back to the best available audio-only stream.
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(watch_url, download=False)
                best_audio_id = select_best_audio_format_id(info)
                if best_audio_id:
                    alt_opts = dict(ydl_opts)
                    alt_opts["format"] = best_audio_id
                    logger.warning(
                        f"Requested mp4 format unavailable for {raw_title}; "
                        f"falling back to audio-only format {best_audio_id}"
                    )
                    ok, output, err = perform_download(alt_opts, watch_url)
            except Exception:
                pass

        if not ok:
            logger.error(f"yt-dlp download error for {raw_title}: {output}")
            raise err
        download_file = find_downloaded_file(download_dir, safe_title)

    # Convert to WAV if not already done
    if not wav_file.exists():
        buf = io.StringIO()
        try:
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if not download_file.exists():
                    download_file = find_downloaded_file(download_dir, safe_title)
                convert_to_wav(download_file, wav_file)
            logger.info(f"Converted to wav: {wav_file}")
        except Exception:
            err_output = buf.getvalue().strip()
            logger.error(f"Error converting {download_file}: {err_output or Exception}")
            raise

    # Remove original file if global config says so
    if not global_conf.keep_raw_downloads:
        if not download_file.exists():
            download_file = find_downloaded_file(download_dir, safe_title)
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
    ensure_js_runtime()

    # Extract video info to get the title
    global_conf = get_global_config()
    cookies_spec = parse_cookies_from(global_conf.youtube.cookies_from or "")
    ydl_info_opts = {"quiet": True, "no_warnings": True, "remote_components": ["ejs:github"]}
    if cookies_spec:
        ydl_info_opts["cookiesfrombrowser"] = cookies_spec
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
    if cookies_spec:
        ydl_download_opts["cookiesfrombrowser"] = cookies_spec

    with YoutubeDL(ydl_download_opts) as ydl:
        ydl.download([url])

    # Convert to WAV
    source_file = find_downloaded_file(download_dir, safe_title)
    wav_file = download_dir / f"{safe_title}.wav"
    convert_to_wav(source_file, wav_file)

    return wav_file


def main():
    """
    Iterate over parsed YouTube metadata, download new videos (with retry on rate limits), and convert to WAV.
    """
    global_config = get_global_config()
    js_runtime = ensure_js_runtime()
    logger.info(f"Using JS runtime: {js_runtime}")
    cookies_spec = parse_cookies_from(global_config.youtube.cookies_from or "")
    if cookies_spec:
        if len(cookies_spec) == 2:
            logger.info(f"Using browser cookies: {cookies_spec[0]} (profile {cookies_spec[1]})")
        else:
            logger.info(f"Using browser cookies: {cookies_spec[0]}")

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

                downloaded = (
                    {p.stem for p in download_dir.glob("*.mp4")}
                    | {p.stem for p in download_dir.glob("*.m4a")}
                    | {p.stem for p in download_dir.glob("*.webm")}
                )
                wav_files = {p.stem for p in wav_dir.glob("*.wav")}

                metadata = cfg.output_path / 'metadata' / 'metadata_parsed.csv'
                if not metadata.exists():
                    continue

                metadata_records = csv_to_dict(metadata)
                if not metadata_records:
                    print(f"{cfg.name} ({cfg.channel_name_or_term}): No records to process.")
                    continue

                to_process = [
                    r for r in sorted(metadata_records,
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
