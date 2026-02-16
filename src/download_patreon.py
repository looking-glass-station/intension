import asyncio
from pathlib import Path
from typing import Dict, List

import aiohttp
from yarl import URL
from tqdm_sound import TqdmSound

from configs import get_global_config, get_patreon_configs
from file_utils import csv_to_dict
from logger import global_logger
from patreon_utils import get_cookie, parse_cookie_string, has_auth_cookie
from to_wav import convert_to_wav

logger = global_logger("download_patreon")


async def download_to_path(session: aiohttp.ClientSession, url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    async with session.get(url) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                f.write(chunk)
    tmp_path.replace(target)


async def process_campaign(cfg, session: aiohttp.ClientSession, progress: TqdmSound, global_config):
    raw_dir = cfg.input_path / "raw"
    wav_dir = cfg.output_path / "wav"
    meta_dir = cfg.output_path / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = meta_dir / "metadata_raw.csv"
    if not meta_csv.exists():
        logger.error(f"Patreon metadata missing: {meta_csv}")
        return

    records = csv_to_dict(meta_csv) or []
    if not records:
        logger.info(f"No Patreon records to download for {cfg.name} ({cfg.channel_name_or_term})")
        return

    bar = progress.progress_bar(
        records,
        total=len(records),
        desc=f"{cfg.name} ({cfg.channel_name_or_term}) Patreon",
        unit="post",
        leave=True,
        ten_percent_ticks=True,
    )

    for rec in bar:
        url = rec.get("download_url")
        if not url:
            continue

        stem = rec.get("file_stem") or rec.get("sanitize_title") or "patreon_file"
        filename = rec.get("filename") or ""
        suffix = Path(filename).suffix or ".bin"
        raw_path = raw_dir / f"{stem}{suffix}"
        wav_path = wav_dir / f"{stem}.wav"

        if wav_path.exists() and not cfg.overwrite:
            continue

        try:
            await download_to_path(session, url, raw_path)
            convert_to_wav(raw_path, wav_path)
            if not global_config.keep_raw_downloads and raw_path.exists():
                raw_path.unlink()
        except Exception as e:
            title = rec.get("title") or stem
            logger.error(f"Patreon download failed for {title}: {e}")


async def run():
    global_config = get_global_config()
    cookie = get_cookie(global_config)
    cookie_dict = parse_cookie_string(cookie) if cookie else {}

    jar = aiohttp.CookieJar()
    if cookie_dict:
        jar.update_cookies(cookie_dict, response_url=URL("https://www.patreon.com/"))
        if not has_auth_cookie(cookie_dict):
            keys = ", ".join(sorted(cookie_dict.keys()))
            logger.warning(
                "PATREON cookie found but missing authenticated session keys "
                "(expected one of: session_id, session_id.sig, __Secure-next-auth.session-token). "
                f"Provided keys: {keys or '(none)'}. Patron-only downloads will be unavailable."
            )
    else:
        logger.warning("PATREON cookie not found; only public posts may be available.")

    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(cookie_jar=jar, headers=headers) as session:
        progress = TqdmSound(
            activity_mute_seconds=0,
            dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
        )
        for cfg in get_patreon_configs():
            await process_campaign(cfg, session, progress, global_config)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
