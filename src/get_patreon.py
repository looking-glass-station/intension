import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import aiohttp
from yarl import URL
from pathvalidate import sanitize_filename
from tqdm_sound import TqdmSound

from configs import get_patreon_configs, get_global_config
from file_utils import csv_to_dict, dict_to_csv
from logger import global_logger
from patreon_dl.data import parse_included, reify_relationships_many
from patreon_utils import (
    get_cookie,
    parse_cookie_string,
    has_auth_cookie,
    vanity_from_config,
    make_base_urls,
    fetch_campaign_id,
    posts_generator,
    should_skip_post,
    pick_download_url,
    pick_filename,
    parse_post_duration_seconds,
    is_patreon_hosted,
)

logger = global_logger("get_patreon")


def build_metadata_rows(posts: List[Dict], cfg) -> List[Dict]:
    rows: List[Dict] = []
    skipped = {
        "not_viewable": 0,
        "filters": 0,
        "min_length": 0,
        "post_type": 0,
        "audio_only": 0,
        "no_url": 0,
        "not_patreon_hosted": 0,
    }
    audio_post_types = {"audio_file", "podcast"}
    downloadable_post_types = {"audio_file", "podcast", "video_external_file"}

    for post in posts:
        attrs = post.get("attributes") or {}
        post_type = (attrs.get("post_type") or "").strip()
        if not attrs.get("current_user_can_view", True):
            skipped["not_viewable"] += 1
            continue

        title = (attrs.get("title") or "").strip()
        if cfg.must_contain and not all(x.lower() in title.lower() for x in cfg.must_contain):
            skipped["filters"] += 1
            continue
        if cfg.must_exclude and any(x.lower() in title.lower() for x in cfg.must_exclude):
            skipped["filters"] += 1
            continue
        if cfg.max_date:
            try:
                max_date = datetime.strptime(cfg.max_date, "%Y-%m-%d")
                published = datetime.fromisoformat(attrs.get("published_at"))
                if published <= max_date:
                    skipped["filters"] += 1
                    continue
            except Exception:
                pass

        if cfg.min_length_mins:
            duration = parse_post_duration_seconds(post)
            if duration is not None and duration < cfg.min_length_mins * 60:
                skipped["min_length"] += 1
                continue

        if post_type == "video_embed":
            skipped["post_type"] += 1
            continue
        if cfg.audio_only and post_type not in audio_post_types:
            skipped["audio_only"] += 1
            continue
        if post_type not in downloadable_post_types:
            skipped["post_type"] += 1
            continue

        url = pick_download_url(post)
        if not url:
            skipped["no_url"] += 1
            continue
        if not is_patreon_hosted(url):
            skipped["not_patreon_hosted"] += 1
            continue

        attrs = post.get("attributes") or {}
        title = (attrs.get("title") or "untitled").strip()
        post_id = str(post.get("id") or "unknown")
        published_at = attrs.get("published_at") or ""

        date_prefix = "unknown-date"
        try:
            dttm = datetime.fromisoformat(published_at)
            date_prefix = dttm.strftime("%Y-%m-%d")
        except Exception:
            pass

        safe_title = sanitize_filename(title) or "untitled"
        file_stem = f"{date_prefix}_{safe_title}_{post_id}"
        filename = pick_filename(post)
        duration = parse_post_duration_seconds(post)

        rows.append({
            "post_id": post_id,
            "title": title,
            "sanitize_title": safe_title,
            "file_stem": file_stem,
            "published_at": published_at,
            "post_type": attrs.get("post_type"),
            "url": attrs.get("patreon_url") or attrs.get("url") or "",
            "download_url": url,
            "filename": filename,
            "duration": duration or 0,
        })

    if skipped:
        logger.info(
            f"[PATREON] {cfg.name} ({cfg.channel_name_or_term}) skipped counts: "
            + ", ".join(f"{k}={v}" for k, v in skipped.items())
        )
        not_viewable = skipped.get("not_viewable", 0)
        total = len(posts)
        if total > 0 and (not_viewable / total) >= 0.5:
            logger.warning(
                f"[PATREON] {cfg.name} ({cfg.channel_name_or_term}) access-limited: "
                f"{not_viewable}/{total} posts are not viewable by current session."
            )
    return rows


async def process_campaign(cfg, session: aiohttp.ClientSession, progress: TqdmSound):
    vanity = vanity_from_config(cfg)
    if not vanity:
        logger.error(f"Missing Patreon vanity/URL for {cfg.name} ({cfg.channel_name_or_term})")
        return

    campaign_id = (getattr(cfg, "campaign_id", None) or "").strip() or None
    base_url = None

    # If campaign_id is configured, avoid scraping creator HTML (often blocked by Cloudflare).
    if campaign_id:
        base_url = make_base_urls()[0]
    else:
        for candidate in make_base_urls():
            try:
                campaign_id = await fetch_campaign_id(session, vanity, candidate)
                if campaign_id:
                    base_url = candidate
                    break
            except aiohttp.ClientConnectorDNSError as e:
                logger.warning(f"DNS failed for {candidate}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Campaign lookup failed for {candidate}: {e}")
                continue

    if not campaign_id or not base_url:
        logger.error(f"Could not resolve campaign id for {vanity}")
        return

    base = cfg.output_path
    raw_dir = base / "raw"
    wav_dir = base / "wav"
    meta_dir = base / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    posts: List[Dict] = []
    async for data in posts_generator(session, campaign_id, base_url):
        if "included" not in data:
            posts.extend(data.get("data") or [])
            continue
        try:
            included = parse_included(data)
            reified = reify_relationships_many(data.get("data") or [], included)
            posts.extend(reified or [])
        except KeyError as e:
            logger.warning(f"[PATREON] skipping reification due to missing key: {e}")
            posts.extend(data.get("data") or [])

    if not posts:
        logger.info(f"No Patreon posts for {cfg.name} ({cfg.channel_name_or_term})")
        return
    logger.info(f"[PATREON] {cfg.name} ({cfg.channel_name_or_term}) total posts fetched: {len(posts)}")

    bar = progress.progress_bar(
        posts,
        total=len(posts),
        desc=f"{cfg.name} ({cfg.channel_name_or_term}) Patreon - metadata",
        unit="post",
        leave=True,
        ten_percent_ticks=True,
    )
    rows = build_metadata_rows(list(bar), cfg)
    bar.close()

    raw_csv = meta_dir / "metadata_raw.csv"
    if raw_csv.exists() and not cfg.overwrite:
        existing = csv_to_dict(raw_csv) or []
        deduped_existing = []
        seen_existing_ids = set()
        for rec in existing:
            rec_id = str(rec.get("post_id") or "")
            if rec_id and rec_id in seen_existing_ids:
                continue
            if rec_id:
                seen_existing_ids.add(rec_id)
            deduped_existing.append(rec)
        existing_ids = {str(r.get("post_id")) for r in deduped_existing if r.get("post_id")}
        rows = deduped_existing + [r for r in rows if str(r.get("post_id")) not in existing_ids]

    rows.sort(key=lambda r: r.get("published_at") or "")
    dict_to_csv(raw_csv, rows)
    logger.info(f"[PATREON] metadata_raw.csv written to {raw_csv} ({len(rows)} rows)")


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
                f"Provided keys: {keys or '(none)'}. Patron-only posts will be unavailable."
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
            await process_campaign(cfg, session, progress)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
