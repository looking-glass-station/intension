import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiohttp

from logger import global_logger
from patreon_dl import api

logger = global_logger("patreon_utils")


def get_cookie(global_config) -> str:
    cookie = os.environ.get("PATREON_COOKIE", "").strip()
    if cookie:
        return cookie

    # Optional local file override (keep secrets out of JSON / git history)
    try:
        root = getattr(global_config, "project_root", None)
        if root:
            p = Path(root) / "confs" / "patreon_cookie.txt"
            if p.exists():
                txt = p.read_text(encoding="utf-8").strip()
                if txt.lower().startswith("cookie:"):
                    txt = txt.split(":", 1)[1].strip()
                if txt:
                    return txt
    except Exception:
        pass

    return (global_config.patreon_cookie or "").strip()


def parse_cookie_string(cookie: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k and v:
            out[k.strip()] = v.strip()
    return out


def has_auth_cookie(cookie_dict: Dict[str, str]) -> bool:
    # Patreon member access requires an authenticated session cookie, not just Cloudflare cookies.
    auth_keys = {"session_id", "session_id.sig", "__Secure-next-auth.session-token"}
    return any(k in cookie_dict for k in auth_keys)


def vanity_from_config(cfg) -> str:
    raw = (getattr(cfg, "url", None) or cfg.channel_name_or_term or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        host = urlparse(raw).netloc.lower()
        if "patreon.com" not in host:
            return ""
        path = urlparse(raw).path.strip("/")
        if not path:
            return ""
        return path.split("/")[0].lower()
    return raw.lower()


def make_base_urls() -> List[str]:
    env_base = os.environ.get("PATREON_BASE_URL", "").strip()
    bases = [b for b in [env_base, "https://www.patreon.com", "https://patreon.com"] if b]
    seen = set()
    ordered = []
    for b in bases:
        if b not in seen:
            ordered.append(b)
            seen.add(b)
    return ordered


async def fetch_campaign_id(session: aiohttp.ClientSession, vanity: str, base_url: str) -> Optional[str]:
    if not vanity:
        return None
    # Patreon has no public API to look up campaign ID by vanity name.
    # Instead, fetch the creator's public page and extract the campaign ID
    # from the embedded JSON data.
    url = f"{base_url}/{vanity}"
    async with session.get(url) as response:
        if response.status != 200:
            logger.error(f"Patreon page fetch failed ({response.status}) for {vanity}")
            return None
        text = await response.text()
        # Look for campaign ID in the page's embedded data
        for pattern in [
            r'"campaign_id"\s*:\s*"?(\d+)"?',
            r'"id"\s*:\s*"(\d+)"\s*,\s*"type"\s*:\s*"campaign"',
            r'campaign/(\d+)',
            r'"patreon-campaign"\s+content="(\d+)"',
        ]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        logger.error(f"Could not find campaign ID in page for {vanity}")
        return None


async def posts_generator(session: aiohttp.ClientSession, campaign_id: str, base_url: str):
    url = api.posts_url(campaign_id)
    if url.startswith("https://www.patreon.com"):
        url = url.replace("https://www.patreon.com", base_url, 1)
    elif url.startswith("https://patreon.com"):
        url = url.replace("https://patreon.com", base_url, 1)
    while url:
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            yield data
            url = data.get("links", {}).get("next")
            if url and url.startswith("https://www.patreon.com"):
                url = url.replace("https://www.patreon.com", base_url, 1)
            elif url and url.startswith("https://patreon.com"):
                url = url.replace("https://patreon.com", base_url, 1)


def parse_post_duration_seconds(post: Dict) -> Optional[int]:
    post_file = (post.get("attributes") or {}).get("post_file") or {}
    meta = post_file.get("metadata") or {}
    for key in ("duration", "duration_seconds", "length_seconds"):
        val = meta.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    return None


def should_skip_post(post: Dict, cfg) -> bool:
    attrs = post.get("attributes") or {}
    title = (attrs.get("title") or "").strip()
    post_type = (attrs.get("post_type") or "").strip()

    if not attrs.get("current_user_can_view", True):
        return True

    if cfg.must_contain:
        if not all(x.lower() in title.lower() for x in cfg.must_contain):
            return True

    if cfg.must_exclude:
        if any(x.lower() in title.lower() for x in cfg.must_exclude):
            return True

    if cfg.max_date:
        try:
            max_date = datetime.strptime(cfg.max_date, "%Y-%m-%d")
            published = datetime.fromisoformat(attrs.get("published_at"))
            if published <= max_date:
                return True
        except Exception:
            pass

    if cfg.min_length_mins:
        duration = parse_post_duration_seconds(post)
        if duration is not None and duration < cfg.min_length_mins * 60:
            return True

    if post_type == "video_embed":
        return True

    audio_post_types = {"audio_file", "podcast"}
    downloadable_post_types = {"audio_file", "podcast", "video_external_file"}

    if cfg.audio_only and post_type not in audio_post_types:
        return True

    if post_type not in downloadable_post_types:
        return True

    return False


def pick_download_url(post: Dict) -> Optional[str]:
    attrs = post.get("attributes") or {}
    post_type = (attrs.get("post_type") or "").strip()
    post_file = attrs.get("post_file") or {}
    if post_type in {"audio_file", "podcast"}:
        return post_file.get("url")
    if post_type == "video_external_file":
        return post_file.get("download_url") or post_file.get("url")
    return None


def pick_filename(post: Dict, fallback_ext: str = ".bin") -> str:
    attrs = post.get("attributes") or {}
    post_file = attrs.get("post_file") or {}
    name = post_file.get("name") or ""
    if name:
        return name
    url = pick_download_url(post)
    if url:
        path = urlparse(url).path
        ext = Path(path).suffix
        if ext:
            return f"file{ext}"
    return f"file{fallback_ext}"


def is_patreon_hosted(url: str) -> bool:
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return "patreon" in host
