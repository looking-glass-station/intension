import io
import json
import sys
from datetime import datetime
from typing import List, Dict, Union, Optional

from tqdm_sound import TqdmSound
from twitchdl.commands.clips import clips as _clips
from twitchdl.commands.videos import videos as _videos
from twitchdl.twitch import VideosSort, VideosType

import file_utils
from configs import get_twitch_configs, get_global_config
from episode_title_parser import sanitize_title, EpisodeTitleParser
from logger import global_logger

logger = global_logger('get_twitch')


def get_channel_videos(
        channel: str,
        *,
        limit: Optional[int] = None,
        fetch_all: bool = True,
        pager: Optional[int] = None,
        games: Optional[List[str]] = None,
        sort: VideosSort = 'time',
        vtype: VideosType = 'archive',
) -> List[Dict[str, Union[str, int, float]]]:
    buf = io.StringIO()
    orig_stdout = sys.stdout
    try:
        sys.stdout = buf
        kwargs: Dict[str, Union[bool, int, tuple, str]] = {
            'all': fetch_all,
            'compact': False,
            'json': True,
            'limit': limit if limit is not None else 0,
            'games': tuple(games) if games else (),
        }
        if pager is not None:
            kwargs['pager'] = pager
        if sort:
            kwargs['sort'] = sort
        if vtype:
            kwargs['type'] = vtype
        _videos(channel, **kwargs)
    finally:
        sys.stdout = orig_stdout
    data = json.loads(buf.getvalue())
    return data.get('videos', [])


def get_channel_clips(
        channel: str,
        *,
        limit: Optional[int] = None,
        fetch_all: bool = True,
) -> List[Dict[str, Union[str, int, float]]]:
    buf = io.StringIO()
    orig_stdout = sys.stdout
    try:
        sys.stdout = buf
        _clips(channel, all=fetch_all, limit=limit or 0, json=True)
    finally:
        sys.stdout = orig_stdout
    data = json.loads(buf.getvalue())
    return data if isinstance(data, list) else data.get('data', [])


def extract_raw_info_from_twitch(video: Dict[str, Union[str, int, float]]) -> Optional[
    Dict[str, Union[str, int, float]]]:
    try:
        vid = video.get('id')
        raw_title = video.get('title')
        sanitized_title = sanitize_title(raw_title)
        desc = (video.get('description') or '').replace('\n', ' ').replace('\r', '')
        pub = video.get('publishedAt')
        if not pub:
            return None
        dt = datetime.fromisoformat(pub.rstrip('Z'))
        date_uploaded = dt.strftime('%Y-%m-%d')
        view_count = int(video.get('viewCount') or 0)
        dur_secs = int(video.get('lengthSeconds') or 0)
        url = video.get('url') or f"https://www.twitch.tv/videos/{vid}"
        return {
            'video_id': vid,
            'title': raw_title,
            'sanitize_title': sanitized_title,
            'description': desc,
            'url': url,
            'date_uploaded': date_uploaded,
            'view_count': view_count,
            'duration': dur_secs,
            'likes': 0,
        }
    except Exception as exc:
        logger.error(f"Error extracting VOD info: {exc}")
        return None


def extract_raw_info_from_clip(clip: Dict[str, Union[str, int, float]]) -> Optional[Dict[str, Union[str, int, float]]]:
    try:
        cid = clip.get('id')
        title = clip.get('title') or ''
        desc = (clip.get('description') or '').replace('\n', ' ').replace('\r', '')
        created_at = clip.get('created_at') or clip.get('createdAt')
        if not created_at:
            return None
        dt = datetime.fromisoformat(created_at.rstrip('Z'))
        date_uploaded = dt.strftime('%Y-%m-%d')
        view_count = int(clip.get('view_count') or 0)
        dur_secs = int(clip.get('duration') or 0)
        sanitized_title = sanitize_title(title)
        url = clip.get('url') or f'https://clips.twitch.tv/{cid}'
        return {
            'video_id': cid,
            'title': title,
            'sanitize_title': sanitized_title,
            'description': desc,
            'url': url,
            'date_uploaded': date_uploaded,
            'view_count': view_count,
            'duration': dur_secs,
            'likes': 0,
        }
    except Exception as exc:
        logger.error(f"Error extracting clip info: {exc}")
        return None


def parse_and_filter_twitch_records(raw_records: List[Dict[str, Union[str, int, float]]], config) -> List[
    Dict[str, Union[str, int, float]]]:
    filtered = []
    max_date_obj = None
    if config.max_date:
        try:
            max_date_obj = datetime.strptime(config.max_date, "%Y-%m-%d")
        except Exception:
            pass
    for item in raw_records:
        title = item.get("title", "")
        if not all(inc.lower() in title.lower() for inc in config.must_contain):
            continue
        if any(exc.lower() in title.lower() for exc in config.must_exclude):
            continue
        date_str = item.get("date_uploaded", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue
        if max_date_obj and dt <= max_date_obj:
            continue
        dur_secs = item.get("duration", 0)
        if dur_secs < config.min_length_mins * 60:
            continue
        title_parser = EpisodeTitleParser(title, config)
        parsed_guest = title_parser.extract_guest()
        parsed_episode_number = title_parser.extract_episode_number()
        filtered.append({
            **item,
            "sanitize_title": sanitize_title(title),
            "parsed_guest": parsed_guest,
            "parsed_episode_number": parsed_episode_number,
        })
    return filtered


def fetch_all_raw_twitch_metadata() -> None:
    configs = get_twitch_configs()
    global_config = get_global_config()
    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )
    seen_channels = set()
    for cfg in configs:
        channel = cfg.channel_name_or_term
        base_input = cfg.input_path
        in_dir = base_input / 'metadata'
        in_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = in_dir / 'metadata_raw.csv'
        if channel not in seen_channels:
            if raw_csv.exists() and not cfg.overwrite:
                existing = file_utils.csv_to_dict(raw_csv) or []
                existing_ids = {rec.get('video_id') for rec in existing if rec.get('video_id')}
            else:
                existing = []
                existing_ids = set()
        else:
            existing = file_utils.csv_to_dict(raw_csv) or []
            existing_ids = {rec.get('video_id') for rec in existing if rec.get('video_id')}
        new_items: List[Dict[str, Union[str, int]]] = []
        items = (
            get_channel_clips(channel, fetch_all=True)
            if cfg.is_clips
            else get_channel_videos(channel, fetch_all=True, pager=100)
        )
        kind = 'clips' if cfg.is_clips else 'videos'
        bar = progress.progress_bar(
            items,
            total=len(items),
            desc=f"{cfg.name} ({channel}) {kind} fetch",
            unit=kind[:-1],
            leave=True,
            position=0,
            ten_percent_ticks=True,
        )
        for item in bar:
            vid = item.get('id')
            bar.set_description(f"{cfg.name} ({channel}) {kind}")
            if vid and vid not in existing_ids:
                raw = (
                    extract_raw_info_from_clip(item)
                    if cfg.is_clips else extract_raw_info_from_twitch(item)
                )
                if raw:
                    raw['config_name'] = channel
                    new_items.append(raw)
                    existing_ids.add(vid)
        combined = existing + new_items
        seen = set()
        unique = []
        for rec in combined:
            vid = rec.get('video_id')
            if vid and vid not in seen:
                seen.add(vid)
                unique.append(rec)
        unique.sort(key=lambda d: d.get('date_uploaded', ''))
        file_utils.dict_to_csv(raw_csv, unique)
        logger.info(
            f"[TWITCH {kind.upper()}] ({channel}) RAW metadata CSV updated at {raw_csv} (added {len(new_items)} new items)")
        seen_channels.add(channel)


def parse_all_raw_metadata(write_files: bool = True) -> List[Dict[str, Union[str, int, float]]]:
    configs = get_twitch_configs()
    global_config = get_global_config()
    progress = TqdmSound(
        activity_mute_seconds=0,
        dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
    )
    parsed_all: List[Dict[str, Union[str, int, float]]] = []
    for cfg in configs:
        base_input = cfg.input_path
        base_output = cfg.output_path
        in_dir = base_input / 'metadata'
        raw_csv = in_dir / 'metadata_raw.csv'
        if not raw_csv.exists():
            logger.warning(f"[TWITCH] Skipping {cfg.channel_name_or_term}: no raw CSV at {raw_csv}")
            continue
        out_dir = base_output / 'metadata'
        out_dir.mkdir(parents=True, exist_ok=True)
        parsed_csv = out_dir / 'metadata_parsed.csv'
        raw_records = file_utils.csv_to_dict(raw_csv) or []
        filtered_records = parse_and_filter_twitch_records(raw_records, cfg)
        bar = progress.progress_bar(
            filtered_records,
            total=len(filtered_records),
            desc=f"{cfg.name} ({cfg.channel_name_or_term}) parse",
            unit="row",
            leave=True,
            position=0,
            ten_percent_ticks=True,
        )
        for rec in bar:
            val = str(rec.get('sanitize_title', ''))
            rec['sanitize_title'] = sanitize_title(val)
            bar.set_description(f"{cfg.name} ({cfg.channel_name_or_term}) - Parsing/filtering")
        if write_files:
            file_utils.dict_to_csv(parsed_csv, filtered_records)
            logger.info(
                f"[TWITCH] Filtered+Parsed metadata CSV written to {parsed_csv} ({len(filtered_records)} rows)")
        parsed_all.extend(filtered_records)
    return parsed_all


def main():
    fetch_all_raw_twitch_metadata()
    parse_all_raw_metadata(True)


if __name__ == "__main__":
    main()
