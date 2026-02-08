import itertools
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import dateparser
import isodate
import scrapetube
import soundfile as sf
from googleapiclient.discovery import build
from pathvalidate import sanitize_filename
from tqdm_sound import TqdmSound

import file_utils
from configs import get_youtube_configs, get_global_config, YouTubeConfig
from episode_title_parser import EpisodeTitleParser, sanitize_title
from logger import global_logger

logger = global_logger('get_youtube')
global_config = get_global_config()

# Initialize TqdmSound for progress bars
progress = TqdmSound(
    activity_mute_seconds=0,
    dynamic_settings_file=str(global_config.project_root / "confs" / "sound.json")
)


def get_wav_duration_seconds(base_output: Path, sanitized_title: str) -> Optional[float]:
    """
    Return WAV duration in seconds if the file exists, else None.
    """
    if not base_output:
        return None
    wav_path = base_output / "wav" / f"{sanitize_filename(sanitized_title)}.wav"
    if not wav_path.exists():
        return None
    try:
        info = sf.info(str(wav_path))
        return float(info.frames) / float(info.samplerate) if info.samplerate else None
    except Exception:
        logger.warning(f"Failed to read duration for {wav_path}")
        return None


def parse_duration(text: str) -> Optional[int]:
    """
    Convert HH:MM:SS or MM:SS to seconds (int).
    Returns None if format is not parseable.
    """
    try:
        parts = list(map(int, text.split(':')))
        h, m, s = (0, 0, *parts)[-3:]
        return h * 3600 + m * 60 + s

    except (ValueError, TypeError):
        return None


def extract_raw_info_from_scrapetube(video: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a scrapetube video dict to the common metadata format.
    Returns None on parse error.
    """
    try:
        vid = video["videoId"]
        raw_title = re.sub(r"(\W)\1+", r"\1", video["title"]["runs"][0]["text"])

        # Date/time parsing: prefer publishedTimeText
        if "publishedTimeText" in video:
            dt = dateparser.parse(video["publishedTimeText"]["simpleText"])
        else:
            dt = dateparser.parse(video["dateText"]["simpleText"])

        if not dt:
            return None

        view_count = 0
        if "viewCountText" in video:
            raw = video["viewCountText"]["simpleText"]
            view_count = int(re.sub(r"[^0-9]", "", raw))

        elif "viewCount" in video:
            raw = video["viewCount"]["videoViewCountRenderer"]["viewCount"]["simpleText"]
            view_count = int(re.sub(r"[^0-9]", "", raw))

        if "lengthText" not in video:
            return None

        dur_str = video["lengthText"]["simpleText"]
        dur_secs = parse_duration(dur_str)
        if dur_secs is None:
            return None

        desc = ""
        if "descriptionSnippet" in video:
            runs = video["descriptionSnippet"]["runs"]
            desc = "".join(run.get("text", "") for run in runs).replace("\n", " ").replace("\r", "")

        like_count = 0
        sanitized_title = sanitize_title(raw_title)

        return {
            "video_id": vid,
            "title": raw_title,
            "sanitize_title": sanitized_title,
            "description": desc,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "date_uploaded": dt.strftime("%Y-%m-%d"),
            "view_count": view_count,
            "duration": dur_secs,
            "likes": like_count,
        }
    except Exception:
        return None


def get_channel_id(api_key: str, channel_name: str) -> str:
    """
    Given a YouTube channel handle (or name), look up and return its channelId string via API.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    handle = channel_name.strip()

    # Use the forHandle endpoint for direct lookup
    youtube = build("youtube", "v3", developerKey=api_key)
    handle = channel_name.strip()

    # Use the forHandle endpoint for direct lookup
    request = youtube.channels().list(part='id', forHandle=handle.lstrip('@'))
    response = request.execute()
    items = response.get('items')
    if not items:
        raise ValueError(f"Channel handle '{handle}' not found.")

    return items[0]['id']


def get_uploads_playlist_id(api_key: str, channel_id: str) -> str:
    """
    Given a YouTube channelId, return the uploads playlistId.
    """
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.channels().list(part="contentDetails", id=channel_id)
    response = request.execute()
    items = response.get("items", [])

    if not items:
        raise ValueError(f"No channel found with ID '{channel_id}'.")

    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_raw_video_details_batch_api(
        youtube_client: Any, video_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Fetch up to 50 YouTube videos at a time using API, returning list of normalized dicts.
    """
    results: List[Dict[str, Any]] = []
    total_chunks = (len(video_ids) + 49) // 50
    for chunk_index in range(total_chunks):
        chunk = video_ids[chunk_index * 50: (chunk_index + 1) * 50]
        try:
            resp = (
                youtube_client.videos()
                .list(part="snippet,statistics,contentDetails", id=",".join(chunk))
                .execute()
            )

        except Exception:
            continue

        for item in resp.get("items", []):
            snippet = item["snippet"]
            published_at = datetime.fromisoformat(
                snippet["publishedAt"].replace("Z", "+00:00")
            )

            title = snippet["title"]
            desc = snippet["description"].replace("\n", " ").replace("\r", "")
            view_count = int(item.get("statistics", {}).get("viewCount", 0))

            try:
                dur_secs = int(
                    isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
                )

            except Exception:
                continue

            like_count = int(item.get("statistics", {}).get("likeCount", 0))
            dislike_count = 0
            sanitized_title = sanitize_title(title)
            results.append({
                "video_id": item["id"],
                "title": title,
                "sanitize_title": sanitized_title,
                "description": desc,
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "date_uploaded": published_at.strftime("%Y-%m-%d"),
                "view_count": view_count,
                "duration": dur_secs,
                "likes": like_count,
                "dislikes": dislike_count,
            })
    results.sort(key=lambda d: d.get("date_uploaded", ""))
    return results


def fetch_all_raw_metadata() -> None:
    """
    For each YouTube config, fetch raw video metadata and save to CSV under each channel's folder.
    If overwrite is disabled, skips rows that already exist.
    """
    configs = get_youtube_configs()
    # OUTER PROGRESS BAR for configs
    config_bar = progress.progress_bar(
        configs,
        desc="YouTube configs - Fetching metadata",
        unit="config",
        total=len(configs),
        leave=True,
        position=0,
        ten_percent_ticks=True,
    )
    for config in config_bar:
        config_bar.set_description(f"{config.name}"
                                   f" ({getattr(config, 'channel_name_or_term', getattr(config, 'channel_name_playlist_id', ''))}) - Fetching metadata")
        base_input = config.input_path
        in_dir = base_input / "metadata"
        in_dir.mkdir(parents=True, exist_ok=True)
        raw_csv_path = in_dir / "metadata_raw.csv"

        # Determine existing records, respecting overwrite flag
        if raw_csv_path.exists() and not config.overwrite:
            existing_records: List[Dict[str, Any]] = file_utils.csv_to_dict(raw_csv_path) or []
            existing_ids = {rec.get("video_id") for rec in existing_records if rec.get("video_id")}
        else:
            existing_records = []
            existing_ids = set()

        missing_ids: List[str] = []
        new_items: List[Dict[str, Any]] = []

        if config.use_google_api:
            if not global_config.google_token:
                raise ValueError("Google API key is missing.")

            youtube_client = build("youtube", "v3", developerKey=global_config.google_token)

            if config.is_playlist:
                uploads_playlist_id = config.channel_name_or_term

            else:
                channel_search_id = get_channel_id(
                    global_config.google_token, config.channel_name_or_term
                )

                uploads_playlist_id = get_uploads_playlist_id(
                    global_config.google_token, channel_search_id
                )

            try:
                if config.is_playlist:
                    playlist_meta = (
                        youtube_client.playlists()
                        .list(part="contentDetails", id=uploads_playlist_id)
                        .execute()
                    )
                    total_videos = int(playlist_meta["items"][0]["contentDetails"]["itemCount"])

                else:
                    channel_meta = (
                        youtube_client.channels()
                        .list(part="statistics", id=channel_search_id)
                        .execute()
                    )

                    total_videos = int(channel_meta["items"][0]["statistics"]["videoCount"])

            except Exception:
                total_videos = 0

            per_page = 50
            num_pages = (total_videos + per_page - 1) // per_page if total_videos > 0 else 0
            next_token: Optional[str] = None

            bar = progress.progress_bar(
                range(num_pages),
                desc=f"{config.name} ({config.channel_name_or_term}) youtube - Fetching pages (0/{num_pages})",
                unit="page",
                position=0,
                leave=True,
            )
            for page_idx in bar:
                bar.set_description(
                    f"{config.name} ({config.channel_name_or_term}) youtube - Fetching pages ({page_idx + 1}/{num_pages})"
                )
                try:
                    resp = (
                        youtube_client.playlistItems()
                        .list(
                            part="contentDetails",
                            playlistId=uploads_playlist_id,
                            maxResults=per_page,
                            pageToken=next_token,
                        )
                        .execute()
                    )

                except Exception:
                    break

                page_video_ids = [item["contentDetails"]["videoId"] for item in resp.get("items", [])]
                if page_video_ids and all(vid in existing_ids for vid in page_video_ids):
                    break

                for vid in page_video_ids:
                    if vid not in existing_ids:
                        missing_ids.append(vid)

                next_token = resp.get("nextPageToken")
                if not next_token:
                    break

            bar.close()
            if missing_ids:
                chunks = (len(missing_ids) + 49) // 50
                for i in range(chunks):
                    chunk = missing_ids[i * 50: (i + 1) * 50]
                    batch = get_raw_video_details_batch_api(youtube_client, chunk)
                    for entry in batch:
                        entry["config_name"] = config.channel_name_or_term
                    new_items.extend(batch)
        else:
            source_id = config.channel_name_or_term
            if config.is_playlist:
                videos_gen = scrapetube.get_playlist(playlist_id=source_id)
            elif config.is_search:
                videos_gen = scrapetube.get_search(channel_id=source_id, limit=250)
            else:
                videos_gen = scrapetube.get_channel(
                    channel_username=source_id, sort_by="newest", content_type="videos"
                )

            infinite_pages = itertools.count()
            page_bar = progress.progress_bar(
                infinite_pages,
                desc=f"{config.name} ({config.channel_name_or_term}) youtube - Fetching pages",
                unit="page",
                position=0,
                leave=True,
            )

            while True:
                buffer: List[Dict[str, Any]] = []
                for _ in range(50):
                    try:
                        vg = next(videos_gen)
                    except StopIteration:
                        break

                    buffer.append(vg)
                if not buffer:
                    break

                page_bar.update(1)
                page_ids = [vid_dict["videoId"] for vid_dict in buffer]
                if all(pid in existing_ids for pid in page_ids):
                    break

                for vid_dict in buffer:
                    vid = vid_dict["videoId"]
                    if vid not in existing_ids:
                        raw_info = extract_raw_info_from_scrapetube(vid_dict)
                        if raw_info:
                            raw_info["config_name"] = config.channel_name_or_term
                            new_items.append(raw_info)

            page_bar.close()

        # Write CSV (overwrite or append)
        if not new_items and not config.overwrite:
            logger.info(f"No new raw metadata for {config.channel_name_or_term}; skipping update.")
        else:
            combined = new_items if config.overwrite else existing_records + new_items
            combined.sort(key=lambda d: d.get("date_uploaded", ""))
            file_utils.dict_to_csv(raw_csv_path, combined)

            logger.info(f"RAW metadata CSV updated at {raw_csv_path} (added {len(new_items)} new items)")

    config_bar.close()


def parse_and_filter_raw_records(
        raw_records: List[Dict[str, Any]], config: YouTubeConfig
) -> List[Dict[str, Any]]:
    """
    Apply all filtering and parsing to raw records for a channel, returning dicts with parsed guest, episode, etc.
    """
    filtered: List[Dict[str, Any]] = []
    max_date_obj: Optional[datetime] = None
    if config.max_date:
        try:
            max_date_obj = datetime.strptime(config.max_date, "%Y-%m-%d")
        except ValueError:
            max_date_obj = None

    for item in raw_records:
        title = item.get("title", "")
        if not all(inc.lower() in title.lower() for inc in config.must_contain):
            continue

        if any(exc.lower() in title.lower() for exc in config.must_exclude):
            continue

        date_str = item.get("date_uploaded")
        if not date_str:
            continue

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        if max_date_obj and dt <= max_date_obj:
            continue

        source_duration = item.get("duration", 0)
        if source_duration < config.min_length_mins * 60:
            continue

        title_parser = EpisodeTitleParser(item["title"], config)
        parsed_guest = title_parser.extract_guest()
        parsed_episode_number = title_parser.extract_episode_number()
        sanitized = sanitize_title(title)
        wav_duration = get_wav_duration_seconds(config.output_path, sanitized)

        filtered.append({
            "title": title,
            "sanitize_title": sanitized,
            "date_uploaded": item.get("date_uploaded", ""),
            "parsed_guest": parsed_guest,
            "parsed_episode_number": parsed_episode_number,
            "video_id": item.get("video_id", ""),
            "likes": item.get("likes", 0),
            "url": f"https://www.youtube.com/watch?v={item.get('video_id', '')}",
            "duration": wav_duration if wav_duration is not None else source_duration,
            "source_duration": source_duration,
        })

    return filtered


def parse_all_raw_metadata(write_files: bool = True) -> List[Dict[str, Any]]:
    """
    For each config, parse + filter all raw records, save parsed CSV, and return as list of dicts.
    Each dict will have 'config' attached for downstream processes.
    """
    configs = get_youtube_configs()
    out: List[Dict[str, Any]] = []
    # OUTER PROGRESS BAR for parsing
    config_bar = progress.progress_bar(
        configs,
        desc="YouTube configs - Parsing/filtering",
        unit="config",
        total=len(configs),
        leave=True,
        position=0,
        ten_percent_ticks=True,
    )
    for config in config_bar:
        config_bar.set_description(
            f"{config.name} ({getattr(config, 'channel_name_or_term', getattr(config, 'channel_name_playlist_id', ''))}) - Parsing/filtering")
        base_input = config.input_path
        base_output = config.output_path
        in_dir = base_input / "metadata"
        raw_csv_path = in_dir / "metadata_raw.csv"

        if not raw_csv_path.exists():
            logger.info(f"[YOUTUBE] Skipping {config.channel_name_or_term}: no raw CSV at {raw_csv_path}")
            continue

        out_dir = base_output / "metadata"
        out_dir.mkdir(parents=True, exist_ok=True)
        parsed_csv_path = out_dir / "metadata_parsed.csv"
        raw_records = file_utils.csv_to_dict(raw_csv_path) or []
        filtered_parsed = parse_and_filter_raw_records(raw_records, config)

        if write_files:
            file_utils.dict_to_csv(parsed_csv_path, filtered_parsed)
            logger.info(
                f"[YOUTUBE] Filtered+Parsed metadata CSV written to {parsed_csv_path} ({len(filtered_parsed)} rows)")

        for entry in filtered_parsed:
            entry['config'] = config  # This is useful for downstream processes like download_youtube

        out.extend(filtered_parsed)
    config_bar.close()
    return out


def main():
    """
    Entry point for CLI and batch: fetch and parse all YouTube metadata for all configs.
    """
    fetch_all_raw_metadata()
    parse_all_raw_metadata(True)


if __name__ == '__main__':
    main()
