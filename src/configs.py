import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import marshmallow.fields as ma_fields
from dataclasses_json import dataclass_json, config

from file_utils import csv_to_dict


@dataclass
class HostEmbedding:
    """
    Represents an embedding file and associated label for a host used in speaker matching.
    """
    embeddings_file: Path
    label: str


@dataclass_json
@dataclass
class GuestSearch:
    separator: str
    position: str


@dataclass_json
@dataclass
class YouTubeConfig:
    channel_name_or_term: str
    overwrite: bool
    get_static_videos: bool
    audio_only: bool
    use_google_api: bool
    is_playlist: bool
    is_search: bool
    max_date: Optional[str]
    min_length_mins: int
    guest_searches: List[GuestSearch]
    episode_prefix: Optional[str]
    guest_replace: List[str]
    must_contain: List[str]
    must_exclude: List[str]
    hosts: List[str]

    # Computed, not serialized
    host_embeddings: List[HostEmbedding] = field(
        default_factory=list,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )

    name: Optional[str] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )
    input_path: Optional[Path] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )
    output_path: Optional[Path] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )

    _metadata_cache: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """
        Loads and returns parsed metadata as a list of dicts from metadata_parsed.csv.
        Caches on first access.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.output_path:
            self._metadata_cache = []
            return self._metadata_cache
        csv_path = self.output_path / "metadata" / "metadata_parsed.csv"
        if not csv_path.exists():
            self._metadata_cache = []
            return self._metadata_cache
        rows = csv_to_dict(csv_path) or []
        self._metadata_cache = rows
        return rows

    def get_metadata_by_filename(self, sanitize_title: str) -> Optional[Dict[str, Any]]:
        """
        Return the metadata row dict for a given sanitized title, or None if not found.
        """
        for row in self.metadata:
            if row.get("sanitize_title") == sanitize_title:
                return row
        return None


@dataclass_json
@dataclass
class TwitchConfig:
    channel_name_or_term: str
    overwrite: bool
    get_static_videos: bool
    audio_only: bool
    is_clips: bool
    max_date: Optional[str]
    min_length_mins: int
    guest_searches: List[GuestSearch]
    episode_prefix: Optional[str]
    guest_replace: List[str]
    must_contain: List[str]
    must_exclude: List[str]
    hosts: List[str]

    # Computed, not serialized
    host_embeddings: List[HostEmbedding] = field(
        default_factory=list,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )
    name: Optional[str] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )
    input_path: Optional[Path] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )
    output_path: Optional[Path] = field(
        default=None,
        init=False,
        metadata=config(exclude=lambda _: True, mm_field=ma_fields.Raw())
    )

    _metadata_cache: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """
        Loads and returns parsed metadata as a list of dicts from metadata_parsed.csv.
        Caches on first access.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        if not self.output_path:
            self._metadata_cache = []
            return self._metadata_cache
        csv_path = self.output_path / "metadata" / "metadata_parsed.csv"
        if not csv_path.exists():
            self._metadata_cache = []
            return self._metadata_cache
        rows = csv_to_dict(csv_path) or []
        self._metadata_cache = rows
        return rows

    def get_metadata_by_filename(self, sanitize_title: str) -> Optional[Dict[str, Any]]:
        """
        Return the metadata row dict for a given sanitized title, or None if not found.
        """
        for row in self.metadata:
            if row.get("sanitize_title") == sanitize_title:
                return row
        return None


@dataclass
class DownloadConfigs:
    youtube: List[YouTubeConfig]
    twitch: List[TwitchConfig] = field(default_factory=list)


@dataclass
class ChannelConfig:
    name: str
    download_configs: DownloadConfigs


@dataclass
class YouTubeGlobalConfig:
    cookies_from: str


@dataclass
class HostMatchConfig:
    """
    Parameters defining how many host audio segments to sample and process for embedding generation.
    """
    sample_file_count: int
    per_file_segment_count: int
    target_file_count: int


@dataclass
class GlobalConfig:
    """
    Application-wide settings loaded from global.json, including data paths, tokens, and host matching parameters.
    """
    data_directory: Path
    youtube: YouTubeGlobalConfig
    keep_raw_downloads: bool
    max_workers_base: int
    min_free_disk_space_gb: int
    reverse_processing_order: bool
    host_match: HostMatchConfig
    google_token: str
    hf_token: str
    log_dir: Path
    project_root: Path


def get_global_config() -> GlobalConfig:
    """
    Load and resolve the global configuration from confs/global.json.
    Reads tokens for Google and Hugging Face, resolves paths relative to the project root,
    and constructs a GlobalConfig instance.
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'confs' / 'global.json'
    raw = json.loads(config_path.read_text(encoding='utf-8'))

    raw_root = raw.get('project_root')
    if raw_root:
        config_root = Path(raw_root)
        if not config_root.is_absolute():
            config_root = project_root / config_root
    else:
        config_root = project_root

    data_root = raw.get('data', 'data')
    p_obj = Path(data_root)
    data = p_obj if p_obj.is_absolute() else config_root / p_obj

    youtube_conf = YouTubeGlobalConfig(**raw.get('youtube', {}))
    max_workers_base = raw.get('max_workers_base', 1)
    min_free_disk_space_gb = raw.get('min_free_disk_space_gb', 5)
    reverse_processing_order = raw.get('reverse_processing_order', False)
    keep_raw_downloads = raw.get('keep_raw_downloads', True)
    host_match_raw = raw.get('host_match', {})
    host_match_conf = HostMatchConfig(
        sample_file_count=host_match_raw.get('sample_file_count', 0),
        per_file_segment_count=host_match_raw.get('per_file_segment_count', 0),
        target_file_count=host_match_raw.get('target_file_count', 0),
    )

    google_token = (config_root / 'tokens' / 'google').read_text().strip()
    hf_token = (config_root / 'tokens' / 'huggingface').read_text().strip()
    log_dir = config_root / 'logs'

    return GlobalConfig(
        data_directory=data,
        youtube=youtube_conf,
        max_workers_base=max_workers_base,
        min_free_disk_space_gb=min_free_disk_space_gb,
        reverse_processing_order=reverse_processing_order,
        keep_raw_downloads=keep_raw_downloads,
        host_match=host_match_conf,
        google_token=google_token,
        hf_token=hf_token,
        log_dir=log_dir,
        project_root=config_root
    )


_configs: Dict[str, ChannelConfig] = {}


def get_configs() -> List[ChannelConfig]:
    """
    Load all channel-specific download configurations from JSON files in confs/download_configurations.
    Caches results to avoid reloading on repeated calls and returns a list of ChannelConfig.
    """
    global _configs
    if _configs:
        return list(_configs.values())

    gc = get_global_config()
    configs_dir = gc.project_root / 'confs' / 'download_configurations'

    config_files = list(configs_dir.glob('*.json'))
    if gc.reverse_processing_order:
        config_files = config_files[::-1]

    for config_file in config_files:
        name = config_file.stem
        raw = json.loads(config_file.read_text(encoding='utf-8'))

        youtube_items = raw.get('youtube', [])
        twitch_items = raw.get('twitch', [])
        youtube_configs: List[YouTubeConfig] = []
        twitch_configs: List[TwitchConfig] = []

        for item in youtube_items:
            yt_cfg = YouTubeConfig.schema().load(item)
            yt_cfg.name = name
            base = gc.data_directory / name / 'youtube' / yt_cfg.channel_name_or_term
            base.mkdir(parents=True, exist_ok=True)
            yt_cfg.input_path = base
            yt_cfg.output_path = base

            training_folder = base / 'training'
            embeddings_list: List[HostEmbedding] = []
            for idx, _host in enumerate(yt_cfg.hosts):
                emb_file = training_folder / f'embeddings_speaker_{idx}.npy'
                label_file = training_folder / f'embeddings_speaker_label_{idx}.txt'
                label_text = ''
                if label_file.exists():
                    try:
                        label_text = label_file.read_text(encoding='utf-8').strip()
                    except Exception:
                        label_text = ''

                if not label_text:
                    label_text = 'Unknown'

                embeddings_list.append(HostEmbedding(embeddings_file=emb_file, label=label_text))

            yt_cfg.host_embeddings = embeddings_list
            youtube_configs.append(yt_cfg)

        for t_dict in twitch_items:
            tw_cfg = TwitchConfig.schema().load(t_dict)
            tw_cfg.name = name
            base = gc.data_directory / name / 'twitch' / tw_cfg.channel_name_or_term
            base.mkdir(parents=True, exist_ok=True)
            tw_cfg.input_path = base
            tw_cfg.output_path = base

            training_folder = base / 'training'
            embeddings_list: List[HostEmbedding] = []
            for idx, _host in enumerate(tw_cfg.hosts):
                emb_file = training_folder / f'embeddings_speaker_{idx}.npy'
                label_file = training_folder / f'embeddings_speaker_label_{idx}.txt'
                label_text = ''

                if label_file.exists():
                    try:
                        label_text = label_file.read_text(encoding='utf-8').strip()
                    except Exception:
                        label_text = ''

                if not label_text:
                    label_text = 'Unknown'

                embeddings_list.append(HostEmbedding(embeddings_file=emb_file, label=label_text))

            tw_cfg.host_embeddings = embeddings_list
            twitch_configs.append(tw_cfg)

        _configs[name] = ChannelConfig(name=name,
                                       download_configs=DownloadConfigs(youtube=youtube_configs, twitch=twitch_configs))

    return list(_configs.values())


def get_config(name: str) -> ChannelConfig:
    """
    Retrieve a ChannelConfig by name, raising KeyError if no matching configuration is found.

    :param name: The name of the channel configuration to retrieve.
    :return: The matching ChannelConfig instance.
    :raises KeyError: If no config with the given name exists.
    """
    for c in get_configs():
        if c.name == name:
            return c

    raise KeyError(f"No config for channel '{name}'")


def get_youtube_configs() -> List[YouTubeConfig]:
    """
    Flatten and return all YouTubeConfig instances from all channel configurations.
    """
    return [yt for c in get_configs() for yt in c.download_configs.youtube]


def get_twitch_configs() -> List[TwitchConfig]:
    """
    Flatten and return all TwitchConfig instances from all channel configurations.
    """
    return [tw for c in get_configs() for tw in c.download_configs.twitch]
