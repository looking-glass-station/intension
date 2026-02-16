import gc
from pathlib import Path
from typing import Any, List, Dict, Optional

import pandas as pd


def free_resources(model) -> None:
    """
    Collects garbage and empties CUDA cache, then deletes the given model.
    """

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    del model


def file_writable_test(file_path: Path) -> None:
    """
    Raises FileNotFoundError if the file doesn't exist, or IOError if it's open elsewhere.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        # test append mode without writing
        with file_path.open("a"):
            pass
    except Exception:
        raise IOError(f"FILE IS PROBABLY OPEN!!!: {file_path}")


def csv_to_dict(csv_file: Path, reverse: bool = False) -> List[Dict[str, Any]] | None:
    """
    Reads a CSV into a list of dicts using pandas, skipping the first row if it's a header row duplicate.
    If reverse is True, the order of the returned records is reversed.

    Args:
        csv_file (Path): Path to the CSV file.
        reverse (bool): Whether to reverse the order of records.

    Returns:
        List[Dict[str, Any]] | None: List of dictionaries representing the rows, or None if the file is empty.
    """
    if csv_file.stat().st_size <= 2:
        return None

    df = pd.read_csv(csv_file, encoding='utf-8', skip_blank_lines=True, on_bad_lines='skip')

    if df.empty:
        return None

    if df.columns.tolist() == df.iloc[0].tolist():
        df = df.iloc[1:]

    records = df.to_dict(orient='records')
    return records[::-1] if reverse else records


def get_audio_files(input_dir: Path) -> List[Path]:
    """
    Returns list of audio files matching supported formats.
    """
    formats = ['*.m4a', '*.mp3', '*.wav']
    return [f for pattern in formats for f in input_dir.glob(pattern)]


def get_media_files_recursive(input_dir: Path) -> List[Path]:
    """
    Returns list of audio/video files under input_dir (recursive).
    """
    if not input_dir.exists():
        return []
    exts = {
        ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus",
        ".mp4", ".mkv", ".webm", ".mov", ".avi",
    }
    return [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]


def parse_rttm(rttm_file: Path) -> List[Dict[str, Any]]:
    """
    Parse RTTM file and extract per-speaker segments.
    """
    segments: List[Dict[str, Any]] = []
    for line in rttm_file.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != 'SPEAKER':
            continue

        start_time = float(parts[3])
        duration = float(parts[4])
        segments.append({
            'speaker': parts[7],
            'start_time': start_time,
            'end_time': start_time + duration,
            'duration': duration,
        })
    return segments


from pathlib import Path
from typing import List

def filter_files_by_stems(
        source_dir: Path,
        source_extension: str,
        filter_stems_dirs: List[Path],
        overwrite: bool = False,
        reverse: bool = False,
) -> List[Path]:
    """
    Return files from source_dir whose stems do NOT exist in all filter_stems_dirs,
    or the reverse if reverse=True.

    Args:
        source_dir (Path): Directory containing source files.
        source_extension (str): File extension for source dir.
        filter_stems_dirs (List[Path]): List of directories to filter by stem presence.
        overwrite (bool, optional): If True, return all files in source_dir. Defaults to False.
        reverse (bool, optional): If True, invert the filter results. Defaults to False.

    Returns:
        List[Path]: Filtered list of files from source_dir.
    """
    files = [p for p in source_dir.glob(f'*.{source_extension}')]

    if overwrite:
        return files

    # Build a set of stems for each filter_stems_dir
    filter_stem_sets = [
        {p.stem for p in filter_dir.glob('*.*') if p.is_file()}
        for filter_dir in filter_stems_dirs
    ]

    # Main filtering
    def stem_in_all_sets(stem: str) -> bool:
        return all(stem in stem_set for stem_set in filter_stem_sets)

    result = [
        f for f in files
        if (stem_in_all_sets(f.stem) if reverse else not stem_in_all_sets(f.stem))
    ]

    return result



def dict_to_csv(
        path: Path,
        out_data: List[Dict[str, Any]],
        delimiter: str = ',',
        fields: Optional[List[str]] = None,
        headers: bool = True,
) -> None:
    """
    Writes a list of dictionaries out_data to a CSV at path.
    If `fields` is provided, columns will be reordered (and missing columns
    will be included as empty) instead of raising KeyError.
    """
    df = pd.DataFrame.from_records(out_data)
    if fields:
        # reindex will include all requested columns, filling missing ones with NaN
        df = df.reindex(columns=fields)

    df.to_csv(
        path,
        sep=delimiter,
        index=False,
        header=headers,
        encoding='utf-8'
    )


def audacity_writer(audacity_path: Path, out_data: List[Dict[str, Any]]) -> None:
    """
    Writes out_data for Audacity label track: start_time, end_time, text (tab-delimited, no headers).
    """
    dict_to_csv(
        audacity_path,
        out_data,
        delimiter='\t',
        fields=['start_time', 'end_time', 'text'],
        headers=False
    )
