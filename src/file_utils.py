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


def csv_to_dict(csv_file: Path) -> List[Dict[str, Any]] | None:
    """
    Reads a CSV into a list of dicts using pandas, skipping the first row if it's a header row duplicate.
    """

    if csv_file.stat().st_size <= 2:
        return None

    df = pd.read_csv(csv_file, encoding='utf-8', skip_blank_lines=True, on_bad_lines='skip')

    if df.empty:
        return None

    if df.columns.tolist() == df.iloc[0].tolist():
        df = df.iloc[1:]

    return df.to_dict(orient='records')


def get_audio_files(input_dir: Path) -> List[Path]:
    """
    Returns list of audio files matching supported formats.
    """
    formats = ['*.m4a', '*.mp3', '*.wav']
    return [f for pattern in formats for f in input_dir.glob(pattern)]


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


def filter_files_by_stems(
        source_dir: Path,
        source_extension: str,
        filter_stems_dirs: List[Path],
        overwrite: bool = False
) -> List[Path]:
    """
    Return files from source_dir whose stems do NOT exist in all filter_stems_dirs.

    Args:
        source_dir (Path): Directory containing source files.
        source_extension (str): File extension for source dir
        filter_stems_dirs (List[Path]): List of directories to filter by stem presence.
        overwrite (bool, optional): If True, return all files in source_dir. Defaults to False.

    Returns:
        List[Path]: List of files from source_dir that are NOT present in all filter_stems_dirs by stem.
    """
    files = [p for p in source_dir.glob(f'*.{source_extension}')]

    if overwrite:
        return files

    # Build a set of stems for each filter_stems_dir
    filter_stem_sets = [
        {p.stem for p in filter_dir.glob('*.*') if p.is_file()}
        for filter_dir in filter_stems_dirs
    ]

    # Only include files whose stem does NOT exist in all filter stem sets
    result = [
        f for f in files
        if not all(f.stem in stem_set for stem_set in filter_stem_sets)
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
