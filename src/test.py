import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from configs import get_configs
from transcribe import Transcriber

# Directories to search (relative to project root, e.g. "data")
TARGET_DIRS = [
    "data",  # will filter below for /transcription, /transcription_labeled, /bias, /topics subdirs
]

TARGET_SUBDIRS = ["transcription", "transcription_labeled", "bias", "topics"]


def find_target_csvs(root_dir):
    for base in TARGET_SUBDIRS:
        for p in Path(root_dir).rglob(f"{base}/*.csv"):
            yield p


def guess_cfg(cfgs, csv_path):
    """
    Guess the config based on file path
    """
    parts = [p for p in csv_path.parts]
    for channel_cfg in cfgs:
        for cfg_list in vars(channel_cfg.download_configs).values():
            if not isinstance(cfg_list, list):
                continue
            for cfg in cfg_list:
                # Look for the output_path in the csv path
                if cfg.output_path and str(cfg.output_path) in str(csv_path):
                    return cfg
    return None


def main():
    root = Path(r"D:\Documents\Code\intension\data")
    all_cfgs = get_configs()
    transcriber = Transcriber()

    for csv_file in find_target_csvs(root):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
            continue

        if 'timestamp_url' in df.columns:
            print(f"Already has timestamp_url: {csv_file}")
            continue

        cfg = guess_cfg(all_cfgs, csv_file)
        if not cfg:
            print(f"Could not find config for {csv_file}")
            continue

        print(f"Fixing: {csv_file}")
        # Assume filename comes from the .stem of the wav file (by convention)
        filename = csv_file.stem.replace("_transcript", "")

        def get_url(row):
            start = row.get('start_time', 0)
            try:
                return transcriber.make_url(cfg, filename, float(start))
            except Exception:
                return ""

        df['timestamp_url'] = df.apply(get_url, axis=1)

        # Reorder columns if wanted: put timestamp_url last
        cols = [c for c in df.columns if c != 'timestamp_url'] + ['timestamp_url']
        df = df[cols]

        try:
            df.to_csv(csv_file, index=False)
            pass
        except Exception as e:
            print(f"Failed to write {csv_file}: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
