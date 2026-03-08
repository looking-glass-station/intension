import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
for path in (ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src import pipeline_helper


if __name__ == '__main__':
    """
    This separates out processes so that we don't have weird coalitions,
    and it frees up models per script.
    """
    root = ROOT
    scripts = [
        root / "src" / "diarize.py",
        root / "src" / "transcribe.py",
        root / "src" / "generate_host_match.py",
        root / "src" / "label_speakers.py",
        root / "src" / "bias.py",
        root / "src" / "topics.py",
    ]

    for script in scripts:
        pipeline_helper.run_script(script)

