import subprocess
import sys
from pathlib import Path


def run_script(script_path):
    """
    This separates out processes so that we don't have weird coalitions,
    and it frees up models per script.
    :param script_path:
    :return:
    """
    label = Path(script_path).stem
    print(f"[*] {label.replace("_", " ").title()}")

    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        print(f"{label} failed with code {result.returncode}")

        sys.exit(result.returncode)
