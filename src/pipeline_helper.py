import subprocess
import sys
from pathlib import Path
from typing import List
from configs import get_global_config

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

def run_scripts(script_paths: List[Path]):
    global_config = get_global_config()

    if global_config.reverse_processing_order:
        script_paths = reversed(script_paths)

    for script_path in script_paths:
        run_script(script_path)
