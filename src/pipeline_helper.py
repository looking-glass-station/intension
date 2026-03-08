import os
import subprocess
import sys
from pathlib import Path
from typing import List
from configs import get_global_config


def _get_python_executable() -> str:
    """
    Prefer the project-local venv interpreter so all pipeline subprocesses
    run with the same environment.
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_script(script_path):
    """
    This separates out processes so that we don't have weird coalitions,
    and it frees up models per script.
    :param script_path:
    :return:
    """
    label = Path(script_path).stem
    print(f"[*] {label.replace('_', ' ').title()}")


    python_exe = _get_python_executable()
    result = subprocess.run([python_exe, str(script_path)])
    transcribe_crash_codes = {3221226505, -1073740791}
    if label == "transcribe" and result.returncode in transcribe_crash_codes:
        print("transcribe crashed natively (0xC0000409); retrying once on GPU safe mode (int8_float16)")
        retry_env = os.environ.copy()
        retry_env["INTENSION_GPU_SAFE_MODE"] = "1"
        result = subprocess.run([python_exe, str(script_path)], env=retry_env)

    if result.returncode != 0:
        print(f"{label} failed with code {result.returncode}")

        sys.exit(result.returncode)

def run_scripts(script_paths: List[Path]):
    global_config = get_global_config()

    if global_config.reverse_processing_order:
        script_paths = reversed(script_paths)

    for script_path in script_paths:
        run_script(script_path)
