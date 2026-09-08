import os
import sys
from pathlib import Path

import torch  # noqa: E402  (torch first, so it binds its own bundled cuDNN 9)

from configs import get_global_config

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


def _add_ctranslate2_cudnn8() -> None:
    """
    ctranslate2 4.4.0 (faster-whisper's backend, CUDA 11) needs cuDNN 8, but
    torch 2.5.1+cu118 only bundles cuDNN 9. Put the nvidia-cudnn-cu11 (8.9.x)
    DLLs on the search path *after* torch has loaded, so only ctranslate2 - which
    imports later - picks them up.
    """
    if os.name != "nt":
        return
    for base in {Path(p) for p in sys.path if p} | {Path(torch.__file__).resolve().parents[1]}:
        d = base / "nvidia" / "cudnn" / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            return


_add_ctranslate2_cudnn8()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_num = 1 if torch.cuda.is_available() else 0
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
is_cuda = True if torch.cuda.is_available() else False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# remember the law of dimensioning returns, less is sometimes more
def max_workers(multiplier: int = None, absolute_count: int = None):
    if multiplier is not None:
        absolute_count = get_global_config().max_workers_base * multiplier

    if multiplier is None and absolute_count is None:
        absolute_count = 1

    return min(absolute_count, os.cpu_count() or 1)
