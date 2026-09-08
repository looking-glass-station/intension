import os
from pathlib import Path

import torch  # noqa: E402  (torch first, so it binds its own bundled cuDNN 9)

from configs import get_global_config

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())


def _add_ctranslate2_cuda_libs() -> None:
    """
    ctranslate2 4.4.0 (faster-whisper's backend) is a CUDA 12 build: it needs
    cuBLAS 12 + cuDNN 8. torch 2.5.1+cu118 only bundles CUDA 11 cuBLAS and
    cuDNN 9. Put nvidia-{cublas,cudnn}-cu12's DLLs on the search path *after*
    torch has loaded, so only ctranslate2 - which imports later - binds them.
    Windows only; a no-op elsewhere (Linux ctranslate2 wheels bundle their libs).
    """
    if os.name != "nt":
        return
    site = Path(torch.__file__).resolve().parents[1]
    for sub in ("cublas", "cudnn"):
        d = site / "nvidia" / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"


_add_ctranslate2_cuda_libs()

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
