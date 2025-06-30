import os

import torch

from configs import get_global_config

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())

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
