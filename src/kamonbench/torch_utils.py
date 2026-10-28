from __future__ import annotations

import os
import random
from contextlib import AbstractContextManager, nullcontext

import numpy as np
import torch


def seed_runtime(seed: int) -> None:
    if seed < 0:
        raise ValueError('--seed must be non-negative')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(device: torch.device, *, deterministic: bool = False) -> None:
    torch.use_deterministic_algorithms(deterministic)
    if device.type != 'cuda':
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def autocast_context(device: torch.device) -> AbstractContextManager[None]:
    if device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
