from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != 'cuda':
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device) -> AbstractContextManager[None]:
    if device.type != 'cuda':
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
