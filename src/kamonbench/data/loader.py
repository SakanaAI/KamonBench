from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_kwargs(
    *, batch_size: int, shuffle: bool, num_workers: int = 4, seed: int | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if seed is not None:
        kwargs['generator'] = torch.Generator().manual_seed(seed)
        kwargs['worker_init_fn'] = _seed_worker
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return kwargs
