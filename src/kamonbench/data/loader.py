from __future__ import annotations

from typing import Any


def dataloader_kwargs(*, batch_size: int, shuffle: bool, num_workers: int = 4) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return kwargs
