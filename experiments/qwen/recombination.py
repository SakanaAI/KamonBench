from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

from torch.utils.data import DistributedSampler as TorchDistributedSampler

from experiments.qwen import train as train_qwen


class LengthProxy:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


class FullBatchTrainingDataset(train_qwen.KamonTrainingDataset):
    def __init__(
        self, records: list[dict[str, Any]], image_root: Path, *, augmentations: int, seed: int, augment_module: Any
    ) -> None:
        super().__init__(records, image_root, augmentations=augmentations, seed=seed, augment_module=augment_module)
        self.full_exposures = super().__len__()
        self.usable_exposures = (self.full_exposures // STATE['target_global_batch_size']) * STATE[
            'target_global_batch_size'
        ]
        self.dropped_exposures = self.full_exposures - self.usable_exposures
        if self.dropped_exposures != STATE['expected_dropped_exposures']:
            raise ValueError(
                f'Derived {self.dropped_exposures} dropped exposures, expected {STATE["expected_dropped_exposures"]}.'
            )

    def __len__(self) -> int:
        return self.usable_exposures

    def sample_key(self, index: int) -> str:
        slots = self.augmentations + 1
        record = self.records[index // slots]
        return f'{record["id"]}:{index % slots}'


class TailDroppingDistributedSampler:
    def __init__(
        self,
        dataset: Sized,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        delegate_dataset: Sized = dataset
        if isinstance(dataset, FullBatchTrainingDataset):
            delegate_dataset = LengthProxy(dataset.full_exposures)
        self.delegate = TorchDistributedSampler(
            delegate_dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.delegate.set_epoch(epoch)

    def __len__(self) -> int:
        if not isinstance(self.dataset, FullBatchTrainingDataset):
            return len(self.delegate)
        return self.dataset.usable_exposures // self.delegate.num_replicas

    def _indices_for_rank(self, rank: int) -> list[int]:
        delegate = TorchDistributedSampler(
            LengthProxy(self.dataset.full_exposures),
            num_replicas=self.delegate.num_replicas,
            rank=rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        delegate.set_epoch(self.epoch)
        return list(delegate)

    def _write_dropped_audit(self, keep: int) -> None:
        if self.delegate.rank != 0:
            return
        tails = [self._indices_for_rank(rank)[keep:] for rank in range(self.delegate.num_replicas)]
        dropped_indices = [
            tails[rank][offset] for offset in range(len(tails[0])) for rank in range(self.delegate.num_replicas)
        ]
        keys = [self.dataset.sample_key(index) for index in dropped_indices]
        encoded = ('\n'.join(keys) + '\n').encode()
        payload = {
            'epoch': self.epoch,
            'policy': 'tail of the full seeded DistributedSampler order',
            'full_exposures': self.dataset.full_exposures,
            'usable_exposures': self.dataset.usable_exposures,
            'dropped_count': len(keys),
            'dropped_sample_keys': keys,
            'dropped_sample_keys_sha256': hashlib.sha256(encoded).hexdigest(),
        }
        path = STATE['output_dir'] / 'dropped_training_exposures' / (f'epoch-{self.epoch:02d}.json')
        train_qwen.write_json(path, payload)

    def __iter__(self) -> Iterator[int]:
        if not isinstance(self.dataset, FullBatchTrainingDataset):
            return iter(self.delegate)
        indices = list(self.delegate)
        keep = len(self)
        expected_tail = self.dataset.dropped_exposures // self.delegate.num_replicas
        if len(indices) - keep != expected_tail:
            raise ValueError(
                f'Rank {self.delegate.rank} sampler tail has {len(indices) - keep} exposures, expected {expected_tail}.'
            )
        self._write_dropped_audit(keep)
        return iter(indices[:keep])


def finalize_audit(config: dict[str, Any]) -> None:
    output_dir = Path(config['run']['output_dir'])
    audit_dir = output_dir / 'dropped_training_exposures'
    paths = [audit_dir / f'epoch-{epoch:02d}.json' for epoch in range(config['run']['max_epochs'])]
    payloads = [train_qwen.read_json(path) for path in paths]
    expected = config['data']['dropped_exposures_per_epoch']
    if any(payload['dropped_count'] != expected for payload in payloads):
        raise ValueError('Dropped-exposure audit count mismatch.')
    combined = hashlib.sha256()
    for path in paths:
        combined.update(path.read_bytes())
    summary = {
        'policy': 'shuffle all exposures, then discard the per-rank tails corresponding to the final incomplete global batch',
        'development_and_test_dropped': 0,
        'epochs': len(paths),
        'dropped_exposures_per_epoch': expected,
        'epoch_audit_paths': [str(path.resolve()) for path in paths],
        'epoch_audits_sha256': combined.hexdigest(),
    }
    train_qwen.write_json(audit_dir / 'summary.json', summary)
    run_summary_path = output_dir / 'run_summary.json'
    run_summary = train_qwen.read_json(run_summary_path)
    run_summary['training_tail_policy'] = {**summary, 'summary_path': str((audit_dir / 'summary.json').resolve())}
    train_qwen.write_json(run_summary_path, run_summary)


def main() -> int:
    args = train_qwen.parse_args()
    config_path = args.config.resolve()
    config = train_qwen.read_json(config_path)
    global STATE
    STATE = {
        'target_global_batch_size': config['distributed']['target_global_batch_size'],
        'expected_dropped_exposures': config['data']['dropped_exposures_per_epoch'],
        'output_dir': Path(config['run']['output_dir']),
    }
    train_qwen.KamonTrainingDataset = FullBatchTrainingDataset
    train_qwen.DistributedSampler = TailDroppingDistributedSampler
    if args.output_dir is not None:
        STATE['output_dir'] = args.output_dir
        config['run']['output_dir'] = str(args.output_dir)
    code = train_qwen.train(config_path, overrides=args)
    if int(os.environ.get('RANK', '0')) == 0:
        finalize_audit(config)
    return code


STATE: dict[str, Any] = {}


if __name__ == '__main__':
    raise SystemExit(main())
