#!/usr/bin/env -S uv run python
"""Run the eleven published Qwen configurations, test inference, and intervals."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path('experiments/qwen/configs')
STANDARD = '4b-standard-native-3ep-20260428'
SEED_RUNS = [f'4b-standard-native-3ep-{seed}' for seed in (20260428, 20260429, 20260430)]


def main() -> int:
    configs = {path.stem: path.relative_to(ROOT) for path in sorted((ROOT / CONFIG_DIR).glob('*.json'))}
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Requires the reconstructed dataset and 2, 4, or 8 CUDA GPUs, as specified by each config. '
            'Runs sequentially with the published batch sizes, seeds, model revisions, and output directories. '
            'Each run writes selected/trainable.safetensors, test/predictions.jsonl, test/metrics.json, '
            'and test/intervals.json under its configured runs/ directory. The standard 4B native-token '
            'three-epoch seed 20260428 also writes runs/qwen-all-test.jsonl, including components, '
            'for run_analysis.py. Selecting all three standard seeds writes runs/qwen-standard-intervals.json. '
            'Run from the repository root: scripts/run_qwen.py'
        ),
    )
    parser.add_argument('--configs', nargs='+', choices=list(configs), help='Config names to run; default: all eleven.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading data or launching jobs.')
    args = parser.parse_args()
    names = args.configs or [STANDARD, *[name for name in configs if name != STANDARD]]

    def run(*command: str | Path) -> None:
        command = [str(part) for part in command]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    outputs = {}
    for name in names:
        config_path = configs[name]
        config = json.loads((ROOT / config_path).read_text())
        distributed = config['distributed']
        divisor = distributed['per_device_batch_size'] * config['run']['expected_gradient_accumulation_steps']
        gpu_count, remainder = divmod(distributed['target_global_batch_size'], divisor)
        if remainder or gpu_count < 1:
            parser.error(f'Invalid distributed batch settings in {config_path}')
        module = 'recombination' if config['data'].get('heldout_combination') else 'train'
        output = Path(config['run']['output_dir'])
        outputs[name] = output
        run(
            sys.executable,
            '-m',
            'torch.distributed.run',
            '--standalone',
            f'--nproc_per_node={gpu_count}',
            '-m',
            f'experiments.qwen.{module}',
            '--config',
            config_path,
        )
        if name == STANDARD:
            run(
                sys.executable,
                '-m',
                'experiments.qwen.infer',
                '--config',
                config_path,
                '--checkpoint',
                output / 'selected/trainable.safetensors',
                '--subset',
                'all',
                '--output',
                'runs/qwen-all-test.jsonl',
            )
        run(
            sys.executable,
            '-m',
            'experiments.intervals',
            output / 'test/predictions.jsonl',
            '--output',
            output / 'test/intervals.json',
        )
    if all(name in outputs for name in SEED_RUNS):
        run(
            sys.executable,
            '-m',
            'experiments.intervals',
            *[outputs[name] / 'test/predictions.jsonl' for name in SEED_RUNS],
            '--label',
            'qwen-standard',
            '--output',
            'runs/qwen-standard-intervals.json',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
