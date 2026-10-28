#!/usr/bin/env -S uv run python
"""Train the seeded program baselines and recombination runs, then compute intervals."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/baselines'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running or writing files.')
    args = parser.parse_args()

    # Allow direct invocation as `python scripts/run_recombination.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments import baselines

    def run(command: list[str]) -> None:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    for model in ('vit', 'vgg-masks', 'vgg-no-masks'):
        baseline_args = ['--model', model, '--device', args.device, '--output-dir', str(args.output_dir)]
        baselines.main(baseline_args + (['--dry-run'] if args.dry_run else []))
        if model != 'vit':
            # Keep the published rm/crm launcher defaults; reuse its recipe for cm.
            run_dir = args.output_dir / f'{model}-cm-{baselines.SEEDS[0]}'
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=False)
            for command in baselines.commands(model, 'cm', baselines.SEEDS[0], run_dir, args.device):
                run(command)

        seeds = baselines.SEEDS if model == 'vit' else baselines.SEEDS[:1]
        splits = baselines.SPLITS if model == 'vit' else baselines.SPLITS[1:]
        for split in splits:
            predictions = []
            for seed in seeds:
                run_dir = args.output_dir / f'{model}-{split}-{seed}'
                prediction = str(run_dir / 'outputs/test_predictions.jsonl')
                predictions.append(prediction)
                run(
                    [
                        sys.executable,
                        '-m',
                        'experiments.intervals',
                        prediction,
                        '--output',
                        str(run_dir / 'results/intervals.json'),
                    ]
                )
            if model == 'vit':
                run(
                    [
                        sys.executable,
                        '-m',
                        'experiments.intervals',
                        *predictions,
                        '--label',
                        f'vit-{split}',
                        '--output',
                        str(args.output_dir / f'vit-{split}-intervals.json'),
                    ]
                )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
