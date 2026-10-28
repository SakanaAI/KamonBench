"""Run the reported seeded ViT and VGG program-label experiments."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

COMMON = {
    'image-size': 224,
    'batch-size': 64,
    'early-stop-patience': 10,
    'weight-decay': 0.0001,
    'label-smoothing': 0.02,
    'start-token': -1,
}
VIT = {
    'model': 'vit_decoder',
    'num-epochs': 50,
    'checkpoint-steps': 2000,
    'num-train-augmentations': 3,
    'learning-rate': 0.00015,
    'vit-encoder-name': 'vit_base_patch16_224',
    'vit-n-heads': 8,
    'vit-d-model': 512,
    'vit-n-layers': 4,
    'vit-dropout': 0.1,
    'vit-token-dropout': 0.0,
    'vit-backbone-learning-rate': 0.00001,
}
VGG = {
    'model': 'vgg',
    'num-epochs': 200,
    'checkpoint-steps': 1250,
    'num-train-augmentations': 5,
    'learning-rate': 0.0001,
    'hidden-dim': 512,
}
SPLITS = ('standard', 'cm', 'rm', 'crm')
SEEDS = (20260428, 20260429, 20260430)


def commands(model: str, split: str, seed: int, run_dir: Path, device: str) -> list[list[str]]:
    metadata = Path('metadata') / ('program.json' if split == 'standard' else f'{split}.json')
    checkpoints = run_dir / 'checkpoints'
    outputs = run_dir / 'outputs'
    predictions = outputs / 'test_predictions.jsonl'
    results = run_dir / 'results'
    base = [sys.executable, '-m', 'kamonbench']
    settings = {**COMMON, **(VIT if model == 'vit' else VGG)}
    if model != 'vit':
        settings['ngram-length'] = 1 if model == 'vgg-masks' else 4
    training = [*base, 'train']
    for key, value in settings.items():
        training.extend([f'--{key}', str(value)])
    training.extend(
        [
            '--eval-composites-only',
            '--seed',
            str(seed),
            '--device',
            device,
            '--metadata-path',
            str(metadata),
            '--checkpoint-dir',
            str(checkpoints),
            '--output-dir',
            str(outputs),
        ]
    )
    training.extend(['--vit-train-backbone'] if model == 'vit' else ['--also-train-vgg'])
    if model == 'vgg-no-masks':
        training.append('--no-masks')
    bootstrap = ['--bootstrap-samples', '10000' if model == 'vit' else '0', '--bootstrap-seed', str(seed)]
    return [
        training,
        [
            *base,
            'infer',
            '--model',
            settings['model'],
            '--metadata-path',
            str(metadata),
            '--checkpoint-path',
            str(checkpoints / 'checkpoint_best_*.pt'),
            '--output-file',
            str(predictions),
            '--dataset-subset',
            'test',
            '--batch-size',
            '64',
            '--seed',
            str(seed),
            '--device',
            device,
        ],
        [
            *base,
            'summarize-predictions',
            str(predictions),
            '--metadata-path',
            str(metadata),
            '--split',
            'test',
            '--include-slices',
            *bootstrap,
            '--output',
            str(results / 'aggregate.json'),
        ],
        [
            *base,
            'evaluate-program-factors',
            str(predictions),
            *bootstrap,
            '--output',
            str(results / 'factors.json'),
            '--markdown-output',
            str(results / 'factors.md'),
        ],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['vit', 'vgg-masks', 'vgg-no-masks'], required=True)
    parser.add_argument('--splits', choices=SPLITS, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/baselines'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training or writing files.')
    args = parser.parse_args(argv)
    splits = args.splits or (SPLITS if args.model == 'vit' else ('rm', 'crm'))
    seeds = args.seeds or (SEEDS if args.model == 'vit' else (SEEDS[0],))
    if args.model != 'vit' and any(split not in ('rm', 'crm') for split in splits):
        parser.error('The reported VGG recombination runs use only rm and crm.')
    for split in splits:
        for seed in seeds:
            run_dir = args.output_dir / f'{args.model}-{split}-{seed}'
            if not args.dry_run:
                # An existing run is never silently retrained or overwritten.
                run_dir.mkdir(parents=True, exist_ok=False)
            for command in commands(args.model, split, seed, run_dir, args.device):
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
