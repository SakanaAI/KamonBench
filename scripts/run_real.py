#!/usr/bin/env -S uv run python
"""Prepare and evaluate all four synthetic-to-real conditions with all four models."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('canonical', 'grayscale', 'foreground_binary', 'aligned_binary')
CHECKPOINTS = {
    'vit': 'runs/baselines/vit-standard-20260428/checkpoints/checkpoint_best_*.pt',
    'vgg_masked_ngram1': 'checkpoints_program_vgg/checkpoint_best_*.pt',
    'vgg_no_mask_ngram4': 'checkpoints_program_vgg_nomask/checkpoint_best_*.pt',
    'qwen': 'runs/4b-standard-native-3ep-20260428/selected/trainable.safetensors',
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Requires CUDA and the checkpoints from run_full_pipeline.sh (program-label VGG variants), '
            'run_recombination.py (standard ViT, seed 20260428), and run_qwen.py '
            '(standard 4B native-token three-epoch Qwen, seed 20260428). '
            'Each baseline checkpoint directory must contain exactly one checkpoint_best_*.pt. '
            'Uses the 114 bundled images in experiments/real/data and verifies their recorded hashes. '
            'Installs source assets for 49 matched synthetic controls, prepares all four conditions, then '
            'writes 16 prediction files and four *-summary.json files under --output-dir. '
            'Existing prepared inputs are rejected; choose a fresh --output-dir for another run. '
            'Example: scripts/run_real.py --source-archive /path/to/mon-white-224.tar.gz'
        ),
    )
    parser.add_argument(
        '--source-archive', type=Path, required=True, help='Upstream mon-white-224.tar.gz archive for matched controls.'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('runs/real'), help='Output directory (default: runs/real).'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Print commands without requiring images or checkpoints.'
    )
    args = parser.parse_args()
    real_root = ROOT / 'experiments/real'
    archive = args.source_archive.resolve()
    output = args.output_dir.resolve()
    inputs = output / 'inputs'
    checkpoints = dict(CHECKPOINTS)
    if not args.dry_run:
        targets = [json.loads(line) for line in (ROOT / 'experiments/real/targets.jsonl').read_text().splitlines()]
        missing = [path for path in [archive, *[real_root / row['image'] for row in targets]] if not path.is_file()]
        errors = [f'Missing required file: {path}' for path in missing]
        for model, pattern in CHECKPOINTS.items():
            matches = sorted(path for path in ROOT.glob(pattern) if path.is_file())
            if len(matches) != 1:
                errors.append(f'{model}: expected one checkpoint at {pattern}; found {len(matches)}.')
            else:
                checkpoints[model] = str(matches[0])
        if inputs.exists():
            errors.append(f'Prepared input directory already exists: {inputs}. Choose a fresh --output-dir.')
        if errors:
            parser.error('\n'.join(errors))

    def run(*command: str | Path) -> None:
        command = [str(part) for part in command]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    run(sys.executable, '-m', 'kamonbench', 'install-mon-data', '--source-archive', archive)
    run(sys.executable, '-m', 'experiments.real.prepare', '--mon-dir', 'data/mon-white-224', '--output-dir', inputs)
    for model, checkpoint in checkpoints.items():
        for condition in CONDITIONS:
            extra = (
                ['--qwen-config', 'experiments/qwen/configs/4b-standard-native-3ep-20260428.json']
                if model == 'qwen'
                else []
            )
            run(
                sys.executable,
                '-m',
                'experiments.real.infer',
                '--model',
                model,
                '--checkpoint',
                checkpoint,
                *extra,
                '--rows',
                inputs / f'{condition}.jsonl',
                '--image-root',
                inputs,
                '--output',
                output / f'{model}-{condition}.jsonl',
            )
        run(
            sys.executable,
            '-m',
            'experiments.real.score',
            '--model',
            model,
            *[
                part
                for condition in CONDITIONS
                for part in (f'--{condition.replace("_", "-")}', output / f'{model}-{condition}.jsonl')
            ],
            '--output',
            output / f'{model}-summary.json',
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
