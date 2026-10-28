#!/usr/bin/env -S uv run python
"""Run program diagnostics, frozen probes, VGG ablations, and motif comparisons."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = Path('metadata/program.json')
RESULTS = Path('runs/analysis')
BASELINES = {'vit': 'vit_decoder', 'vgg-masks': 'vgg', 'vgg-no-masks': 'vgg_nomask'}
QWEN = Path('runs/4b-standard-native-3ep-20260428/test/predictions.jsonl')
QWEN_COMPONENTS = Path('runs/qwen-all-test.jsonl')
VIT_COMPONENTS = Path('runs/baselines/vit-standard-20260428/outputs/test_predictions.jsonl')
SECTIONS = ('diagnostics', 'probes', 'ablations', 'components')
ABLATIONS = ('vgg-initial-ngram2', 'vgg-no-masks-ngram2')


def require_file(path: Path, dry_run: bool) -> Path:
    """Resolve checkpoint globs; extraction requires an actual file path."""
    if dry_run:
        return path
    matches = [match for match in ROOT.glob(str(path)) if match.is_file()]
    if not matches:
        raise FileNotFoundError(f'Missing required artifact: {path}. Run its experiment group first.')
    if len(matches) != 1:
        raise ValueError(f'Expected one artifact matching {path}, found {len(matches)}.')
    return matches[0].relative_to(ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sections', choices=SECTIONS, nargs='+', default=SECTIONS)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without artifacts or execution.')
    args = parser.parse_args(argv)

    def run(*arguments: object, module: str = 'kamonbench') -> None:
        command = [sys.executable, '-m', module, *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    predictions = {model: Path(f'outputs_program_{suffix}/test_decode.jsonl') for model, suffix in BASELINES.items()}
    checkpoints = {}
    try:
        # Check prerequisites before starting any long-running extraction or training.
        if set(args.sections) - {'diagnostics'}:
            require_file(METADATA, args.dry_run)
        if 'diagnostics' in args.sections or 'probes' in args.sections:
            for path in predictions.values():
                require_file(path, args.dry_run)
        if 'diagnostics' in args.sections:
            require_file(QWEN, args.dry_run)
        if 'probes' in args.sections:
            checkpoints = {
                model: require_file(Path(f'checkpoints_program_{suffix}/checkpoint_best_*.pt'), args.dry_run)
                for model, suffix in BASELINES.items()
            }
        if 'components' in args.sections:
            for path in (QWEN_COMPONENTS, VIT_COMPONENTS):
                require_file(path, args.dry_run)
        if 'ablations' in args.sections and not args.dry_run:
            for name in ABLATIONS:
                if (ROOT / 'runs' / name).exists():
                    raise FileExistsError(f'Run directory already exists: runs/{name}. Preserve or move it first.')

        if 'diagnostics' in args.sections:
            for model, prediction in {**predictions, 'qwen': QWEN}.items():
                for command, output in (
                    ('evaluate-program-factors', 'factors'),
                    ('summarize-program-counterfactuals', 'counterfactuals'),
                    ('summarize-program-confusions', 'confusions'),
                ):
                    run(command, prediction, '--output', RESULTS / model / f'{output}.json')
            for model, prediction in predictions.items():
                run(prediction, '--output', RESULTS / model / 'intervals.json', module='experiments.intervals')

        if 'probes' in args.sections:
            features = RESULTS / 'frozen_features'
            probes = RESULTS / 'frozen_probe_metrics'
            for model, checkpoint in checkpoints.items():
                run(
                    'extract-representation-features',
                    '--checkpoint-path',
                    checkpoint,
                    '--model-name',
                    model,
                    '--metadata-path',
                    METADATA,
                    '--image-root',
                    '.',
                    '--output-dir',
                    features,
                    '--device',
                    args.device,
                )
                run(
                    'train-representation-probes',
                    '--feature-dir',
                    features,
                    '--model-name',
                    model,
                    '--output-dir',
                    probes,
                    '--device',
                    args.device,
                )
                common = [
                    '--feature-dir',
                    features,
                    '--probe-dir',
                    probes,
                    '--model-name',
                    model,
                    '--output-dir',
                    probes,
                ]
                run('compare-representation-probes', *common, '--prediction-jsonl', predictions[model])
                run('compare-representation-probe-support', *common, '--metadata-path', METADATA)

        if 'ablations' in args.sections:
            for name in ABLATIONS:
                initial = name == 'vgg-initial-ngram2'
                run_dir = Path('runs') / name
                checkpoint_dir = run_dir / 'checkpoints'
                outputs = run_dir / 'outputs'
                prediction = outputs / 'test_decode.jsonl'
                if not args.dry_run:
                    (ROOT / run_dir).mkdir(parents=True, exist_ok=False)
                run(
                    'train',
                    '--model',
                    'vgg',
                    '--ngram-length',
                    2,
                    '--batch-size',
                    16 if initial else 64,
                    '--num-epochs',
                    200,
                    '--checkpoint-steps',
                    5000 if initial else 1250,
                    '--early-stop-patience',
                    10,
                    '--num-train-augmentations',
                    5,
                    '--learning-rate',
                    0.0001,
                    '--weight-decay',
                    0 if initial else 0.0001,
                    '--label-smoothing',
                    0 if initial else 0.02,
                    '--also-train-vgg',
                    '--metadata-path',
                    METADATA,
                    '--checkpoint-dir',
                    checkpoint_dir,
                    '--output-dir',
                    outputs,
                    '--device',
                    args.device,
                    *([] if initial else ['--no-masks', '--eval-composites-only']),
                )
                checkpoint = require_file(checkpoint_dir / 'checkpoint_best_*.pt', args.dry_run)
                run(
                    'infer',
                    '--metadata-path',
                    METADATA,
                    '--checkpoint-path',
                    checkpoint,
                    '--output-file',
                    prediction,
                    '--device',
                    args.device,
                )
                run(
                    'summarize-predictions',
                    prediction,
                    '--metadata-path',
                    METADATA,
                    '--split',
                    'test',
                    '--output',
                    run_dir / 'aggregate.json',
                )
                run('evaluate-program-factors', prediction, '--output', run_dir / 'factors.json')
                run('summarize-program-confusions', prediction, '--output', run_dir / 'confusions.json')

        if 'components' in args.sections:
            run(
                '--metadata',
                METADATA,
                '--qwen',
                QWEN_COMPONENTS,
                '--vit',
                VIT_COMPONENTS,
                '--output',
                RESULTS / 'component-comparison.json',
                module='experiments.qwen.components',
            )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
