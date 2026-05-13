from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

STEP_RE = re.compile(r'^step_(\d+)$')


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _metric_value(summary: dict[str, Any], slice_name: str, metric_name: str) -> float | None:
    return summary.get('slices', {}).get(slice_name, {}).get('metrics', {}).get(metric_name)


def _factor_value(factors: dict[str, Any] | None, metric_name: str) -> float | None:
    if factors is None:
        return None
    return factors.get('metrics', {}).get(metric_name, {}).get('value')


def _step_from_dir(path: Path) -> int | None:
    match = STEP_RE.match(path.name)
    return int(match.group(1)) if match else None


def summarize_validation_curve(run_dir: Path) -> dict[str, Any]:
    rows = []
    for step_dir in sorted(run_dir.glob('step_*'), key=lambda path: (_step_from_dir(path) or -1, path.name)):
        step = _step_from_dir(step_dir)
        summary_path = step_dir / 'summary_slices.json'
        if step is None or not summary_path.exists():
            continue

        summary = _load_json(summary_path)
        factors_path = step_dir / 'factors.json'
        factors = _load_json(factors_path) if factors_path.exists() else None
        rows.append(
            {
                'step': step,
                'summary_path': str(summary_path),
                'all_acc': _metric_value(summary, 'all', 'acc'),
                'composite_acc': _metric_value(summary, 'composite', 'acc'),
                'composite_error': _metric_value(summary, 'composite', 'cer'),
                'component_acc': _metric_value(summary, 'component', 'acc'),
                'valid_prediction': _factor_value(factors, 'valid_prediction'),
                'factor_tuple': _factor_value(factors, 'factor_tuple'),
                'container': _factor_value(factors, 'container'),
                'modifier': _factor_value(factors, 'modifier'),
                'motif': _factor_value(factors, 'motif'),
            }
        )

    best_acc = max(rows, key=lambda row: row['composite_acc'] or -1, default=None)
    best_error = min(
        (row for row in rows if row['composite_error'] is not None),
        key=lambda row: row['composite_error'],
        default=None,
    )
    return {
        'run_dir': str(run_dir),
        'rows': rows,
        'best_composite_acc_step': None if best_acc is None else best_acc['step'],
        'best_composite_acc': None if best_acc is None else best_acc['composite_acc'],
        'lowest_composite_error_step': None if best_error is None else best_error['step'],
        'lowest_composite_error': None if best_error is None else best_error['composite_error'],
    }


def _format_value(value: float | None) -> str:
    return '-' if value is None else f'{value:.3f}'


def validation_curve_markdown(result: dict[str, Any]) -> str:
    lines = [
        '| Step | All acc | Composite acc | Composite error | Component acc | Tuple | C | R | M | Valid |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in result['rows']:
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(row['step']),
                    _format_value(row['all_acc']),
                    _format_value(row['composite_acc']),
                    _format_value(row['composite_error']),
                    _format_value(row['component_acc']),
                    _format_value(row['factor_tuple']),
                    _format_value(row['container']),
                    _format_value(row['modifier']),
                    _format_value(row['motif']),
                    _format_value(row['valid_prediction']),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('summarize-validation-curve', help='Summarize per-step validation metrics.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = summarize_validation_curve(args.run_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(validation_curve_markdown(result), encoding='utf-8')

    print(validation_curve_markdown(result), end='')
    return 0
