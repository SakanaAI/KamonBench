from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kamonbench.reporting.html import include_in_visual_report

MAIN_METRICS = ['valid_prediction', 'factor_tuple', 'container', 'modifier', 'motif']


@dataclass(frozen=True)
class ProgramFactors:
    containers: tuple[str, ...]
    modifier: str
    motif: str

    @property
    def has_container(self) -> bool:
        return bool(self.containers)


@dataclass(frozen=True)
class ProgramFactorRow:
    reference: ProgramFactors | None
    predicted: ProgramFactors | None
    train_images_reference: tuple[str, ...]
    train_images_predicted: tuple[str, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_program_factors(text: str) -> ProgramFactors | None:
    containers: list[str] = []
    modifier = None
    motif = None
    for token in text.split():
        if token.startswith('C:'):
            containers.append(token)
        elif token.startswith('X:') and modifier is None:
            modifier = token
        elif token.startswith('M:') and motif is None:
            motif = token
        else:
            return None
    if modifier is None or motif is None:
        return None
    return ProgramFactors(tuple(containers), modifier, motif)


def program_factor_rows(predictions: list[dict[str, Any]]) -> list[ProgramFactorRow]:
    rows = []
    for item in predictions:
        if not include_in_visual_report(item):
            continue
        rows.append(
            ProgramFactorRow(
                reference=parse_program_factors(item.get('reference', '')),
                predicted=parse_program_factors(item.get('predicted', '')),
                train_images_reference=tuple(item.get('train_images_reference') or []),
                train_images_predicted=tuple(item.get('train_images_predicted') or []),
            )
        )
    return rows


def _metric_indicators(row: ProgramFactorRow) -> dict[str, bool | None]:
    ref = row.reference
    pred = row.predicted
    if ref is None:
        return {name: None for name in MAIN_METRICS}
    valid_prediction = pred is not None
    return {
        'valid_prediction': valid_prediction,
        'factor_tuple': valid_prediction and ref == pred,
        'container': None if not ref.has_container else valid_prediction and ref.containers == pred.containers,
        'modifier': valid_prediction and ref.modifier == pred.modifier,
        'motif': valid_prediction and ref.motif == pred.motif,
    }


def metric_summary(rows: list[ProgramFactorRow]) -> dict[str, Any]:
    counts = {name: {'correct': 0, 'total': 0} for name in MAIN_METRICS}
    valid_reference = 0
    for row in rows:
        valid_reference += int(row.reference is not None)
        for name, value in _metric_indicators(row).items():
            if value is None:
                continue
            counts[name]['total'] += 1
            counts[name]['correct'] += int(value)

    return {
        'total_rows': len(rows),
        'valid_reference': valid_reference,
        'metrics': {
            name: {
                'correct': count['correct'],
                'total': count['total'],
                'value': count['correct'] / count['total'] if count['total'] else None,
            }
            for name, count in counts.items()
        },
    }


def slice_rows(rows: list[ProgramFactorRow]) -> dict[str, list[ProgramFactorRow]]:
    slices: dict[str, list[ProgramFactorRow]] = defaultdict(list)
    for row in rows:
        ref = row.reference
        if ref is None:
            slices['invalid_reference'].append(row)
            continue
        slices['contained' if ref.has_container else 'containerless'].append(row)
        if not row.train_images_reference and not row.train_images_predicted:
            slices['no_training_support'].append(row)
    return dict(slices)


def bootstrap_intervals(rows: list[ProgramFactorRow], *, samples: int, seed: int) -> dict[str, dict[str, float]]:
    if samples <= 0 or not rows:
        return {}

    indicators = {name: [] for name in MAIN_METRICS}
    for row in rows:
        row_indicators = _metric_indicators(row)
        for name, value in row_indicators.items():
            if value is not None:
                indicators[name].append(float(value))

    rng = random.Random(seed)
    intervals = {}
    for name, values in indicators.items():
        if not values:
            continue
        n = len(values)
        means = []
        for _ in range(samples):
            means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
        means.sort()
        intervals[name] = {'low': means[int(0.025 * (samples - 1))], 'high': means[int(0.975 * (samples - 1))]}
    return intervals


def evaluate_program_factors(
    predictions: list[dict[str, Any]], *, bootstrap_samples: int = 0, bootstrap_seed: int = 20260428
) -> dict[str, Any]:
    rows = program_factor_rows(predictions)
    result = metric_summary(rows)
    result['slices'] = {name: metric_summary(slice_group) for name, slice_group in slice_rows(rows).items()}
    if bootstrap_samples:
        result['bootstrap'] = {
            'samples': bootstrap_samples,
            'seed': bootstrap_seed,
            'metrics': bootstrap_intervals(rows, samples=bootstrap_samples, seed=bootstrap_seed),
            'slices': {
                name: bootstrap_intervals(slice_group, samples=bootstrap_samples, seed=bootstrap_seed)
                for name, slice_group in slice_rows(rows).items()
            },
        }
    return result


def format_metric(metric: dict[str, Any]) -> str:
    value = metric.get('value')
    return '—' if value is None else f'{value:.3f}'


def format_ci(interval: dict[str, float] | None) -> str:
    if not interval:
        return '—'
    return f'[{interval["low"]:.3f}, {interval["high"]:.3f}]'


def markdown_table(result: dict[str, Any]) -> str:
    rows = [('all', result)]
    for name in ['contained', 'containerless', 'no_training_support']:
        if name in result['slices']:
            rows.append((name, result['slices'][name]))

    lines = ['| Slice | N | Tuple | C | R | M |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, summary in rows:
        metrics = summary['metrics']
        lines.append(
            '| '
            + ' | '.join(
                [
                    name,
                    str(summary['total_rows']),
                    format_metric(metrics['factor_tuple']),
                    format_metric(metrics['container']),
                    format_metric(metrics['modifier']),
                    format_metric(metrics['motif']),
                ]
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('evaluate-program-factors', help='Evaluate program-label factor accuracy.')
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)
    parser.add_argument('--bootstrap-samples', type=int, default=0)
    parser.add_argument('--bootstrap-seed', type=int, default=20260428)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = evaluate_program_factors(
        load_jsonl(args.input_jsonl), bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed
    )
    result['input_jsonl'] = str(args.input_jsonl)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_table(result), encoding='utf-8')

    metrics = result['metrics']
    print(f'Composite rows: {result["total_rows"]}')
    print(f'Tuple: {format_metric(metrics["factor_tuple"])}')
    print(f'Modifier: {format_metric(metrics["modifier"])}')
    print(f'Motif: {format_metric(metrics["motif"])}')
    print(f'Valid prediction: {format_metric(metrics["valid_prediction"])}')
    return 0
