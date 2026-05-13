from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from kamonbench.evaluation.program_factors import ProgramFactorRow, ProgramFactors, load_jsonl, program_factor_rows

FACTOR_NAMES = ('container', 'modifier', 'motif')
SLICE_NAMES = ('all', 'contained', 'containerless')


def _slice_rows(rows: list[ProgramFactorRow], slice_name: str) -> list[ProgramFactorRow]:
    if slice_name == 'all':
        return [row for row in rows if row.reference is not None]
    if slice_name == 'contained':
        return [row for row in rows if row.reference is not None and row.reference.has_container]
    if slice_name == 'containerless':
        return [row for row in rows if row.reference is not None and not row.reference.has_container]
    raise ValueError(f'Unknown slice: {slice_name}')


def _factor_value(factors: ProgramFactors | None, factor: str) -> str | None:
    if factors is None:
        return None
    if factor == 'container':
        return factors.containers[0] if factors.containers else None
    if factor == 'modifier':
        return factors.modifier
    if factor == 'motif':
        return factors.motif
    raise ValueError(f'Unknown factor: {factor}')


def _top_items(counts: Counter[str], total: int, top_k: int) -> list[dict[str, Any]]:
    return [
        {'label': label, 'count': count, 'share': count / total if total else None}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


def _factor_summary(rows: list[ProgramFactorRow], factor: str, top_k: int) -> dict[str, Any]:
    references: Counter[str] = Counter()
    predictions: Counter[str] = Counter()
    wrong_predictions: Counter[str] = Counter()
    correct = total = missing = 0

    for row in rows:
        gold = _factor_value(row.reference, factor)
        if gold is None:
            continue
        total += 1
        references[gold] += 1
        pred = _factor_value(row.predicted, factor)
        if pred is None:
            missing += 1
            continue
        predictions[pred] += 1
        if pred == gold:
            correct += 1
        else:
            wrong_predictions[pred] += 1

    return {
        'correct': correct,
        'total': total,
        'missing_predictions': missing,
        'accuracy': correct / total if total else None,
        'top_references': _top_items(references, total, top_k),
        'top_predictions': _top_items(predictions, total, top_k),
        'top_wrong_predictions': _top_items(wrong_predictions, total, top_k),
    }


def summarize_program_confusions(predictions: Iterable[dict[str, Any]], *, top_k: int = 10) -> dict[str, Any]:
    rows = program_factor_rows(list(predictions))
    return {
        'total_rows': len(rows),
        'top_k': top_k,
        'slices': {
            slice_name: {
                'total_rows': len(slice_group),
                'factors': {factor: _factor_summary(slice_group, factor, top_k) for factor in FACTOR_NAMES},
            }
            for slice_name in SLICE_NAMES
            for slice_group in [_slice_rows(rows, slice_name)]
        },
    }


def _format_accuracy(value: float | None) -> str:
    return '—' if value is None else f'{value:.3f}'


def _format_top(items: list[dict[str, Any]]) -> str:
    if not items:
        return '—'
    item = items[0]
    return f'{item["label"]} ({item["count"]})'


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Slice | Factor | N | Acc | Missing | Top prediction | Top wrong prediction |',
        '| --- | --- | --- | ---: | ---: | ---: | --- | --- |',
    ]
    for model, result in results.items():
        for slice_name in SLICE_NAMES:
            factors = result['slices'][slice_name]['factors']
            for factor in FACTOR_NAMES:
                summary = factors[factor]
                if summary['total'] == 0:
                    continue
                lines.append(
                    '| '
                    + ' | '.join(
                        [
                            model,
                            slice_name,
                            factor,
                            str(summary['total']),
                            _format_accuracy(summary['accuracy']),
                            str(summary['missing_predictions']),
                            _format_top(summary['top_predictions']),
                            _format_top(summary['top_wrong_predictions']),
                        ]
                    )
                    + ' |'
                )
    return '\n'.join(lines) + '\n'


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('summarize-program-confusions', help='Summarize program-label factor confusions.')
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = summarize_program_confusions(load_jsonl(args.input_jsonl), top_k=args.top_k)
    result['input_jsonl'] = str(args.input_jsonl)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        model_name = args.model_name or args.input_jsonl.stem
        args.markdown_output.write_text(markdown_table({model_name: result}), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)
    args = parser.parse_args(argv)
    return int(args._fn(args))


if __name__ == '__main__':
    raise SystemExit(main())
