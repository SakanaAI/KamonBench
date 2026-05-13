from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from kamonbench.evaluation.program_factors import (
    ProgramFactorRow,
    format_metric,
    load_jsonl,
    metric_summary,
    program_factor_rows,
)


def _modifier_sort_key(modifier: str) -> tuple[int, str]:
    prefix, _, suffix = modifier.partition(':')
    if prefix == 'X' and suffix.isdigit():
        return int(suffix), modifier
    return 999, modifier


def _modifier_groups(rows: Iterable[ProgramFactorRow]) -> dict[str, list[ProgramFactorRow]]:
    groups: dict[str, list[ProgramFactorRow]] = defaultdict(list)
    for row in rows:
        if row.reference is None:
            continue
        groups[row.reference.modifier].append(row)
    return dict(groups)


def _slice_rows(rows: list[ProgramFactorRow], slice_name: str) -> list[ProgramFactorRow]:
    if slice_name == 'all':
        return rows
    if slice_name == 'contained':
        return [row for row in rows if row.reference is not None and row.reference.has_container]
    if slice_name == 'containerless':
        return [row for row in rows if row.reference is not None and not row.reference.has_container]
    raise ValueError(f'Unknown slice: {slice_name}')


def evaluate_modifier_breakdown(predictions: Iterable[dict[str, Any]], *, min_split_rows: int = 1) -> dict[str, Any]:
    rows = program_factor_rows(list(predictions))
    slices: dict[str, Any] = {}
    for slice_name in ['all', 'contained', 'containerless']:
        slice_group = _slice_rows(rows, slice_name)
        modifier_summaries = {}
        for modifier, modifier_rows in sorted(
            _modifier_groups(slice_group).items(), key=lambda item: _modifier_sort_key(item[0])
        ):
            if slice_name != 'all' and len(modifier_rows) < min_split_rows:
                continue
            modifier_summaries[modifier] = metric_summary(modifier_rows)
        slices[slice_name] = {
            'total_rows': len(slice_group),
            'min_split_rows': min_split_rows if slice_name != 'all' else None,
            'modifiers': modifier_summaries,
        }
    return {'total_rows': len(rows), 'slices': slices}


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Slice | Modifier | N | Tuple | C | R | M |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model, result in results.items():
        for slice_name in ['all', 'contained', 'containerless']:
            for modifier, summary in result['slices'][slice_name]['modifiers'].items():
                metrics = summary['metrics']
                lines.append(
                    '| '
                    + ' | '.join(
                        [
                            model,
                            slice_name,
                            modifier,
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


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--min-split-rows', type=int, default=1)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('evaluate-program-modifiers', help='Evaluate program factor accuracy by modifier.')
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = evaluate_modifier_breakdown(load_jsonl(args.input_jsonl), min_split_rows=args.min_split_rows)
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
