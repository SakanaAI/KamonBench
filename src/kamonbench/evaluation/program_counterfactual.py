from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from kamonbench.evaluation.program_factors import ProgramFactorRow, load_jsonl, program_factor_rows

SLICE_NAMES = ('all', 'contained', 'containerless')


def _slice_rows(rows: list[ProgramFactorRow], slice_name: str) -> list[ProgramFactorRow]:
    if slice_name == 'all':
        return [row for row in rows if row.reference is not None]
    if slice_name == 'contained':
        return [row for row in rows if row.reference is not None and row.reference.has_container]
    if slice_name == 'containerless':
        return [row for row in rows if row.reference is not None and not row.reference.has_container]
    raise ValueError(f'Unknown slice: {slice_name}')


def _outer_key(row: ProgramFactorRow) -> tuple[tuple[str, ...], str]:
    if row.reference is None:
        raise ValueError('Counterfactual grouping requires a valid reference.')
    return row.reference.containers, row.reference.modifier


def _summarize_slice(rows: list[ProgramFactorRow]) -> dict[str, Any]:
    groups: dict[tuple[tuple[str, ...], str], list[ProgramFactorRow]] = defaultdict(list)
    for row in rows:
        groups[_outer_key(row)].append(row)

    usable_groups = []
    for group_rows in groups.values():
        motifs = {row.reference.motif for row in group_rows if row.reference is not None}
        if len(motifs) >= 2:
            usable_groups.append(group_rows)

    pair_count = valid_pair_count = separated_pairs = pair_exact_motif = 0
    collapsed_groups = 0
    for group_rows in usable_groups:
        valid_predicted_motifs = {row.predicted.motif for row in group_rows if row.predicted is not None}
        if len(valid_predicted_motifs) <= 1:
            collapsed_groups += 1
        for left, right in combinations(group_rows, 2):
            if left.reference is None or right.reference is None or left.reference.motif == right.reference.motif:
                continue
            pair_count += 1
            if left.predicted is None or right.predicted is None:
                continue
            valid_pair_count += 1
            separated_pairs += int(left.predicted.motif != right.predicted.motif)
            pair_exact_motif += int(
                left.predicted.motif == left.reference.motif and right.predicted.motif == right.reference.motif
            )

    return {
        'groups': len(usable_groups),
        'pairs': pair_count,
        'valid_pairs': valid_pair_count,
        'valid_pair_share': valid_pair_count / pair_count if pair_count else None,
        'motif_separation': separated_pairs / pair_count if pair_count else None,
        'pair_exact_motif': pair_exact_motif / pair_count if pair_count else None,
        'collapsed_groups': collapsed_groups,
        'collapsed_group_share': collapsed_groups / len(usable_groups) if usable_groups else None,
    }


def summarize_program_counterfactuals(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = program_factor_rows(list(predictions))
    return {
        'total_rows': len(rows),
        'grouping': 'same reference containers and modifier, different reference motifs',
        'slices': {slice_name: _summarize_slice(_slice_rows(rows, slice_name)) for slice_name in SLICE_NAMES},
    }


def _format(value: float | None) -> str:
    return '—' if value is None else f'{value:.3f}'


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Slice | Groups | Pairs | Valid pairs | Motif separation | Pair exact M | Collapsed groups |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model, result in results.items():
        for slice_name in SLICE_NAMES:
            item = result['slices'][slice_name]
            if item['pairs'] == 0:
                continue
            lines.append(
                '| '
                + ' | '.join(
                    [
                        model,
                        slice_name,
                        str(item['groups']),
                        str(item['pairs']),
                        _format(item['valid_pair_share']),
                        _format(item['motif_separation']),
                        _format(item['pair_exact_motif']),
                        _format(item['collapsed_group_share']),
                    ]
                )
                + ' |'
            )
    return '\n'.join(lines) + '\n'


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        'summarize-program-counterfactuals',
        help='Summarize motif sensitivity across program examples with shared outer factors.',
    )
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = summarize_program_counterfactuals(load_jsonl(args.input_jsonl))
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
