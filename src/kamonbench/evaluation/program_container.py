from __future__ import annotations

import argparse
import csv
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


def load_container_names(csv_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with csv_path.open('r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[2] == 'container':
                names[f'C:{Path(row[0]).stem}'] = row[1]
    return names


def _container_groups(rows: Iterable[ProgramFactorRow]) -> dict[str, list[ProgramFactorRow]]:
    groups: dict[str, list[ProgramFactorRow]] = defaultdict(list)
    for row in rows:
        if row.reference is None or not row.reference.has_container:
            continue
        groups[' '.join(row.reference.containers)].append(row)
    return dict(groups)


def evaluate_container_breakdown(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = program_factor_rows(list(predictions))
    groups = _container_groups(rows)
    containers = {
        container: metric_summary(container_rows)
        for container, container_rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }
    return {
        'total_rows': len(rows),
        'contained_rows': sum(summary['total_rows'] for summary in containers.values()),
        'containers': containers,
    }


def markdown_table(
    results: dict[str, dict[str, Any]], *, container_names: dict[str, str] | None = None, max_rows: int | None = None
) -> str:
    container_names = container_names or {}
    lines = [
        '| Model | Container | Name | N | C | Tuple | R | M |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model, result in results.items():
        items = list(result['containers'].items())
        if max_rows is not None:
            items = items[:max_rows]
        for container, summary in items:
            metrics = summary['metrics']
            lines.append(
                '| '
                + ' | '.join(
                    [
                        model,
                        container,
                        container_names.get(container, ''),
                        str(summary['total_rows']),
                        format_metric(metrics['container']),
                        format_metric(metrics['factor_tuple']),
                        format_metric(metrics['modifier']),
                        format_metric(metrics['motif']),
                    ]
                )
                + ' |'
            )
    return '\n'.join(lines) + '\n'


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--basic-charges-csv', type=Path, default=None)
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--max-markdown-rows', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        'evaluate-program-containers', help='Evaluate program container accuracy by container.'
    )
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    result = evaluate_container_breakdown(load_jsonl(args.input_jsonl))
    result['input_jsonl'] = str(args.input_jsonl)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        names = load_container_names(args.basic_charges_csv) if args.basic_charges_csv else {}
        model_name = args.model_name or args.input_jsonl.stem
        args.markdown_output.write_text(
            markdown_table({model_name: result}, container_names=names, max_rows=args.max_markdown_rows),
            encoding='utf-8',
        )
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
