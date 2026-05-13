from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kamonbench.reporting.html import (
    calculate_character_error_rate,
    calculate_token_error_rate,
    is_program_label,
    preprocess_text_for_comparison,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_croissant_split(path: Path, split: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return [record for record in data['recordSet'][0]['data'] if record['split'] == split]


@dataclass(frozen=True)
class RowStats:
    edits: int
    ref_chars: int
    correct: bool
    no_training_support: bool


def _is_correct(row: dict[str, Any]) -> bool:
    return preprocess_text_for_comparison(_reference_text(row)) == preprocess_text_for_comparison(_predicted_text(row))


def _is_no_training_support(row: dict[str, Any]) -> bool:
    return not row.get('train_images_reference') and not row.get('train_images_predicted')


def _reference_text(row: dict[str, Any]) -> str:
    return row.get('reference') or row.get('reference_description') or ''


def _predicted_text(row: dict[str, Any]) -> str:
    return row.get('predicted') or row.get('predicted_description') or ''


def _is_program_rows(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(is_program_label(_reference_text(row)) for row in rows)


def _row_stats(row: dict[str, Any], *, use_token_error: bool) -> RowStats:
    reference = _reference_text(row)
    predicted = _predicted_text(row)
    if use_token_error:
        ins, dels, subs, _ = calculate_token_error_rate(reference, predicted)
        ref_units = len(reference.split())
    else:
        ins, dels, subs, _ = calculate_character_error_rate(reference, predicted)
        ref_units = len(preprocess_text_for_comparison(reference))
    return RowStats(
        edits=ins + dels + subs,
        ref_chars=ref_units,
        correct=_is_correct(row),
        no_training_support=_is_no_training_support(row),
    )


def _is_composite_row(row: dict[str, Any]) -> bool:
    image = row.get('image') or row.get('image_path') or ''
    return '_base' not in image and '_container' not in image


def _with_croissant_images(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(records):
        raise ValueError(f'Prediction rows ({len(rows)}) do not match Croissant rows ({len(records)})')
    paired = []
    for row, record in zip(rows, records, strict=True):
        updated = dict(row)
        updated['image'] = record['image_path']
        updated['image_path'] = record['image_path']
        paired.append(updated)
    return paired


def _summarize_rows(data: list[dict[str, Any]]) -> dict[str, Any]:
    use_token_error = _is_program_rows(data)
    stats = [_row_stats(row, use_token_error=use_token_error) for row in data]
    edits = sum(row.edits for row in stats)
    ref_chars = sum(row.ref_chars for row in stats)
    correct = sum(row.correct for row in stats)
    nit_stats = [row for row in stats if row.no_training_support]
    nit_correct = sum(row.correct for row in nit_stats)
    return {
        'total_rows': len(data),
        'nit_rows': len(nit_stats),
        'error_unit': 'token' if use_token_error else 'character',
        'metrics': {
            'cer': edits / ref_chars if ref_chars else 0.0,
            'acc': correct / len(data) if data else 0.0,
            'acc_nit': nit_correct / len(nit_stats) if nit_stats else None,
        },
    }


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_rows([row for row in rows if _is_composite_row(row)])


def summarize_prediction_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    composites = [row for row in rows if _is_composite_row(row)]
    components = [row for row in rows if not _is_composite_row(row)]
    return {
        'all': _summarize_rows(rows),
        'composite': _summarize_rows(composites),
        'component': _summarize_rows(components),
    }


def bootstrap_intervals(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, dict[str, float]]:
    data = [row for row in rows if _is_composite_row(row)]
    if samples <= 0 or not data:
        return {}

    stats = [_row_stats(row, use_token_error=_is_program_rows(data)) for row in data]
    rng = random.Random(seed)
    values = {'cer': [], 'acc': [], 'acc_nit': []}
    n = len(stats)
    for _ in range(samples):
        edits = ref_chars = correct = nit_total = nit_correct = 0
        for _ in range(n):
            row = stats[rng.randrange(n)]
            edits += row.edits
            ref_chars += row.ref_chars
            correct += int(row.correct)
            if row.no_training_support:
                nit_total += 1
                nit_correct += int(row.correct)
        values['cer'].append(edits / ref_chars if ref_chars else 0.0)
        values['acc'].append(correct / n)
        if nit_total:
            values['acc_nit'].append(nit_correct / nit_total)

    intervals = {}
    for name, metric_values in values.items():
        if not metric_values:
            continue
        metric_values.sort()
        low = metric_values[int(0.025 * (len(metric_values) - 1))]
        high = metric_values[int(0.975 * (len(metric_values) - 1))]
        intervals[name] = {'low': low, 'high': high}
    return intervals


def evaluate_predictions(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 20260428,
    croissant_records: list[dict[str, Any]] | None = None,
    include_slices: bool = False,
):
    if croissant_records is not None:
        rows = _with_croissant_images(rows, croissant_records)
    result = summarize_predictions(rows)
    if include_slices:
        result['slices'] = summarize_prediction_slices(rows)
    if bootstrap_samples:
        result['bootstrap'] = {
            'samples': bootstrap_samples,
            'seed': bootstrap_seed,
            'metrics': bootstrap_intervals(rows, samples=bootstrap_samples, seed=bootstrap_seed),
        }
    return result


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('summarize-predictions', help='Summarize prediction JSONL metrics.')
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--croissant-path', type=Path, default=None)
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default='test')
    parser.add_argument('--include-slices', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=0)
    parser.add_argument('--bootstrap-seed', type=int, default=20260428)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    croissant_records = load_croissant_split(args.croissant_path, args.split) if args.croissant_path else None
    result = evaluate_predictions(
        load_jsonl(args.input_jsonl),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        croissant_records=croissant_records,
        include_slices=args.include_slices,
    )
    result['input_jsonl'] = str(args.input_jsonl)
    if args.croissant_path:
        result['croissant_path'] = str(args.croissant_path)
        result['split'] = args.split

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
