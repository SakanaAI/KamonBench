from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kamonbench.evaluation.program_support import (
    COMBO_NAMES,
    ProgramFactors,
    ProgramRecord,
    load_json,
    program_records_from_croissant,
)

SPLITS = ('train', 'dev', 'test')


def _combo_key(factors: ProgramFactors, combo: str) -> tuple[Any, ...]:
    if combo == 'C,M':
        return factors.containers, factors.motif
    if combo == 'R,M':
        return factors.relation, factors.motif
    if combo == 'C,R,M':
        return factors.containers, factors.relation, factors.motif
    raise ValueError(f'Unknown combo: {combo}')


def _factor_tokens(factors: ProgramFactors) -> tuple[str, ...]:
    return (*factors.containers, factors.relation, factors.motif)


def _primitive_counts(records: list[ProgramRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_factor_tokens(record.factors))
    return counts


def _can_remove(train_counts: Counter[str], group_counts: Counter[str], *, min_train_count: int) -> bool:
    return all(train_counts[token] - count >= min_train_count for token, count in group_counts.items())


def _tokens_to_analysis(tokens: tuple[str, ...]) -> list[dict[str, str]]:
    return [{'expr': token} for token in tokens]


def _set_program_field_descriptions(croissant: dict[str, Any]) -> None:
    for field in croissant['recordSet'][0].get('field', []):
        if field.get('name') == 'translation':
            field['description'] = 'Whitespace-tokenized program label text'
        elif field.get('name') == 'analysis':
            field['description'] = 'Structured program label tokens'


def _apply_program_labels(croissant: dict[str, Any], records_by_image: dict[str, ProgramRecord]) -> None:
    data = croissant['recordSet'][0]['data']
    component_tokens: dict[int, tuple[str, ...]] = {}

    for record in data:
        if not record.get('is_composite'):
            continue
        factors = records_by_image.get(record['image_path'])
        if factors is None:
            continue
        component_ids = record.get('component_ids', [])
        if component_ids:
            component_tokens[component_ids[0]] = (factors.factors.motif,)
            for component_id, container in zip(component_ids[1:], factors.factors.containers, strict=True):
                component_tokens[component_id] = (container,)

    for record in data:
        if record.get('is_composite'):
            factors = records_by_image.get(record['image_path'])
            if factors is None:
                continue
            tokens = _factor_tokens(factors.factors)
        else:
            tokens = component_tokens.get(record['id'])
            if tokens is None:
                continue
        record['analysis'] = _tokens_to_analysis(tokens)
        record['translation'] = ' '.join(tokens)

    _set_program_field_descriptions(croissant)


def _split_summary(
    records: list[ProgramRecord],
    group_assignment: dict[tuple[Any, ...], str],
    combo: str,
    *,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    min_train_primitive_count: int,
) -> dict[str, Any]:
    split_counts = {split: 0 for split in SPLITS}
    split_group_counts = {split: 0 for split in SPLITS}
    primitive_counts = {split: Counter() for split in SPLITS}

    for record in records:
        split = group_assignment[_combo_key(record.factors, combo)]
        split_counts[split] += 1
        primitive_counts[split].update(_factor_tokens(record.factors))

    for split in group_assignment.values():
        split_group_counts[split] += 1

    train_keys = {key for key, split in group_assignment.items() if split == 'train'}
    heldout_counts = {
        split: sum(
            1 for key, assigned_split in group_assignment.items() if assigned_split == split and key not in train_keys
        )
        for split in ('dev', 'test')
    }
    train_min = min(primitive_counts['train'].values()) if primitive_counts['train'] else 0

    return {
        'combo': combo,
        'seed': seed,
        'ratios': {'train': train_ratio, 'dev': dev_ratio, 'test': test_ratio},
        'min_train_primitive_count': min_train_primitive_count,
        'composite_counts': split_counts,
        'combo_group_counts': split_group_counts,
        'heldout_combo_groups': heldout_counts,
        'train_min_primitive_count': train_min,
        'train_primitive_count': len(primitive_counts['train']),
        'dev_primitive_count': len(primitive_counts['dev']),
        'test_primitive_count': len(primitive_counts['test']),
    }


def create_recombination_split(
    croissant: dict[str, Any],
    *,
    basic_charges_csv: Path,
    combo: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    min_train_primitive_count: int = 1,
    label_mode: str = 'preserve',
) -> tuple[dict[str, Any], dict[str, Any]]:
    if combo not in COMBO_NAMES:
        raise ValueError(f'Unknown combo: {combo}')
    if label_mode not in {'preserve', 'program'}:
        raise ValueError(f'Unknown label mode: {label_mode}')
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('Split ratios must sum to 1.0')

    program_records = program_records_from_croissant(croissant, basic_charges_csv=basic_charges_csv)
    groups: dict[tuple[Any, ...], list[ProgramRecord]] = defaultdict(list)
    for record in program_records:
        groups[_combo_key(record.factors, combo)].append(record)

    train_counts = _primitive_counts(program_records)
    group_counts = {key: _primitive_counts(records) for key, records in groups.items()}
    group_assignment = {key: 'train' for key in groups}

    total = len(program_records)
    targets = {'dev': int(total * dev_ratio), 'test': total - int(total * train_ratio) - int(total * dev_ratio)}
    split_counts = {'dev': 0, 'test': 0}

    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        candidate_splits = sorted(('dev', 'test'), key=lambda split: targets[split] - split_counts[split], reverse=True)
        for split in candidate_splits:
            if split_counts[split] >= targets[split]:
                continue
            if not _can_remove(train_counts, group_counts[key], min_train_count=min_train_primitive_count):
                continue
            group_assignment[key] = split
            train_counts.subtract(group_counts[key])
            split_counts[split] += len(groups[key])
            break

    new_croissant = copy.deepcopy(croissant)
    records_by_image = {record.image: record for record in program_records}
    split_by_image = {record.image: group_assignment[_combo_key(record.factors, combo)] for record in program_records}
    component_splits: dict[int, str] = {}

    data = new_croissant['recordSet'][0]['data']
    for record in data:
        if not record.get('is_composite'):
            continue
        split = split_by_image.get(record['image_path'])
        if split is None:
            continue
        record['split'] = split
        for component_id in record.get('component_ids', []):
            previous = component_splits.get(component_id)
            if previous is not None and previous != split:
                raise ValueError(f'Component {component_id} is shared across multiple splits')
            component_splits[component_id] = split

    for record in data:
        if not record.get('is_composite') and record['id'] in component_splits:
            record['split'] = component_splits[record['id']]

    if label_mode == 'program':
        _apply_program_labels(new_croissant, records_by_image)

    summary = _split_summary(
        program_records,
        group_assignment,
        combo,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        min_train_primitive_count=min_train_primitive_count,
    )
    return new_croissant, summary


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        'create-recombination-split',
        help='Create a strict factor-recombination Croissant split from existing metadata.',
    )
    parser.add_argument('--croissant-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-output', type=Path, default=None)
    parser.add_argument('--basic-charges-csv', type=Path, default=Path('data/basic_charges.csv'))
    parser.add_argument('--combo', choices=COMBO_NAMES, default='C,M')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-train-primitive-count', type=int, default=1)
    parser.add_argument('--label-mode', choices=['preserve', 'program'], default='preserve')
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    croissant, summary = create_recombination_split(
        load_json(args.croissant_path),
        basic_charges_csv=args.basic_charges_csv,
        combo=args.combo,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_train_primitive_count=args.min_train_primitive_count,
        label_mode=args.label_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(croissant, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
