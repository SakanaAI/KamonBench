from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kamonbench.data.metadata import load_records, write_records
from kamonbench.evaluation.program_factors import ProgramFactors
from kamonbench.evaluation.program_support import COMBO_NAMES, ProgramRecord, program_records

SPLITS = ('train', 'dev', 'test')


def _combo_key(factors: ProgramFactors, combo: str) -> tuple[Any, ...]:
    if combo == 'C,M':
        return factors.containers, factors.motif
    if combo == 'R,M':
        return factors.modifier, factors.motif
    if combo == 'C,R,M':
        return factors.containers, factors.modifier, factors.motif
    raise ValueError(f'Unknown combo: {combo}')


def _factor_tokens(factors: ProgramFactors) -> tuple[str, ...]:
    return (*factors.containers, factors.modifier, factors.motif)


def _primitive_counts(records: list[ProgramRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(_factor_tokens(record.factors))
    return counts


def _can_remove(train_counts: Counter[str], group_counts: Counter[str], *, min_train_count: int) -> bool:
    return all(train_counts[token] - count >= min_train_count for token, count in group_counts.items())


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
    records: list[dict[str, Any]],
    *,
    combo: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    min_train_primitive_count: int = 1,
    label_mode: str = 'preserve',
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if combo not in COMBO_NAMES:
        raise ValueError(f'Unknown combo: {combo}')
    if label_mode not in {'preserve', 'program'}:
        raise ValueError(f'Unknown label mode: {label_mode}')
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('Split ratios must sum to 1.0')

    factor_records = program_records(records)
    groups: dict[tuple[Any, ...], list[ProgramRecord]] = defaultdict(list)
    for record in factor_records:
        groups[_combo_key(record.factors, combo)].append(record)

    train_counts = _primitive_counts(factor_records)
    group_counts = {key: _primitive_counts(records) for key, records in groups.items()}
    group_assignment = {key: 'train' for key in groups}

    total = len(factor_records)
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

    data = copy.deepcopy(records)
    split_by_image = {record.image: group_assignment[_combo_key(record.factors, combo)] for record in factor_records}
    component_splits: dict[int, str] = {}

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
        for record in data:
            record['analysis'] = [{'expr': token} for token in record['program_tokens']]
            record['translation'] = ' '.join(record['program_tokens'])

    summary = _split_summary(
        factor_records,
        group_assignment,
        combo,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        min_train_primitive_count=min_train_primitive_count,
    )
    return data, summary


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        'create-recombination-split', help='Create strict factor-recombination JSONL metadata from existing metadata.'
    )
    parser.add_argument('--metadata-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary-output', type=Path, default=None)
    parser.add_argument('--combo', choices=COMBO_NAMES, default='C,M')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--dev-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-train-primitive-count', type=int, default=1)
    parser.add_argument('--label-mode', choices=['preserve', 'program'], default='preserve')
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    records, summary = create_recombination_split(
        load_records(args.metadata_path),
        combo=args.combo,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_train_primitive_count=args.min_train_primitive_count,
        label_mode=args.label_mode,
    )
    split_name = args.combo.replace(',', '').lower()
    for record in records:
        record.setdefault('splits', {})[split_name] = record['split']
    write_records(args.output, records)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
