from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from kamonbench.data.metadata import load_records, write_records


def train_composite_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get('split') == 'train' and r.get('is_composite')]


def selected_train_ids(records: list[dict[str, Any]], selected_composite_paths: set[str]) -> set[int]:
    by_id = {int(r['id']): r for r in records}
    ids: set[int] = set()
    for record in records:
        if (
            record.get('split') == 'train'
            and record.get('is_composite')
            and record.get('image_path') in selected_composite_paths
        ):
            record_id = int(record['id'])
            ids.add(record_id)
            for component_id in record.get('component_ids', []):
                component = by_id.get(int(component_id))
                if component is None:
                    raise ValueError(f'Missing component id {component_id} referenced by record {record_id}')
                ids.add(int(component_id))
    return ids


def count_splits(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, bool]] = Counter(
        (record.get('split', ''), bool(record.get('is_composite'))) for record in records
    )
    return {
        split: {
            'composite': counts[(split, True)],
            'component': counts[(split, False)],
            'total': counts[(split, True)] + counts[(split, False)],
        }
        for split in ['train', 'dev', 'test']
    }


def write_subset(*, source_path: Path, output_path: Path, selected_composite_paths: set[str]) -> dict[str, Any]:
    records = load_records(source_path)
    train_ids = selected_train_ids(records, selected_composite_paths)
    subset = [record for record in records if record.get('split') != 'train' or int(record['id']) in train_ids]
    write_records(output_path, subset)
    return count_splits(subset)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--program-metadata', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=20260429)
    args = parser.parse_args()

    base_records = load_records(args.source_metadata)
    train_composites = train_composite_records(base_records)
    paths = sorted(record['image_path'] for record in train_composites)
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    if max(args.sizes) > len(paths):
        raise ValueError(f'Requested {max(args.sizes)} train composites, but only {len(paths)} are available')

    summary = {
        'seed': args.seed,
        'source_metadata': str(args.source_metadata),
        'program_metadata': str(args.program_metadata),
        'sizes': {},
    }
    for size in args.sizes:
        selected = set(paths[:size])
        size_dir = args.output_dir / f'train_{size}'
        base_counts = write_subset(
            source_path=args.source_metadata, output_path=size_dir / 'records.jsonl', selected_composite_paths=selected
        )
        program_counts = write_subset(
            source_path=args.program_metadata, output_path=size_dir / 'program.jsonl', selected_composite_paths=selected
        )
        summary['sizes'][str(size)] = {
            'base_counts': base_counts,
            'program_counts': program_counts,
            'selected_train_composites': size,
        }
        print(f'train_{size}: {base_counts["train"]["total"]} train rows')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
