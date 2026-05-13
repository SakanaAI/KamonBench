from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)['recordSet'][0]['data']


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


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


def subset_metadata(
    metadata: dict[str, Any], *, train_ids: set[int], train_composites: int, seed: int, source_path: Path
) -> dict[str, Any]:
    data = metadata['recordSet'][0]['data']
    subset_records = [record for record in data if record.get('split') != 'train' or int(record['id']) in train_ids]
    result = dict(metadata)
    result['recordSet'] = [dict(metadata['recordSet'][0])]
    result['recordSet'][0]['data'] = subset_records
    result['training_subset'] = {
        'source_croissant': str(source_path),
        'train_composite_examples': train_composites,
        'sampling_seed': seed,
        'selection': 'deterministic shuffled subset of original train composites; dev/test unchanged',
    }
    return result


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


def write_subset(
    *, source_path: Path, output_path: Path, selected_composite_paths: set[str], train_composites: int, seed: int
) -> dict[str, Any]:
    metadata = load_metadata(source_path)
    records = metadata['recordSet'][0]['data']
    train_ids = selected_train_ids(records, selected_composite_paths)
    subset = subset_metadata(
        metadata, train_ids=train_ids, train_composites=train_composites, seed=seed, source_path=source_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return count_splits(subset['recordSet'][0]['data'])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-croissant', type=Path, required=True)
    parser.add_argument('--program-croissant', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=20260429)
    args = parser.parse_args()

    base_records = load_records(args.source_croissant)
    train_composites = train_composite_records(base_records)
    paths = sorted(record['image_path'] for record in train_composites)
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    if max(args.sizes) > len(paths):
        raise ValueError(f'Requested {max(args.sizes)} train composites, but only {len(paths)} are available')

    summary = {
        'seed': args.seed,
        'source_croissant': str(args.source_croissant),
        'program_croissant': str(args.program_croissant),
        'sizes': {},
    }
    for size in args.sizes:
        selected = set(paths[:size])
        size_dir = args.output_dir / f'train_{size}'
        base_counts = write_subset(
            source_path=args.source_croissant,
            output_path=size_dir / 'kamon_croissant.json',
            selected_composite_paths=selected,
            train_composites=size,
            seed=args.seed,
        )
        program_counts = write_subset(
            source_path=args.program_croissant,
            output_path=size_dir / 'kamon_croissant_program.json',
            selected_composite_paths=selected,
            train_composites=size,
            seed=args.seed,
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
