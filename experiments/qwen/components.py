"""Compare motif correctness on paired isolated components and composites."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from kamonbench.data.metadata import load_records


def target(record: dict[str, Any]) -> str:
    return ' '.join(item['expr'] for item in record['analysis'])


def motif(text: str) -> str | None:
    tokens = [token for token in text.split() if token.startswith('M:')]
    return tokens[0] if len(tokens) == 1 else None


def transition(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    counts = Counter(
        'both'
        if isolated and composite
        else 'isolated_only'
        if isolated
        else 'composite_only'
        if composite
        else 'neither'
        for isolated, composite in rows
    )
    isolated_correct = counts['both'] + counts['isolated_only']
    composite_correct = counts['both'] + counts['composite_only']
    return {
        'rows': len(rows),
        **{name: counts[name] for name in ('both', 'isolated_only', 'composite_only', 'neither')},
        'isolated_correct': isolated_correct,
        'isolated_exact': isolated_correct / len(rows) if rows else None,
        'composite_correct': composite_correct,
        'composite_motif': composite_correct / len(rows) if rows else None,
        'retention_given_isolated_correct': counts['both'] / isolated_correct if isolated_correct else None,
    }


def read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {int(r['id'] if 'id' in r else r['record_id']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate record IDs in {path}')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('metadata/program.json'))
    parser.add_argument('--qwen', type=Path, required=True, help='Predictions for all standard test records.')
    parser.add_argument('--vit', type=Path, required=True, help='Predictions for all standard test records.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.metadata)
    by_id = {r['id']: r for r in records}
    models = {'qwen': read_predictions(args.qwen), 'vit': read_predictions(args.vit)}
    pairs = {name: [] for name in models}
    for composite in records:
        if composite['split'] != 'test' or not composite['is_composite']:
            continue
        ids = [i for i in composite['component_ids'] if target(by_id[i]).startswith('M:')]
        if len(ids) != 1:
            raise ValueError(f'Expected one motif component for {composite["id"]}')
        component_id = ids[0]
        reference = motif(target(composite))
        for name, predictions in models.items():
            isolated, combined = predictions[component_id], predictions[composite['id']]
            for row, record in [(isolated, by_id[component_id]), (combined, composite)]:
                if ' '.join(row['reference'].split()) != target(record):
                    raise ValueError(f'Reference mismatch for {record["id"]}')
            pairs[name].append((motif(isolated['predicted']) == reference, motif(combined['predicted']) == reference))
    result = {name: transition(rows) for name, rows in pairs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
