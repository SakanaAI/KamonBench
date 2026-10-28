"""Load dataset records and label/split selections."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records without modifying the source, IDs, or ordering."""
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    with path.open(encoding='utf-8') as stream:
        view = json.load(stream)
    if 'records' not in view or not isinstance(view['records'], str):
        raise ValueError(f'Expected JSONL records or a metadata view: {path}')
    source = path.parent / view['records']
    if source.suffix != '.jsonl':
        raise ValueError(f'Metadata view must reference a JSONL file: {path}')
    records = load_records(source)
    labels = view.get('labels')
    if labels not in (None, 'program'):
        raise ValueError(f'Unknown metadata label view: {labels!r}')
    split = view.get('split', 'standard')
    for record in records:
        if split not in record.get('splits', {}):
            raise ValueError(f'Record {record["id"]} has no {split!r} split')
        record['split'] = record['splits'][split]
        if labels == 'program':
            tokens = record.get('program_tokens')
            if not tokens or not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
                raise ValueError(f'Record {record["id"]} has no valid program_tokens')
            record['analysis'] = [{'expr': token} for token in tokens]
            record['translation'] = ' '.join(tokens)
    return records


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
