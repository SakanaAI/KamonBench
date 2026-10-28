from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kamonbench.evaluation.program_factors import ProgramFactors, parse_program_factors

COMBO_NAMES = ('C,M', 'R,M', 'C,R,M')


@dataclass(frozen=True)
class ProgramRecord:
    image: str
    split: str
    factors: ProgramFactors


def program_records(records: list[dict[str, Any]]) -> list[ProgramRecord]:
    result: list[ProgramRecord] = []
    for record in records:
        if not record.get('is_composite'):
            continue
        factors = parse_program_factors(' '.join(record['program_tokens']))
        if factors is None:
            raise ValueError(f'Invalid program tokens for {record["image_path"]}')
        result.append(ProgramRecord(image=record['image_path'], split=record['split'], factors=factors))
    return result
