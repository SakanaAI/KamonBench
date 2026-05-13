from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .rules import EXCLUDED_CONTAINERS


@dataclass(frozen=True)
class ChargeAssets:
    containers: dict[str, str]
    motifs: dict[str, str]
    all: dict[str, str]


def load_basic_charges(csv_path: Path) -> ChargeAssets:
    containers: dict[str, str] = {}
    motifs: dict[str, str] = {}

    with csv_path.open('r', encoding='utf-8') as stream:
        reader = csv.reader(stream, delimiter=',', quotechar='"')
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f'Invalid row in {csv_path}: {row}')
            path, name, kind = row[0], row[1], row[2]
            has_container_relation = 'に' in name

            if kind == 'container':
                if name in EXCLUDED_CONTAINERS:
                    if not has_container_relation:
                        motifs[name] = path
                else:
                    containers[name] = path
            else:
                if not has_container_relation:
                    motifs[name] = path

    all_charges = dict(containers)
    all_charges.update(motifs)
    return ChargeAssets(containers=containers, motifs=motifs, all=all_charges)


def default_basic_charges_path() -> Path:
    return Path('data/basic_charges.csv')
