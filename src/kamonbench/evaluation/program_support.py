from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from kamonbench.data.croissant import PROGRAM_MODIFIER_TO_ID, _compact_description
from kamonbench.generator.rules import EXCLUDED_CONTAINERS

COMBO_NAMES = ('C,M', 'R,M', 'C,R,M')


@dataclass(frozen=True)
class ProgramFactors:
    containers: tuple[str, ...]
    relation: str
    motif: str

    @property
    def text(self) -> str:
        return ' '.join([*self.containers, self.relation, self.motif])


@dataclass(frozen=True)
class ProgramRecord:
    image: str
    split: str
    factors: ProgramFactors


@dataclass(frozen=True)
class PredictionRow:
    image: str
    reference: ProgramFactors
    predicted: ProgramFactors | None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _charge_ids(csv_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    containers: dict[str, str] = {}
    motifs: dict[str, str] = {}
    with csv_path.open('r', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            path, name, kind = row[:3]
            has_container_relation = 'に' in name
            if kind == 'container':
                if name in EXCLUDED_CONTAINERS:
                    if not has_container_relation:
                        motifs[name] = Path(path).stem
                else:
                    containers[name] = Path(path).stem
            elif not has_container_relation:
                motifs[name] = Path(path).stem
    return containers, motifs


def parse_program_factors(text: str) -> ProgramFactors | None:
    containers: list[str] = []
    relation = None
    motif = None
    for token in text.split():
        if token.startswith('C:'):
            containers.append(token)
        elif token.startswith('X:') and relation is None:
            relation = token
        elif token.startswith('M:') and motif is None:
            motif = token
        else:
            return None
    if relation is None or motif is None:
        return None
    return ProgramFactors(tuple(containers), relation, motif)


def _include_composite_image(image_path: str) -> bool:
    return '_base' not in image_path and '_container' not in image_path


def program_records_from_croissant(croissant: dict[str, Any], *, basic_charges_csv: Path) -> list[ProgramRecord]:
    records = croissant['recordSet'][0]['data']
    by_id = {record['id']: record for record in records}
    container_ids, motif_ids = _charge_ids(basic_charges_csv)
    program_records: list[ProgramRecord] = []

    for record in records:
        if not record.get('is_composite'):
            continue

        components = [by_id[component_id] for component_id in record.get('component_ids', [])]
        if not components:
            continue

        base_term = components[0]['description']
        container_terms = [component['description'] for component in components[1:]]
        description = _compact_description(record['description'])
        base_suffix = _compact_description(base_term)
        container_prefix = 'に'.join(_compact_description(term) for term in container_terms)
        if container_prefix:
            container_prefix += 'に'

        if not description.startswith(container_prefix) or not description.endswith(base_suffix):
            raise ValueError(f'Could not decompose {record["image_path"]}: {record["description"]!r}')

        relation_name = description[len(container_prefix) : len(description) - len(base_suffix)]
        if relation_name not in PROGRAM_MODIFIER_TO_ID:
            raise ValueError(f'Unknown relation in {record["image_path"]}: {relation_name!r}')
        if base_term not in motif_ids:
            raise ValueError(f'Unknown motif in {record["image_path"]}: {base_term!r}')

        containers = []
        for term in container_terms:
            if term not in container_ids:
                raise ValueError(f'Unknown container in {record["image_path"]}: {term!r}')
            containers.append(f'C:{container_ids[term]}')

        factors = ProgramFactors(
            containers=tuple(containers),
            relation=f'X:{PROGRAM_MODIFIER_TO_ID[relation_name]}',
            motif=f'M:{motif_ids[base_term]}',
        )
        program_records.append(ProgramRecord(image=record['image_path'], split=record['split'], factors=factors))

    return program_records


def prediction_rows(predictions: Iterable[dict[str, Any]]) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for item in predictions:
        image = item.get('image', '')
        if not _include_composite_image(image):
            continue
        reference = parse_program_factors(item.get('reference', ''))
        if reference is None:
            continue
        rows.append(
            PredictionRow(image=image, reference=reference, predicted=parse_program_factors(item.get('predicted', '')))
        )
    return rows


def _combo_key(factors: ProgramFactors, combo: str) -> tuple[Any, ...]:
    if combo == 'C,M':
        return factors.containers, factors.motif
    if combo == 'R,M':
        return factors.relation, factors.motif
    if combo == 'C,R,M':
        return factors.containers, factors.relation, factors.motif
    raise ValueError(f'Unknown combo: {combo}')


def _metric(rows: list[PredictionRow]) -> dict[str, Any]:
    correct = sum(row.predicted == row.reference for row in rows)
    total = len(rows)
    return {'correct': correct, 'total': total, 'value': correct / total if total else None}


def evaluate_support_slices(
    predictions: Iterable[dict[str, Any]], train_records: Iterable[ProgramRecord]
) -> dict[str, Any]:
    pred_rows = prediction_rows(predictions)
    train = list(train_records)
    support = {combo: {_combo_key(record.factors, combo) for record in train} for combo in COMBO_NAMES}

    slices: dict[str, Any] = {}
    for combo in COMBO_NAMES:
        absent = [row for row in pred_rows if _combo_key(row.reference, combo) not in support[combo]]
        present = [row for row in pred_rows if _combo_key(row.reference, combo) in support[combo]]
        slices[combo] = {
            'absent_from_train': _metric(absent),
            'present_in_train': _metric(present),
            'train_support_count': len(support[combo]),
        }

    return {'train_rows': len(train), 'prediction_rows': len(pred_rows), 'slices': slices}


def markdown_table(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Held-out combo | N absent | Acc absent | N present | Acc present |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model, result in results.items():
        for combo in COMBO_NAMES:
            summary = result['slices'][combo]
            absent = summary['absent_from_train']
            present = summary['present_in_train']
            absent_value = '—' if absent['value'] is None else f'{absent["value"]:.3f}'
            present_value = '—' if present['value'] is None else f'{present["value"]:.3f}'
            lines.append(
                f'| {model} | {combo} | {absent["total"]} | {absent_value} | {present["total"]} | {present_value} |'
            )
    return '\n'.join(lines) + '\n'


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('input_jsonl', type=Path)
    parser.add_argument('--croissant-path', type=Path, required=True)
    parser.add_argument('--basic-charges-csv', type=Path, default=Path('data/basic_charges.csv'))
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('evaluate-program-support', help='Evaluate held-out program factor combinations.')
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    records = program_records_from_croissant(load_json(args.croissant_path), basic_charges_csv=args.basic_charges_csv)
    train = [record for record in records if record.split == 'train']
    result = evaluate_support_slices(load_jsonl(args.input_jsonl), train)
    result['input_jsonl'] = str(args.input_jsonl)
    result['croissant_path'] = str(args.croissant_path)
    result['basic_charges_csv'] = str(args.basic_charges_csv)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        model_name = args.model_name or args.input_jsonl.stem
        args.markdown_output.write_text(markdown_table({model_name: result}), encoding='utf-8')

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
