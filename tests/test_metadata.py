import json

import pytest
from kamonbench.data.metadata import load_records, write_records


@pytest.fixture
def records():
    return [
        {
            'id': 17,
            'image_path': 'dataset01/crest.png',
            'description': '丸に蔦',
            'analysis': [{'expr': '丸'}, {'expr': 'に'}, {'expr': '蔦'}],
            'translation': 'ivy in a circle',
            'is_composite': True,
            'component_ids': [20, 22],
            'split': 'train',
            'program_tokens': ['C:01', 'X:0', 'M:02'],
            'splits': {'standard': 'train', 'cm': 'test', 'rm': 'dev', 'crm': 'train'},
        }
    ]


def test_jsonl_preserves_records(tmp_path, records):
    path = tmp_path / 'records.jsonl'
    write_records(path, records)
    assert load_records(path) == records


@pytest.mark.parametrize('split', ['standard', 'cm', 'rm', 'crm'])
def test_program_view_preserves_identity_and_source(tmp_path, records, split):
    source = tmp_path / 'records.jsonl'
    write_records(source, records)
    before = source.read_bytes()
    view = tmp_path / 'program.json'
    view.write_text(json.dumps({'records': 'records.jsonl', 'labels': 'program', 'split': split}))
    row = load_records(view)[0]
    assert row['id'] == 17
    assert row['component_ids'] == [20, 22]
    assert row['image_path'] == records[0]['image_path']
    assert row['description'] == records[0]['description']
    assert row['analysis'] == [{'expr': t} for t in records[0]['program_tokens']]
    assert row['translation'] == 'C:01 X:0 M:02'
    assert row['split'] == records[0]['splits'][split]
    assert source.read_bytes() == before
    assert load_records(source) == records


def test_missing_split_does_not_silently_use_standard(tmp_path, records):
    write_records(tmp_path / 'records.jsonl', records)
    view = tmp_path / 'view.json'
    view.write_text(json.dumps({'records': 'records.jsonl', 'split': 'unknown'}))
    with pytest.raises(ValueError, match='has no.*split'):
        load_records(view)


def test_recombination_preserves_records_and_component_splits(tmp_path):
    from kamonbench.data.recombination_split import create_recombination_split

    records = []
    for c in range(4):
        for m in range(4):
            record_id = len(records)
            records.extend(
                [
                    {
                        'id': record_id,
                        'image_path': f'{c}-{m}.png',
                        'is_composite': True,
                        'split': 'train',
                        'program_tokens': [f'C:{c}', 'X:0', f'M:{m}'],
                        'component_ids': [record_id + 1, record_id + 2],
                    },
                    {
                        'id': record_id + 1,
                        'image_path': f'{c}-{m}_base.png',
                        'is_composite': False,
                        'split': 'train',
                        'program_tokens': [f'M:{m}'],
                    },
                    {
                        'id': record_id + 2,
                        'image_path': f'{c}-{m}_container.png',
                        'is_composite': False,
                        'split': 'train',
                        'program_tokens': [f'C:{c}'],
                    },
                ]
            )
    path = tmp_path / 'records.jsonl'
    write_records(path, records)
    rows, _ = create_recombination_split(load_records(path), combo='C,M', seed=3)
    assert [row['id'] for row in rows] == [row['id'] for row in records]
    assert [row['program_tokens'] for row in rows] == [row['program_tokens'] for row in records]
    assert {row['split'] for row in rows} == {'train', 'dev', 'test'}
    by_id = {row['id']: row for row in rows}
    for row in rows:
        for component_id in row.get('component_ids', []):
            assert by_id[component_id]['split'] == row['split']
    assert load_records(path) == records


def test_converter_links_components_and_preserves_labels(tmp_path):
    import argparse

    from kamonbench.data.convert import add_parser

    (tmp_path / 'synthetic.jsonl').write_text(
        json.dumps(
            {
                'description': '丸に蔦',
                'images': [
                    {
                        'path': 'dataset01/crest.png',
                        'base_motif': {'path': 'dataset01/crest_base.png', 'description': '蔦'},
                        'containers': [{'path': 'dataset01/crest_container.png', 'description': '丸'}],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + '\n'
    )
    csv = tmp_path / 'charges.csv'
    csv.write_text('images/01.png,丸,container\nimages/02.png,蔦,motif\n')
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            'convert-to-metadata',
            '--dataset-dir',
            str(tmp_path),
            '--basic-charges-csv',
            str(csv),
            '--label-mode',
            'program',
            '--seed',
            '5',
        ]
    )
    assert args._fn(args) == 0
    rows = load_records(tmp_path / 'records.jsonl')
    assert [row['id'] for row in rows] == [0, 1, 2]
    assert rows[0]['component_ids'] == [1, 2]
    assert [row['program_tokens'] for row in rows] == [['C:01', 'X:0', 'M:02'], ['M:02'], ['C:01']]
    assert {row['split'] for row in rows} == {'test'}
    for row in rows:
        assert row['analysis'] == [{'expr': token} for token in row['program_tokens']]
        assert row['translation'] == ' '.join(row['program_tokens'])
        assert row['splits'] == {'standard': row['split']}


def test_reduced_training_metadata_keeps_selected_components_and_heldout_rows(tmp_path):
    from runpy import run_path

    subset = run_path('scripts/make_train_size_metadata.py')['write_subset']
    rows = [
        {'id': 1, 'image_path': 'a.png', 'split': 'train', 'is_composite': True, 'component_ids': [2]},
        {'id': 2, 'image_path': 'a_base.png', 'split': 'train', 'is_composite': False},
        {'id': 3, 'image_path': 'b.png', 'split': 'train', 'is_composite': True, 'component_ids': [4]},
        {'id': 4, 'image_path': 'b_base.png', 'split': 'train', 'is_composite': False},
        {'id': 5, 'image_path': 'dev.png', 'split': 'dev', 'is_composite': True},
        {'id': 6, 'image_path': 'test.png', 'split': 'test', 'is_composite': True},
    ]
    source = tmp_path / 'records.jsonl'
    output = tmp_path / 'subset.jsonl'
    write_records(source, rows)
    counts = subset(source_path=source, output_path=output, selected_composite_paths={'b.png'})
    assert load_records(output) == rows[2:]
    assert load_records(source) == rows
    assert counts['train'] == {'composite': 1, 'component': 1, 'total': 2}
