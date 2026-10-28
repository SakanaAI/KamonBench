import hashlib
import io
import json
import subprocess
import sys
import tarfile

import pytest
from kamonbench.data.install_mon_data import install_mon_data
from kamonbench.data.reconstruct import (
    SourceArchive,
    file_sha256,
    reconstruct,
    render_record,
    verify_image,
    verify_records,
)
from PIL import Image


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


@pytest.fixture
def release(tmp_path, monkeypatch):
    image = Image.new('RGB', (224, 224), 'white')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    source_bytes = buffer.getvalue()
    member = 'mon-white-224/train/example.png'
    archive = tmp_path / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo(member)
        info.size = len(source_bytes)
        tar.addfile(info, io.BytesIO(source_bytes))
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    sources = [{'archive_path': member, 'sha256': hashlib.sha256(source_bytes).hexdigest()}]
    record = {
        'image_path': 'dataset01/component.png',
        'source_files': [member],
        'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
        'width': 224,
        'height': 224,
        'mode': 'RGB',
    }
    records = [
        record,
        {
            **record,
            'image_path': 'dataset01/composite.png',
            'reconstruction': {'motif': member, 'container': member, 'modifier': '豆'},
        },
    ]
    write_jsonl(metadata / 'records.jsonl', records)
    write_jsonl(metadata / 'sources.jsonl', sources)
    monkeypatch.setattr('kamonbench.data.reconstruct.ARCHIVE_SHA256', file_sha256(archive))
    return {
        'archive': archive,
        'metadata': metadata,
        'output': tmp_path / 'output',
        'records': records,
        'sources': sources,
        'source_bytes': source_bytes,
    }


def build(release, **kwargs):
    return reconstruct(
        source_archive=release['archive'],
        metadata_path=release['metadata'] / 'records.jsonl',
        output_dir=release['output'],
        **kwargs,
    )


@pytest.mark.parametrize('workers', [1, 2])
def test_reconstruct_verify_and_resume(release, workers):
    assert build(release, workers=workers) == {'written': 2, 'resumed': 0}
    assert verify_records(release['records'], release['output']) == 2
    output = release['output'] / release['records'][0]['image_path']
    before = output.stat().st_mtime_ns
    assert build(release, workers=workers) == {'written': 0, 'resumed': 2}
    assert output.stat().st_mtime_ns == before


def test_wrong_archive_hash(release):
    release['archive'].write_bytes(b'wrong archive')
    with pytest.raises(ValueError, match='Source archive SHA-256 mismatch'):
        build(release)
    assert not release['output'].exists()


def test_wrong_source_hash(release):
    write_jsonl(release['metadata'] / 'sources.jsonl', [{**release['sources'][0], 'sha256': '0' * 64}])
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        build(release)


def test_missing_source_member(release):
    write_jsonl(release['metadata'] / 'sources.jsonl', [{**release['sources'][0], 'archive_path': 'missing.png'}])
    with pytest.raises(FileNotFoundError, match='Missing source archive member'):
        build(release)


@pytest.mark.parametrize('name', ['../escape.png', '/absolute.png', 'nested/../../escape.png', 'C:\\escape.png'])
def test_unsafe_source_paths(release, name):
    sources = [{**release['sources'][0], 'archive_path': name}]
    with pytest.raises(ValueError, match='Unsafe relative path'):
        SourceArchive(release['archive'], sources)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_nonregular_source_member(release, kind, monkeypatch):
    with tarfile.open(release['archive'], 'w:gz') as tar:
        info = tarfile.TarInfo(release['sources'][0]['archive_path'])
        info.type = kind
        info.linkname = 'elsewhere.png'
        tar.addfile(info)
    monkeypatch.setattr('kamonbench.data.reconstruct.ARCHIVE_SHA256', file_sha256(release['archive']))
    with pytest.raises(ValueError, match='unique regular archive member'):
        SourceArchive(release['archive'], release['sources'])


def test_output_traversal(release):
    records = [{**release['records'][0], 'image_path': '../escape.png'}]
    write_jsonl(release['metadata'] / 'records.jsonl', records)
    with pytest.raises(ValueError, match='Unsafe relative path'):
        build(release)


def test_output_symlink_escape(release, tmp_path):
    release['output'].mkdir()
    (release['output'] / 'dataset01').symlink_to(tmp_path)
    with pytest.raises(ValueError, match='Unsafe output path'):
        build(release)


def test_corrupt_output_requires_force(release):
    build(release)
    output = release['output'] / release['records'][0]['image_path']
    output.write_bytes(b'broken PNG')
    with pytest.raises(ValueError, match='Output verification failed'):
        build(release)
    assert output.read_bytes() == b'broken PNG'
    with pytest.raises(OSError):
        verify_records(release['records'], release['output'])
    assert build(release, force=True) == {'written': 1, 'resumed': 1}
    assert verify_records(release['records'], release['output']) == 2


def test_missing_output_is_detected(release):
    with pytest.raises(FileNotFoundError):
        verify_records(release['records'], release['output'])


def test_pixel_mismatch_does_not_write(release):
    records = [{**release['records'][0], 'pixel_sha256': '0' * 64}]
    write_jsonl(release['metadata'] / 'records.jsonl', records)
    with pytest.raises(ValueError, match='RGB pixels'):
        build(release)
    assert not release['output'].exists()


def test_shape_and_mode_are_checked(release):
    for changes in ({'width': 223}, {'mode': 'L'}):
        with pytest.raises(ValueError, match='mode or dimensions'):
            verify_image(release['source_bytes'], {**release['records'][0], **changes})


def test_pixel_verification_accepts_different_png_encoding(release):
    with Image.open(io.BytesIO(release['source_bytes'])) as image:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', compress_level=0)
    assert buffer.getvalue() != release['source_bytes']
    verify_image(buffer.getvalue(), release['records'][0])


def test_unknown_modifier_is_rejected(release):
    record = release['records'][1]
    record['reconstruction']['modifier'] = 'unknown'
    with pytest.raises(ValueError, match='Unknown modifier'):
        render_record(record, {record['source_files'][0]: release['source_bytes']})


def test_source_install_is_local_verified_and_resumable(release, tmp_path):
    charges = tmp_path / 'basic_charges.csv'
    charges.write_text('mon-white-224/example.png,example,motif\n')
    kwargs = {
        'source_archive': release['archive'],
        'output_dir': release['output'],
        'basic_charges_csv': charges,
        'metadata_dir': release['metadata'],
    }
    install_mon_data(**kwargs)
    install_mon_data(**kwargs)
    installed = release['output'] / 'mon-white-224/example.png'
    assert installed.read_bytes() == release['source_bytes']
    assert (release['output'] / 'basic_charges.csv').read_text().split(',')[0] == str(installed)
    installed.write_bytes(b'conflict')
    with pytest.raises(FileExistsError, match='Conflicting output'):
        install_mon_data(**kwargs)
    install_mon_data(**kwargs, force=True)
    assert installed.read_bytes() == release['source_bytes']


def test_reconstruction_cli_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; from kamonbench.cli import _build_parser; '
                '_build_parser("reconstruct"); assert "torch" not in sys.modules'
            ),
        ],
        check=True,
    )
