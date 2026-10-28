"""Reconstruct fixed benchmark recipes from a user-supplied upstream archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Self

from PIL import Image

from kamonbench.data.metadata import load_records
from kamonbench.generator.ops import BEAN, NORMAL, StackMode, inside, stack

DEFAULT_METADATA = Path('metadata/records.jsonl')
ARCHIVE_SHA256 = '0491b97c22e0bf240d77acdda0a84b6cae378ff938546e933b1783b32abff199'
STACK_MODES = {'三つ盛り': StackMode.BASIC, '尻合せ三つ': StackMode.SHIRI, '頭合せ三つ': StackMode.ATAMA}
_WORKER_SOURCES: dict[str, bytes] = {}


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_sha256(data: bytes, expected: str, label: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {label}: expected {expected}, got {actual}')


def safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe relative path: {name!r}')
    return Path(*path.parts)


def output_path(root: Path, name: str) -> Path:
    path = root / safe_relative(name)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe output path: {name!r}')
    return path


class SourceArchive:
    """Read only named, regular archive members after checking their hashes."""

    def __init__(self, path: Path, sources: list[dict]):
        actual = file_sha256(path)
        if actual != ARCHIVE_SHA256:
            raise ValueError(f'Source archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}')
        self.sources = {}
        for source in sources:
            name = source['archive_path']
            safe_relative(name)
            if name in self.sources:
                raise ValueError(f'Duplicate source manifest entry: {name}')
            self.sources[name] = source
        self.tar = tarfile.open(path, 'r:*')  # noqa: SIM115 - closed by this context manager
        self.members = {}
        self.cache: dict[str, bytes] = {}
        try:
            for member in self.tar.getmembers():
                if member.name not in self.sources:
                    continue
                if not member.isfile() or member.name in self.members:
                    raise ValueError(f'Expected a unique regular archive member: {member.name}')
                self.members[member.name] = member
            missing = self.sources.keys() - self.members.keys()
            if missing:
                raise FileNotFoundError(f'Missing source archive member: {min(missing)}')
        except Exception:
            self.tar.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.tar.close()

    def read(self, name: str) -> bytes:
        safe_relative(name)
        if name not in self.sources:
            raise ValueError(f'Unlisted source archive member: {name}')
        if name not in self.cache:
            with self.tar.extractfile(self.members[name]) as stream:
                data = stream.read()
            check_sha256(data, self.sources[name]['sha256'], name)
            self.cache[name] = data
        return self.cache[name]

    def preload(self) -> dict[str, bytes]:
        # Archive order avoids repeated backward seeks through the gzip stream.
        for name in self.members:
            self.read(name)
        return self.cache


def source_image(sources: dict[str, bytes], name: str) -> Image.Image:
    safe_relative(name)
    if name not in sources:
        raise ValueError(f'Unlisted source archive member: {name}')
    with Image.open(io.BytesIO(sources[name])) as image:
        return image.convert('RGB')


def render_record(record: dict, sources: dict[str, bytes]) -> Image.Image:
    source_files = record['source_files']
    recipe = record.get('reconstruction')
    if recipe is None:
        if len(source_files) != 1:
            raise ValueError(f'Component must have exactly one source: {record["image_path"]}')
        return source_image(sources, source_files[0])
    motif_path, container_path, modifier = recipe['motif'], recipe['container'], recipe['modifier']
    expected_sources = {motif_path} | ({container_path} if container_path is not None else set())
    if set(source_files) != expected_sources:
        raise ValueError(f'Reconstruction sources disagree: {record["image_path"]}')
    if modifier not in ('', '豆', '覗き', *STACK_MODES):
        raise ValueError(f'Unknown modifier: {modifier!r}')
    motif = source_image(sources, motif_path)
    if modifier in STACK_MODES:
        motif = stack(motif, mode=STACK_MODES[modifier])
    if container_path is None:
        if modifier not in STACK_MODES:
            raise ValueError(f'Invalid containerless modifier: {modifier!r}')
        return motif
    container = source_image(sources, container_path)
    if modifier == '豆':
        motif = motif.resize((int(motif.height * BEAN), int(motif.width * BEAN)))
        return inside(container, motif, scale=1.0)
    return inside(container, motif, scale=NORMAL, peek=modifier == '覗き')


def verify_image(data: bytes, record: dict) -> None:
    """Verify decoded RGB pixels, dimensions, and mode."""
    name = record['image_path']
    with Image.open(io.BytesIO(data)) as image:
        expected_size = (record['width'], record['height'])
        if image.mode != record.get('mode', 'RGB') or image.size != expected_size:
            raise ValueError(f'Image mode or dimensions disagree: {name}')
        check_sha256(image.convert('RGB').tobytes(), record['pixel_sha256'], f'{name} RGB pixels')


def write_bytes(path: Path, data: bytes, *, force: bool) -> bool:
    """Resume identical files; atomically replace conflicts only when requested."""
    if path.exists():
        if path.read_bytes() == data:
            return False
        if not force:
            raise FileExistsError(f'Conflicting output: {path}; pass --force to replace it')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def verify_records(records: list[dict], output_dir: Path) -> int:
    for record in records:
        verify_image(output_path(output_dir, record['image_path']).read_bytes(), record)
    return len(records)


def _worker_init(sources: dict[str, bytes]) -> None:
    global _WORKER_SOURCES
    _WORKER_SOURCES = sources


def _render_png(record: dict, sources: dict[str, bytes] | None = None) -> bytes:
    image = render_record(record, _WORKER_SOURCES if sources is None else sources)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    data = buffer.getvalue()
    verify_image(data, record)
    return data


def reconstruct(
    *,
    source_archive: Path,
    metadata_path: Path = DEFAULT_METADATA,
    output_dir: Path = Path('.'),
    force: bool = False,
    workers: int = 1,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError('workers must be at least 1')
    records = load_records(metadata_path)
    sources = load_records(metadata_path.parent / 'sources.jsonl')
    with SourceArchive(source_archive, sources) as archive:
        source_bytes = archive.preload()
    pending = []
    resumed = 0
    seen = set()
    for record in records:
        path = output_path(output_dir, record['image_path'])
        if path in seen:
            raise ValueError(f'Duplicate image path: {record["image_path"]}')
        seen.add(path)
        if path.exists():
            try:
                verify_image(path.read_bytes(), record)
            except (OSError, ValueError):
                if not force:
                    raise ValueError(f'Output verification failed: {path}; pass --force to reconstruct it') from None
            else:
                resumed += 1
                continue
        pending.append(record)
    with ExitStack() as context:
        if workers == 1:
            rendered = map(partial(_render_png, sources=source_bytes), pending)
        else:
            executor = context.enter_context(
                ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(source_bytes,))
            )
            rendered = executor.map(_render_png, pending, chunksize=8)
        for index, (record, data) in enumerate(zip(pending, rendered, strict=True), 1):
            write_bytes(output_path(output_dir, record['image_path']), data, force=force)
            if index % 500 == 0:
                print(f'Reconstructed and verified {index}/{len(pending)} images', flush=True)
    return {'written': len(pending), 'resumed': resumed}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--metadata-path', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    parser.set_defaults(_fn=_run)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('reconstruct', help='Reconstruct fixed images from a local upstream archive.')
    add_arguments(parser)
    verify = subparsers.add_parser('verify-data', help='Verify reconstructed image pixels, dimensions, and modes.')
    verify.add_argument('--metadata-path', type=Path, default=DEFAULT_METADATA)
    verify.add_argument('--output-dir', type=Path, default=Path('.'))
    verify.set_defaults(_fn=_verify)


def _run(args: argparse.Namespace) -> int:
    result = reconstruct(
        source_archive=args.source_archive,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        force=args.force,
        workers=args.workers,
    )
    print(f'Reconstructed {result["written"]} images; verified {result["resumed"]} existing images.')
    return 0


def _verify(args: argparse.Namespace) -> int:
    count = verify_records(load_records(args.metadata_path), args.output_dir)
    print(f'Verified {count} images.')
    return 0
