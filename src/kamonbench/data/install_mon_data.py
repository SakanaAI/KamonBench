"""Install verified source images from a user-supplied local archive."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from .metadata import load_records
from .reconstruct import SourceArchive, output_path, write_bytes


def install_mon_data(
    *,
    source_archive: Path,
    output_dir: Path = Path('data'),
    basic_charges_csv: Path = Path('data/basic_charges.csv'),
    metadata_dir: Path = Path('metadata'),
    force: bool = False,
) -> None:
    sources = load_records(metadata_dir / 'sources.jsonl')
    charges = basic_charges_csv.read_bytes()
    rows = list(csv.reader(io.StringIO(charges.decode('utf-8'))))
    relocated = False
    for row in rows:
        if len(row) != 3:
            raise ValueError(f'Invalid basic charge row: {row}')
        path = str(output_dir / 'mon-white-224' / Path(row[0]).name)
        relocated |= path != row[0]
        row[0] = path
    if relocated:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator='\n').writerows(rows)
        charges = buffer.getvalue().encode('utf-8')
    names = set()
    with SourceArchive(source_archive, sources) as archive:
        source_bytes = archive.preload()
    for member, data in source_bytes.items():
        name = Path(member).name
        if name in names:
            raise ValueError(f'Duplicate source basename: {name}')
        names.add(name)
        destination = output_path(output_dir, f'mon-white-224/{name}')
        write_bytes(destination, data, force=force)
    write_bytes(output_path(output_dir, 'basic_charges.csv'), charges, force=force)
    print(f'Installed and verified {len(names)} source images in {output_dir / "mon-white-224"}.')


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser('install-mon-data', help='Install source assets from a verified local archive.')
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data'))
    parser.add_argument('--basic-charges-csv', type=Path, default=Path('data/basic_charges.csv'))
    parser.add_argument('--metadata-dir', type=Path, default=Path('metadata'))
    parser.add_argument('--force', action='store_true')
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    install_mon_data(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        basic_charges_csv=args.basic_charges_csv,
        metadata_dir=args.metadata_dir,
        force=args.force,
    )
    return 0
