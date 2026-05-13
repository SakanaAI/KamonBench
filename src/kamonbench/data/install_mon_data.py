from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_MON_TAR_URL = 'https://raw.githubusercontent.com/Rebolforces/kamondataset/main/mon-white-224.tar.gz'


def _remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _safe_extract_tar(tar: tarfile.TarFile, dst: Path) -> None:
    dst = dst.resolve()
    for member in tar.getmembers():
        member_path = (dst / member.name).resolve()
        if dst not in member_path.parents and member_path != dst:
            raise RuntimeError(f'Refusing to extract outside target directory: {member.name}')
    tar.extractall(dst)  # noqa: S202


def _install_mon_images_from_tar(*, tar_url: str, target_mon_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tar_path = tmp / 'mon-white-224.tar.gz'
        urllib.request.urlretrieve(tar_url, tar_path)  # noqa: S310

        extract_root = tmp / 'extract'
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, mode='r:gz') as tf:
            _safe_extract_tar(tf, extract_root)

        jpgs = list(extract_root.rglob('*.jpg'))
        if not jpgs:
            raise FileNotFoundError(f'No .jpg images found after extracting: {tar_url}')

        by_parent: dict[Path, int] = {}
        for p in jpgs:
            by_parent[p.parent] = by_parent.get(p.parent, 0) + 1
        src_dir = max(by_parent.items(), key=lambda kv: kv[1])[0]

        target_mon_dir.mkdir(parents=True, exist_ok=False)
        for src in sorted(src_dir.glob('*.jpg')):
            shutil.copy2(src, target_mon_dir / src.name)


def install_mon_data(*, output_dir: Path, basic_charges_csv: Path, mon_tar_url: str, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_basic_charges = output_dir / 'basic_charges.csv'
    target_mon_dir = output_dir / 'mon-white-224'

    same_basic_charges_path = basic_charges_csv.resolve() == target_basic_charges.resolve()

    if force:
        if not same_basic_charges_path:
            _remove_if_exists(target_basic_charges)
        _remove_if_exists(target_mon_dir)
    else:
        if target_mon_dir.exists():
            raise FileExistsError('mon data appears to be installed already. Pass --force to overwrite.')

    if not basic_charges_csv.is_file():
        raise FileNotFoundError(basic_charges_csv)

    if not target_basic_charges.exists() or (force and not same_basic_charges_path):
        if same_basic_charges_path:
            raise FileNotFoundError(target_basic_charges)
        shutil.copy2(basic_charges_csv, target_basic_charges)

    _install_mon_images_from_tar(tar_url=mon_tar_url, target_mon_dir=target_mon_dir)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('install-mon-data', help='Install required mon assets into ./data.')
    p.add_argument('--output-dir', type=Path, default=Path('data'))
    p.add_argument('--basic-charges-csv', type=Path, default=Path('data/basic_charges.csv'))
    p.add_argument('--mon-tar-url', type=str, default=DEFAULT_MON_TAR_URL)
    p.add_argument('--force', action='store_true')
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    install_mon_data(
        output_dir=args.output_dir,
        basic_charges_csv=args.basic_charges_csv,
        mon_tar_url=args.mon_tar_url,
        force=args.force,
    )
    return 0
