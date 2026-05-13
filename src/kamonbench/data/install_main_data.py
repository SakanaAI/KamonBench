from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_DATASET_URL = 'https://huggingface.co/datasets/SakanaAI/KamonBench/resolve/main/kamon_bench.zip'
DEFAULT_METADATA_URL = 'https://huggingface.co/datasets/SakanaAI/KamonBench/resolve/main/kamon_croissant.json'
DEFAULT_DATASET_DIR = Path('dataset01')
DEFAULT_ZIP_PATH = Path('kamon_bench.zip')
REQUIRED_FILES = ('kamon_croissant.json',)


class DownloadProgress:
    def __init__(self) -> None:
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))
        if percent != self.last_percent and percent % 5 == 0:
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f'Downloaded {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent}%)')
            self.last_percent = percent


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}')
    print(f'Saving to {dst}')
    try:
        urllib.request.urlretrieve(url, dst, DownloadProgress())  # noqa: S310
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP error while downloading {url}: {e.code} {e.reason}') from e
    except urllib.error.URLError as e:
        raise RuntimeError(f'URL error while downloading {url}: {e.reason}') from e

    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f'Downloaded file is missing or empty: {dst}')


def _safe_extract_zip(zip_path: Path, dst: Path) -> None:
    dst = dst.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            member_path = (dst / info.filename).resolve()
            if dst not in member_path.parents and member_path != dst:
                raise RuntimeError(f'Refusing to extract outside target directory: {info.filename}')
        zf.extractall(dst)


def _has_expected_files(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES)


def _has_dataset_payload(path: Path) -> bool:
    return any(path.glob('synth_*.png'))


def _find_dataset_root(extract_root: Path) -> Path:
    preferred = [extract_root / DEFAULT_DATASET_DIR.name, extract_root]
    for candidate in preferred:
        if _has_expected_files(candidate) or _has_dataset_payload(candidate):
            return candidate

    matches = [p.parent for p in extract_root.rglob(REQUIRED_FILES[0]) if _has_expected_files(p.parent)]
    if not matches:
        matches = [p.parent for p in extract_root.rglob('synth_*.png') if _has_dataset_payload(p.parent)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        expected = ', '.join([*REQUIRED_FILES, 'synth_*.png'])
        raise FileNotFoundError(f'Could not find extracted dataset directory with: {expected}')
    raise RuntimeError(f'Found multiple possible dataset directories in {extract_root}')


def install_main_data(
    *, dataset_dir: Path, zip_path: Path, dataset_url: str, keep_zip: bool, force: bool, metadata_url: str | None = None
) -> None:
    if dataset_dir.resolve() == Path.cwd().resolve():
        raise ValueError('--dataset-dir must name a dataset directory, not the current working directory')

    if dataset_dir.exists() and not force:
        raise FileExistsError(f'{dataset_dir} already exists. Pass --force to overwrite.')

    if not zip_path.exists() or force:
        _download_file(dataset_url, zip_path)
    elif zip_path.stat().st_size == 0:
        raise RuntimeError(f'Existing zip file is empty: {zip_path}')
    else:
        print(f'Using existing zip file: {zip_path}')

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_root = Path(tmpdir) / 'extract'
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f'Extracting {zip_path}')
        _safe_extract_zip(zip_path, extract_root)
        src_dir = _find_dataset_root(extract_root)
        if dataset_dir.exists():
            if dataset_dir.is_dir():
                shutil.rmtree(dataset_dir)
            else:
                dataset_dir.unlink()
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dataset_dir)

    for required_file in REQUIRED_FILES:
        target = dataset_dir / required_file
        if not target.exists():
            if not metadata_url:
                raise FileNotFoundError(f'{target} is missing and no metadata URL was provided')
            _download_file(metadata_url, target)

    if not keep_zip and zip_path.exists():
        zip_path.unlink()

    print(f'Installed KamonBench dataset in {dataset_dir}')


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('install-main-data', help='Install the main Hugging Face dataset into ./dataset01.')
    p.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument('--zip-path', type=Path, default=DEFAULT_ZIP_PATH)
    p.add_argument('--dataset-url', type=str, default=DEFAULT_DATASET_URL)
    p.add_argument('--metadata-url', type=str, default=DEFAULT_METADATA_URL)
    p.add_argument('--keep-zip', action='store_true')
    p.add_argument('--force', action='store_true')
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    install_main_data(
        dataset_dir=args.dataset_dir,
        zip_path=args.zip_path,
        dataset_url=args.dataset_url,
        keep_zip=args.keep_zip,
        force=args.force,
        metadata_url=args.metadata_url,
    )
    return 0
