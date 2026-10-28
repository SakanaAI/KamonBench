"""Construct matched controls and the reported real-image preprocessing variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from kamonbench.data.images import retrieve_image
from PIL import Image, ImageFilter, ImageOps

MODIFIER_NAMES = {'X:0': '', 'X:1': '覗き', 'X:2': '豆', 'X:3': '三つ盛り', 'X:4': '尻合せ三つ', 'X:5': '頭合せ三つ'}
SIZE = 224
TARGET_EXTENT = 212
MIN_COMPONENT_AREA = 8
CONDITIONS = ('canonical', 'grayscale', 'foreground_binary', 'aligned_binary')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def otsu_threshold(values: np.ndarray) -> int:
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = values.size
    weighted_total = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(255):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def border_median(values: np.ndarray, width: int = 8) -> float:
    border = np.concatenate(
        (
            values[:width].ravel(),
            values[-width:].ravel(),
            values[width:-width, :width].ravel(),
            values[width:-width, -width:].ravel(),
        )
    )
    return float(np.median(border))


def variants(image: Image.Image) -> dict[str, tuple[Image.Image, dict[str, Any]]]:
    grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    median = border_median(grayscale)
    oriented = 255 - grayscale if median < 127.5 else grayscale
    threshold = otsu_threshold(oriented)
    binary = np.where(oriented <= threshold, 0, 255).astype(np.uint8)
    return {
        'grayscale': (
            Image.fromarray(grayscale, mode='L').convert('RGB'),
            {
                'minimum': int(grayscale.min()),
                'maximum': int(grayscale.max()),
                'mean': float(grayscale.mean()),
                'border_median': median,
            },
        ),
        'foreground_binary': (
            Image.fromarray(binary, mode='L').convert('RGB'),
            {
                'input_border_median': median,
                'polarity_inverted': median < 127.5,
                'otsu_threshold_after_orientation': threshold,
                'black_fraction': float(np.mean(binary == 0)),
            },
        ),
    }


def morphology(mask: np.ndarray, operation: str) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=operation == 'erode')
    neighbors = [padded[y : y + SIZE, x : x + SIZE] for y in range(3) for x in range(3)]
    return np.logical_or.reduce(neighbors) if operation == 'dilate' else np.logical_and.reduce(neighbors)


def remove_small_components(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    visited = np.zeros_like(mask)
    kept = np.zeros_like(mask)
    removed_components = 0
    removed_pixels = 0
    height, width = mask.shape
    for start_y, start_x in zip(*np.where(mask & ~visited), strict=True):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component = []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if len(component) >= MIN_COMPONENT_AREA:
            ys, xs = zip(*component, strict=True)
            kept[ys, xs] = True
        else:
            removed_components += 1
            removed_pixels += len(component)
    return kept, removed_components, removed_pixels


def load_square_luma(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as source:
        source.load()
        original_size = source.size
        if source.mode in {'RGBA', 'LA'}:
            canvas = Image.new('RGB', source.size, 'white')
            canvas.paste(source, mask=source.getchannel('A'))
            source = canvas
        else:
            source = source.convert('RGB')
        grayscale = ImageOps.grayscale(source)
    values = np.asarray(grayscale)
    width = max(1, min(8, values.shape[1] // 4))
    background = round(border_median(values, width))
    side = max(grayscale.size)
    square = Image.new('L', (side, side), background)
    square.paste(grayscale, ((side - grayscale.width) // 2, (side - grayscale.height) // 2))
    resized = square.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.asarray(resized), {
        'source_size': list(original_size),
        'source_aspect_ratio': original_size[0] / original_size[1],
        'square_padding_value': background,
    }


def transform(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    grayscale, statistics = load_square_luma(path)
    median = border_median(grayscale)
    oriented = 255 - grayscale if median < 127.5 else grayscale
    filtered = np.asarray(Image.fromarray(oriented, mode='L').filter(ImageFilter.MedianFilter(size=3)), dtype=np.uint8)
    threshold = otsu_threshold(filtered)
    mask = filtered <= threshold
    closed = morphology(morphology(mask, 'dilate'), 'erode')
    cleaned, removed_components, removed_pixels = remove_small_components(closed)
    ys, xs = np.where(cleaned)
    if not len(xs):
        raise ValueError(f'No foreground survived: {path}')
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    crop = Image.fromarray((~cleaned[top:bottom, left:right] * 255).astype(np.uint8), mode='L')
    scale = min(TARGET_EXTENT / crop.width, TARGET_EXTENT / crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.NEAREST
    )
    output = Image.new('L', (SIZE, SIZE), 255)
    output.paste(resized, ((SIZE - resized.width) // 2, (SIZE - resized.height) // 2))
    final = np.asarray(output) == 0
    statistics.update(
        {
            'input_border_median_after_square_padding': median,
            'polarity_inverted': median < 127.5,
            'median_filter_size': 3,
            'otsu_threshold': threshold,
            'morphological_close_kernel': 3,
            'minimum_component_area': MIN_COMPONENT_AREA,
            'removed_components': removed_components,
            'removed_pixels': removed_pixels,
            'foreground_bbox_before_alignment': [left, top, right, bottom],
            'target_maximum_extent': TARGET_EXTENT,
            'final_black_fraction': float(final.mean()),
        }
    )
    return output.convert('RGB'), statistics


def parse_program(program: str):
    from kamonbench.evaluation.program_factors import parse_program_factors

    factors = parse_program_factors(program)
    if factors is None or len(factors.containers) > 1:
        raise ValueError(f'Expected a valid one-level program, got {program!r}')
    if factors.modifier not in MODIFIER_NAMES:
        raise ValueError(f'Unknown modifier in {program!r}')
    if not factors.containers and factors.modifier not in {'X:3', 'X:4', 'X:5'}:
        raise ValueError(f'Invalid containerless program: {program!r}')
    return factors


@dataclass(frozen=True)
class Asset:
    name: str
    kind: str
    filename: str


def load_asset_index(path: Path) -> dict[str, Asset]:
    assets: dict[str, Asset] = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.reader(stream):
            if len(row) < 3:
                continue
            asset_id = Path(row[0]).stem
            asset = Asset(name=row[1], kind=row[2], filename=Path(row[0]).name)
            if asset_id in assets and assets[asset_id] != asset:
                raise ValueError(f'Conflicting asset ID {asset_id}: {assets[asset_id]} vs {asset}')
            assets[asset_id] = asset
    return assets


def render_control(program: str, asset_index: dict[str, Asset], mon_dir: Path):
    from kamonbench.generator.assets import ChargeAssets
    from kamonbench.generator.render import generate

    factors = parse_program(program)
    motif_id = factors.motif.removeprefix('M:')
    motif = asset_index.get(motif_id)
    if motif is None or motif.kind == 'container':
        raise ValueError(f'Missing or invalid motif asset {motif_id}')
    motif_path = mon_dir / motif.filename
    if not motif_path.is_file():
        raise FileNotFoundError(motif_path)

    containers = []
    container_assets: dict[str, str] = {}
    if factors.containers:
        container_id = factors.containers[0].removeprefix('C:')
        container = asset_index.get(container_id)
        if container is None or container.kind != 'container':
            raise ValueError(f'Missing or invalid container asset {container_id}')
        container_path = mon_dir / container.filename
        if not container_path.is_file():
            raise FileNotFoundError(container_path)
        containers = [container.name]
        container_assets[container.name] = str(container_path)

    assets = ChargeAssets(
        containers=container_assets,
        motifs={motif.name: str(motif_path)},
        all={**container_assets, motif.name: str(motif_path)},
    )
    image, expression, _, _ = generate(
        assets, containers=containers, motif=motif.name, modifier=MODIFIER_NAMES[factors.modifier]
    )
    return image.convert('RGB'), expression


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, default=Path(__file__).with_name('targets.jsonl'))
    parser.add_argument(
        '--real-root', type=Path, default=Path(__file__).parent, help='Image root (default: bundled evaluation images).'
    )
    parser.add_argument('--mon-dir', type=Path, default=Path('data/mon-white-224'))
    parser.add_argument('--charges', type=Path, default=Path('data/basic_charges.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    targets = [json.loads(line) for line in args.targets.read_text().splitlines() if line.strip()]
    args.output_dir.mkdir(parents=True)
    assets = load_asset_index(args.charges)
    controls = []
    for index, program in enumerate(sorted({r['reference'] for r in targets})):
        image, description = render_control(program, assets, args.mon_dir)
        relative = Path('controls') / f'{index:03d}.png'
        path = args.output_dir / relative
        path.parent.mkdir(exist_ok=True)
        image.save(path)
        controls.append(
            {
                'row_id': f'control_{index:03d}',
                'domain': 'control',
                'source': 'synthetic',
                'reference_kind': 'matched_control',
                'image': relative.as_posix(),
                'reference': program,
                'description': description,
                'image_sha256': sha256(path),
            }
        )
    records = {condition: [] for condition in CONDITIONS}
    for row in targets:
        source = args.real_root / row['image']
        if sha256(source) != row['image_sha256']:
            raise ValueError(f'Source image differs from the recorded image: {row["row_id"]}')
        canonical = retrieve_image(str(source), size=224)
        images = {'canonical': (canonical, {}), **variants(canonical), 'aligned_binary': transform(source)}
        for condition, (image, statistics) in images.items():
            relative = Path(condition) / f'{row["row_id"]}.png'
            path = args.output_dir / relative
            path.parent.mkdir(exist_ok=True)
            image.save(path, format='PNG', compress_level=9, optimize=False)
            records[condition].append(
                {
                    **row,
                    'domain': 'real',
                    'source_image': row['image'],
                    'image': relative.as_posix(),
                    'image_sha256': sha256(path),
                    'preprocessing': statistics,
                }
            )
    for condition, rows in records.items():
        path = args.output_dir / f'{condition}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in [*rows, *controls]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
