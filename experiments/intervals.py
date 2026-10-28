"""Test-image bootstrap intervals, with paired sampling across training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from kamonbench.evaluation.program_factors import parse_program_factors
from kamonbench.metrics import token_error_rate
from kamonbench.text import normalize_japanese_for_comparison

METRICS = ('ter', 'exact', 'motif')
CHUNK = 250


def metric_features(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[int]] = {name: [] for name in ('edits', 'reference_tokens', 'exact', 'motif')}
    for row in rows:
        reference = row['reference']
        predicted = row['predicted']
        insertions, deletions, substitutions, _ = token_error_rate(reference, predicted)
        reference_factors = parse_program_factors(reference)
        predicted_factors = parse_program_factors(predicted)
        if reference_factors is None:
            raise ValueError(f'Invalid reference program: {reference!r}')
        values['edits'].append(insertions + deletions + substitutions)
        values['reference_tokens'].append(len(reference.split()))
        values['exact'].append(
            int(normalize_japanese_for_comparison(reference) == normalize_japanese_for_comparison(predicted))
        )
        values['motif'].append(
            int(predicted_factors is not None and reference_factors.motif == predicted_factors.motif)
        )
    return {name: np.asarray(items, dtype=np.int64) for name, items in values.items()}


def point(features: dict[str, np.ndarray], metric: str) -> float:
    if metric == 'ter':
        return float(features['edits'].sum() / features['reference_tokens'].sum())
    return float(features[metric].mean())


def bootstrap_values(features: dict[str, np.ndarray], metric: str, sample_indices: np.ndarray) -> np.ndarray:
    if metric == 'ter':
        numerator = features['edits'][sample_indices].sum(axis=1)
        denominator = features['reference_tokens'][sample_indices].sum(axis=1)
    else:
        numerator = features[metric][sample_indices].sum(axis=1)
        denominator = np.full(sample_indices.shape[0], sample_indices.shape[1], dtype=np.int64)
    result = np.full(sample_indices.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def interval(values: np.ndarray) -> dict[str, float]:
    clean = np.sort(values[~np.isnan(values)])
    if not clean.size:
        raise ValueError('No finite bootstrap samples.')
    return {'low': float(clean[int(0.025 * (clean.size - 1))]), 'high': float(clean[int(0.975 * (clean.size - 1))])}


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = []
    for row in rows:
        image = row.get('image') or row.get('image_path', '')
        if '_base' in image or '_container' in image:
            continue
        result.append(
            {
                **row,
                'image': image,
                'reference': row.get('reference') or row.get('reference_description', ''),
                'predicted': row.get('predicted') or row.get('predicted_description', ''),
            }
        )
    if not result or len({r['image'] for r in result}) != len(result):
        raise ValueError(f'Expected unique composite predictions in {path}')
    return result


def compute(paths: list[Path], *, samples: int, seed: int, label: str | None) -> dict[str, Any]:
    runs = [read_rows(path) for path in paths]
    identity = [(r['image'], r['reference']) for r in runs[0]]
    if any([(r['image'], r['reference']) for r in rows] != identity for rows in runs[1:]):
        raise ValueError('Predictions must have the same test images, order and references across seeds.')
    if len(runs) > 1 and label is None:
        raise ValueError('Seed means require a label, e.g. vit-cm, vit-rm, vit-crm or qwen-standard.')
    features = [metric_features(rows) for rows in runs]
    n = len(runs[0])
    draws = {metric: [] for metric in METRICS}
    if label is None:
        rng = random.Random(seed)
    else:
        digest = hashlib.sha256(f'{seed}:{label}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    for offset in range(0, samples, CHUNK):
        count = min(CHUNK, samples - offset)
        if label is None:
            indices = np.fromiter((rng.randrange(n) for _ in range(count * n)), dtype=np.int64).reshape(count, n)
        else:
            indices = rng.integers(0, n, size=(count, n))
            # Preserve the random stream of the original cross-seed estimator.
            rng.integers(0, len(runs), size=(count, len(runs)))
        for metric in METRICS:
            per_seed = np.stack([bootstrap_values(f, metric, indices) for f in features], axis=1)
            draws[metric].append(np.nanmean(per_seed, axis=1))
    result = {}
    for metric in METRICS:
        points = [point(f, metric) for f in features]
        result[metric] = {
            'point': statistics.fmean(points),
            'ci95': interval(np.concatenate(draws[metric])),
            'seed_values': points,
            'seed_sd': statistics.stdev(points) if len(points) > 1 else None,
        }
    return {
        'rows': n,
        'samples': samples,
        'seed': seed,
        'label': label,
        'method': 'percentile test-row bootstrap, fixed checkpoints; shared rows across seeds',
        'metrics': result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path, nargs='+')
    parser.add_argument(
        '--label', help='Use the recorded cross-seed RNG: vit-standard, vit-cm, vit-rm, vit-crm or qwen-standard.'
    )
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (20260725 if args.label else 20260428)
    result = compute(args.predictions, samples=args.samples, seed=seed, label=args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
