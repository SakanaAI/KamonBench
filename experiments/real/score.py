"""Score transfer and preprocessing using paired program-cluster bootstraps."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRIC_NAMES = ('ter', 'exact', 'parseability', 'container', 'modifier', 'motif')
MODELS = ('vit', 'vgg_masked_ngram1', 'vgg_no_mask_ngram4', 'qwen')
CONDITIONS = ('canonical', 'grayscale', 'foreground_binary', 'aligned_binary')


@dataclass(frozen=True)
class ScoredRow:
    raw: dict[str, Any]
    program: str
    ref_tokens: int
    edits: int
    exact: float
    parseability: float
    container: float | None
    modifier: float
    motif: float


def score_row(row: dict[str, Any]) -> ScoredRow:
    from kamonbench.evaluation.program_factors import parse_program_factors
    from kamonbench.metrics import edit_counts

    reference = ' '.join(row['reference'].split())
    predicted = ' '.join(row['predicted'].split())
    ref = parse_program_factors(reference)
    if ref is None:
        raise ValueError(f'Invalid reference program in {row["row_id"]}: {reference!r}')
    pred = parse_program_factors(predicted)
    insertions, deletions, substitutions = edit_counts(reference.split(), predicted.split())
    valid = pred is not None
    return ScoredRow(
        raw=row,
        program=reference,
        ref_tokens=len(reference.split()),
        edits=insertions + deletions + substitutions,
        exact=float(reference == predicted),
        parseability=float(valid),
        container=None if not ref.has_container else float(valid and ref.containers == pred.containers),
        modifier=float(valid and ref.modifier == pred.modifier),
        motif=float(valid and ref.motif == pred.motif),
    )


def summarize_micro(rows: list[ScoredRow]) -> dict[str, Any]:
    ref_tokens = sum(row.ref_tokens for row in rows)
    metrics: dict[str, float | None] = {
        'ter': sum(row.edits for row in rows) / ref_tokens if ref_tokens else None,
        'exact': sum(row.exact for row in rows) / len(rows) if rows else None,
        'parseability': sum(row.parseability for row in rows) / len(rows) if rows else None,
        'modifier': sum(row.modifier for row in rows) / len(rows) if rows else None,
        'motif': sum(row.motif for row in rows) / len(rows) if rows else None,
    }
    containers = [row.container for row in rows if row.container is not None]
    metrics['container'] = sum(containers) / len(containers) if containers else None
    return {
        'rows': len(rows),
        'programs': len({row.program for row in rows}),
        'reference_tokens': ref_tokens,
        'metrics': metrics,
    }


def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | None]:
    result = {}
    for name in METRIC_NAMES:
        a, b = left['metrics'][name], right['metrics'][name]
        result[name] = a - b if a is not None and b is not None else None
    return result


def percentile(values: list[float], quantile: float) -> float:
    values.sort()
    return values[round(quantile * (len(values) - 1))]


def cluster_bootstrap(
    real: list[ScoredRow], controls: dict[str, ScoredRow], *, samples: int, seed: int, aggregation: str
) -> dict[str, Any]:
    rng = random.Random(seed)
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    groups: dict[str, list[ScoredRow]] = defaultdict(list)
    for row in real:
        groups[row.program].append(row)
    programs = sorted(groups)
    if aggregation == 'micro':
        for _ in range(samples):
            sampled_programs = [programs[rng.randrange(len(programs))] for _ in programs]
            sampled_real = [row for program in sampled_programs for row in groups[program]]
            sampled_controls = [controls[row.program] for row in sampled_real]
            delta = difference(summarize_micro(sampled_real), summarize_micro(sampled_controls))
            for name, value in delta.items():
                if value is not None:
                    values[name].append(value)
    else:
        raise ValueError(f'Unknown aggregation: {aggregation}')
    return {
        'unit': 'program_cluster',
        'aggregation': aggregation,
        'samples': samples,
        'seed': seed,
        'intervals': {
            name: {'low': percentile(metric, 0.025), 'high': percentile(metric, 0.975)}
            for name, metric in values.items()
            if metric
        },
    }


def paired_effect(
    canonical: list[Any], variant: list[Any], *, samples: int, seed: int
) -> dict[str, dict[str, float | int | None]]:
    if [row.raw['row_id'] for row in canonical] != [row.raw['row_id'] for row in variant]:
        raise ValueError('Canonical/variant real rows are not paired.')
    groups: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for left, right in zip(canonical, variant, strict=True):
        if left.program != right.program:
            raise ValueError('Canonical/variant references differ.')
        groups[left.program].append((left, right))
    programs = sorted(groups)
    baseline = summarize_micro(canonical)['metrics']
    transformed = summarize_micro(variant)['metrics']
    draws: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    rng = random.Random(seed)
    for _ in range(samples):
        sampled = [programs[rng.randrange(len(programs))] for _ in programs]
        left = [pair[0] for program in sampled for pair in groups[program]]
        right = [pair[1] for program in sampled for pair in groups[program]]
        left_metrics = summarize_micro(left)['metrics']
        right_metrics = summarize_micro(right)['metrics']
        for metric in METRIC_NAMES:
            if left_metrics[metric] is not None and right_metrics[metric] is not None:
                draws[metric].append(right_metrics[metric] - left_metrics[metric])
    result = {}
    for metric in METRIC_NAMES:
        values = sorted(draws[metric])
        result[metric] = {
            'variant_minus_canonical': (
                transformed[metric] - baseline[metric]
                if transformed[metric] is not None and baseline[metric] is not None
                else None
            ),
            'ci95_low': values[round(0.025 * (len(values) - 1))] if values else None,
            'ci95_high': values[round(0.975 * (len(values) - 1))] if values else None,
            'bootstrap_samples': samples,
            'bootstrap_seed': seed,
        }
    return result


def read_predictions(path: Path) -> list[ScoredRow]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate image IDs in {path}')
    return [score_row(r) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--grayscale', type=Path)
    parser.add_argument('--foreground-binary', type=Path)
    parser.add_argument('--aligned-binary', type=Path)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260725)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model_index = MODELS.index(args.model)
    baseline = [r for r in read_predictions(args.canonical) if r.raw['domain'] == 'real']
    results = {}
    for condition_index, condition in enumerate(CONDITIONS):
        path = getattr(args, condition)
        if path is None:
            continue
        rows = read_predictions(path)
        real = [r for r in rows if r.raw['domain'] == 'real']
        controls = {r.program: r for r in rows if r.raw['domain'] == 'control'}
        if set(controls) != {r.program for r in real}:
            raise ValueError('Each real target program requires exactly one matched control.')
        summary = {
            'real': summarize_micro(real),
            'matched_control': summarize_micro([controls[r.program] for r in real]),
            'real_minus_control': difference(
                summarize_micro(real), summarize_micro([controls[r.program] for r in real])
            ),
            'ci95': cluster_bootstrap(
                real,
                controls,
                samples=args.samples,
                seed=args.seed + model_index * 10 + condition_index,
                aggregation='micro',
            ),
        }
        full_support = [r for r in real if r.raw.get('all_factors_in_standard_train')]
        summary['full_training_support'] = summarize_micro(full_support)
        summary['sources'] = {
            source: summarize_micro([r for r in real if r.raw['source'] == source]) for source in ('edo', 'wiki')
        }
        if condition != 'canonical':
            summary['preprocessing_effect'] = paired_effect(
                baseline, real, samples=args.samples, seed=args.seed + 100 + model_index * 10 + condition_index
            )
        results[condition] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
