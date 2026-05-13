from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kamonbench.evaluation.representation_probes import _load_split, _standardize

BASE_MODIFIER = 'X:0'


@dataclass(frozen=True)
class ModifierOperatorConfig:
    feature_dir: Path
    model_name: str
    output_dir: Path


def _context_groups(labels: dict[str, list[str]]) -> dict[tuple[str, str], dict[str, list[int]]]:
    groups: dict[tuple[str, str], dict[str, list[int]]] = {}
    for idx, (container, modifier, motif) in enumerate(
        zip(labels['container'], labels['modifier'], labels['motif'], strict=True)
    ):
        groups.setdefault((container, motif), {}).setdefault(modifier, []).append(idx)
    return groups


def _centroid(features: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return features[torch.tensor(indices, dtype=torch.long)].mean(dim=0)


def _fit_operator_sums(
    features: torch.Tensor, labels: dict[str, list[str]]
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for by_modifier in _context_groups(labels).values():
        if BASE_MODIFIER not in by_modifier:
            continue
        base = _centroid(features, by_modifier[BASE_MODIFIER])
        for modifier, indices in by_modifier.items():
            if modifier == BASE_MODIFIER:
                continue
            delta = _centroid(features, indices) - base
            sums[modifier] = sums.get(modifier, torch.zeros_like(delta)) + delta
            counts[modifier] = counts.get(modifier, 0) + 1
    return sums, counts


def _fit_operators(
    features: torch.Tensor, labels: dict[str, list[str]]
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    sums, counts = _fit_operator_sums(features, labels)
    return {modifier: sums[modifier] / counts[modifier] for modifier in sums}, counts


def _empty_modifier_result() -> dict[str, Any]:
    return {
        'pairs': 0,
        'retrieval_pairs': 0,
        'retrieval_accuracy': None,
        'delta_cosine': None,
        'target_cosine_gain': None,
        'contexts': 0,
    }


def _evaluate_split(
    features: torch.Tensor, labels: dict[str, list[str]], operators: dict[str, torch.Tensor]
) -> dict[str, Any]:
    totals: dict[str, dict[str, Any]] = {}
    aggregate = {
        'pairs': 0,
        'retrieval_pairs': 0,
        'retrieval_correct': 0,
        'delta_cosine_sum': 0.0,
        'target_cosine_gain_sum': 0.0,
        'contexts': 0,
    }

    for by_modifier in _context_groups(labels).values():
        candidates = {
            modifier: _centroid(features, indices)
            for modifier, indices in by_modifier.items()
            if modifier != BASE_MODIFIER and modifier in operators
        }
        if BASE_MODIFIER not in by_modifier or not candidates:
            continue
        base = _centroid(features, by_modifier[BASE_MODIFIER])
        aggregate['contexts'] += 1

        for modifier, target in candidates.items():
            operator = operators[modifier]
            predicted = base + operator
            scores = {
                name: float(F.cosine_similarity(predicted, candidate, dim=0).item())
                for name, candidate in candidates.items()
            }
            actual_delta = target - base
            delta_cosine = float(F.cosine_similarity(operator, actual_delta, dim=0).item())
            target_gain = float(
                (F.cosine_similarity(predicted, target, dim=0) - F.cosine_similarity(base, target, dim=0)).item()
            )
            correct = int(max(scores, key=scores.get) == modifier)

            item = totals.setdefault(
                modifier,
                {
                    'pairs': 0,
                    'retrieval_pairs': 0,
                    'retrieval_correct': 0,
                    'delta_cosine_sum': 0.0,
                    'target_cosine_gain_sum': 0.0,
                    'contexts': 0,
                },
            )
            item['pairs'] += 1
            item['delta_cosine_sum'] += delta_cosine
            item['target_cosine_gain_sum'] += target_gain
            item['contexts'] += 1

            aggregate['pairs'] += 1
            aggregate['delta_cosine_sum'] += delta_cosine
            aggregate['target_cosine_gain_sum'] += target_gain
            if len(candidates) > 1:
                item['retrieval_pairs'] += 1
                item['retrieval_correct'] += correct
                aggregate['retrieval_pairs'] += 1
                aggregate['retrieval_correct'] += correct

    def finish(item: dict[str, Any]) -> dict[str, Any]:
        pairs = item['pairs']
        if pairs == 0:
            return _empty_modifier_result()
        retrieval_pairs = item['retrieval_pairs']
        return {
            'pairs': pairs,
            'retrieval_pairs': retrieval_pairs,
            'retrieval_accuracy': item['retrieval_correct'] / retrieval_pairs if retrieval_pairs else None,
            'delta_cosine': item['delta_cosine_sum'] / pairs,
            'target_cosine_gain': item['target_cosine_gain_sum'] / pairs,
            'contexts': item['contexts'],
        }

    return {
        'all': finish(aggregate),
        'modifiers': {modifier: finish(item) for modifier, item in sorted(totals.items())},
    }


def _evaluate_context_cv(features: torch.Tensor, labels: dict[str, list[str]]) -> dict[str, Any]:
    sums, counts = _fit_operator_sums(features, labels)
    totals: dict[str, dict[str, Any]] = {}
    aggregate = {
        'pairs': 0,
        'retrieval_pairs': 0,
        'retrieval_correct': 0,
        'delta_cosine_sum': 0.0,
        'target_cosine_gain_sum': 0.0,
        'contexts': 0,
    }

    for by_modifier in _context_groups(labels).values():
        candidates = {
            modifier: _centroid(features, indices)
            for modifier, indices in by_modifier.items()
            if modifier != BASE_MODIFIER and counts.get(modifier, 0) > 1
        }
        if BASE_MODIFIER not in by_modifier or not candidates:
            continue
        base = _centroid(features, by_modifier[BASE_MODIFIER])
        aggregate['contexts'] += 1

        for modifier, target in candidates.items():
            actual_delta = target - base
            operator = (sums[modifier] - actual_delta) / (counts[modifier] - 1)
            predicted = base + operator
            scores = {
                name: float(F.cosine_similarity(predicted, candidate, dim=0).item())
                for name, candidate in candidates.items()
            }
            delta_cosine = float(F.cosine_similarity(operator, actual_delta, dim=0).item())
            target_gain = float(
                (F.cosine_similarity(predicted, target, dim=0) - F.cosine_similarity(base, target, dim=0)).item()
            )
            correct = int(max(scores, key=scores.get) == modifier)

            item = totals.setdefault(
                modifier,
                {
                    'pairs': 0,
                    'retrieval_pairs': 0,
                    'retrieval_correct': 0,
                    'delta_cosine_sum': 0.0,
                    'target_cosine_gain_sum': 0.0,
                    'contexts': 0,
                },
            )
            item['pairs'] += 1
            item['delta_cosine_sum'] += delta_cosine
            item['target_cosine_gain_sum'] += target_gain
            item['contexts'] += 1

            aggregate['pairs'] += 1
            aggregate['delta_cosine_sum'] += delta_cosine
            aggregate['target_cosine_gain_sum'] += target_gain
            if len(candidates) > 1:
                item['retrieval_pairs'] += 1
                item['retrieval_correct'] += correct
                aggregate['retrieval_pairs'] += 1
                aggregate['retrieval_correct'] += correct

    def finish(item: dict[str, Any]) -> dict[str, Any]:
        pairs = item['pairs']
        if pairs == 0:
            return _empty_modifier_result()
        retrieval_pairs = item['retrieval_pairs']
        return {
            'pairs': pairs,
            'retrieval_pairs': retrieval_pairs,
            'retrieval_accuracy': item['retrieval_correct'] / retrieval_pairs if retrieval_pairs else None,
            'delta_cosine': item['delta_cosine_sum'] / pairs,
            'target_cosine_gain': item['target_cosine_gain_sum'] / pairs,
            'contexts': item['contexts'],
        }

    return {
        'all': finish(aggregate),
        'modifiers': {modifier: finish(item) for modifier, item in sorted(totals.items())},
    }


def summarize_modifier_operators_from_splits(
    train: dict[str, Any], dev: dict[str, Any], test: dict[str, Any], *, model_name: str
) -> dict[str, Any]:
    train_x, dev_x, test_x = _standardize(train['features'], dev['features'], test['features'])
    operators, train_counts = _fit_operators(train_x, train['labels'])
    result: dict[str, Any] = {
        'model_name': model_name,
        'base_modifier': BASE_MODIFIER,
        'feature_dim': int(train_x.shape[1]),
        'operators': {
            modifier: {'train_pairs': train_counts[modifier], 'norm': float(operator.norm().item())}
            for modifier, operator in sorted(operators.items())
        },
        'splits': {
            'train_context_cv': _evaluate_context_cv(train_x, train['labels']),
            'dev': _evaluate_split(dev_x, dev['labels'], operators),
            'test': _evaluate_split(test_x, test['labels'], operators),
        },
    }
    return result


def summarize_modifier_operators(config: ModifierOperatorConfig) -> dict[str, Any]:
    train = _load_split(config.feature_dir, config.model_name, 'train')
    dev = _load_split(config.feature_dir, config.model_name, 'dev')
    test = _load_split(config.feature_dir, config.model_name, 'test')
    return summarize_modifier_operators_from_splits(train, dev, test, model_name=config.model_name)


def modifier_operator_markdown(
    results: dict[str, dict[str, Any]], *, splits: tuple[str, ...] = ('train_context_cv', 'dev', 'test')
) -> str:
    lines = [
        '| Model | Split | Modifier | Train pairs | Eval pairs | Retrieval pairs | Retrieval Acc | Delta Cos | Target Cos Gain |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]

    def fmt(value: float | None) -> str:
        return '—' if value is None else f'{value:.3f}'

    for model_name, result in results.items():
        for split in splits:
            if split not in result['splits']:
                continue
            all_eval = result['splits'][split]['all']
            if all_eval['pairs']:
                all_retrieval_pairs = all_eval.get('retrieval_pairs', all_eval['pairs'])
                lines.append(
                    f'| {model_name} | {split} | all | '
                    f'{sum(item["train_pairs"] for item in result["operators"].values())} | '
                    f'{all_eval["pairs"]} | {all_retrieval_pairs} | {fmt(all_eval["retrieval_accuracy"])} | '
                    f'{all_eval["delta_cosine"]:.3f} | {all_eval["target_cosine_gain"]:.3f} |'
                )
            for modifier, operator in result['operators'].items():
                item = result['splits'][split]['modifiers'].get(modifier, _empty_modifier_result())
                if item['pairs'] == 0:
                    continue
                retrieval_pairs = item.get('retrieval_pairs', item['pairs'])
                lines.append(
                    f'| {model_name} | {split} | {modifier} | {operator["train_pairs"]} | '
                    f'{item["pairs"]} | {retrieval_pairs} | {fmt(item["retrieval_accuracy"])} | '
                    f'{item["delta_cosine"]:.3f} | {item["target_cosine_gain"]:.3f} |'
                )
    return '\n'.join(lines) + '\n'


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('etc/modifier_operator_metrics'))
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        'summarize-modifier-operators', help='Evaluate additive modifier operators on frozen representation features.'
    )
    _add_arguments(parser)
    parser.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize_modifier_operators(
        ModifierOperatorConfig(feature_dir=args.feature_dir, model_name=args.model_name, output_dir=args.output_dir)
    )
    output_path = args.output_dir / f'{args.model_name}_modifier_operators.json'
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    existing = {}
    for path in sorted(args.output_dir.glob('*_modifier_operators.json')):
        existing[path.name.removesuffix('_modifier_operators.json')] = json.loads(path.read_text(encoding='utf-8'))
    markdown = modifier_operator_markdown(existing)
    markdown_path = args.markdown_output or args.output_dir / 'modifier_operators.md'
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding='utf-8')
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
