from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FAMILIES = [('vit', 'ViT decoder'), ('vgg_mask', 'VGG masks'), ('vgg_no_mask', 'VGG no masks')]
LABELS = [('jp', 'Japanese'), ('en', 'English'), ('program', 'Program')]
FACTOR_SLICES = [('all', 'All'), ('contained', 'Contained'), ('containerless', 'Containerless')]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def metric_ci(summary: dict[str, Any], name: str) -> str:
    metric = summary['metrics'][name]
    interval = summary.get('bootstrap', {}).get('metrics', {}).get(name)
    if interval:
        return f'\\tsci{{{metric:.3f}}}{{[{interval["low"]:.3f}, {interval["high"]:.3f}]}}'
    return f'{metric:.3f}'


def result_rows(sweep_root: Path, sizes: list[int]) -> list[str]:
    rows: list[str] = []
    for size in sizes:
        first_size = True
        for family, family_label in FAMILIES:
            first_family = True
            for label, label_name in LABELS:
                summary = load_json(
                    sweep_root / f'train_{size}' / 'runs' / f'{family}_{label}' / 'test_summary_slices.json'
                )
                composite = summary
                family_cell = family_label if first_family else ''
                size_cell = f'{size:,}' if first_size else ''
                rows.append(
                    f'{size_cell} & {family_cell} & {label_name} & '
                    f'{metric_ci(composite, "cer")} & {metric_ci(composite, "acc")} & '
                    f'{metric_ci(composite, "acc_nit")}\\\\'
                )
                first_size = False
                first_family = False
            if family != FAMILIES[-1][0]:
                rows.append('\\addlinespace[0.25ex]')
        if size != sizes[-1]:
            rows.append('\\midrule')
    return rows


def factor_metric(summary: dict[str, Any], metric_name: str) -> str:
    value = summary['metrics'][metric_name]['value']
    return '---' if value is None else f'{value:.3f}'


def factor_rows(sweep_root: Path, sizes: list[int]) -> list[str]:
    rows: list[str] = []
    for size in sizes:
        first_size = True
        for family, family_label in FAMILIES:
            result = load_json(sweep_root / f'train_{size}' / 'runs' / f'{family}_program' / 'test_factors.json')
            for slice_key, slice_label in FACTOR_SLICES:
                summary = result if slice_key == 'all' else result['slices'][slice_key]
                family_cell = family_label if slice_key == 'all' else ''
                size_cell = f'{size:,}' if first_size else ''
                rows.append(
                    f'{size_cell} & {family_cell} & {slice_label} & {summary["total_rows"]} & '
                    f'{factor_metric(summary, "factor_tuple")} & {factor_metric(summary, "container")} & '
                    f'{factor_metric(summary, "modifier")} & {factor_metric(summary, "motif")}\\\\'
                )
                first_size = False
            if family != FAMILIES[-1][0]:
                rows.append('\\addlinespace[0.25ex]')
        if size != sizes[-1]:
            rows.append('\\midrule')
    return rows


def write_results_table(path: Path, rows: list[str]) -> None:
    body = '\n'.join(rows)
    path.write_text(
        '\\begin{table*}[t]\n'
        '\\centering\n'
        '\\small\n'
        '\\caption{\\label{tab:train_size_results}Reduced-training-data '
        'performance of the archived baseline recipes on the unchanged held-out '
        'composite test data. Train size counts original training composites; '
        'associated component rows are included in training. Gray brackets give '
        '95\\% bootstrap intervals from 1,000 resamples.}\n'
        '\\newcommand{\\tsci}[2]{\\begin{tabular}[t]{@{}c@{}}#1\\\\[-0.2ex]{\\scriptsize\\textcolor{gray}{#2}}\\end{tabular}}\n'
        '\\begin{tabular}{@{}rllccc@{}}\n'
        '\\toprule\n'
        'Train & Model & Label & CER/TER & Acc & Acc$_{\\rm NIT}$\\\\\n'
        '\\midrule\n'
        f'{body}\n'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table*}\n',
        encoding='utf-8',
    )


def write_factor_table(path: Path, rows: list[str]) -> None:
    body = '\n'.join(rows)
    path.write_text(
        '\\begin{table*}[t]\n'
        '\\centering\n'
        '\\small\n'
        '\\caption{\\label{tab:train_size_factors}Reduced-training-data '
        'program-label factorwise performance on the unchanged held-out composite '
        'test data. Train size counts original training composites; associated '
        'component rows are included in training.}\n'
        '\\begin{tabular}{@{}rllrrrrr@{}}\n'
        '\\toprule\n'
        'Train & Model & Slice & N & Tuple & C & R & M\\\\\n'
        '\\midrule\n'
        f'{body}\n'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table*}\n',
        encoding='utf-8',
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-root', type=Path, required=True)
    parser.add_argument('--latex-root', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    args = parser.parse_args()

    tables_dir = args.latex_root / 'Tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_results_table(tables_dir / 'train_size_sweep_results.tex', result_rows(args.sweep_root, args.sizes))
    write_factor_table(tables_dir / 'train_size_sweep_factors.tex', factor_rows(args.sweep_root, args.sizes))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
