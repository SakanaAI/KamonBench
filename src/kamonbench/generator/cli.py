from __future__ import annotations

import argparse
import collections
import os
import random
import re
from pathlib import Path

import jsonlines

from .assets import default_basic_charges_path, load_basic_charges
from .render import generate
from .rules import MODS


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('generate', help='Generate synthetic crests.')
    p.add_argument('--num', type=int, default=10)
    p.add_argument('--dataset-dir', type=Path, default=Path('dataset01'))
    p.add_argument('--append', action='store_true', help='Append images to an existing dataset dir.')
    p.add_argument('--force', action='store_true', help='Overwrite existing synthetic images/JSONL in dataset dir.')
    p.add_argument('--seed', type=int, default=None, help='Seed Python random for reproducible generation.')
    p.add_argument('--max-containers', type=int, default=2)
    p.add_argument('--containerless-prob', type=float, default=0.3)
    p.add_argument('--vanilla-mod-multiplier', type=int, default=1)
    p.add_argument('--save-components', action='store_true')
    p.add_argument('--basic-charges-csv', type=Path, default=default_basic_charges_path())
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    mods = [''] * int(args.vanilla_mod_multiplier) + list(MODS)
    assets = load_basic_charges(args.basic_charges_csv)

    if args.append and args.force:
        raise ValueError('--append and --force are mutually exclusive')

    if args.seed is not None:
        random.seed(int(args.seed))

    os.makedirs(args.dataset_dir, exist_ok=True)

    synthetic_path = args.dataset_dir / 'synthetic.jsonl'
    existing_pngs = list(args.dataset_dir.glob('synth_*.png'))
    if (synthetic_path.exists() or existing_pngs) and not (args.append or args.force):
        raise FileExistsError(f'{args.dataset_dir} already contains synthetic artifacts; pass --append or --force')

    data: dict[str, list[dict]] = collections.defaultdict(list)
    start_idx = 0

    if args.append:
        if not synthetic_path.exists():
            raise FileNotFoundError(f'--append requires {synthetic_path}')
        with jsonlines.open(synthetic_path) as reader:
            for item in reader:
                desc = item.get('description', '')
                imgs = item.get('images', [])
                if desc and isinstance(imgs, list):
                    data[desc].extend(imgs)

        idx_re = re.compile(r'^synth_(\d+)')
        indices = []
        for p in existing_pngs:
            m = idx_re.match(p.name)
            if m:
                indices.append(int(m.group(1)))
        start_idx = (max(indices) + 1) if indices else 0

    if args.force and not args.append:
        if synthetic_path.exists():
            synthetic_path.unlink()
        for p in existing_pngs:
            p.unlink()

    for i in range(start_idx, start_idx + int(args.num)):
        result = generate(
            assets,
            save_components=bool(args.save_components),
            max_containers=int(args.max_containers),
            containerless_prob=float(args.containerless_prob),
            mods=mods,
        )
        if args.save_components:
            img, expr, _, _, base_motif_img, container_imgs, base_term, container_terms = result
        else:
            img, expr, _, _ = result

        path = str(args.dataset_dir / f'synth_{i:04d}.png')
        img.save(path)

        entry = {'path': path, 'source': 'synthetic'}

        if args.save_components:
            base_path = str(args.dataset_dir / f'synth_{i:04d}_base.png')
            base_motif_img.save(base_path)
            entry['base_motif'] = {'path': base_path, 'description': base_term}

            if container_imgs:
                containers_list = []
                for j, (container_img, container_term) in enumerate(zip(container_imgs, container_terms)):
                    container_path = str(args.dataset_dir / f'synth_{i:04d}_container{j:02d}.png')
                    container_img.save(container_path)
                    containers_list.append({'path': container_path, 'description': container_term})
                entry['containers'] = containers_list

        data[expr].append(entry)

    jsonl = [{'description': desc, 'images': data[desc]} for desc in data]
    output_path = args.dataset_dir / 'synthetic.jsonl'
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(jsonl)
    return 0
