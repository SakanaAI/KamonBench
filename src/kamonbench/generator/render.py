from __future__ import annotations

import random
from typing import Any

from PIL import Image

from .assets import ChargeAssets
from .ops import BEAN, NORMAL, StackMode, inside, stack
from .rules import CONTAINERLESS_MODS, MODS, is_already_stacked


def generate(
    assets: ChargeAssets,
    *,
    keep_intermediate: bool = False,
    save_components: bool = False,
    containers: list[str] | None = None,
    motif: str | None = None,
    modifier: str | None = None,
    max_containers: int = 2,
    containerless_prob: float = 0.3,
    mods: list[str] | None = None,
) -> tuple[Any, ...]:
    if mods is None:
        mods = MODS

    if containers is not None:
        num_containers = len(containers)
        container_list: list[tuple[str, Image.Image]] = []
        for container_name in containers:
            if container_name not in assets.containers:
                raise ValueError(f'Unknown container: {container_name}')
            container_list.append((container_name, Image.open(assets.containers[container_name])))
        main_container_term = container_list[0][0] if container_list else None
    else:
        if random.random() < containerless_prob:
            num_containers = 0
        else:
            cont = list(range(1, max_containers + 1))
            num_containers = random.choice(cont)

        container_list = []
        if num_containers > 0:
            container_keys = list(assets.containers.keys())
            for _ in range(num_containers):
                key = random.choice(container_keys)
                container_list.append((key, Image.open(assets.containers[key])))
            main_container_term = container_list[0][0]
        else:
            main_container_term = None

    containers_loaded = container_list

    if modifier is not None:
        if num_containers == 0:
            if modifier not in CONTAINERLESS_MODS:
                raise ValueError(
                    f"Modifier '{modifier}' not valid for containerless patterns. Must be one of {CONTAINERLESS_MODS}"
                )
    else:
        if num_containers == 0:
            modifier = random.choice(CONTAINERLESS_MODS)
        else:
            modifier = random.choice(mods)

    is_stacking_modifier = modifier in CONTAINERLESS_MODS

    if motif is not None:
        if motif not in assets.motifs:
            raise ValueError(f'Unknown motif: {motif}')
        if is_stacking_modifier and is_already_stacked(motif):
            raise ValueError(f"Motif '{motif}' is already stacked and cannot be stacked again")
        key = motif
    else:
        other_keys = list(assets.motifs.keys())
        if is_stacking_modifier:
            other_keys = [k for k in other_keys if not is_already_stacked(k)]
            if not other_keys:
                raise ValueError('No non-stacked motifs available')
        key = random.choice(other_keys)

    final_term = key
    other = key, Image.open(assets.motifs[key])
    base_motif_img = other[1].copy() if save_components else None

    intermediates: list[dict[str, Any]] = []
    is_stacked = False

    def add_intermediate(img, expr: str, *, prepend: bool = False) -> None:
        if not keep_intermediate:
            return
        entry = {'img': img, 'expr': expr}
        if prepend:
            intermediates.insert(0, entry)
        else:
            intermediates.append(entry)

    if num_containers == 0:
        if modifier == '三つ盛り':
            img = stack(other[1], mode=StackMode.BASIC)
            add_intermediate(other[1], other[0])
            add_intermediate(img, f'三つ盛り {other[0]}')
            expr = f'三つ盛り{other[0]}'
            is_stacked = True
        elif modifier == '尻合せ三つ':
            img = stack(other[1], mode=StackMode.SHIRI)
            add_intermediate(other[1], other[0])
            add_intermediate(img, f'尻合せ三つ {other[0]}')
            expr = f'尻合せ三つ{other[0]}'
            is_stacked = True
        elif modifier == '頭合せ三つ':
            img = stack(other[1], mode=StackMode.ATAMA)
            add_intermediate(other[1], other[0])
            add_intermediate(img, f'頭合せ三つ {other[0]}')
            expr = f'頭合せ三つ{other[0]}'
            is_stacked = True
        else:
            raise ValueError(f"Invalid containerless modifier: '{modifier}'. Expected one of {CONTAINERLESS_MODS}")
    else:
        idx = len(containers_loaded) - 1
        container = containers_loaded[idx]

        base_scale = NORMAL

        if modifier == '覗き':
            add_intermediate(container[1], container[0])
            add_intermediate(None, 'に 覗き')
            add_intermediate(other[1], other[0])
            img = inside(container[1], other[1], scale=base_scale, peek=True)
            expr = f'{container[0]}に覗き{other[0]}'
        elif modifier == '豆':
            h, w = other[1].height, other[1].width
            img = other[1].resize((int(h * BEAN), int(w * BEAN)))
            add_intermediate(container[1], container[0])
            add_intermediate(img, f'に 豆 {other[0]}')
            add_intermediate(other[1], other[0])
            img = inside(container[1], img, scale=1.0, peek=False)
            expr = f'{container[0]}に豆{other[0]}'
        elif modifier == '三つ盛り':
            img = stack(other[1], mode=StackMode.BASIC)
            add_intermediate(container[1], container[0])
            add_intermediate(img, f'に 三つ盛り {other[0]}')
            add_intermediate(other[1], other[0])
            img = inside(container[1], img, scale=base_scale, peek=False)
            expr = f'{container[0]}に三つ盛り{other[0]}'
            is_stacked = True
        elif modifier == '尻合せ三つ':
            img = stack(other[1], mode=StackMode.SHIRI)
            add_intermediate(container[1], container[0])
            add_intermediate(img, f'に 尻合せ三つ {other[0]}')
            add_intermediate(other[1], other[0])
            img = inside(container[1], img, scale=base_scale, peek=False)
            expr = f'{container[0]}に尻合せ三つ{other[0]}'
            is_stacked = True
        elif modifier == '頭合せ三つ':
            img = stack(other[1], mode=StackMode.ATAMA)
            add_intermediate(container[1], container[0])
            add_intermediate(img, f'に 頭合せ三つ {other[0]}')
            add_intermediate(other[1], other[0])
            img = inside(container[1], img, scale=base_scale, peek=False)
            expr = f'{container[0]}に頭合せ三つ{other[0]}'
            is_stacked = True
        else:
            add_intermediate(container[1], container[0])
            add_intermediate(other[1], other[0])
            img = inside(container[1], other[1], scale=base_scale, peek=False)
            expr = f'{container[0]}に{other[0]}'

        idx -= 1
        while idx > -1:
            container = containers_loaded[idx]
            add_intermediate(img, f'に {expr}', prepend=True)
            add_intermediate(container[1], container[0], prepend=True)
            if is_stacked:
                nested_scale = base_scale * 0.85
            else:
                nested_scale = base_scale * 0.8
            img = inside(container[1], img, scale=nested_scale, peek=False)
            expr = f'{container[0]}に{expr}'
            idx -= 1

    expr = ''.join(expr.split())
    tup: tuple[Any, ...] = (img, expr, main_container_term, final_term)
    if keep_intermediate:
        tup = tup + (intermediates,)
    if save_components:
        container_imgs = [cont[1].copy() for cont in containers_loaded] if containers_loaded else []
        container_terms = [cont[0] for cont in containers_loaded] if containers_loaded else []
        tup = tup + (base_motif_img, container_imgs, final_term, container_terms)
    return tup
