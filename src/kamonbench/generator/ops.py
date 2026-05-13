from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Dict

import numpy as np
from PIL import Image

BEAN = 0.25
NORMAL = 0.75
STACK_SCALE = 0.4
WHITE = 255, 255, 255
BLACK = 0, 0, 0

Mask = list[list[int], list[int], list[int]]


def image_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.getdata()).reshape((img.height, img.width, -1))


def array_to_image(ar: np.ndarray) -> Image.Image:
    return Image.fromarray(ar.astype(np.uint8))


def create_nonwhite_masks(ar: np.ndarray, *, shape: tuple[int, int, int], offset: tuple[int, int]) -> Dict[int, Mask]:
    canvas = np.ones(shape) * 255
    canvas[offset[0] : offset[0] + ar.shape[0], offset[1] : offset[1] + ar.shape[1], :] = ar
    masks = defaultdict(Mask)
    for h in range(shape[0]):
        for w in range(shape[1]):
            for d in range(shape[2]):
                pix = canvas[h, w, d]
                if pix not in masks:
                    masks[pix] = [], [], []
                if pix != 255:
                    masks[pix][0].append(h)
                    masks[pix][1].append(w)
                    masks[pix][2].append(d)
    return masks


def apply_nonwhite_masks(ar: np.ndarray, masks: Dict[int, Mask]) -> np.ndarray:
    ar = deepcopy(ar)
    for pix in masks:
        ar[masks[pix]] = pix
    return ar


def _exterior_masks(ar: np.ndarray) -> tuple[Mask, Mask]:
    assert len(ar.shape) == 3
    h, w = ar.shape[:2]
    interior = set()
    mid = int(w / 2)
    for i in range(h):
        j = mid
        while j >= 0:
            if np.array_equal(ar[i, j, :], [0, 0, 0]):
                break
            interior.add((i, j))
            j -= 1
        j = mid + 1
        while j < w:
            if np.array_equal(ar[i, j, :], [0, 0, 0]):
                break
            interior.add((i, j))
            j += 1
    blackhs = []
    blackws = []
    blackds = []
    whitehs = []
    whitews = []
    whiteds = []
    for i in range(h):
        for j in range(w):
            if (i, j) in interior:
                continue
            for k in range(3):
                if ar[i, j, k] == 255:
                    whitehs.append(i)
                    whitews.append(j)
                    whiteds.append(k)
                else:
                    blackhs.append(i)
                    blackws.append(j)
                    blackds.append(k)
    bmask = blackhs, blackws, blackds
    wmask = whitehs, whitews, whiteds
    return bmask, wmask


def inside(a: Image.Image, b: Image.Image, *, scale: float = NORMAL, peek: bool = False) -> Image.Image:
    b = b.resize((int(b.height * scale), int(b.width * scale)))
    adata = image_to_array(a)
    bdata = image_to_array(b)
    h = int((adata.shape[0] - bdata.shape[0]) / 2)
    w = int((adata.shape[1] - bdata.shape[1]) / 2)
    if peek:
        h += int(bdata.shape[0] * 0.5)
        over = h + bdata.shape[0] - adata.shape[0]
        if over > 0:
            h -= over
    masks = create_nonwhite_masks(bdata, shape=adata.shape, offset=(h, w))
    exterior_bmask, exterior_wmask = _exterior_masks(adata)
    adata = apply_nonwhite_masks(adata, masks)
    adata[exterior_bmask] = 0
    adata[exterior_wmask] = 255
    return array_to_image(adata)


def safe_rotate(img: Image.Image, angle: int, *, pad: int = 20) -> Image.Image:
    ar = image_to_array(img)
    shape = ar.shape[0] + 2 * pad, ar.shape[1] + 2 * pad, ar.shape[-1]
    canvas = np.ones(shape) * 255
    canvas[pad : pad + ar.shape[0], pad : pad + ar.shape[1], :] = ar
    return array_to_image(canvas).rotate(angle, fillcolor=WHITE)


class StackMode(Enum):
    BASIC = 1
    SHIRI = 2
    ATAMA = 3


def stack(img: Image.Image, *, mode: StackMode = StackMode.BASIC) -> Image.Image:
    h, w = img.size
    sh, sw = int(h * STACK_SCALE), int(w * STACK_SCALE)
    small_img = img.resize((sh, sw))
    ar = image_to_array(img)
    sar = image_to_array(small_img)
    sw_half = int(sw * 0.5)
    rotpad = 20

    def set_points(which: str) -> tuple[int, int]:
        left = top = 0
        if which == 'top':
            left = int(w * 0.5) - sw_half
            if mode == StackMode.ATAMA:
                top = int(h * 0.09)
            else:
                top = int(h * 0.05)
        elif which == 'bottom_left':
            if mode == StackMode.ATAMA:
                top = int(h * 0.3)
                left = int(w * 0.28) - sw_half - 10
            elif mode == StackMode.SHIRI:
                top = int(h * 0.35)
                left = int(w * 0.28) - sw_half - 18
            else:
                top = int(h * 0.45)
                left = int(w * 0.28) - sw_half
        elif which == 'bottom_right':
            if mode == StackMode.ATAMA:
                top = int(h * 0.3)
                left = int(w * 0.72) - sw_half - rotpad - 10
            elif mode == StackMode.SHIRI:
                top = int(h * 0.35)
                left = int(w * 0.72) - sw_half - rotpad - 2
            else:
                top = int(h * 0.45)
                left = int(w * 0.72) - sw_half
        return top, left

    top, left = set_points('top')
    sar1 = sar
    if mode == StackMode.ATAMA:
        sar1 = image_to_array(small_img.rotate(180, fillcolor=WHITE))
    masks1 = create_nonwhite_masks(sar1, shape=ar.shape, offset=(top, left))

    if mode == StackMode.ATAMA:
        sar2 = image_to_array(safe_rotate(small_img, 300, pad=rotpad))
    elif mode == StackMode.SHIRI:
        sar2 = image_to_array(safe_rotate(small_img, 120, pad=rotpad))
    else:
        sar2 = sar
    top, left = set_points('bottom_left')
    masks2 = create_nonwhite_masks(sar2, shape=ar.shape, offset=(top, left))

    sar3 = sar
    if mode == StackMode.ATAMA:
        sar3 = image_to_array(safe_rotate(small_img, 60, pad=rotpad))
    elif mode == StackMode.SHIRI:
        sar3 = image_to_array(safe_rotate(small_img, 240, pad=rotpad))
    top, left = set_points('bottom_right')
    masks3 = create_nonwhite_masks(sar3, shape=ar.shape, offset=(top, left))

    canvas = np.ones_like(ar) * 255
    canvas = apply_nonwhite_masks(canvas, masks1)
    canvas = apply_nonwhite_masks(canvas, masks2)
    canvas = apply_nonwhite_masks(canvas, masks3)
    return array_to_image(canvas)
