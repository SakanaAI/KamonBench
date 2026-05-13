from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance


def rotate(img: Image.Image, *, pad: int = 2, window: int = 10) -> Image.Image:
    size = img.size

    def random_angle() -> int:
        return random.randint(-window, window)

    def image_to_array(img: Image.Image) -> np.ndarray:
        return np.asarray(img)

    def array_to_image(ar: np.ndarray) -> Image.Image:
        return Image.fromarray(ar.astype(np.uint8))

    white = 255, 255, 255

    ar = image_to_array(img)
    shape = ar.shape[0] + 2 * pad, ar.shape[1] + 2 * pad, ar.shape[-1]
    canvas = np.ones(shape) * 255
    canvas[pad : pad + ar.shape[0], pad : pad + ar.shape[1], :] = ar
    new_img = array_to_image(canvas).rotate(random_angle(), fillcolor=white)
    return new_img.resize(size)


def randrange(bot: float, top: float) -> float:
    if not top > bot:
        raise ValueError('top must be > bot')
    return random.random() * (top - bot) + bot


def salt_and_pepper(img: Image.Image, *, bot: float = 0.002, top: float = 0.006) -> Image.Image:
    output = np.copy(np.array(img))
    amount = randrange(bot, top)
    nb_salt = int(np.ceil(amount * output.size * 0.5))
    coords = [(random.randint(0, output.shape[0] - 1), random.randint(0, output.shape[1] - 1)) for _ in range(nb_salt)]
    for x, y in coords:
        output[x, y] = 1
    nb_pepper = int(np.ceil(amount * output.size * 0.5))
    coords = [
        (random.randint(0, output.shape[0] - 1), random.randint(0, output.shape[1] - 1)) for _ in range(nb_pepper)
    ]
    for x, y in coords:
        output[x, y] = 0
    return Image.fromarray(output)


def gaussian(img: Image.Image, *, bot: float = 0.1, top: float = 0.3) -> Image.Image:
    output = np.copy(np.array(img, dtype=np.float64))
    row, col, depth = output.shape
    mean = 0
    var = randrange(bot, top)
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, depth)).reshape(row, col, depth)
    output += gauss
    return Image.fromarray(np.array(output, dtype=np.uint8))


def adjust_brightness(img: Image.Image) -> Image.Image:
    var = random.random() + 0.5
    return ImageEnhance.Brightness(img).enhance(var)


def adjust_contrast(img: Image.Image) -> Image.Image:
    var = random.random() + 0.5
    return ImageEnhance.Contrast(img).enhance(var)


ADJUSTMENTS = [rotate, salt_and_pepper, gaussian, adjust_brightness, adjust_contrast]


def _apply_adjustments(img: Image.Image) -> Image.Image:
    new_img = img
    for adjustment in ADJUSTMENTS:
        if random.random() < 0.5:
            new_img = adjustment(new_img)
    return new_img


def apply_adjustments(img: Image.Image, niter: int = 3) -> Image.Image:
    new_img = _apply_adjustments(img)
    for _ in range(niter - 1):
        if random.random() < 0.5:
            new_img = _apply_adjustments(new_img)
    return new_img
