from __future__ import annotations

import os

from PIL import Image


def retrieve_image(path: str, *, size: int, root: str = '.') -> Image.Image:
    size_tuple = (size, size)
    full_path = os.path.join(root, path)
    img = Image.open(full_path).resize(size_tuple)

    if img.mode in ['RGBA', 'LA']:
        img.load()
        canvas = Image.new('RGB', img.size, (255, 255, 255))
        idx = 3 if img.mode == 'RGBA' else 1
        canvas.paste(img, mask=img.split()[idx])
        img = canvas

    return img.convert('RGB')
