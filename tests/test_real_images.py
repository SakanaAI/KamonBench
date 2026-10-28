import hashlib
import json
from pathlib import Path

from kamonbench.data.images import retrieve_image


def test_real_evaluation_images_are_complete_and_unchanged():
    root = Path(__file__).resolve().parents[1] / 'experiments' / 'real'
    targets = [json.loads(line) for line in (root / 'targets.jsonl').read_text().splitlines()]
    expected = {root / row['image']: row['image_sha256'] for row in targets}
    assert len(targets) == len(expected) == 114
    assert {path for path in (root / 'data').rglob('*') if path.is_file()} == set(expected)
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
    for row in targets:
        image = retrieve_image(str(root / row['image']), size=224)
        assert hashlib.sha256(image.tobytes()).hexdigest() == row['canonical_pixel_sha256'], row['row_id']
