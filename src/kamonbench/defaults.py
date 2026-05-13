from __future__ import annotations

DEFAULT_CROISSANT_PATH = 'dataset01/kamon_croissant.json'
DEFAULT_INFERENCE_RESULT_NAME = 'test_decode.jsonl'
DEFAULT_INFER_MODEL_FOR_PATHS = 'vgg'


def label_suffix(*, use_translation: bool) -> str:
    return 'en' if use_translation else 'ja'


def checkpoint_dir(model: str, *, use_translation: bool) -> str:
    return f'checkpoints_{model}_{label_suffix(use_translation=use_translation)}'


def output_dir(model: str, *, use_translation: bool) -> str:
    return f'outputs_{model}_{label_suffix(use_translation=use_translation)}'


def inference_model_for_paths(model: str) -> str:
    return DEFAULT_INFER_MODEL_FOR_PATHS if model == 'auto' else model
