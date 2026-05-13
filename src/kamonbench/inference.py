from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, replace
from typing import Any

import jsonlines
import torch
from torch.utils.data import DataLoader

from kamonbench import defaults
from kamonbench.checkpoints import load_checkpoint
from kamonbench.data.dataset import KamonDataset
from kamonbench.data.loader import dataloader_kwargs
from kamonbench.text import (
    normalize_english_whitespace,
    normalize_japanese_for_comparison,
    strip_after_token,
    tokens_to_text,
)
from kamonbench.torch_utils import autocast_context, configure_torch_runtime


@dataclass(frozen=True)
class InferConfig:
    checkpoint_path: str | None = None
    model: str = 'auto'
    start_token: int = -1
    dataset_subset: str = 'test'
    output_file: str | None = None
    batch_size: int = 16
    device: str = 'auto'
    croissant_path: str = defaults.DEFAULT_CROISSANT_PATH
    use_translation: bool = False


def resolve_infer_config(config: InferConfig) -> InferConfig:
    model_for_paths = defaults.inference_model_for_paths(config.model)
    checkpoint_path = config.checkpoint_path or os.path.join(
        defaults.checkpoint_dir(model_for_paths, use_translation=config.use_translation), 'checkpoint_best_*.pt'
    )
    output_file = config.output_file or os.path.join(
        defaults.output_dir(model_for_paths, use_translation=config.use_translation),
        defaults.DEFAULT_INFERENCE_RESULT_NAME,
    )
    return replace(config, checkpoint_path=checkpoint_path, output_file=output_file)


def _resolve_checkpoint_path(path: str) -> str:
    if not glob.has_magic(path):
        return path
    matches = sorted(glob.glob(path))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return path
    raise ValueError(f'Checkpoint glob matched multiple files: {path}')


def _select_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _generate_tokens(
    model: torch.nn.Module, model_name: str, images: torch.Tensor, end_token: int, *, start_token: int, max_length: int
) -> torch.Tensor:
    if model_name == 'vgg':
        tokens, _ = model.generate(images, end_token, max_length=max_length)  # type: ignore[attr-defined]
        return tokens
    if model_name == 'vit_decoder':
        return model.generate(  # type: ignore[attr-defined]
            images, end_token=end_token, start_token=start_token, max_length=max_length
        )
    raise ValueError(f'Unknown model: {model_name}')


def build_description_to_images_map(train: KamonDataset) -> dict[str, list[str]]:
    desc_to_images: dict[str, list[str]] = {}
    for item in train.metadata:
        labels = strip_after_token(item.get('labels', []), train.end_token)
        desc = tokens_to_text(labels, train.active_label_to_expr)
        img_path = item.get('image_path', '')
        if not desc or not img_path:
            continue
        if train.use_translation:
            normalized_desc = normalize_english_whitespace(desc)
        else:
            normalized_desc = normalize_japanese_for_comparison(desc)
        desc_to_images.setdefault(normalized_desc, []).append(img_path)
    return desc_to_images


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    label_to_expr: dict[int, str],
    end_token: int,
    dataset_metadata: list[dict[str, Any]],
    train_desc_to_images: dict[str, list[str]],
    model_name: str,
    start_token: int,
    use_translation: bool,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (images, target_tokens) in enumerate(dataloader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)

            with autocast_context(device):
                pred_tokens = _generate_tokens(
                    model, model_name, images, end_token, start_token=start_token, max_length=target_tokens.shape[1]
                )

            batch_size = images.size(0)
            for i in range(batch_size):
                pred_tokens_list = strip_after_token(pred_tokens[i].cpu().tolist(), end_token)
                predicted_description = tokens_to_text(pred_tokens_list, label_to_expr)

                gt_tokens_list = strip_after_token(target_tokens[i].cpu().tolist(), end_token)
                reference_description = tokens_to_text(gt_tokens_list, label_to_expr)

                example_idx = batch_idx * dataloader.batch_size + i
                if example_idx < len(dataset_metadata):
                    item_metadata = dataset_metadata[example_idx]
                    image_path = item_metadata.get('image_path', '')
                    translation = item_metadata.get('translation', '')
                else:
                    image_path = ''
                    translation = ''

                if use_translation:
                    normalized_reference = normalize_english_whitespace(reference_description)
                else:
                    normalized_reference = normalize_japanese_for_comparison(reference_description)
                train_images_reference = train_desc_to_images.get(normalized_reference, [])

                if use_translation:
                    normalized_predicted = normalize_english_whitespace(predicted_description)
                else:
                    normalized_predicted = normalize_japanese_for_comparison(predicted_description)
                train_images_predicted = train_desc_to_images.get(normalized_predicted, [])

                results.append(
                    {
                        'reference': reference_description,
                        'predicted': predicted_description,
                        'image': image_path,
                        'translation': translation,
                        'train_images_reference': train_images_reference,
                        'train_images_predicted': train_images_predicted,
                        'reference_tokens': gt_tokens_list,
                        'predicted_tokens': pred_tokens_list,
                        'batch_idx': batch_idx,
                        'example_idx': i,
                    }
                )

    return results


def infer(config: InferConfig) -> int:
    config = resolve_infer_config(config)
    if config.dataset_subset not in {'train', 'dev', 'test'}:
        raise ValueError('--dataset-subset must be one of train/dev/test')

    device = _select_device(config.device)
    configure_torch_runtime(device)
    checkpoint_path = _resolve_checkpoint_path(config.checkpoint_path)
    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path, device, model_override=config.model, start_token_override=config.start_token
    )
    checkpoint_use_translation = checkpoint_metadata.get('use_translation')
    if checkpoint_use_translation is not None and bool(checkpoint_use_translation) != config.use_translation:
        expected_flag = '--use-translation' if checkpoint_use_translation else 'without --use-translation'
        raise ValueError(
            f'Checkpoint was trained {expected_flag}; rerun inference with matching label source, '
            'or use a checkpoint trained for the requested label source.'
        )

    train_dataset = KamonDataset(
        croissant_path=config.croissant_path,
        division='train',
        image_size=checkpoint_metadata['image_size'],
        num_augmentations=0,
        one_hot=False,
        use_translation=config.use_translation,
    )
    train_desc_to_images = build_description_to_images_map(train_dataset)

    dataset = KamonDataset(
        croissant_path=config.croissant_path,
        division=config.dataset_subset,
        image_size=checkpoint_metadata['image_size'],
        num_augmentations=0,
        one_hot=False,
        use_translation=config.use_translation,
    )

    dataloader = DataLoader(dataset, **dataloader_kwargs(batch_size=config.batch_size, shuffle=False))

    results = run_inference(
        model,
        dataloader,
        device,
        label_to_expr=checkpoint_metadata['label_to_expr'],
        end_token=checkpoint_metadata['end_token'],
        dataset_metadata=dataset.metadata,
        train_desc_to_images=train_desc_to_images,
        model_name=checkpoint_metadata['model'],
        start_token=checkpoint_metadata['start_token'],
        use_translation=config.use_translation,
    )

    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with jsonlines.open(config.output_file, 'w') as writer:
        for result in results:
            writer.write(result)

    total_examples = len(results)
    correct_predictions = sum(1 for r in results if r['reference'] == r['predicted'])
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    print(f'Total examples: {total_examples}')
    print(f'Correct predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)')
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('infer', help='Run inference from a checkpoint.')
    p.add_argument('--checkpoint-path', type=str, default=None)
    p.add_argument('--model', choices=['auto', 'vgg', 'vit_decoder'], default='auto')
    p.add_argument('--start-token', type=int, default=-1)
    p.add_argument('--dataset-subset', choices=['train', 'dev', 'test'], default='test')
    p.add_argument('--output-file', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--croissant-path', type=str, default=defaults.DEFAULT_CROISSANT_PATH)
    p.add_argument(
        '--use-translation', action='store_true', help='Use Croissant translation tokens instead of analysis.'
    )
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    config = InferConfig(
        checkpoint_path=args.checkpoint_path,
        model=args.model,
        start_token=args.start_token,
        dataset_subset=args.dataset_subset,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=args.device,
        croissant_path=args.croissant_path,
        use_translation=bool(args.use_translation),
    )
    return infer(config)
