from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Any

import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from kamonbench import defaults
from kamonbench.data.dataset import PAD_LABEL, KamonDataset
from kamonbench.data.loader import dataloader_kwargs
from kamonbench.metrics import character_edit_distance, token_edit_distance
from kamonbench.models.vgg_ngram import VGGImageToTextModel
from kamonbench.models.vit_decoder import DecoderImageCaptioner
from kamonbench.text import strip_after_token, tokens_to_text
from kamonbench.torch_utils import autocast_context, configure_torch_runtime


@dataclass(frozen=True)
class TrainConfig:
    model: str = 'vgg'
    image_size: int = 224
    num_train_augmentations: int | None = None
    batch_size: int | None = None
    num_epochs: int | None = None
    learning_rate: float | None = None
    checkpoint_steps: int | None = None
    early_stop_patience: int | None = None
    checkpoint_dir: str | None = None
    output_dir: str | None = None
    also_train_vgg: bool | None = None
    use_masks: bool = True
    ngram_length: int = 2
    hidden_dim: int = 512
    position_embedding_dim: int = 0
    device: str = 'auto'
    vit_encoder_name: str = 'vit_base_patch16_224'
    vit_n_heads: int = 8
    vit_d_model: int = 512
    vit_n_layers: int | None = None
    vit_dropout: float = 0.1
    vit_token_dropout: float = 0.0
    vit_train_backbone: bool | None = None
    vit_backbone_learning_rate: float | None = None
    vit_enc_proj_rank: int = 0
    start_token: int = -1
    label_smoothing: float | None = None
    weight_decay: float | None = None
    croissant_path: str = defaults.DEFAULT_CROISSANT_PATH
    use_translation: bool = False
    eval_composites_only: bool = False


@dataclass(frozen=True)
class _TrainDefaults:
    num_train_augmentations: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    checkpoint_steps: int
    early_stop_patience: int
    also_train_vgg: bool
    vit_n_layers: int
    vit_train_backbone: bool
    vit_backbone_learning_rate: float | None
    label_smoothing: float
    weight_decay: float


_VGG_DEFAULTS = _TrainDefaults(
    num_train_augmentations=5,
    batch_size=16,
    num_epochs=200,
    learning_rate=1e-4,
    checkpoint_steps=5_000,
    early_stop_patience=10,
    also_train_vgg=True,
    vit_n_layers=2,
    vit_train_backbone=False,
    vit_backbone_learning_rate=None,
    label_smoothing=0.0,
    weight_decay=0.0,
)

_VIT_DECODER_DEFAULTS = _TrainDefaults(
    num_train_augmentations=3,
    batch_size=64,
    num_epochs=50,
    learning_rate=1.5e-4,
    checkpoint_steps=2_000,
    early_stop_patience=8,
    also_train_vgg=False,
    vit_n_layers=4,
    vit_train_backbone=True,
    vit_backbone_learning_rate=1e-5,
    label_smoothing=0.02,
    weight_decay=1e-4,
)


def _defaults_for_model(model: str) -> _TrainDefaults:
    if model == 'vgg':
        return _VGG_DEFAULTS
    if model == 'vit_decoder':
        return _VIT_DECODER_DEFAULTS
    raise ValueError(f'Unknown model: {model}')


def resolve_train_config(config: TrainConfig) -> TrainConfig:
    model_defaults = _defaults_for_model(config.model)
    checkpoint_dir = config.checkpoint_dir or defaults.checkpoint_dir(
        config.model, use_translation=config.use_translation
    )
    output_dir = config.output_dir or defaults.output_dir(config.model, use_translation=config.use_translation)
    return replace(
        config,
        num_train_augmentations=(
            config.num_train_augmentations
            if config.num_train_augmentations is not None
            else model_defaults.num_train_augmentations
        ),
        batch_size=config.batch_size if config.batch_size is not None else model_defaults.batch_size,
        num_epochs=config.num_epochs if config.num_epochs is not None else model_defaults.num_epochs,
        learning_rate=config.learning_rate if config.learning_rate is not None else model_defaults.learning_rate,
        checkpoint_steps=(
            config.checkpoint_steps if config.checkpoint_steps is not None else model_defaults.checkpoint_steps
        ),
        early_stop_patience=(
            config.early_stop_patience if config.early_stop_patience is not None else model_defaults.early_stop_patience
        ),
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        also_train_vgg=config.also_train_vgg if config.also_train_vgg is not None else model_defaults.also_train_vgg,
        vit_n_layers=config.vit_n_layers if config.vit_n_layers is not None else model_defaults.vit_n_layers,
        vit_train_backbone=(
            config.vit_train_backbone if config.vit_train_backbone is not None else model_defaults.vit_train_backbone
        ),
        vit_backbone_learning_rate=(
            config.vit_backbone_learning_rate
            if config.vit_backbone_learning_rate is not None
            else model_defaults.vit_backbone_learning_rate
        ),
        label_smoothing=(
            config.label_smoothing if config.label_smoothing is not None else model_defaults.label_smoothing
        ),
        weight_decay=config.weight_decay if config.weight_decay is not None else model_defaults.weight_decay,
    )


def _config_from_args(args: argparse.Namespace) -> TrainConfig:
    return resolve_train_config(
        TrainConfig(
            model=args.model,
            image_size=args.image_size,
            num_train_augmentations=args.num_train_augmentations,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            checkpoint_steps=args.checkpoint_steps,
            early_stop_patience=args.early_stop_patience,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            also_train_vgg=args.also_train_vgg,
            use_masks=not bool(args.no_masks),
            ngram_length=args.ngram_length,
            hidden_dim=args.hidden_dim,
            position_embedding_dim=args.position_embedding_dim,
            device=args.device,
            vit_encoder_name=args.vit_encoder_name,
            vit_n_heads=args.vit_n_heads,
            vit_d_model=args.vit_d_model,
            vit_n_layers=args.vit_n_layers,
            vit_dropout=args.vit_dropout,
            vit_token_dropout=args.vit_token_dropout,
            vit_train_backbone=args.vit_train_backbone,
            vit_backbone_learning_rate=args.vit_backbone_learning_rate,
            vit_enc_proj_rank=args.vit_enc_proj_rank,
            start_token=args.start_token,
            label_smoothing=args.label_smoothing,
            weight_decay=args.weight_decay,
            croissant_path=args.croissant_path,
            use_translation=bool(args.use_translation),
            eval_composites_only=bool(args.eval_composites_only),
        )
    )


def _select_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _save_mask_images(masks: torch.Tensor, output_dir: str, prefix: str) -> None:
    batch_size, seq_len, _, _ = masks.shape
    for batch_idx in range(batch_size):
        example_dir = os.path.join(output_dir, f'{prefix}_{batch_idx:03d}')
        os.makedirs(example_dir, exist_ok=True)
        for seq_idx in range(seq_len):
            mask = masks[batch_idx, seq_idx].cpu().numpy()
            mask_img = (mask * 255).astype(np.uint8)
            Image.fromarray(mask_img, mode='L').save(os.path.join(example_dir, f'img_{seq_idx:03d}.png'))


def _forward_with_optional_masks(
    model: torch.nn.Module, model_name: str, images: torch.Tensor, target_tokens: torch.Tensor, *, start_token: int
):
    if model_name == 'vgg':
        logits, masks = model(images, seq_len=target_tokens.shape[1])
        return logits, masks
    if model_name == 'vit_decoder':
        logits = model(images, target_tokens, start_token=start_token)
        return logits, None
    raise ValueError(f'Unknown model: {model_name}')


def _trim_batch_targets(target_tokens: torch.Tensor) -> torch.Tensor:
    seq_len = int((target_tokens != PAD_LABEL).sum(dim=1).max().item())
    return target_tokens[:, :seq_len]


def _uses_program_labels(label_to_expr: dict[int, str], end_token: int) -> bool:
    labels = [expr for label, expr in label_to_expr.items() if label != end_token]
    return bool(labels) and all(expr.startswith(('C:', 'X:', 'M:')) for expr in labels)


def _generate_with_optional_masks(
    model: torch.nn.Module, model_name: str, images: torch.Tensor, end_token: int, *, start_token: int, max_length: int
):
    if model_name == 'vgg':
        tokens, masks = model.generate(images, end_token, max_length=max_length)
        return tokens, masks
    if model_name == 'vit_decoder':
        tokens = model.generate(images, end_token=end_token, start_token=start_token, max_length=max_length)
        return tokens, None
    raise ValueError(f'Unknown model: {model_name}')


def _checkpoint_config(config: TrainConfig) -> dict[str, Any]:
    return {
        'model': config.model,
        'image_size': config.image_size,
        'ngram_length': config.ngram_length,
        'hidden_dim': config.hidden_dim,
        'position_embedding_dim': config.position_embedding_dim,
        'also_train_vgg': config.also_train_vgg,
        'use_masks': config.use_masks,
        'vit_encoder_name': config.vit_encoder_name,
        'vit_n_heads': config.vit_n_heads,
        'vit_d_model': config.vit_d_model,
        'vit_n_layers': config.vit_n_layers,
        'vit_dropout': config.vit_dropout,
        'vit_token_dropout': config.vit_token_dropout,
        'vit_train_backbone': config.vit_train_backbone,
        'vit_backbone_learning_rate': config.vit_backbone_learning_rate,
        'vit_enc_proj_rank': config.vit_enc_proj_rank,
        'start_token': config.start_token,
        'use_translation': config.use_translation,
        'eval_composites_only': config.eval_composites_only,
    }


def _save_checkpoint(
    path: str,
    *,
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    epoch: int,
    vocab_size: int,
    max_seq_len: int,
    end_token: int,
    label_to_expr: dict[int, str],
    train_loss: float | None = None,
    val_loss: float | None = None,
    edit_distance: int | None = None,
) -> None:
    config = resolve_train_config(config)
    payload: dict[str, Any] = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'end_token': end_token,
        'label_to_expr': label_to_expr,
        'config': _checkpoint_config(config),
    }
    if train_loss is not None:
        payload['train_loss'] = train_loss
    if val_loss is not None:
        payload['val_loss'] = val_loss
    if edit_distance is not None:
        payload['edit_distance'] = edit_distance
    torch.save(payload, path)


def _validate_checkpoint_schedule(config: TrainConfig, num_batches: int) -> None:
    config = resolve_train_config(config)
    if config.num_epochs <= 0:
        raise ValueError('--num-epochs must be positive')
    if config.checkpoint_steps <= 0:
        raise ValueError('--checkpoint-steps must be positive')
    if num_batches <= 0:
        raise ValueError('Training split is empty')

    total_steps = num_batches * config.num_epochs
    if total_steps < config.checkpoint_steps:
        raise ValueError(
            f'Training would run only {total_steps} steps, so no checkpoint_best_*.pt would be created '
            f'with --checkpoint-steps={config.checkpoint_steps}. Increase --num-epochs, add data, '
            'or lower --checkpoint-steps.'
        )


def _evaluate_model(
    config: TrainConfig,
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    *,
    label_to_expr: dict[int, str],
    end_token: int,
    output_dir: str,
    step: int | str,
    use_token_edit_metric: bool = False,
) -> tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_edit_distance = 0
    total_samples = 0
    all_predictions: list[dict[str, Any]] = []
    metric_samples = 0
    metadata = getattr(val_loader.dataset, 'metadata', [])

    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=config.label_smoothing)

    with torch.no_grad():
        for batch_idx, (images, target_tokens) in enumerate(val_loader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            loss_target_tokens = _trim_batch_targets(target_tokens)

            with autocast_context(device):
                logits, _ = _forward_with_optional_masks(
                    model, config.model, images, loss_target_tokens, start_token=config.start_token
                )

                batch_size, _, vocab_size = logits.shape
                loss = criterion(logits.reshape(-1, vocab_size), loss_target_tokens.reshape(-1))
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            with autocast_context(device):
                pred_tokens, pred_masks = _generate_with_optional_masks(
                    model,
                    config.model,
                    images,
                    end_token,
                    start_token=config.start_token,
                    max_length=target_tokens.shape[1],
                )

            for i in range(batch_size):
                metadata_idx = batch_idx * val_loader.batch_size + i
                item = metadata[metadata_idx] if metadata_idx < len(metadata) else {}
                image_path = item.get('image_path', '')
                pred_tokens_list = strip_after_token(pred_tokens[i].cpu().tolist(), end_token)
                predicted_description = tokens_to_text(pred_tokens_list, label_to_expr)

                gt_tokens_list = strip_after_token(target_tokens[i].cpu().tolist(), end_token)
                reference_description = tokens_to_text(gt_tokens_list, label_to_expr)

                is_composite = '_base' not in image_path and '_container' not in image_path
                if not config.eval_composites_only or is_composite:
                    metric_samples += 1
                    if use_token_edit_metric:
                        total_edit_distance += token_edit_distance(reference_description, predicted_description)
                    else:
                        total_edit_distance += character_edit_distance(reference_description, predicted_description)

                all_predictions.append(
                    {
                        'batch_idx': batch_idx,
                        'example_idx': i,
                        'image': image_path,
                        'image_path': image_path,
                        'reference': reference_description,
                        'predicted': predicted_description,
                        'predicted_description': predicted_description,
                        'predicted_tokens': pred_tokens_list,
                        'reference_description': reference_description,
                        'reference_tokens': gt_tokens_list,
                    }
                )

            if pred_masks is not None and batch_idx < 5:
                _save_mask_images(pred_masks.cpu(), os.path.join(output_dir, f'step_{step}'), f'val_{batch_idx:03d}')

    avg_loss = total_loss / max(total_samples, 1)
    if config.eval_composites_only and metric_samples == 0:
        raise ValueError('No composite validation rows found for --eval-composites-only')

    predictions_path = os.path.join(output_dir, f'step_{step}', 'predictions.jsonl')
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
    with jsonlines.open(predictions_path, 'w') as writer:
        for pred in all_predictions:
            writer.write(pred)

    return avg_loss, total_edit_distance


def train(config: TrainConfig) -> int:
    config = resolve_train_config(config)

    device = _select_device(config.device)
    configure_torch_runtime(device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    train_data = KamonDataset(
        croissant_path=config.croissant_path,
        division='train',
        image_size=config.image_size,
        num_augmentations=config.num_train_augmentations,
        use_translation=config.use_translation,
        one_hot=False,
    )
    val_data = KamonDataset(
        croissant_path=config.croissant_path,
        division='dev',
        image_size=config.image_size,
        num_augmentations=0,
        use_translation=config.use_translation,
        one_hot=False,
    )

    label_to_expr = train_data.active_label_to_expr
    vocab_size = train_data.vocab_size
    end_token = train_data.end_token
    max_seq_len = train_data.max_len
    use_token_edit_metric = _uses_program_labels(label_to_expr, end_token)

    train_loader = DataLoader(train_data, **dataloader_kwargs(batch_size=config.batch_size, shuffle=True))
    val_loader = DataLoader(val_data, **dataloader_kwargs(batch_size=config.batch_size, shuffle=False))
    _validate_checkpoint_schedule(config, len(train_loader))

    if config.model == 'vgg':
        model: torch.nn.Module = VGGImageToTextModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            image_size=config.image_size,
            ngram_length=config.ngram_length,
            hidden_dim=config.hidden_dim,
            position_embedding_dim=config.position_embedding_dim,
            also_train_vgg=config.also_train_vgg,
            use_masks=config.use_masks,
        ).to(device)
    elif config.model == 'vit_decoder':
        model = DecoderImageCaptioner(
            encoder_name=config.vit_encoder_name,
            seq_len=max_seq_len,
            vocab_size=vocab_size,
            n_heads=config.vit_n_heads,
            d_model=config.vit_d_model,
            n_layers=config.vit_n_layers,
            dropout=config.vit_dropout,
            token_dropout=config.vit_token_dropout,
            train_backbone=config.vit_train_backbone,
            enc_proj_rank=config.vit_enc_proj_rank,
        ).to(device)
    else:
        raise ValueError(f'Unknown model: {config.model}')

    if config.model == 'vit_decoder' and config.vit_train_backbone and config.vit_backbone_learning_rate is not None:
        backbone_params = list(model.encoder.parameters())  # type: ignore[attr-defined]
        backbone_param_ids = {id(p) for p in backbone_params}
        decoder_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        optimizer = optim.AdamW(
            [
                {'params': decoder_params, 'lr': config.learning_rate},
                {'params': backbone_params, 'lr': config.vit_backbone_learning_rate},
            ],
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=config.label_smoothing)

    global_step = 0
    best_edit_distance = float('inf')
    best_checkpoint_path: str | None = None
    no_improve_checks = 0
    last_loss: float | None = None
    stop_training = False

    for epoch in range(config.num_epochs):
        model.train()
        stop_training = False

        for batch_idx, (images, target_tokens) in enumerate(train_loader):
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            loss_target_tokens = _trim_batch_targets(target_tokens)

            with autocast_context(device):
                logits, _ = _forward_with_optional_masks(
                    model, config.model, images, loss_target_tokens, start_token=config.start_token
                )

                batch_size, _, vocab_size_out = logits.shape
                ce_loss = criterion(logits.reshape(-1, vocab_size_out), loss_target_tokens.reshape(-1))

                if config.model == 'vgg' and config.use_masks:
                    diversity_loss = model.get_mask_diversity_loss(weight=0.001)  # type: ignore[attr-defined]
                    loss = ce_loss + diversity_loss
                else:
                    diversity_loss = None
                    loss = ce_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

            global_step += 1

            if batch_idx % 100 == 0:
                if diversity_loss is None:
                    print(
                        f'Epoch {epoch}, Batch {batch_idx}, CE Loss: {ce_loss.item():.4f}, Total Loss: {loss.item():.4f}'
                    )
                else:
                    print(
                        f'Epoch {epoch}, Batch {batch_idx}, CE Loss: {ce_loss.item():.4f}, '
                        f'Diversity Loss: {diversity_loss.item():.6f}, Total Loss: {loss.item():.4f}'
                    )

            if global_step % config.checkpoint_steps == 0:
                val_loss, total_edit_distance = _evaluate_model(
                    config,
                    model,
                    val_loader,
                    device,
                    label_to_expr=label_to_expr,
                    end_token=end_token,
                    output_dir=config.output_dir,
                    step=global_step,
                    use_token_edit_metric=use_token_edit_metric,
                )

                if total_edit_distance < best_edit_distance:
                    best_edit_distance = total_edit_distance
                    no_improve_checks = 0

                    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                        try:
                            os.remove(best_checkpoint_path)
                        except OSError:
                            pass

                    best_checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_best_{global_step}.pt')
                    _save_checkpoint(
                        best_checkpoint_path,
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        train_loss=last_loss,
                        val_loss=val_loss,
                        edit_distance=total_edit_distance,
                        vocab_size=vocab_size,
                        max_seq_len=max_seq_len,
                        end_token=end_token,
                        label_to_expr=label_to_expr,
                    )
                else:
                    no_improve_checks += 1
                    if config.early_stop_patience > 0 and no_improve_checks >= config.early_stop_patience:
                        stop_training = True

                model.train()
                if stop_training:
                    break

        if stop_training:
            break

    if stop_training and best_checkpoint_path and os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])

    if best_checkpoint_path is None:
        raise AssertionError('Training finished without creating a best checkpoint.')

    _evaluate_model(
        config,
        model,
        val_loader,
        device,
        label_to_expr=label_to_expr,
        end_token=end_token,
        output_dir=config.output_dir,
        step='final',
        use_token_edit_metric=use_token_edit_metric,
    )

    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('train', help='Train a baseline model.')
    p.add_argument('--model', choices=['vgg', 'vit_decoder'], default='vgg')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--num-train-augmentations', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--learning-rate', type=float, default=None)
    p.add_argument('--checkpoint-steps', type=int, default=None)
    p.add_argument('--early-stop-patience', type=int, default=None)
    p.add_argument('--checkpoint-dir', type=str, default=None)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--also-train-vgg', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--no-masks', action='store_true')
    p.add_argument('--ngram-length', type=int, default=2)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--position-embedding-dim', type=int, default=0)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--vit-encoder-name', type=str, default='vit_base_patch16_224')
    p.add_argument('--vit-n-heads', type=int, default=8)
    p.add_argument('--vit-d-model', type=int, default=512)
    p.add_argument('--vit-n-layers', type=int, default=None)
    p.add_argument('--vit-dropout', type=float, default=0.1)
    p.add_argument('--vit-token-dropout', type=float, default=0.0)
    p.add_argument('--vit-train-backbone', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--vit-backbone-learning-rate', type=float, default=None)
    p.add_argument('--vit-enc-proj-rank', type=int, default=0)
    p.add_argument('--start-token', type=int, default=-1)
    p.add_argument('--label-smoothing', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--croissant-path', type=str, default=defaults.DEFAULT_CROISSANT_PATH)
    p.add_argument(
        '--use-translation', action='store_true', help='Use Croissant translation tokens instead of analysis.'
    )
    p.add_argument(
        '--eval-composites-only',
        action='store_true',
        help='Select best checkpoints using only composite validation examples.',
    )
    p.set_defaults(_fn=_run)


def _run(args: argparse.Namespace) -> int:
    return train(_config_from_args(args))
