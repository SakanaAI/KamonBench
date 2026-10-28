from __future__ import annotations

import os
from typing import Any

import torch

from kamonbench.models.vgg_ngram import VGGImageToTextModel
from kamonbench.models.vit_decoder import DecoderImageCaptioner


def load_checkpoint(
    checkpoint_path: str, device: torch.device, *, model_override: str = 'auto', start_token_override: int = -1
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab_size = checkpoint['vocab_size']
    max_seq_len = checkpoint['max_seq_len']
    end_token = checkpoint['end_token']
    label_to_expr = checkpoint['label_to_expr']

    model_config = checkpoint['config']
    image_size = model_config['image_size']

    checkpoint_model_name = model_config['model']
    model_name = model_override if model_override != 'auto' else checkpoint_model_name

    checkpoint_start_token = model_config['start_token']
    start_token = start_token_override if start_token_override != -1 else checkpoint_start_token

    if model_name == 'vgg':
        hidden_dim = model_config['hidden_dim']
        also_train_vgg = model_config['also_train_vgg']
        use_masks = model_config['use_masks']

        ngram_length = model_config['ngram_length']

        model = VGGImageToTextModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            image_size=image_size,
            ngram_length=ngram_length,
            hidden_dim=hidden_dim,
            also_train_vgg=also_train_vgg,
            use_masks=use_masks,
        )
    elif model_name == 'vit_decoder':
        model = DecoderImageCaptioner(
            encoder_name=model_config['vit_encoder_name'],
            seq_len=max_seq_len,
            vocab_size=vocab_size,
            n_heads=model_config['vit_n_heads'],
            d_model=model_config['vit_d_model'],
            n_layers=model_config['vit_n_layers'],
            dropout=model_config['vit_dropout'],
            token_dropout=model_config['vit_token_dropout'],
            train_backbone=model_config['vit_train_backbone'],
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    metadata = {
        'model': model_name,
        'vocab_size': vocab_size,
        'max_seq_len': max_seq_len,
        'end_token': end_token,
        'label_to_expr': label_to_expr,
        'image_size': image_size,
        'start_token': start_token,
        'use_translation': model_config['use_translation'],
        'seed': model_config.get('seed'),
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
    }
    return model, metadata
