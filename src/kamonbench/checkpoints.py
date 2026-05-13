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

    model_config = checkpoint.get('config', {})
    image_size = model_config.get('image_size', 224)

    checkpoint_model_name = model_config.get('model', 'vgg')
    model_name = model_override if model_override != 'auto' else checkpoint_model_name
    if model_name not in {'vgg', 'vit_decoder'}:
        model_name = 'vgg'

    checkpoint_start_token = model_config.get('start_token', 0)
    start_token = start_token_override if start_token_override != -1 else checkpoint_start_token

    if model_name == 'vgg':
        hidden_dim = model_config.get('hidden_dim', 512)
        position_embedding_dim = model_config.get('position_embedding_dim', 0)
        also_train_vgg = model_config.get('also_train_vgg', False)
        use_masks = model_config.get('use_masks', True)

        if 'ngram_length' in model_config:
            ngram_length = model_config['ngram_length']
        else:
            vgg_feature_dim = 4096
            feature_combiner_input_dim = checkpoint['model_state_dict']['feature_combiner.0.weight'].shape[1]
            ngram_length = 1 + (feature_combiner_input_dim - vgg_feature_dim) // (vgg_feature_dim + vocab_size)

        model = VGGImageToTextModel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            image_size=image_size,
            ngram_length=ngram_length,
            hidden_dim=hidden_dim,
            position_embedding_dim=position_embedding_dim,
            also_train_vgg=also_train_vgg,
            use_masks=use_masks,
        )
    elif model_name == 'vit_decoder':
        model = DecoderImageCaptioner(
            encoder_name=model_config.get('vit_encoder_name', 'vit_base_patch16_224'),
            seq_len=max_seq_len,
            vocab_size=vocab_size,
            n_heads=model_config.get('vit_n_heads', 8),
            d_model=model_config.get('vit_d_model', 512),
            n_layers=model_config.get('vit_n_layers', 2),
            dropout=model_config.get('vit_dropout', 0.1),
            token_dropout=model_config.get('vit_token_dropout', 0.0),
            train_backbone=model_config.get('vit_train_backbone', False),
            enc_proj_rank=model_config.get('vit_enc_proj_rank', 0),
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')

    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'state_dict'):
        expected = model.state_dict()
        if 'position_embeddings' in state_dict and 'position_embeddings' not in expected:
            saved_positions = state_dict['position_embeddings']
            if hasattr(saved_positions, 'numel') and saved_positions.numel() == 0:
                state_dict = dict(state_dict)
                del state_dict['position_embeddings']
        key = 'text_embedding.weight'
        if key in state_dict and key in expected:
            saved_weight = state_dict[key]
            expected_weight = expected[key]
            if saved_weight.shape[0] + 1 == expected_weight.shape[0]:
                state_dict = dict(state_dict)
                state_dict[key] = torch.cat([saved_weight, saved_weight.mean(dim=0, keepdim=True)], dim=0)

    model.load_state_dict(state_dict)
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
        'use_translation': model_config.get('use_translation'),
        'step': checkpoint.get('step', 'unknown'),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', checkpoint.get('loss', 'unknown')),
    }
    return model, metadata
