"""VGG-based image-to-text model for Japanese Kamon description generation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class VGGImageToTextModel(nn.Module):
    """VGG-based n-gram language model for image-to-text generation."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        image_size: int = 224,
        ngram_length: int = 2,
        vgg_feature_dim: int = 4096,
        hidden_dim: int = 512,
        position_embedding_dim: int = 0,
        also_train_vgg: bool = False,
        use_masks: bool = True,
    ):
        super().__init__()
        if position_embedding_dim < 0:
            raise ValueError(f'position_embedding_dim must be non-negative, got {position_embedding_dim}')

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.ngram_length = ngram_length
        self.vgg_feature_dim = vgg_feature_dim
        self.hidden_dim = hidden_dim
        self.position_embedding_dim = position_embedding_dim
        self.decoder_feature_dim = vgg_feature_dim + position_embedding_dim
        self.also_train_vgg = also_train_vgg
        self.use_masks = use_masks

        self.construct_vgg_classifier()

        if self.use_masks:
            self.position_masks = nn.Parameter(torch.randn(max_seq_len, 1, image_size, image_size) * 0.1)
        else:
            self.register_buffer('position_masks', torch.ones(max_seq_len, 1, image_size, image_size))

        if position_embedding_dim > 0:
            self.position_embeddings = nn.Parameter(torch.randn(max_seq_len, position_embedding_dim) * 0.02)
        else:
            self.register_buffer('position_embeddings', torch.zeros(max_seq_len, 0), persistent=False)

        input_dim = self.decoder_feature_dim + (ngram_length - 1) * (self.decoder_feature_dim + vocab_size)
        self.feature_combiner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Linear(hidden_dim, vocab_size)

        self.register_buffer('dummy_features', torch.zeros(1, self.decoder_feature_dim))
        self.register_buffer('dummy_logits', torch.zeros(1, vocab_size))

    def construct_vgg_classifier(self) -> None:
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg_model.classifier = vgg_model.classifier[:-1]

        params = list(vgg_model.parameters())
        if not self.also_train_vgg:
            for p in params:
                p.requires_grad = False
            vgg_model.eval()

        self.vgg_nfeatures = params[-1].shape[0] if params else 4096
        self.vgg = vgg_model

    def apply_mask_to_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(mask)

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.size(0) == 1 and image.size(0) > 1:
            mask = mask.expand(image.size(0), -1, -1, -1)

        mask = mask.expand(-1, 3, -1, -1)
        return image * mask

    def extract_vgg_features(self, image: torch.Tensor) -> torch.Tensor:
        if not self.also_train_vgg:
            with torch.no_grad():
                return self.vgg(image)
        return self.vgg(image)

    def position_indices(self, seq_len: int) -> torch.Tensor:
        if seq_len <= 0:
            raise ValueError(f'seq_len must be positive, got {seq_len}')
        return torch.arange(seq_len, device=self.position_masks.device).clamp_max(self.max_seq_len - 1)

    def extract_position_features(self, images: torch.Tensor, seq_len: int) -> torch.Tensor:
        if not self.use_masks:
            if seq_len <= 0:
                raise ValueError(f'seq_len must be positive, got {seq_len}')
            features = self.extract_vgg_features(images)
            features = features.unsqueeze(1).expand(-1, seq_len, -1)
            return self.add_position_embeddings(features, seq_len)

        batch_size = images.size(0)
        masks = torch.sigmoid(self.position_masks[self.position_indices(seq_len)]).unsqueeze(0)
        masked = images.unsqueeze(1) * masks
        flat_masked = masked.reshape(batch_size * seq_len, *images.shape[1:])
        flat_features = self.extract_vgg_features(flat_masked)
        features = flat_features.reshape(batch_size, seq_len, -1)
        return self.add_position_embeddings(features, seq_len)

    def add_position_embeddings(self, features: torch.Tensor, seq_len: int) -> torch.Tensor:
        if self.position_embedding_dim == 0:
            return features
        positions = self.position_embeddings[self.position_indices(seq_len)]
        positions = positions.unsqueeze(0).expand(features.size(0), -1, -1)
        return torch.cat([features, positions], dim=-1)

    def expanded_masks(self, batch_size: int, seq_len: int) -> torch.Tensor:
        masks = torch.sigmoid(self.position_masks[self.position_indices(seq_len), 0])
        return masks.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, images: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = self.max_seq_len

        batch_size = images.size(0)

        all_logits = []
        position_features = self.extract_position_features(images, seq_len)

        feature_history = []
        logits_history = []

        for pos in range(seq_len):
            vgg_features = position_features[:, pos, :]

            context_features = []
            context_logits = []
            for i in range(self.ngram_length - 1):
                if i < len(feature_history):
                    hist_idx = len(feature_history) - 1 - i
                    context_features.append(feature_history[hist_idx])
                    context_logits.append(logits_history[hist_idx])
                else:
                    context_features.append(self.dummy_features.expand(batch_size, -1))
                    context_logits.append(self.dummy_logits.expand(batch_size, -1))

            combined_parts = [vgg_features]
            for feat, logit in zip(context_features, context_logits):
                combined_parts.extend([feat, logit])
            combined_input = torch.cat(combined_parts, dim=1)

            hidden_features = self.feature_combiner(combined_input)
            logits = self.classifier(hidden_features)

            all_logits.append(logits)

            feature_history.append(vgg_features)
            logits_history.append(logits)

            if len(feature_history) > self.ngram_length - 1:
                feature_history.pop(0)
                logits_history.pop(0)

        return torch.stack(all_logits, dim=1), self.expanded_masks(batch_size, seq_len)

    def get_mask_diversity_loss(self, weight: float = 0.01) -> torch.Tensor:
        if not self.use_masks:
            return torch.tensor(0.0, device=self.position_masks.device)

        masks = torch.sigmoid(self.position_masks)
        masks_flat = masks.view(self.max_seq_len, -1)
        similarities = torch.mm(masks_flat, masks_flat.t())

        mask_diag = torch.eye(self.max_seq_len, device=masks.device, dtype=masks.dtype)
        off_diagonal = similarities * (1 - mask_diag)
        diversity_loss = off_diagonal.sum() / (self.max_seq_len * (self.max_seq_len - 1))
        return weight * diversity_loss

    def generate(
        self, images: torch.Tensor, end_token: int, max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_length is None:
            max_length = self.max_seq_len

        batch_size = images.size(0)

        generated_tokens = []
        position_features = self.extract_position_features(images, max_length)

        feature_history = []
        logits_history = []

        for pos in range(max_length):
            vgg_features = position_features[:, pos, :]

            context_features = []
            context_logits = []
            for i in range(self.ngram_length - 1):
                if i < len(feature_history):
                    hist_idx = len(feature_history) - 1 - i
                    context_features.append(feature_history[hist_idx])
                    context_logits.append(logits_history[hist_idx])
                else:
                    context_features.append(self.dummy_features.expand(batch_size, -1))
                    context_logits.append(self.dummy_logits.expand(batch_size, -1))

            combined_parts = [vgg_features]
            for feat, logit in zip(context_features, context_logits):
                combined_parts.extend([feat, logit])
            combined_input = torch.cat(combined_parts, dim=1)

            hidden_features = self.feature_combiner(combined_input)
            logits = self.classifier(hidden_features)

            tokens = torch.argmax(logits, dim=1)
            generated_tokens.append(tokens)

            feature_history.append(vgg_features)
            logits_history.append(logits)
            if len(feature_history) > self.ngram_length - 1:
                feature_history.pop(0)
                logits_history.pop(0)

            if torch.all(tokens == end_token):
                break

        output_len = len(generated_tokens)
        return torch.stack(generated_tokens, dim=1), self.expanded_masks(batch_size, output_len)
