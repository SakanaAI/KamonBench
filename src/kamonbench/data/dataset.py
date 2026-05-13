from __future__ import annotations

import json
import random
import time
from typing import Any

import torch
from torchvision import transforms
from tqdm import tqdm

from . import augment
from .images import retrieve_image
from .labels import END_TOKEN, analysis_tokens, create_label_sets, translation_tokens

PAD_LABEL = -1


class KamonDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        croissant_path: str,
        division: str = 'train',
        image_size: int = 224,
        dataset_mean: list[float] = [0.5, 0.5, 0.5],
        dataset_std: list[float] = [0.5, 0.5, 0.5],
        one_hot: bool = False,
        pad: bool = True,
        num_augmentations: int = 5,
        use_translation: bool = False,
        root_dir: str = '.',
        verbose: bool = True,
    ):
        if division not in ['train', 'dev', 'test']:
            raise ValueError(f'division must be train/dev/test, got {division!r}')

        self.division = division
        self.image_size = image_size
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.one_hot = one_hot
        self.pad = pad
        self.num_augmentations = num_augmentations
        self.use_translation = use_translation
        self.root_dir = root_dir
        self.verbose = verbose

        with open(croissant_path, 'r', encoding='utf-8') as f:
            croissant_data = json.load(f)
        all_records = croissant_data['recordSet'][0]['data']

        if self.verbose:
            print(f'Creating label sets from {len(all_records)} total records...')
        if self.use_translation:
            missing = [r.get('description', '<missing description>') for r in all_records if not translation_tokens(r)]
            missing_field = 'translation'
        else:
            missing = [r.get('description', '<missing description>') for r in all_records if not analysis_tokens(r)]
            missing_field = 'analysis'
        if missing:
            examples = ', '.join(repr(x) for x in missing[:5])
            raise ValueError(
                f'{len(missing)} records have empty {missing_field}; examples: {examples}. '
                'Regenerate Croissant labels explicitly with --label-mode jp-en or --label-mode program.'
            )

        (self.expr_to_label, self.label_to_expr, self.translation_expr_to_label, self.translation_label_to_expr) = (
            create_label_sets(all_records)
        )
        # Backward-compatible attribute names for older callers.
        self.en_expr_to_label = self.translation_expr_to_label
        self.en_label_to_expr = self.translation_label_to_expr
        if self.verbose:
            print(
                f'Vocabularies created: analysis={len(self.expr_to_label)}, '
                f'translation={len(self.translation_expr_to_label)}'
            )

        if self.use_translation:
            self.active_expr_to_label = self.translation_expr_to_label
            self.active_label_to_expr = self.translation_label_to_expr
        else:
            self.active_expr_to_label = self.expr_to_label
            self.active_label_to_expr = self.label_to_expr

        self.end_token = self.active_expr_to_label[END_TOKEN]
        self.vocab_size = len(self.active_expr_to_label)
        self.max_v = self.vocab_size

        self.max_len = 0
        for record in all_records:
            if self.use_translation:
                tokens = translation_tokens(record)
                labels = [self.translation_expr_to_label[t] for t in tokens if t in self.translation_expr_to_label]
            else:
                exprs = analysis_tokens(record)
                labels = [self.expr_to_label[e] for e in exprs if e in self.expr_to_label]
            labels_len = len(labels) + 1
            if labels_len > self.max_len:
                self.max_len = labels_len

        split_records = [r for r in all_records if r['split'] == division]

        self.metadata: list[dict[str, Any]] = []
        if self.verbose:
            print(f'Loading {len(split_records)} images for {division} split...')
        record_iter = tqdm(split_records, desc=f'Loading {division} images', unit='img', disable=not self.verbose)
        for record in record_iter:
            if self.use_translation:
                tokens = translation_tokens(record)
                labels = [self.translation_expr_to_label[t] for t in tokens if t in self.translation_expr_to_label]
            else:
                exprs = analysis_tokens(record)
                labels = [self.expr_to_label[e] for e in exprs if e in self.expr_to_label]

            labels = labels + [self.end_token]

            image = retrieve_image(record['image_path'], size=self.image_size, root=self.root_dir)

            self.metadata.append(
                {
                    'description': record['description'],
                    'translation': record.get('translation', ''),
                    'labels': labels,
                    'image': image,
                    'image_path': record['image_path'],
                }
            )

        self._indices: list[tuple[int, bool]] = [(i, False) for i in range(len(self.metadata))]
        if division == 'train' and num_augmentations > 0:
            if self.verbose:
                print(f'Applying {num_augmentations}x data augmentation...')
            random.seed(time.time())
            base_len = len(self.metadata)
            item_iter = tqdm(range(base_len), desc='Indexing augmentations', unit='img', disable=not self.verbose)
            for idx in item_iter:
                for _ in range(num_augmentations):
                    self._indices.append((idx, True))
            random.shuffle(self._indices)
            if self.verbose:
                print(f'Training set expanded to {len(self._indices)} images')

        self.padded = [PAD_LABEL] * self.max_len
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.dataset_mean, self.dataset_std),
            ]
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        item_idx, should_augment = self._indices[idx]
        item = self.metadata[item_idx]
        image = item['image']
        if should_augment:
            image = augment.apply_adjustments(image)
        labels = item['labels']

        if self.pad:
            labels_t = torch.tensor((labels + self.padded)[: self.max_len], dtype=torch.long)
        else:
            labels_t = torch.tensor(labels, dtype=torch.long)

        if self.one_hot:
            pad_mask = labels_t == PAD_LABEL
            labels_t = torch.nn.functional.one_hot(labels_t.clamp_min(0), self.max_v)
            labels_t[pad_mask] = 0

        return self.transform(image), labels_t

    def dump_text(self, path: str) -> None:
        seen = set()
        for item in self.metadata:
            text = ' '.join([self.active_label_to_expr[t] for t in item['labels']])
            seen.add(text)

        with open(path, 'w', encoding='utf-8') as f:
            for text in sorted(seen):
                f.write(f'{text}\n')
