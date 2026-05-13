from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from kamonbench.checkpoints import load_checkpoint
from kamonbench.evaluation.program_factors import parse_program_factors
from kamonbench.evaluation.program_support import load_json, program_records_from_croissant
from kamonbench.torch_utils import autocast_context, configure_torch_runtime

NULL_CONTAINER = 'C:none'
FACTOR_NAMES = ('container', 'modifier', 'motif')


@dataclass(frozen=True)
class FeatureConfig:
    checkpoint_path: Path
    model_name: str
    croissant_path: Path
    basic_charges_csv: Path
    image_root: Path
    output_dir: Path
    splits: tuple[str, ...] = ('train', 'dev', 'test')
    batch_size: int = 128
    device: str = 'cuda:0'
    num_workers: int = 4


@dataclass(frozen=True)
class ProbeConfig:
    feature_dir: Path
    model_name: str
    output_dir: Path
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cuda:0'
    seed: int = 20260428


@dataclass(frozen=True)
class PositionProbeConfig:
    feature_dir: Path
    model_name: str
    output_dir: Path
    positions: int | None = None
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cuda:0'
    seed: int = 20260428


@dataclass(frozen=True)
class CompareConfig:
    feature_dir: Path
    probe_dir: Path
    model_name: str
    prediction_jsonl: Path
    output_dir: Path
    batch_size: int = 512


@dataclass(frozen=True)
class ProbeSupportConfig:
    feature_dir: Path
    probe_dir: Path
    model_name: str
    croissant_path: Path
    basic_charges_csv: Path
    output_dir: Path
    batch_size: int = 512


class CompositeImageDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], *, image_root: Path, image_size: int):
        self.items = items
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        item = self.items[idx]
        path = self.image_root / item['image']
        image = Image.open(path).convert('RGB')
        return self.transform(image), item


def _select_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _factor_labels(factors: Any) -> dict[str, str]:
    return {
        'container': factors.containers[0] if factors.containers else NULL_CONTAINER,
        'modifier': factors.relation,
        'motif': factors.motif,
    }


def _split_items(config: FeatureConfig) -> dict[str, list[dict[str, Any]]]:
    records = program_records_from_croissant(
        load_json(config.croissant_path), basic_charges_csv=config.basic_charges_csv
    )
    items: dict[str, list[dict[str, Any]]] = {split: [] for split in config.splits}
    for record in records:
        if record.split not in items:
            continue
        labels = _factor_labels(record.factors)
        items[record.split].append({'image': record.image, **labels})
    return items


def _feature_batch(model: nn.Module, model_name: str, images: torch.Tensor) -> torch.Tensor:
    if model_name == 'vgg':
        features = model.extract_position_features(images, model.max_seq_len)  # type: ignore[attr-defined]
        return features.flatten(1)
    if model_name == 'vit_decoder':
        memory = model._encode_memory(images)  # type: ignore[attr-defined]
        return memory.mean(dim=1)
    raise ValueError(f'Unsupported model for feature extraction: {model_name}')


def extract_features(config: FeatureConfig) -> dict[str, Any]:
    device = _select_device(config.device)
    configure_torch_runtime(device)
    model, metadata = load_checkpoint(str(config.checkpoint_path), device)
    model_name = metadata['model']
    if model_name not in {'vgg', 'vit_decoder'}:
        raise ValueError(f'Unsupported checkpoint model: {model_name}')

    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_items = _split_items(config)
    manifest = {
        'checkpoint_path': str(config.checkpoint_path),
        'model_name': config.model_name,
        'checkpoint_model': model_name,
        'image_size': metadata['image_size'],
        'feature_kind': 'vgg_position_flatten' if model_name == 'vgg' else 'vit_decoder_memory_mean',
        'splits': {},
    }

    model.eval()
    for split, items in split_items.items():
        dataset = CompositeImageDataset(items, image_root=config.image_root, image_size=metadata['image_size'])
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == 'cuda',
        )
        features = []
        labels: dict[str, list[str]] = {name: [] for name in FACTOR_NAMES}
        images_out: list[str] = []
        with torch.no_grad():
            for images, batch_items in tqdm(loader, desc=f'{config.model_name}:{split}', unit='batch'):
                images = images.to(device, non_blocking=True)
                with autocast_context(device):
                    batch_features = _feature_batch(model, model_name, images)
                features.append(batch_features.detach().cpu().to(torch.float16))
                images_out.extend(batch_items['image'])
                for factor in FACTOR_NAMES:
                    labels[factor].extend(batch_items[factor])

        feature_tensor = torch.cat(features, dim=0) if features else torch.empty((0, 0), dtype=torch.float16)
        out_path = config.output_dir / f'{config.model_name}_{split}.pt'
        torch.save({'features': feature_tensor, 'labels': labels, 'images': images_out}, out_path)
        manifest['splits'][split] = {
            'path': str(out_path),
            'rows': len(images_out),
            'feature_dim': int(feature_tensor.shape[1]) if feature_tensor.ndim == 2 else 0,
        }

    manifest_path = config.output_dir / f'{config.model_name}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def _load_split(feature_dir: Path, model_name: str, split: str) -> dict[str, Any]:
    return torch.load(feature_dir / f'{model_name}_{split}.pt', map_location='cpu')


def _label_maps(train_labels: list[str]) -> tuple[dict[str, int], list[str]]:
    classes = sorted(set(train_labels))
    return {label: idx for idx, label in enumerate(classes)}, classes


def _encode_labels(labels: list[str], label_to_idx: dict[str, int], *, allow_unknown: bool = False) -> torch.Tensor:
    missing = sorted({label for label in labels if label not in label_to_idx})
    if missing and not allow_unknown:
        raise ValueError(f'Labels missing from train classes: {missing[:5]}')
    return torch.tensor([label_to_idx.get(label, -1) for label in labels], dtype=torch.long)


def _standardize(
    train_x: torch.Tensor, dev_x: torch.Tensor, test_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.float().mean(dim=0)
    std = train_x.float().std(dim=0).clamp_min(1e-4)
    return (train_x.float() - mean) / std, (dev_x.float() - mean) / std, (test_x.float() - mean) / std


def _evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, device: torch.device, batch_size: int
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    known_total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            known = yb >= 0
            if bool(known.any()):
                known_logits = logits[known]
                known_y = yb[known]
                loss_sum += float(criterion(known_logits, known_y).item())
                correct += int((known_logits.argmax(dim=1) == known_y).sum().item())
                known_total += int(known_y.numel())
            total += int(yb.numel())
    return {
        'loss': loss_sum / known_total if known_total else None,
        'accuracy': correct / total if total else None,
        'known_accuracy': correct / known_total if known_total else None,
        'correct': correct,
        'total': total,
        'known_total': known_total,
        'unknown_total': total - known_total,
    }


def _train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    config: ProbeConfig,
    num_classes: int,
) -> tuple[dict[str, Any], nn.Module]:
    device = _select_device(config.device)
    probe = nn.Linear(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_state = None
    best_dev = float('inf')
    epochs_without_improvement = 0

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=config.batch_size, shuffle=True)
    for epoch in range(config.max_epochs):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

        dev_metrics = _evaluate(probe, dev_x, dev_y, device=device, batch_size=config.batch_size)
        dev_loss = float(dev_metrics['loss'])
        if dev_loss < best_dev:
            best_dev = dev_loss
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
        probe.to(device)

    metrics = {
        'epochs': epoch + 1,
        'train': _evaluate(probe, train_x, train_y, device=device, batch_size=config.batch_size),
        'dev': _evaluate(probe, dev_x, dev_y, device=device, batch_size=config.batch_size),
        'test': _evaluate(probe, test_x, test_y, device=device, batch_size=config.batch_size),
    }
    return metrics, probe


def train_probes(config: ProbeConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_split(config.feature_dir, config.model_name, 'train')
    dev = _load_split(config.feature_dir, config.model_name, 'dev')
    test = _load_split(config.feature_dir, config.model_name, 'test')
    train_x, dev_x, test_x = _standardize(train['features'], dev['features'], test['features'])

    result: dict[str, Any] = {
        'model_name': config.model_name,
        'feature_dir': str(config.feature_dir),
        'feature_dim': int(train_x.shape[1]),
        'factors': {},
    }

    for factor in FACTOR_NAMES:
        label_to_idx, classes = _label_maps(train['labels'][factor])
        train_y = _encode_labels(train['labels'][factor], label_to_idx)
        dev_y = _encode_labels(dev['labels'][factor], label_to_idx, allow_unknown=True)
        test_y = _encode_labels(test['labels'][factor], label_to_idx, allow_unknown=True)
        metrics, probe = _train_probe(
            train_x, train_y, dev_x, dev_y, test_x, test_y, config=config, num_classes=len(classes)
        )
        result['factors'][factor] = {'classes': len(classes), **metrics}
        torch.save(probe.cpu().state_dict(), config.output_dir / f'{config.model_name}_{factor}_probe.pt')

    result_path = config.output_dir / f'{config.model_name}_probe_metrics.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def _infer_position_count(feature_dim: int, positions: int | None) -> int:
    if positions is not None:
        if positions <= 0:
            raise ValueError(f'positions must be positive, got {positions}')
        if feature_dim % positions != 0:
            raise ValueError(f'Feature dimension {feature_dim} is not divisible by positions={positions}')
        return positions
    if feature_dim % 4096 == 0:
        return feature_dim // 4096
    raise ValueError('Could not infer VGG position count; pass --positions explicitly')


def _position_slice(x: torch.Tensor, position: int, positions: int) -> torch.Tensor:
    feature_dim = x.shape[1]
    per_position = feature_dim // positions
    start = position * per_position
    return x[:, start : start + per_position]


def train_position_probes(config: PositionProbeConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train = _load_split(config.feature_dir, config.model_name, 'train')
    dev = _load_split(config.feature_dir, config.model_name, 'dev')
    test = _load_split(config.feature_dir, config.model_name, 'test')
    feature_dim = int(train['features'].shape[1])
    positions = _infer_position_count(feature_dim, config.positions)
    per_position_dim = feature_dim // positions

    probe_config = ProbeConfig(
        feature_dir=config.feature_dir,
        model_name=config.model_name,
        output_dir=config.output_dir,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
        seed=config.seed,
    )
    result: dict[str, Any] = {
        'model_name': config.model_name,
        'feature_dir': str(config.feature_dir),
        'feature_dim': feature_dim,
        'positions': positions,
        'per_position_dim': per_position_dim,
        'position_feature_kind': 'vgg_position_slice',
        'position_probes': [],
    }

    for position in range(positions):
        train_x, dev_x, test_x = _standardize(
            _position_slice(train['features'], position, positions),
            _position_slice(dev['features'], position, positions),
            _position_slice(test['features'], position, positions),
        )
        for factor in FACTOR_NAMES:
            label_to_idx, classes = _label_maps(train['labels'][factor])
            train_y = _encode_labels(train['labels'][factor], label_to_idx)
            dev_y = _encode_labels(dev['labels'][factor], label_to_idx, allow_unknown=True)
            test_y = _encode_labels(test['labels'][factor], label_to_idx, allow_unknown=True)
            metrics, probe = _train_probe(
                train_x, train_y, dev_x, dev_y, test_x, test_y, config=probe_config, num_classes=len(classes)
            )
            result['position_probes'].append(
                {'position': position, 'factor': factor, 'classes': len(classes), **metrics}
            )
            torch.save(
                probe.cpu().state_dict(), config.output_dir / f'{config.model_name}_pos{position}_{factor}_probe.pt'
            )

    output_path = config.output_dir / f'{config.model_name}_position_probe_metrics.json'
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def _probe_predictions(config: CompareConfig, factor: str) -> tuple[dict[str, Any], list[str], set[str]]:
    train = _load_split(config.feature_dir, config.model_name, 'train')
    dev = _load_split(config.feature_dir, config.model_name, 'dev')
    test = _load_split(config.feature_dir, config.model_name, 'test')
    _, _, test_x = _standardize(train['features'], dev['features'], test['features'])

    classes = sorted(set(train['labels'][factor]))
    idx_to_label = {idx: label for idx, label in enumerate(classes)}
    probe = nn.Linear(test_x.shape[1], len(classes))
    probe.load_state_dict(torch.load(config.probe_dir / f'{config.model_name}_{factor}_probe.pt', map_location='cpu'))
    probe.eval()

    predictions = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(test_x), batch_size=config.batch_size, shuffle=False):
            predictions.extend(idx_to_label[idx] for idx in probe(xb).argmax(dim=1).tolist())
    return test, predictions, set(classes)


def _prediction_jsonl_factor_map(path: Path, factor: str) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            image = item.get('image', '')
            if '_base' in image or '_container' in image:
                continue
            parsed = parse_program_factors(item.get('predicted', ''))
            if parsed is None:
                predictions[image] = None
            elif factor == 'container':
                predictions[image] = parsed.containers[0] if parsed.containers else NULL_CONTAINER
            elif factor == 'modifier':
                predictions[image] = parsed.modifier
            elif factor == 'motif':
                predictions[image] = parsed.motif
            else:
                raise ValueError(f'Unknown factor: {factor}')
    return predictions


def _accuracy_by_slice(rows: dict[str, Any], predictions: list[str | None], factor: str) -> dict[str, dict[str, Any]]:
    result = {}
    masks = {
        'all': [True] * len(rows['images']),
        'contained': [container != NULL_CONTAINER for container in rows['labels']['container']],
        'containerless': [container == NULL_CONTAINER for container in rows['labels']['container']],
    }
    for slice_name, mask in masks.items():
        correct = total = missing = 0
        for keep, gold, pred in zip(mask, rows['labels'][factor], predictions):
            if not keep:
                continue
            total += 1
            if pred is None:
                missing += 1
            elif pred == gold:
                correct += 1
        result[slice_name] = {
            'correct': correct,
            'total': total,
            'missing_predictions': missing,
            'accuracy': correct / total if total else None,
        }
    return result


def compare_probes_to_outputs(config: CompareConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        'model_name': config.model_name,
        'prediction_jsonl': str(config.prediction_jsonl),
        'factors': {},
    }
    for factor in FACTOR_NAMES:
        rows, probe_predictions, train_classes = _probe_predictions(config, factor)
        output_by_image = _prediction_jsonl_factor_map(config.prediction_jsonl, factor)
        output_predictions = [output_by_image.get(image) for image in rows['images']]
        result['factors'][factor] = {
            'train_classes': len(train_classes),
            'test_labels_absent_from_train': sum(label not in train_classes for label in rows['labels'][factor]),
            'probe': _accuracy_by_slice(rows, probe_predictions, factor),
            'output': _accuracy_by_slice(rows, output_predictions, factor),
        }
    return result


def _support_combo_key(labels: dict[str, str], combo: str) -> tuple[str, ...]:
    if combo == 'C,M':
        return labels['container'], labels['motif']
    if combo == 'R,M':
        return labels['modifier'], labels['motif']
    if combo == 'C,R,M':
        return labels['container'], labels['modifier'], labels['motif']
    raise ValueError(f'Unknown combo: {combo}')


def compare_probe_support(config: ProbeSupportConfig) -> dict[str, Any]:
    train_records = [
        record
        for record in program_records_from_croissant(
            load_json(config.croissant_path), basic_charges_csv=config.basic_charges_csv
        )
        if record.split == 'train'
    ]
    support = {
        combo: {_support_combo_key(_factor_labels(record.factors), combo) for record in train_records}
        for combo in ['C,M', 'R,M', 'C,R,M']
    }

    rows = None
    probe_predictions = {}
    for factor in FACTOR_NAMES:
        factor_rows, predictions, _ = _probe_predictions(
            CompareConfig(
                feature_dir=config.feature_dir,
                probe_dir=config.probe_dir,
                model_name=config.model_name,
                prediction_jsonl=Path(),
                output_dir=config.output_dir,
                batch_size=config.batch_size,
            ),
            factor,
        )
        rows = factor_rows
        probe_predictions[factor] = predictions

    if rows is None:
        raise ValueError('No probe rows loaded')
    labels = [{factor: rows['labels'][factor][idx] for factor in FACTOR_NAMES} for idx in range(len(rows['images']))]

    result: dict[str, Any] = {'model_name': config.model_name, 'train_rows': len(train_records), 'combos': {}}
    for combo in ['C,M', 'R,M', 'C,R,M']:
        result['combos'][combo] = {'train_support_count': len(support[combo]), 'factors': {}}
        masks = {
            'absent_from_train': [_support_combo_key(label, combo) not in support[combo] for label in labels],
            'present_in_train': [_support_combo_key(label, combo) in support[combo] for label in labels],
        }
        for factor in FACTOR_NAMES:
            result['combos'][combo]['factors'][factor] = {}
            for slice_name, mask in masks.items():
                correct = total = 0
                for keep, label, pred in zip(mask, labels, probe_predictions[factor]):
                    if not keep:
                        continue
                    total += 1
                    correct += int(pred == label[factor])
                result['combos'][combo]['factors'][factor][slice_name] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': correct / total if total else None,
                }
    return result


def probe_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Factor | Classes | Test CE | Test Acc | Known test | Unknown test |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model_name, result in results.items():
        for factor in FACTOR_NAMES:
            metrics = result['factors'][factor]
            test = metrics['test']
            lines.append(
                f'| {model_name} | {factor} | {metrics["classes"]} | '
                f'{test["loss"]:.3f} | {test["accuracy"]:.3f} | '
                f'{test["known_total"]} | {test["unknown_total"]} |'
            )
    return '\n'.join(lines) + '\n'


def position_probe_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Position | Factor | Test CE | Test Acc | Known test | Unknown test |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for model_name, result in results.items():
        for item in result['position_probes']:
            test = item['test']
            lines.append(
                f'| {model_name} | {item["position"]} | {item["factor"]} | '
                f'{test["loss"]:.3f} | {test["accuracy"]:.3f} | '
                f'{test["known_total"]} | {test["unknown_total"]} |'
            )
    return '\n'.join(lines) + '\n'


def comparison_markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        '| Model | Factor | Slice | Probe Acc | Output Acc | N | Test labels absent from train |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model_name, result in results.items():
        for factor in FACTOR_NAMES:
            for slice_name in ['all', 'contained', 'containerless']:
                probe = result['factors'][factor]['probe'][slice_name]
                output = result['factors'][factor]['output'][slice_name]
                if probe['total'] == 0:
                    continue
                absent = result['factors'][factor]['test_labels_absent_from_train'] if slice_name == 'all' else ''
                lines.append(
                    f'| {model_name} | {factor} | {slice_name} | {probe["accuracy"]:.3f} | '
                    f'{output["accuracy"]:.3f} | {probe["total"]} | {absent} |'
                )
    return '\n'.join(lines) + '\n'


def probe_support_markdown(results: dict[str, dict[str, Any]], *, factor: str = 'motif') -> str:
    lines = [
        '| Model | Held-out combo | N absent | Probe Acc absent | N present | Probe Acc present |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model_name, result in results.items():
        for combo in ['C,M', 'R,M', 'C,R,M']:
            summary = result['combos'][combo]['factors'][factor]
            absent = summary['absent_from_train']
            present = summary['present_in_train']
            lines.append(
                f'| {model_name} | {combo} | {absent["total"]} | {absent["accuracy"]:.3f} | '
                f'{present["total"]} | {present["accuracy"]:.3f} |'
            )
    return '\n'.join(lines) + '\n'


def summarize_probe_seed_stability(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for comparison in comparisons:
        model_name = comparison['model_name']
        model_values = values.setdefault(model_name, {})
        for factor in FACTOR_NAMES:
            factor_values = model_values.setdefault(factor, {})
            for slice_name in ['all', 'contained', 'containerless']:
                accuracy = comparison['factors'][factor]['probe'][slice_name]['accuracy']
                if accuracy is None:
                    continue
                factor_values.setdefault(slice_name, {}).setdefault('accuracies', []).append(float(accuracy))

    summary: dict[str, Any] = {'runs': len(comparisons), 'models': {}}
    for model_name, factors in values.items():
        summary['models'][model_name] = {'factors': {}}
        for factor, slices in factors.items():
            summary['models'][model_name]['factors'][factor] = {}
            for slice_name, item in slices.items():
                xs = item['accuracies']
                summary['models'][model_name]['factors'][factor][slice_name] = {
                    'runs': len(xs),
                    'mean': statistics.fmean(xs),
                    'sd': statistics.stdev(xs) if len(xs) > 1 else 0.0,
                    'min': min(xs),
                    'max': max(xs),
                }
    return summary


def seed_stability_markdown(result: dict[str, Any], *, factor: str = 'motif') -> str:
    def fmt(value: float) -> str:
        return f'{value + 1e-12:.3f}'

    lines = [
        '| Model | Slice | Probe Acc Mean | SD | Min | Max | Runs |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model_name, model in result['models'].items():
        for slice_name in ['all', 'contained', 'containerless']:
            item = model['factors'][factor][slice_name]
            lines.append(
                f'| {model_name} | {slice_name} | {fmt(item["mean"])} | {fmt(item["sd"])} | '
                f'{fmt(item["min"])} | {fmt(item["max"])} | {item["runs"]} |'
            )
    return '\n'.join(lines) + '\n'


def _extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--croissant-path', type=Path, default=Path('etc/source/kamon_croissant.json'))
    parser.add_argument('--basic-charges-csv', type=Path, default=Path('dev/KamonBench/data/basic_charges.csv'))
    parser.add_argument('--image-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=4)


def _probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('etc/frozen_probe_metrics'))
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260428)


def _position_probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('etc/position_probe_metrics'))
    parser.add_argument('--positions', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20260428)


def _compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--probe-dir', type=Path, default=Path('etc/frozen_probe_metrics'))
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--prediction-jsonl', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('etc/frozen_probe_metrics'))
    parser.add_argument('--batch-size', type=int, default=512)


def _probe_support_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--feature-dir', type=Path, default=Path('etc/frozen_features'))
    parser.add_argument('--probe-dir', type=Path, default=Path('etc/frozen_probe_metrics'))
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--croissant-path', type=Path, default=Path('etc/source/kamon_croissant.json'))
    parser.add_argument('--basic-charges-csv', type=Path, default=Path('dev/KamonBench/data/basic_charges.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('etc/frozen_probe_metrics'))
    parser.add_argument('--batch-size', type=int, default=512)


def _seed_stability_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('comparison_json', nargs='+', type=Path)
    parser.add_argument('--factor', choices=FACTOR_NAMES, default='motif')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--markdown-output', type=Path, default=None)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    extract = subparsers.add_parser('extract-representation-features', help='Extract frozen baseline features.')
    _extract_args(extract)
    extract.set_defaults(_fn=_run_extract)

    probe = subparsers.add_parser('train-representation-probes', help='Train linear probes on frozen features.')
    _probe_args(probe)
    probe.set_defaults(_fn=_run_probe)

    position_probe = subparsers.add_parser(
        'train-position-probes', help='Train linear probes on each VGG decoder-position feature slice.'
    )
    _position_probe_args(position_probe)
    position_probe.set_defaults(_fn=_run_position_probe)

    compare = subparsers.add_parser(
        'compare-representation-probes', help='Compare frozen linear probes to generated program outputs.'
    )
    _compare_args(compare)
    compare.set_defaults(_fn=_run_compare)

    support = subparsers.add_parser(
        'compare-representation-probe-support',
        help='Compare frozen linear probes on train-absent and train-present factor combinations.',
    )
    _probe_support_args(support)
    support.set_defaults(_fn=_run_probe_support)

    stability = subparsers.add_parser(
        'summarize-representation-probe-seeds', help='Summarize probe accuracy across comparison JSON files.'
    )
    _seed_stability_args(stability)
    stability.set_defaults(_fn=_run_seed_stability)


def _run_extract(args: argparse.Namespace) -> int:
    result = extract_features(
        FeatureConfig(
            checkpoint_path=args.checkpoint_path,
            model_name=args.model_name,
            croissant_path=args.croissant_path,
            basic_charges_csv=args.basic_charges_csv,
            image_root=args.image_root,
            output_dir=args.output_dir,
            splits=tuple(args.splits),
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_probe(args: argparse.Namespace) -> int:
    result = train_probes(
        ProbeConfig(
            feature_dir=args.feature_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )
    )
    markdown_path = args.output_dir / 'probe_metrics.md'
    existing: dict[str, dict[str, Any]] = {}
    for path in sorted(args.output_dir.glob('*_probe_metrics.json')):
        existing[path.name.removesuffix('_probe_metrics.json')] = json.loads(path.read_text(encoding='utf-8'))
    markdown_path.write_text(probe_markdown(existing), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_position_probe(args: argparse.Namespace) -> int:
    result = train_position_probes(
        PositionProbeConfig(
            feature_dir=args.feature_dir,
            model_name=args.model_name,
            output_dir=args.output_dir,
            positions=args.positions,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )
    )
    markdown_path = args.output_dir / 'position_probe_metrics.md'
    existing: dict[str, dict[str, Any]] = {}
    for path in sorted(args.output_dir.glob('*_position_probe_metrics.json')):
        existing[path.name.removesuffix('_position_probe_metrics.json')] = json.loads(path.read_text(encoding='utf-8'))
    markdown_path.write_text(position_probe_markdown(existing), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_probes_to_outputs(
        CompareConfig(
            feature_dir=args.feature_dir,
            probe_dir=args.probe_dir,
            model_name=args.model_name,
            prediction_jsonl=args.prediction_jsonl,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    )
    output_path = args.output_dir / f'{args.model_name}_probe_output_comparison.json'
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    existing = {}
    for path in sorted(args.output_dir.glob('*_probe_output_comparison.json')):
        existing[path.name.removesuffix('_probe_output_comparison.json')] = json.loads(path.read_text(encoding='utf-8'))
    (args.output_dir / 'probe_output_comparison.md').write_text(comparison_markdown(existing), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_probe_support(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_probe_support(
        ProbeSupportConfig(
            feature_dir=args.feature_dir,
            probe_dir=args.probe_dir,
            model_name=args.model_name,
            croissant_path=args.croissant_path,
            basic_charges_csv=args.basic_charges_csv,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    )
    output_path = args.output_dir / f'{args.model_name}_probe_support.json'
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    existing = {}
    for path in sorted(args.output_dir.glob('*_probe_support.json')):
        existing[path.name.removesuffix('_probe_support.json')] = json.loads(path.read_text(encoding='utf-8'))
    (args.output_dir / 'probe_support.md').write_text(probe_support_markdown(existing), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_seed_stability(args: argparse.Namespace) -> int:
    comparisons = [json.loads(path.read_text(encoding='utf-8')) for path in args.comparison_json]
    result = summarize_probe_seed_stability(comparisons)
    result['comparison_json'] = [str(path) for path in args.comparison_json]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(seed_stability_markdown(result, factor=args.factor), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    add_parser(subparsers)
    args = parser.parse_args(argv)
    return int(args._fn(args))


if __name__ == '__main__':
    raise SystemExit(main())
