"""Qwen training recipes used in the supplementary experiments."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.distributed as dist
from kamonbench.data.metadata import load_records
from kamonbench.evaluation.predictions import evaluate_predictions
from kamonbench.evaluation.program_factors import evaluate_program_factors
from kamonbench.metrics import token_error_rate
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AddedToken, AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune the reported Qwen baselines.')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    encoded = ':'.join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')


def target(record: dict[str, Any]) -> str:
    return ' '.join(item['expr'] for item in record['analysis'])


def messages(image: Image.Image, prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}
    ]
    if answer is not None:
        result.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
    return result


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class KamonTrainingDataset(Dataset):
    def __init__(
        self, records: list[dict[str, Any]], image_root: Path, *, augmentations: int, seed: int, augment_module: Any
    ) -> None:
        self.records = records
        self.image_root = image_root
        self.augmentations = augmentations
        self.seed = seed
        self.augment_module = augment_module
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * (self.augmentations + 1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slots = self.augmentations + 1
        record = self.records[index // slots]
        slot = index % slots
        path = self.image_root / record['image_path']
        with Image.open(path) as source:
            image = source.convert('RGB')
        if slot:
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            sample_seed = stable_seed(self.seed, self.epoch, record['id'], slot)
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32))
            try:
                image = self.augment_module.apply_adjustments(image)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        return {
            'id': record['id'],
            'image': image,
            'image_path': record['image_path'],
            'target': target(record),
            'augmentation_slot': slot,
        }


class KamonEvaluationDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], image_root: Path) -> None:
        self.records = records
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.image_root / record['image_path']
        with Image.open(path) as source:
            image = source.convert('RGB')
        return {
            'id': record['id'],
            'order': index,
            'image': image,
            'image_path': record['image_path'],
            'target': target(record),
        }


class TrainingCollator:
    def __init__(self, processor: Any, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt
        self.prompt_length: int | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.prompt_length is None:
            prompt_inputs = self.processor.apply_chat_template(
                messages(examples[0]['image'], self.prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors='pt',
            )
            self.prompt_length = int(prompt_inputs['input_ids'].shape[1])

        conversations = [messages(example['image'], self.prompt, example['target']) for example in examples]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            padding=True,
            return_dict=True,
            return_tensors='pt',
        )
        labels = inputs['input_ids'].clone()
        labels[:, : self.prompt_length] = -100
        labels[inputs['attention_mask'] == 0] = -100
        if not bool((labels != -100).any(dim=1).all()):
            raise RuntimeError('At least one training row has no supervised assistant token.')
        inputs['labels'] = labels
        inputs['metadata'] = [
            {key: example[key] for key in ('id', 'image_path', 'target', 'augmentation_slot')} for example in examples
        ]
        return inputs


class EvaluationCollator:
    def __init__(self, processor: Any, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self.processor.apply_chat_template(
            [messages(example['image'], self.prompt) for example in examples],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs['metadata'] = [
            {key: example[key] for key in ('id', 'order', 'image_path', 'target')} for example in examples
        ]
        return inputs


def configure_runtime(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32))
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def init_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world_size > 1:
        dist.init_process_group(backend='nccl', device_id=device)
    return rank, local_rank, world_size, device


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def broadcast_object(value: Any, rank: int) -> Any:
    if not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _adapt_model(
    model: torch.nn.Module, config: dict[str, Any], base_parameter_count: int
) -> tuple[torch.nn.Module, dict[str, Any]]:
    adaptation = config['adaptation']
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=adaptation['lora']['rank'],
            lora_alpha=adaptation['lora']['alpha'],
            lora_dropout=adaptation['lora']['dropout'],
            target_modules=adaptation['lora']['target_modules'],
            modules_to_save=adaptation['modules_to_save'],
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    base = model.get_base_model()
    vision_indices = adaptation['vision_block_indices']
    if vision_indices:
        depth = len(base.visual.blocks)
        if vision_indices != list(range(depth - len(vision_indices), depth)):
            raise ValueError(
                f'Vision candidate must be a contiguous final suffix; got {vision_indices} for depth {depth}.'
            )
        for index in vision_indices:
            for parameter in base.visual.blocks[index].parameters():
                parameter.requires_grad = True

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    groups = parameter_groups(trainable, config)
    lora_modules = [
        name
        for name, module in model.named_modules()
        if name.endswith('.lora_A.default') and isinstance(module, torch.nn.Linear)
    ]
    details = {
        'base_parameters': base_parameter_count,
        'trainable_parameters': sum(parameter.numel() for parameter in trainable.values()),
        'trainable_fraction': (sum(parameter.numel() for parameter in trainable.values()) / base_parameter_count),
        'trainable_tensor_count': len(trainable),
        'lora_module_count': len(lora_modules),
        'vision_depth': len(base.visual.blocks),
        'vision_block_indices': vision_indices,
        'group_parameter_counts': {
            name: sum(parameter.numel() for parameter in parameters) for name, parameters in groups.items()
        },
        'group_parameter_dtypes': {
            name: sorted({str(parameter.dtype) for parameter in parameters}) for name, parameters in groups.items()
        },
        'trainable_names': sorted(trainable),
    }
    return model, details


def _load_native_model(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_config = config['model']
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_config['id'],
        revision=model_config['revision'],
        dtype=torch.bfloat16,
        attn_implementation=model_config['attention_implementation'],
        device_map={'': device},
        low_cpu_mem_usage=True,
    )
    base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model, details = _adapt_model(model, config, base_parameter_count)
    base = model.get_base_model()
    input_embedding = base.model.language_model.embed_tokens.weight
    original_output_head = base.lm_head.original_module.weight
    trainable_output_head = base.lm_head.modules_to_save['default'].weight
    expected_base_parameters = model_config.get('expected_base_parameters')
    if expected_base_parameters is not None and base_parameter_count != expected_base_parameters:
        raise ValueError(f'Base parameter count {base_parameter_count} != expected {expected_base_parameters}.')
    configured_tying = bool(base.config.tie_word_embeddings)
    expected_tying = model_config.get('expected_tie_word_embeddings')
    if expected_tying is not None and configured_tying != expected_tying:
        raise ValueError(f'Configured tying {configured_tying} != expected {expected_tying}.')
    if configured_tying != (input_embedding is original_output_head):
        raise RuntimeError('Input/output weight identity disagrees with tie_word_embeddings.')
    if input_embedding is trainable_output_head:
        raise RuntimeError('PEFT trainable output head must be decoupled from the frozen input embedding.')
    if input_embedding.requires_grad or original_output_head.requires_grad or not trainable_output_head.requires_grad:
        raise RuntimeError('Expected frozen input/original head and a trainable PEFT output-head copy.')
    details.update(
        {
            'configured_tie_word_embeddings': configured_tying,
            'input_embedding_is_original_output_head': input_embedding is original_output_head,
            'input_embedding_is_trainable_output_head': input_embedding is trainable_output_head,
            'input_embedding_trainable': input_embedding.requires_grad,
            'original_output_head_trainable': original_output_head.requires_grad,
            'peft_output_head_trainable': trainable_output_head.requires_grad,
            'input_embedding_shape': list(input_embedding.shape),
            'output_head_shape': list(trainable_output_head.shape),
        }
    )
    return model, details


def add_atomic_tokens(processor: Any, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = processor.tokenizer
    factors = sorted({item['expr'] for record in records for item in record['analysis']})
    original_size = len(tokenizer)
    manifest = [
        {'factor': factor, 'native_token_ids': tokenizer.encode(factor, add_special_tokens=False)} for factor in factors
    ]
    added = tokenizer.add_tokens([AddedToken(factor, normalized=False, single_word=False) for factor in factors])
    for row in manifest:
        row['atomic_token_id'] = tokenizer.convert_tokens_to_ids(row['factor'])
    atomic_ids = [row['atomic_token_id'] for row in manifest]
    if added != len(factors) or len(set(atomic_ids)) != len(factors) or min(atomic_ids) < original_size:
        raise RuntimeError(
            f'Atomic-token addition failed: factors={len(factors)}, added={added}, '
            f'unique_ids={len(set(atomic_ids))}, minimum_id={min(atomic_ids)}.'
        )
    factor_lengths = {len(tokenizer.encode(factor, add_special_tokens=False)) for factor in factors}
    if factor_lengths != {1}:
        raise RuntimeError(f'Factors are not atomic after addition: {factor_lengths}.')
    target_lengths: set[int] = set()
    for record in records:
        text = target(record)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise RuntimeError(f'Atomic target does not round-trip: {decoded!r} != {text!r}.')
        target_lengths.add(len(token_ids))
    return manifest, {
        'factor_count': len(factors),
        'tokenizer_size_before': original_size,
        'tokenizer_size_after': len(tokenizer),
        'factor_token_lengths': sorted(factor_lengths),
        'complete_target_token_lengths': sorted(target_lengths),
        'initialization': 'mean of each factor native-token input embeddings',
        'input_embeddings_trainable': False,
        'output_head_trainability': 'same fully trainable PEFT-decoupled head as native baseline',
    }


def _load_atomic_model(
    config: dict[str, Any], device: torch.device, processor: Any, atomic_manifest: list[dict[str, Any]]
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_config = config['model']
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_config['id'],
        revision=model_config['revision'],
        dtype=torch.bfloat16,
        attn_implementation=model_config['attention_implementation'],
        device_map={'': device},
        low_cpu_mem_usage=True,
    )
    original_base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    tied_before_resize = model.get_input_embeddings().weight is model.lm_head.weight
    model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    tied_after_resize = model.get_input_embeddings().weight is model.lm_head.weight
    input_weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for row in atomic_manifest:
            source_ids = torch.tensor(row['native_token_ids'], device=device)
            input_weight[row['atomic_token_id']].copy_(input_weight[source_ids].mean(dim=0))
    initialization_max_error = max(
        (
            input_weight[row['atomic_token_id']]
            - input_weight[torch.tensor(row['native_token_ids'], device=device)].mean(dim=0)
        )
        .abs()
        .max()
        .item()
        for row in atomic_manifest
    )
    if initialization_max_error != 0:
        raise RuntimeError(f'Atomic native-mean initialization changed: {initialization_max_error}.')
    expanded_base_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model, details = _adapt_model(model, config, expanded_base_parameter_count)
    base = model.get_base_model()
    active_output_weight = base.lm_head.modules_to_save['default'].weight
    if base.get_input_embeddings().weight.requires_grad or active_output_weight is base.get_input_embeddings().weight:
        raise RuntimeError('Atomic input embeddings are not frozen or the output head is not decoupled.')
    details.update(
        {
            'original_base_parameters': original_base_parameter_count,
            'expanded_vocabulary_parameters': expanded_base_parameter_count - original_base_parameter_count,
            'atomic_tokens': {
                'count': len(atomic_manifest),
                'initialization_max_error': initialization_max_error,
                'embeddings_tied_before_resize': tied_before_resize,
                'embeddings_tied_after_resize': tied_after_resize,
                'input_embeddings_trainable': base.get_input_embeddings().weight.requires_grad,
                'active_output_head_decoupled': active_output_weight is not base.get_input_embeddings().weight,
                'active_output_head_trainable': active_output_weight.requires_grad,
            },
        }
    )
    return model, details


def load_model(config, device, processor=None, atomic_manifest=None):
    if config.get('output_tokens') == 'factor':
        if processor is None or atomic_manifest is None:
            raise ValueError('Factor-token models require their processor and token manifest.')
        return _load_atomic_model(config, device, processor, atomic_manifest)
    return _load_native_model(config, device)


def parameter_groups(
    trainable: dict[str, torch.nn.Parameter], config: dict[str, Any]
) -> dict[str, list[torch.nn.Parameter]]:
    vision_indices = config['adaptation']['vision_block_indices']
    groups: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
    unmatched = []
    for name, parameter in trainable.items():
        if 'lora_' in name:
            group = 'lora'
        elif '.visual.merger.' in name:
            group = 'merger'
        elif '.lm_head.' in name:
            group = 'lm_head'
        elif any(f'.visual.blocks.{index}.' in name for index in vision_indices):
            group = 'vision'
        else:
            unmatched.append(name)
            continue
        groups[group].append(parameter)
    if unmatched:
        raise RuntimeError(f'Unassigned trainable parameters: {unmatched[:20]}')
    required = {'lora', 'merger', 'lm_head'}
    if vision_indices:
        required.add('vision')
    missing = sorted(required - groups.keys())
    if missing:
        raise RuntimeError(f'Empty optimizer groups: {missing}')
    return dict(groups)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    trainable = {name: parameter for name, parameter in unwrap(model).named_parameters() if parameter.requires_grad}
    grouped = parameter_groups(trainable, config)
    optimization = config['optimization']
    learning_rates = {
        'lora': optimization['lora_learning_rate'],
        'merger': optimization['merger_and_head_learning_rate'],
        'lm_head': optimization['merger_and_head_learning_rate'],
        'vision': optimization['vision_learning_rate'],
    }
    optimizer = torch.optim.AdamW(
        [{'params': parameters, 'lr': learning_rates[name], 'name': name} for name, parameters in grouped.items()],
        weight_decay=optimization['weight_decay'],
    )
    return optimizer, {name: sum(parameter.numel() for parameter in parameters) for name, parameters in grouped.items()}


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != 'metadata'}


def gather_rows(local_rows: list[dict[str, Any]], world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return sorted(local_rows, key=lambda row: row['order'])
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    rows = [row for rank_rows in gathered if rank_rows is not None for row in rank_rows]
    unique = {row['id']: row for row in rows}
    return sorted(unique.values(), key=lambda row: row['order'])


def support_maps(train_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for record in train_records:
        result[target(record)].append(record['image_path'])
    return dict(result)


def add_support_fields(rows: list[dict[str, Any]], train_targets: dict[str, list[str]]) -> None:
    for row in rows:
        row['train_images_reference'] = train_targets.get(row['reference'], [])
        row['train_images_predicted'] = train_targets.get(row['predicted'], [])
        row['complete_target_seen_in_train'] = bool(row['train_images_reference'])


def metric_bundle(rows: list[dict[str, Any]], config: dict[str, Any], *, bootstrap_samples: int) -> dict[str, Any]:
    edits = reference_tokens = exact = 0
    for row in rows:
        insertions, deletions, substitutions, _ = token_error_rate(row['reference'], row['predicted'])
        edits += insertions + deletions + substitutions
        reference_tokens += len(row['reference'].split())
        exact += int(''.join(row['reference'].split()) == ''.join(row['predicted'].split()))
    seed = config['runtime']['bootstrap_seed']
    headline = evaluate_predictions(rows, bootstrap_samples=bootstrap_samples, bootstrap_seed=seed)
    headline['metrics']['ter'] = headline['metrics']['cer']
    return {
        'selection': {
            'total_token_edits': edits,
            'reference_tokens': reference_tokens,
            'ter': edits / reference_tokens if reference_tokens else 0.0,
            'exact_correct': exact,
            'exact': exact / len(rows) if rows else 0.0,
            'rows': len(rows),
        },
        'released_predictions': headline,
        'released_factors': evaluate_program_factors(rows, bootstrap_samples=bootstrap_samples, bootstrap_seed=seed),
    }


def evaluate(
    model: torch.nn.Module,
    processor: Any,
    records: list[dict[str, Any]],
    train_targets: dict[str, list[str]],
    config: dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    *,
    split: str,
    optimizer_step: int,
    bootstrap_samples: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    dataset = KamonEvaluationDataset(records, Path(config['paths']['image_root']))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=config['distributed']['evaluation_per_device_batch_size'],
        sampler=sampler,
        num_workers=config['runtime']['num_workers'],
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=worker_seed,
        collate_fn=EvaluationCollator(processor, config['prompt']),
    )
    raw_model = unwrap(model)
    raw_model.eval()
    raw_model.config.use_cache = True
    raw_model.generation_config.temperature = None
    raw_model.generation_config.top_p = None
    raw_model.generation_config.top_k = None
    rows = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch_index, batch in enumerate(loader):
            metadata = batch.pop('metadata')
            inputs = to_device(batch, device)
            generated = raw_model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=config['generation']['max_new_tokens']
            )
            suffixes = generated[:, inputs['input_ids'].shape[1] :]
            raw_texts = processor.batch_decode(suffixes, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            clean_texts = processor.batch_decode(suffixes, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for item, token_ids, raw_text, clean_text in zip(
                metadata, suffixes.cpu().tolist(), raw_texts, clean_texts, strict=True
            ):
                rows.append(
                    {
                        'id': item['id'],
                        'order': item['order'],
                        'image': item['image_path'],
                        'reference': item['target'],
                        'predicted': clean_text.strip(),
                        'raw_decoded': raw_text,
                        'generated_token_ids': token_ids,
                        'split': split,
                        'optimizer_step': optimizer_step,
                        'rank': rank,
                        'batch_idx': batch_index,
                    }
                )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    rows = gather_rows(rows, world_size)
    metrics = None
    if rank == 0:
        add_support_fields(rows, train_targets)
        if len(rows) != len(records):
            raise RuntimeError(f'Evaluation returned {len(rows)} unique rows for {len(records)} records.')
        metrics = metric_bundle(rows, config, bootstrap_samples=bootstrap_samples)
    raw_model.config.use_cache = False
    raw_model.train()
    return rows, metrics, seconds


def selection_key(metrics: dict[str, Any], optimizer_step: int) -> tuple[int, int, int]:
    selected = metrics['selection']
    return (int(selected['total_token_edits']), -int(selected['exact_correct']), optimizer_step)


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')


def save_trainable_checkpoint(model: torch.nn.Module, path: Path) -> str:
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in unwrap(model).named_parameters()
        if parameter.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.temporary.safetensors')
    save_file(state, temporary)
    temporary.replace(path)
    return sha256(path)


def restore_trainable_checkpoint(model: torch.nn.Module, path: Path, rank: int) -> None:
    trainable = {name: parameter for name, parameter in unwrap(model).named_parameters() if parameter.requires_grad}
    if rank == 0:
        state = load_file(path)
        if set(state) != set(trainable):
            missing = sorted(set(trainable) - set(state))
            unexpected = sorted(set(state) - set(trainable))
            raise RuntimeError(f'Checkpoint parameter mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}')
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(state[name].to(parameter.device))
    if dist.is_initialized():
        for parameter in trainable.values():
            dist.broadcast(parameter.data, src=0)
    barrier()


def append_trace(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def distributed_parameter_probe(model: torch.nn.Module, rank: int, world_size: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in unwrap(model).named_parameters():
            if not parameter.requires_grad:
                continue
            flattened = parameter.detach().flatten()
            indices = sorted({0, flattened.numel() // 2, flattened.numel() - 1})
            digest.update(name.encode())
            for index in indices:
                digest.update(struct.pack('!f', float(flattened[index].float().cpu())))
    local_hash = digest.hexdigest()
    if world_size == 1:
        hashes = [local_hash]
    else:
        gathered: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered, local_hash)
        hashes = [value for value in gathered if value is not None]
    result = {
        'probe_definition': 'SHA-256 over three FP32-cast scalar probes from every trainable tensor',
        'rank_hashes': hashes,
        'all_ranks_equal': len(set(hashes)) == 1,
    }
    if rank == 0 and not result['all_ranks_equal']:
        raise RuntimeError(f'DDP trainable parameter probes differ across ranks: {hashes}')
    return result


def validate_config(config: dict[str, Any], config_path: Path, world_size: int) -> dict[str, Any]:
    if config['schema_version'] != 1:
        raise ValueError(f'Unsupported schema: {config["schema_version"]}')
    if not Path(config['paths']['metadata']).is_file():
        raise FileNotFoundError(config['paths']['metadata'])
    if config['processor']['min_pixels'] != 224 * 224:
        raise ValueError('min_pixels must be 224².')
    if config['processor']['max_pixels'] != 224 * 224:
        raise ValueError('max_pixels must be 224².')
    if config['data']['augmentations_per_record'] != 3:
        raise ValueError('Full protocol requires exactly three augmented copies.')
    distributed = config['distributed']
    denominator = world_size * distributed['per_device_batch_size']
    target_batch = distributed['target_global_batch_size']
    if target_batch % denominator:
        raise ValueError(
            f'Global batch {target_batch} is not divisible by world size '
            f'{world_size} × per-device batch {distributed["per_device_batch_size"]}.'
        )
    accumulation = target_batch // denominator
    expected = config['run'].get('expected_gradient_accumulation_steps')
    if expected is not None and accumulation != expected:
        raise ValueError(f'Derived accumulation {accumulation} does not match expected {expected}.')
    return {
        'config_path': str(config_path.resolve()),
        'config_sha256': sha256(config_path),
        'world_size': world_size,
        'gradient_accumulation_steps': accumulation,
        'effective_global_batch_size': denominator * accumulation,
    }


def prepare_records(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    records = load_records(config['paths']['metadata'])
    train = [record for record in records if record['split'] == 'train']
    dev = [record for record in records if record['split'] == 'dev' and record['is_composite']]
    test = [record for record in records if record['split'] == 'test' and record['is_composite']]
    expected = config['data']
    if len(train) != expected['train_rows']:
        raise ValueError(f'Expected {expected["train_rows"]} train rows, got {len(train)}.')
    if len(dev) != expected['dev_composites']:
        raise ValueError(f'Expected {expected["dev_composites"]} dev composites, got {len(dev)}.')
    if len(test) != expected['test_composites']:
        raise ValueError(f'Expected {expected["test_composites"]} test composites, got {len(test)}.')
    return train, dev, test, support_maps(train)


def processor_audit(processor: Any, record: dict[str, Any], image_root: Path, prompt: str) -> dict[str, Any]:
    with Image.open(image_root / record['image_path']) as source:
        image = source.convert('RGB')
    inputs = processor.apply_chat_template(
        messages(image, prompt), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt'
    )
    return {
        'source_size': list(image.size),
        'prompt_tokens': int(inputs['input_ids'].shape[1]),
        'image_grid_thw': inputs['image_grid_thw'].tolist(),
        'pixel_values_shape': list(inputs['pixel_values'].shape),
    }


def metadata(
    config,
    config_validation,
    model_details,
    optimizer_groups,
    processor_details,
    records,
    *,
    updates_per_epoch,
    evaluation_steps,
):
    return {
        'status': 'running',
        'run': config['run'],
        'torch': {
            'version': torch.__version__,
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        },
        'hardware': {'gpu_name': torch.cuda.get_device_name(), 'gpu_count': config_validation['world_size']},
        'inputs': {
            'metadata_sha256': sha256(Path(config['paths']['metadata'])),
            'runner_sha256': sha256(Path(__file__)),
            **config_validation,
        },
        'model': {'id': config['model']['id'], 'revision': config['model']['revision'], **model_details},
        'processor': processor_details,
        'optimizer_group_parameters': optimizer_groups,
        'data': {
            'train_rows': records[0],
            'dev_composites': records[1],
            'test_composites': records[2],
            'exposures_per_epoch': records[0] * (config['data']['augmentations_per_record'] + 1),
            'updates_per_epoch': updates_per_epoch,
            'evaluation_steps': evaluation_steps,
        },
        'config': config,
    }


def train(config_path: Path, *, overrides: argparse.Namespace | None = None) -> int:
    process_started = time.perf_counter()
    config = read_json(config_path)
    if overrides is not None:
        if overrides.metadata is not None:
            config['paths']['metadata'] = str(overrides.metadata)
        if overrides.image_root is not None:
            config['paths']['image_root'] = str(overrides.image_root)
        if overrides.output_dir is not None:
            config['run']['output_dir'] = str(overrides.output_dir)
    rank, local_rank, world_size, device = init_distributed()
    configure_runtime(config['runtime']['seed'], rank)
    validation = validate_config(config, config_path, world_size)
    output_dir = Path(config['run']['output_dir'])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / 'resolved_config.json', config)
    barrier()

    from kamonbench.data import augment

    train_records, dev_records, test_records, train_targets = prepare_records(config)
    processor = AutoProcessor.from_pretrained(
        config['model']['id'],
        revision=config['model']['revision'],
        min_pixels=config['processor']['min_pixels'],
        max_pixels=config['processor']['max_pixels'],
    )
    processor.tokenizer.padding_side = 'right'
    atomic_manifest = None
    if config.get('output_tokens') == 'factor':
        atomic_manifest, atomic_details = add_atomic_tokens(processor, load_records(config['paths']['metadata']))
        if rank == 0:
            write_json(output_dir / 'atomic_tokens.json', atomic_manifest)
            processor.save_pretrained(output_dir / 'processor')
        barrier()
    processor_details = processor_audit(
        processor, train_records[0], Path(config['paths']['image_root']), config['prompt']
    )
    if atomic_manifest is not None:
        processor_details['atomic_tokens'] = atomic_details
    if processor_details['source_size'] != [224, 224]:
        raise ValueError(f'Unexpected source image size: {processor_details["source_size"]}')
    if processor_details['image_grid_thw'] != [[1, 14, 14]]:
        raise ValueError(f'Processor did not realize 224² grid: {processor_details["image_grid_thw"]}')

    model, model_details = load_model(config, device, processor, atomic_manifest)
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    optimizer, optimizer_groups = build_optimizer(model, config)

    training_dataset = KamonTrainingDataset(
        train_records,
        Path(config['paths']['image_root']),
        augmentations=config['data']['augmentations_per_record'],
        seed=config['runtime']['seed'],
        augment_module=augment,
    )
    distributed = config['distributed']
    samples_per_rank = math.ceil(len(training_dataset) / world_size)
    microbatches_per_epoch = samples_per_rank // distributed['per_device_batch_size']
    accumulation = validation['gradient_accumulation_steps']
    if microbatches_per_epoch % accumulation:
        raise ValueError(
            f'{microbatches_per_epoch} microbatches per epoch is not divisible by accumulation {accumulation}.'
        )
    updates_per_epoch = microbatches_per_epoch // accumulation
    max_epochs = config['run']['max_epochs']
    max_updates = updates_per_epoch * max_epochs
    schedule_updates = updates_per_epoch * config['optimization']['schedule_reference_epochs']
    warmup_steps = math.ceil(schedule_updates * config['optimization']['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=schedule_updates
    )
    evaluation_steps = sorted(
        {math.ceil(updates_per_epoch * fraction) for fraction in config['optimization']['evaluate_at_epoch_fractions']}
    )

    run_metadata = metadata(
        config,
        validation,
        model_details,
        optimizer_groups,
        processor_details,
        (len(train_records), len(dev_records), len(test_records)),
        updates_per_epoch=updates_per_epoch,
        evaluation_steps=evaluation_steps,
    )
    if rank == 0:
        write_json(output_dir / 'metadata.json', run_metadata)

    best: dict[str, Any] | None = None
    best_path = output_dir / 'selected' / 'trainable.safetensors'
    global_update = 0
    training_seconds = 0.0
    evaluation_seconds = 0.0
    loss_sum = 0.0
    loss_microbatches = 0
    last_gradient_norm = None
    stop = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(max_epochs):
        if stop:
            break
        training_dataset.set_epoch(epoch)
        sampler = DistributedSampler(
            training_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config['runtime']['seed'],
            drop_last=True,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            training_dataset,
            batch_size=distributed['per_device_batch_size'],
            sampler=sampler,
            num_workers=config['runtime']['num_workers'],
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=worker_seed,
            collate_fn=TrainingCollator(processor, config['prompt']),
            drop_last=True,
        )
        model.train()
        unwrap(model).config.use_cache = False
        epoch_started = time.perf_counter()
        for microbatch_index, batch in enumerate(loader):
            batch.pop('metadata')
            inputs = to_device(batch, device)
            accumulation_index = microbatch_index % accumulation
            synchronize = accumulation_index == accumulation - 1
            context = (
                contextlib.nullcontext()
                if synchronize or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with context:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = model(**inputs).loss / accumulation
                loss.backward()
            loss_sum += float(loss.detach()) * accumulation
            loss_microbatches += 1
            if not synchronize:
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config['optimization']['max_grad_norm'],
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f'Non-finite gradient norm at optimizer step {global_update + 1}.')
            last_gradient_norm = float(gradient_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            within_epoch = global_update - epoch * updates_per_epoch
            should_evaluate = within_epoch in evaluation_steps

            if should_evaluate:
                torch.cuda.synchronize(device)
                training_seconds += time.perf_counter() - epoch_started
                rows, metrics, seconds = evaluate(
                    model,
                    processor,
                    dev_records,
                    train_targets,
                    config,
                    device,
                    rank,
                    world_size,
                    split='dev',
                    optimizer_step=global_update,
                )
                evaluation_seconds += seconds
                if rank == 0 and metrics is not None:
                    checkpoint_dir = output_dir / 'dev' / f'step-{global_update:06d}'
                    save_predictions(checkpoint_dir / 'predictions.jsonl', rows)
                    write_json(checkpoint_dir / 'metrics.json', metrics)
                    trace = {
                        'optimizer_step': global_update,
                        'epoch': global_update / updates_per_epoch,
                        'total_token_edits': metrics['selection']['total_token_edits'],
                        'reference_tokens': metrics['selection']['reference_tokens'],
                        'ter': metrics['selection']['ter'],
                        'exact_correct': metrics['selection']['exact_correct'],
                        'exact': metrics['selection']['exact'],
                        'parseability': metrics['released_factors']['metrics']['valid_prediction']['value'],
                        'mean_training_loss': loss_sum / loss_microbatches,
                        'learning_rates': json.dumps(
                            {group['name']: group['lr'] for group in optimizer.param_groups}, sort_keys=True
                        ),
                    }
                    append_trace(output_dir / 'dev_trace.csv', trace)
                    candidate = {
                        **trace,
                        'metrics_path': str((checkpoint_dir / 'metrics.json').resolve()),
                        'predictions_path': str((checkpoint_dir / 'predictions.jsonl').resolve()),
                    }
                    if best is None or selection_key(metrics, global_update) < selection_key(
                        {
                            'selection': {
                                'total_token_edits': best['total_token_edits'],
                                'exact_correct': best['exact_correct'],
                            }
                        },
                        best['optimizer_step'],
                    ):
                        best = candidate
                        best['checkpoint_sha256'] = save_trainable_checkpoint(model, best_path)
                        best['checkpoint_path'] = str(best_path.resolve())
                        write_json(output_dir / 'selected' / 'selection.json', best)
                best = broadcast_object(best, rank)
                epoch_started = time.perf_counter()

            if global_update >= max_updates:
                stop = True
                break
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - epoch_started

    if best is None:
        raise RuntimeError('Training ended without a development evaluation.')

    restore_trainable_checkpoint(model, best_path, rank)
    rows, metrics, seconds = evaluate(
        model,
        processor,
        test_records,
        train_targets,
        config,
        device,
        rank,
        world_size,
        split='test',
        optimizer_step=best['optimizer_step'],
        bootstrap_samples=config['runtime']['bootstrap_samples'],
    )
    evaluation_seconds += seconds
    test_result = None
    if rank == 0 and metrics is not None:
        save_predictions(output_dir / 'test' / 'predictions.jsonl', rows)
        write_json(output_dir / 'test' / 'metrics.json', metrics)
        test_result = {
            'predictions_path': str((output_dir / 'test' / 'predictions.jsonl').resolve()),
            'metrics_path': str((output_dir / 'test' / 'metrics.json').resolve()),
            'metrics': metrics['selection'],
        }

    barrier()
    wall_seconds = time.perf_counter() - process_started
    parameter_probe = distributed_parameter_probe(model, rank, world_size)
    summary = {
        'status': 'complete',
        'run_id': config['run']['id'],
        'purpose': config['run']['purpose'],
        'selection_split': 'development composites',
        'best': best,
        'test': test_result,
        'timing': {
            'wall_seconds': wall_seconds,
            'training_seconds_per_rank': training_seconds,
            'evaluation_seconds_per_rank': evaluation_seconds,
            'gpu_hours': wall_seconds * world_size / 3600,
        },
        'steps': {
            'optimizer_updates_completed': global_update,
            'microbatches_completed': loss_microbatches,
            'updates_per_augmented_epoch': updates_per_epoch,
            'gradient_accumulation_steps': accumulation,
            'last_preclip_gradient_norm': last_gradient_norm,
        },
        'ddp_parameter_probe': parameter_probe,
        'gpu_peak_allocated_bytes_per_rank': torch.cuda.max_memory_allocated(device),
    }
    if rank == 0:
        write_json(output_dir / 'run_summary.json', summary)
        run_metadata['status'] = 'complete'
        run_metadata['summary_path'] = str((output_dir / 'run_summary.json').resolve())
        write_json(output_dir / 'metadata.json', run_metadata)

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def main() -> int:
    args = parse_args()
    return train(args.config, overrides=args)


if __name__ == '__main__':
    raise SystemExit(main())
