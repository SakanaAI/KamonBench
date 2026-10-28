"""Selected-checkpoint inference for composite, component and real images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from kamonbench.data.metadata import load_records
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from experiments.qwen import train as train_qwen


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, source_root: Path) -> None:
        self.rows = rows
        self.source_root = source_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from kamonbench.data.images import retrieve_image

        row = self.rows[index]
        resolved = Path(row.get('resolved_path', self.source_root / row['image']))
        image = retrieve_image(str(resolved), size=224)
        return {'image': image, 'metadata': {key: value for key, value in row.items() if key != 'resolved_path'}}


class Collator:
    def __init__(self, processor: Any, prompt: str) -> None:
        self.processor = processor
        self.prompt = prompt

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self.processor.apply_chat_template(
            [train_qwen.messages(example['image'], self.prompt) for example in examples],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors='pt',
        )
        inputs['metadata'] = [example['metadata'] for example in examples]
        return inputs


def load_selected(config: dict[str, Any], checkpoint: Path, device_spec: str = 'cuda:0'):
    device = torch.device(device_spec)
    if device.type != 'cuda':
        raise ValueError('Qwen inference requires a CUDA device.')
    train_qwen.configure_runtime(config['runtime']['seed'], 0)
    processor = AutoProcessor.from_pretrained(
        config['model']['id'],
        revision=config['model']['revision'],
        min_pixels=config['processor']['min_pixels'],
        max_pixels=config['processor']['max_pixels'],
    )
    processor.tokenizer.padding_side = 'right'
    manifest = None
    if config.get('output_tokens') == 'factor':
        manifest, _ = train_qwen.add_atomic_tokens(processor, load_records(config['paths']['metadata']))
    model, details = train_qwen.load_model(config, device, processor, manifest)
    train_qwen.restore_trainable_checkpoint(model, checkpoint, 0)
    model.eval()
    model.config.use_cache = True
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, processor, device, details


def infer(
    model: torch.nn.Module,
    processor: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
    num_workers: int,
) -> tuple[list[dict[str, Any]], float]:
    dataset = ImageRows(rows, source_root=source_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=Collator(processor, prompt),
    )
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch_index, batch in enumerate(loader):
            metadata = batch.pop('metadata')
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            grids = inputs['image_grid_thw'].cpu().tolist()
            if any(grid != [1, 14, 14] for grid in grids):
                raise ValueError(f'Canonical 224² inputs did not produce a 14×14 grid: {grids[:4]}')
            generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
            suffixes = generated[:, inputs['input_ids'].shape[1] :]
            raw = processor.batch_decode(suffixes, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            clean = processor.batch_decode(suffixes, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for item, ids, raw_text, clean_text in zip(metadata, suffixes.cpu().tolist(), raw, clean, strict=True):
                predictions.append(
                    {
                        **item,
                        'predicted': clean_text.strip(),
                        'raw_decoded': raw_text,
                        'generated_token_ids': ids,
                        'batch_idx': batch_index,
                    }
                )
    torch.cuda.synchronize(device)
    return predictions, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--subset', choices=('all', 'composites', 'components'), default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = train_qwen.read_json(args.config)
    if args.metadata is not None:
        config['paths']['metadata'] = str(args.metadata)
    image_root = args.image_root or Path(config['paths']['image_root'])
    records = load_records(config['paths']['metadata'])
    train_targets = train_qwen.support_maps([r for r in records if r['split'] == 'train'])
    rows = [
        {'id': r['id'], 'image': r['image_path'], 'reference': train_qwen.target(r)}
        for r in records
        if r['split'] == 'test' and (args.subset == 'all' or r['is_composite'] == (args.subset == 'composites'))
    ]
    model, processor, device, _ = load_selected(config, args.checkpoint, args.device)
    predictions, _ = infer(
        model,
        processor,
        device,
        rows,
        source_root=image_root,
        prompt=config['prompt'],
        max_new_tokens=config['generation']['max_new_tokens'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_qwen.add_support_fields(predictions, train_targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train_qwen.save_predictions(args.output, predictions)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
