"""Run standard-split checkpoints on real images and matched synthetic controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def run_cnn(
    model_name: str, rows: list[dict[str, Any]], device_spec: str, batch_size: int, checkpoint_path: Path
) -> list[dict[str, Any]]:
    import torch
    from kamonbench.checkpoints import load_checkpoint
    from kamonbench.data.images import retrieve_image
    from kamonbench.text import strip_after_token, tokens_to_text
    from kamonbench.torch_utils import autocast_context, configure_torch_runtime, seed_runtime
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class Images(Dataset):
        def __init__(self) -> None:
            self.transform = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
            )

        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            image = retrieve_image(str(rows[index]['resolved_path']), size=224)
            return self.transform(image), index

    expected_model = 'vit_decoder' if model_name == 'vit' else 'vgg'
    device = torch.device(device_spec)
    seed_runtime(20260428)
    configure_torch_runtime(device, deterministic=True)
    model, checkpoint = load_checkpoint(str(checkpoint_path), device, model_override=expected_model)
    model.requires_grad_(False)
    model.eval()
    label_to_expr = {int(key): value for key, value in checkpoint['label_to_expr'].items()}
    loader = DataLoader(
        Images(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(20260428),
    )
    predictions = []
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device)
            with autocast_context(device):
                arguments = {'end_token': checkpoint['end_token'], 'max_length': checkpoint['max_seq_len']}
                if expected_model == 'vit_decoder':
                    arguments['start_token'] = checkpoint['start_token']
                generated = model.generate(images, **arguments)
                tokens = generated if expected_model == 'vit_decoder' else generated[0]
            for token_row, index in zip(tokens.cpu().tolist(), indices.tolist(), strict=True):
                ids = strip_after_token(token_row, checkpoint['end_token'])
                row = rows[index]
                predictions.append(
                    {
                        **{key: value for key, value in row.items() if key != 'resolved_path'},
                        'predicted': tokens_to_text(ids, label_to_expr),
                        'predicted_tokens': ids,
                    }
                )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('vit', 'vgg_masked_ngram1', 'vgg_no_mask_ngram4', 'qwen'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--qwen-config', type=Path)
    parser.add_argument('--rows', type=Path, required=True, help='Prepared condition JSONL.')
    parser.add_argument('--image-root', type=Path, required=True, help='Directory written by experiments.real.prepare.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    for row in rows:
        path = args.image_root / row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError(f'Prepared image changed: {row["row_id"]}')
        row['resolved_path'] = str(path)
    if args.model == 'qwen':
        if args.qwen_config is None:
            parser.error('--qwen-config is required for Qwen inference')
        from experiments.qwen.infer import infer, load_selected

        config = json.loads(args.qwen_config.read_text())
        model, processor, device, _ = load_selected(config, args.checkpoint, args.device)
        predictions, _ = infer(
            model,
            processor,
            device,
            rows,
            source_root=args.image_root,
            prompt=config['prompt'],
            max_new_tokens=config['generation']['max_new_tokens'],
            batch_size=args.batch_size,
            num_workers=0,
        )
    else:
        predictions = run_cnn(args.model, rows, args.device, args.batch_size, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
