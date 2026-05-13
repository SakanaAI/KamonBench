# KamonBench

**KamonBench: A Grammar-Based Dataset for Evaluating Compositional Factor
Recovery in Vision-Language Models**

KamonBench is a benchmark for image-to-structure prediction on Japanese kamon
(family crest) images. Each composite crest is paired with a formal kamon
description (KDL, _kamon yōgo_ 家紋用語), a segmented Japanese analysis, an
English translation, and a non-linguistic program code over the generator
factors container `C`, modifier `R`, and motif `M`.

This repository contains the dataset utilities, PyTorch data loader, the three
reference baseline architectures, inference and evaluation tools, and the
pipeline scripts used to reproduce the paper's results.

## Quick start

KamonBench requires Python 3.11–3.13 and uses `uv` for project management.

```bash
uv sync
uv run kamonbench install-main-data
scripts/run_full_pipeline.sh
```

The pipeline trains all three baselines (ViT decoder, masked VGG n-gram-1,
no-mask VGG n-gram-4) on the three label targets (Japanese, English, program),
runs inference on the test split, and renders HTML reports.

## Dataset

The released dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/SakanaAI/KamonBench
```

Install it into the repository root with:

```bash
uv run kamonbench install-main-data
```

The installer downloads `kamon_bench.zip` and the main `kamon_croissant.json`,
then unpacks them into `dataset01/`:

```text
dataset01/
  kamon_croissant.json
  synth_*.png
  synth_*_base.png
  synth_*_containerNN.png
```

The release contains 54,116 images: 20,000 composite crests, 20,000
base-motif components, and 14,116 container components. Each composite is
paired with a Japanese KDL description, an English translation, a segmented
Japanese analysis, and a `(C, R, M)` program-code label.

The Hugging Face dataset also hosts three Croissant variants with held-out
factor combinations for the recombination splits described in the paper:

```text
kamon_croissant_program_cm_holdout.json
kamon_croissant_program_rm_holdout.json
kamon_croissant_program_crm_holdout.json
```

Useful installer options:

```bash
uv run kamonbench install-main-data --force
uv run kamonbench install-main-data --keep-zip
```

## Models

KamonBench includes three reference baseline families:

- **ViT decoder**: a timm Vision Transformer encoder paired with a 4-layer
  Transformer decoder.
- **VGG with learned position masks**: a VGG16 image encoder with
  position-dependent learned masks and an n-gram-1 classifier head.
- **VGG without learned masks**: the same VGG16 backbone with no positional
  masking and an n-gram-4 autoregressive classifier.

The default label source is `analysis[*].expr` (Japanese analysis tokens).
Passing `--use-translation` switches the target sequence to whitespace tokens
from the Croissant `translation` field. Passing `--croissant-path` to the
program croissant trains on non-linguistic program-code targets.

## Reproducing the paper baselines

The three baseline pipelines are in `scripts/`:

```bash
scripts/run_full_pipeline.sh        # all three baselines, all three labels
scripts/run_vit_pipeline.sh         # ViT decoder only
scripts/run_vgg_pipeline.sh         # VGG with masks (n-gram 1)
scripts/run_vgg_nomask_pipeline.sh  # VGG without masks (n-gram 4)
```

Each pipeline passes the full hyperparameter set explicitly so the
reproduction is independent of CLI defaults: batch size 64, label smoothing
0.02, weight decay 1e-4, composite-only checkpoint selection, and the
appropriate `--ngram-length` and `--no-masks` switches. See the appendix of
the paper for the full architecture and optimization tables.

The reduced-data sweep reported in the appendix is reproduced with:

```bash
scripts/run_train_size_sweep.sh
```

## Training

Train a single baseline directly:

```bash
uv run kamonbench train --model vit_decoder
uv run kamonbench train --use-translation
uv run kamonbench train --croissant-path dataset01/kamon_croissant_program.json \
  --eval-composites-only
```

Default output directories are selected from the model and label source:

```text
vgg         + analysis     -> checkpoints_vgg_ja/          outputs_vgg_ja/
vgg         + translation  -> checkpoints_vgg_en/          outputs_vgg_en/
vit_decoder + analysis     -> checkpoints_vit_decoder_ja/  outputs_vit_decoder_ja/
vit_decoder + translation  -> checkpoints_vit_decoder_en/  outputs_vit_decoder_en/
```

The CLI defaults are the legacy unregularized configuration; the regularized
recipe used in the paper is encoded in the pipeline scripts above.

## Inference

Run inference on the test split:

```bash
uv run kamonbench infer
uv run kamonbench infer --model vit_decoder --use-translation
```

Inference loads `checkpoint_best_*.pt` from the matching default directory.
Checkpoints record their architecture, so inference does not need to be told
about `--ngram-length`, `--no-masks`, or similar switches; pass only
`--use-translation` to match the training-side label source.

## Evaluation

Generate the HTML report and aggregate metrics:

```bash
uv run kamonbench visualize
uv run kamonbench visualize outputs_vgg_ja/test_decode.jsonl -o outputs_vgg_ja/test_decode.html
```

The report shows the input image, reference text, prediction, CER, Acc, and
Acc_NIT; `_base` and `_container` component images are filtered out of the
visualized subset.

Program-label runs expose deterministic `(C, R, M)` factors and support
additional diagnostics:

```bash
uv run kamonbench evaluate-program-factors outputs_program/test_decode.jsonl
uv run kamonbench evaluate-program-support  outputs_program/test_decode.jsonl \
  --croissant-path dataset01/kamon_croissant.json
uv run kamonbench evaluate-program-modifiers   outputs_program/test_decode.jsonl
uv run kamonbench evaluate-program-containers  outputs_program/test_decode.jsonl
uv run kamonbench summarize-program-confusions outputs_program/test_decode.jsonl
```

Recombination splits over existing Croissant metadata:

```bash
uv run kamonbench create-recombination-split \
  --croissant-path dataset01/kamon_croissant.json \
  --output dataset01/kamon_croissant_program_cm_holdout.json \
  --summary-output dataset01/kamon_croissant_program_cm_holdout.summary.json \
  --combo C,M --label-mode program --seed 20260428
```

Frozen-representation linear probes from saved program checkpoints:

```bash
uv run kamonbench extract-representation-features --checkpoint-path checkpoint.pt \
  --model-name run_name --croissant-path dataset01/kamon_croissant.json \
  --image-root .
uv run kamonbench train-representation-probes --feature-dir frozen_features \
  --model-name run_name --output-dir frozen_probe_metrics
uv run kamonbench compare-representation-probes --feature-dir frozen_features \
  --probe-dir frozen_probe_metrics --model-name run_name \
  --prediction-jsonl outputs_program/test_decode.jsonl
uv run kamonbench compare-representation-probe-support --feature-dir frozen_features \
  --probe-dir frozen_probe_metrics --model-name run_name
uv run kamonbench summarize-representation-probe-seeds \
  frozen_probe_metrics/*_probe_output_comparison.json
```

## PyTorch dataset

```python
from kamonbench.data.dataset import KamonDataset

train = KamonDataset(
    croissant_path="dataset01/kamon_croissant.json",
    division="train",
    num_augmentations=5,
)

image, labels = train[0]
```

The loader builds vocabularies from all records in the Croissant file and
appends an `<EOS>` token. Japanese analysis labels are normalized from
katakana to hiragana; translation labels are split on whitespace.

## Synthetic generation

The repository can also generate new synthetic kamon. Generation requires the
upstream motif assets:

```bash
uv run kamonbench install-mon-data
uv run kamonbench generate --num 10 --dataset-dir dataset01 --save-components
```

Convert the generated images to a Croissant file with `--label-mode program`,
which derives non-linguistic composition tokens directly from the saved
generator components, e.g. `C:00100 X:3 M:00200`:

```bash
uv run kamonbench convert-to-croissant --dataset-dir dataset01 --label-mode program
```

## Command reference

```bash
uv run kamonbench --help
uv run kamonbench train --help
uv run kamonbench infer --help
uv run kamonbench visualize --help
uv run kamonbench convert-to-croissant --help
```

## License

Code is released under the MIT License. The dataset is released under
CC BY-NC 4.0. See `LICENSE` for the code license, and the Hugging Face
dataset card and Croissant metadata for the dataset license.

The component images bundled with KamonBench (one isolated motif per
composite and one container per contained composite) are repackaged in PNG
form from the _Rebolforces kamondataset_, a publicly available collection of
Japanese kamon motifs originally scraped from a catalogue website that is no
longer accessible online (preserved via the Internet Archive); upstream
provenance cannot be tracked further. We make no copyright claim over those
source images and release KamonBench solely for non-commercial research use.
