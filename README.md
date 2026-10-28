# KamonBench

This is the repository for the paper **KamonBench: A Grammar-Based Dataset for
Evaluating Compositional Factor Recovery in Vision-Language Models**.

KamonBench evaluates image-to-structure prediction on Japanese kamon (family
crest) images. Each composite has a formal description (KDL, _kamon yōgo_
家紋用語), segmented Japanese analysis, English translation, and program code
for the container `C`, modifier `R`, and motif `M`.

This repository provides dataset reconstruction, PyTorch data loading, and
training and evaluation tools for the paper's experiments.

## 1. Setup

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/SakanaAI/KamonBench.git
cd KamonBench
uv sync
```

Run the commands below from the repository root. Qwen training requires up to eight CUDA GPUs.

## 2. Data preparation

Download the
[upstream image archive](https://raw.githubusercontent.com/Rebolforces/kamondataset/21df41681288e30c1add8d8f7bc478d34c7b85af/mon-white-224.tar.gz)
(SHA-256: `0491b97c22e0bf240d77acdda0a84b6cae378ff938546e933b1783b32abff199`),
then run:

```bash
uv run kamonbench reconstruct --source-archive /path/to/mon-white-224.tar.gz --workers 8
```

Reconstruction writes 54,116 images to `dataset01/` and verifies the source
archive and the SHA-256 of each generated image’s pixel data.

## 3. Reproducing the paper results

| Experiments                                 | Script                                                               |
| ------------------------------------------- | -------------------------------------------------------------------- |
| ViT and VGG baselines                       | `scripts/run_full_pipeline.sh`                                       |
| ViT and VGG on held-out factor combinations | `scripts/run_recombination.py`                                       |
| Qwen baselines and held-out combinations    | `scripts/run_qwen.py`                                                |
| Reduced training data                       | `scripts/run_train_size_sweep.sh`                                    |
| Error analysis, probes, and VGG ablations   | `scripts/run_analysis.py`                                            |
| Real-image transfer and preprocessing       | `scripts/run_real.py --source-archive /path/to/mon-white-224.tar.gz` |

Run the baseline, recombination, and Qwen scripts before the analysis and
real-image scripts, which use their checkpoints and predictions. Metrics and
confidence intervals are computed by the scripts.

Use `--help` on the Python scripts for options. Human-study materials and
few-shot LLM prompts are provided in the paper's supplementary material.

## 4. Extras / API

### PyTorch dataset

```python
from kamonbench.data.dataset import KamonDataset

train = KamonDataset("metadata/records.jsonl", division="train", num_augmentations=5)
image, labels = train[0]
```

The loader builds its vocabulary from all selected records and appends
`<EOS>`. Japanese analysis labels are normalized from katakana to hiragana;
English labels are split on whitespace.

## License

Code is released under the [MIT License](LICENSE). CC BY-NC 4.0 applies only
to the author-created annotations, split definitions, and record/provenance
metadata, as specified in [DATA_LICENSE.md](DATA_LICENSE.md). The included real
images retain their source licences.

The synthetic source files come from the Rebolforces kamondataset, which documents a
2021 collection from a 2014 archived capture of Kamon Market.
[metadata/sources.jsonl](metadata/sources.jsonl) records their paths and hashes.
