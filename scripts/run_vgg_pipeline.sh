#!/bin/sh

# Reproduces the masked VGG baseline reported in the paper appendix:
#   regularized recipe, n-gram length 1, learned position-dependent masks.
# All training hyperparameters are passed explicitly so the reproduction does
# not depend on CLI defaults.

script_dir=$(cd "$(dirname "$0")" && pwd)
pipeline_repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/pipeline_common.sh"

pipeline_setup
ensure_base_croissant

vgg_train_flags="--ngram-length 1 \
    --batch-size 64 \
    --num-epochs 200 \
    --checkpoint-steps 1250 \
    --early-stop-patience 10 \
    --num-train-augmentations 5 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --label-smoothing 0.02 \
    --eval-composites-only"

ensure_train train_vgg_ja checkpoints_vgg_ja outputs_vgg_ja/step_final/predictions.jsonl uv run kamonbench train $vgg_train_flags
ensure_file_step infer_vgg_ja outputs_vgg_ja/test_decode.jsonl uv run kamonbench infer
ensure_file_step report_vgg_ja reports/vgg_ja.html uv run kamonbench visualize --model vgg -o reports/vgg_ja.html

ensure_train train_vgg_en checkpoints_vgg_en outputs_vgg_en/step_final/predictions.jsonl uv run kamonbench train $vgg_train_flags --use-translation
ensure_file_step infer_vgg_en outputs_vgg_en/test_decode.jsonl uv run kamonbench infer --use-translation
ensure_file_step report_vgg_en reports/vgg_en.html uv run kamonbench visualize --model vgg --use-translation -o reports/vgg_en.html

ensure_program_croissant

program_vgg_ckpt=checkpoints_program_vgg
program_vgg_out=outputs_program_vgg

ensure_train train_program_vgg "$program_vgg_ckpt" "$program_vgg_out/step_final/predictions.jsonl" uv run kamonbench train $vgg_train_flags --croissant-path "$pipeline_program_croissant" --checkpoint-dir "$program_vgg_ckpt" --output-dir "$program_vgg_out"
ensure_file_step infer_program_vgg "$program_vgg_out/test_decode.jsonl" uv run kamonbench infer --checkpoint-path "$program_vgg_ckpt/checkpoint_best_*.pt" --output-file "$program_vgg_out/test_decode.jsonl" --croissant-path "$pipeline_program_croissant"
ensure_file_step report_program_vgg reports/program_vgg.html uv run kamonbench visualize "$program_vgg_out/test_decode.jsonl" -o reports/program_vgg.html
