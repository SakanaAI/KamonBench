#!/bin/sh

# Reproduces the ViT decoder baseline reported in the paper appendix.
# Hyperparameters are passed explicitly to keep this pipeline a self-contained,
# default-independent reproducer.

script_dir=$(cd "$(dirname "$0")" && pwd)
pipeline_repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/pipeline_common.sh"

pipeline_setup
ensure_base_croissant

vit_train_flags="--model vit_decoder \
    --batch-size 64 \
    --num-epochs 50 \
    --checkpoint-steps 2000 \
    --early-stop-patience 8 \
    --num-train-augmentations 3 \
    --learning-rate 0.00015 \
    --vit-backbone-learning-rate 0.00001 \
    --weight-decay 0.0001 \
    --label-smoothing 0.02"

ensure_train train_vit_ja checkpoints_vit_decoder_ja outputs_vit_decoder_ja/step_final/predictions.jsonl uv run kamonbench train $vit_train_flags
ensure_file_step infer_vit_ja outputs_vit_decoder_ja/test_decode.jsonl uv run kamonbench infer --model vit_decoder
ensure_file_step report_vit_ja reports/vit_ja.html uv run kamonbench visualize --model vit_decoder -o reports/vit_ja.html

ensure_train train_vit_en checkpoints_vit_decoder_en outputs_vit_decoder_en/step_final/predictions.jsonl uv run kamonbench train $vit_train_flags --use-translation
ensure_file_step infer_vit_en outputs_vit_decoder_en/test_decode.jsonl uv run kamonbench infer --model vit_decoder --use-translation
ensure_file_step report_vit_en reports/vit_en.html uv run kamonbench visualize --model vit_decoder --use-translation -o reports/vit_en.html

ensure_program_croissant

program_vit_ckpt=checkpoints_program_vit_decoder
program_vit_out=outputs_program_vit_decoder

ensure_train train_program_vit "$program_vit_ckpt" "$program_vit_out/step_final/predictions.jsonl" uv run kamonbench train $vit_train_flags --croissant-path "$pipeline_program_croissant" --checkpoint-dir "$program_vit_ckpt" --output-dir "$program_vit_out"
ensure_file_step infer_program_vit "$program_vit_out/test_decode.jsonl" uv run kamonbench infer --model vit_decoder --checkpoint-path "$program_vit_ckpt/checkpoint_best_*.pt" --output-file "$program_vit_out/test_decode.jsonl" --croissant-path "$pipeline_program_croissant"
ensure_file_step report_program_vit reports/program_vit.html uv run kamonbench visualize "$program_vit_out/test_decode.jsonl" --model vit_decoder -o reports/program_vit.html
