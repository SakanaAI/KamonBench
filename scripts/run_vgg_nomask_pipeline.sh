#!/bin/sh

# Reproduces the no-mask VGG baseline reported in the paper appendix:
#   regularized recipe, n-gram length 4, no learned position-dependent masks.
# Companion to run_vgg_pipeline.sh; writes to a distinct set of
# checkpoint/output/report directories so its artifacts do not collide with
# the masked variant.

script_dir=$(cd "$(dirname "$0")" && pwd)
pipeline_repo_root=$(cd "$script_dir/.." && pwd)
. "$script_dir/pipeline_common.sh"

pipeline_setup
ensure_base_croissant

vgg_nomask_train_flags="--no-masks \
    --ngram-length 4 \
    --batch-size 64 \
    --num-epochs 200 \
    --checkpoint-steps 1250 \
    --early-stop-patience 10 \
    --num-train-augmentations 5 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --label-smoothing 0.02 \
    --eval-composites-only"

ja_ckpt=checkpoints_vgg_nomask_ja
ja_out=outputs_vgg_nomask_ja
ensure_train train_vgg_nomask_ja "$ja_ckpt" "$ja_out/step_final/predictions.jsonl" uv run kamonbench train $vgg_nomask_train_flags --checkpoint-dir "$ja_ckpt" --output-dir "$ja_out"
ensure_file_step infer_vgg_nomask_ja "$ja_out/test_decode.jsonl" uv run kamonbench infer --checkpoint-path "$ja_ckpt/checkpoint_best_*.pt" --output-file "$ja_out/test_decode.jsonl"
ensure_file_step report_vgg_nomask_ja reports/vgg_nomask_ja.html uv run kamonbench visualize "$ja_out/test_decode.jsonl" -o reports/vgg_nomask_ja.html

en_ckpt=checkpoints_vgg_nomask_en
en_out=outputs_vgg_nomask_en
ensure_train train_vgg_nomask_en "$en_ckpt" "$en_out/step_final/predictions.jsonl" uv run kamonbench train $vgg_nomask_train_flags --use-translation --checkpoint-dir "$en_ckpt" --output-dir "$en_out"
ensure_file_step infer_vgg_nomask_en "$en_out/test_decode.jsonl" uv run kamonbench infer --use-translation --checkpoint-path "$en_ckpt/checkpoint_best_*.pt" --output-file "$en_out/test_decode.jsonl"
ensure_file_step report_vgg_nomask_en reports/vgg_nomask_en.html uv run kamonbench visualize "$en_out/test_decode.jsonl" -o reports/vgg_nomask_en.html

ensure_program_croissant

program_ckpt=checkpoints_program_vgg_nomask
program_out=outputs_program_vgg_nomask

ensure_train train_program_vgg_nomask "$program_ckpt" "$program_out/step_final/predictions.jsonl" uv run kamonbench train $vgg_nomask_train_flags --croissant-path "$pipeline_program_croissant" --checkpoint-dir "$program_ckpt" --output-dir "$program_out"
ensure_file_step infer_program_vgg_nomask "$program_out/test_decode.jsonl" uv run kamonbench infer --checkpoint-path "$program_ckpt/checkpoint_best_*.pt" --output-file "$program_out/test_decode.jsonl" --croissant-path "$pipeline_program_croissant"
ensure_file_step report_program_vgg_nomask reports/program_vgg_nomask.html uv run kamonbench visualize "$program_out/test_decode.jsonl" -o reports/program_vgg_nomask.html
