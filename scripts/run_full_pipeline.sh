#!/bin/sh

script_dir=$(cd "$(dirname "$0")" && pwd)

sh "$script_dir/run_vgg_pipeline.sh" || exit $?

sh "$script_dir/run_vgg_nomask_pipeline.sh" || exit $?

sh "$script_dir/run_vit_pipeline.sh" || exit $?

cd "$script_dir/.." || exit $?

for model in vgg vgg_nomask vit_decoder; do
    for labels in ja en program; do
        if [ "$labels" = program ]; then
            predictions="outputs_program_${model}/test_decode.jsonl"
            metadata=${KAMON_PROGRAM_METADATA_PATH:-metadata/program.json}
        else
            predictions="outputs_${model}_${labels}/test_decode.jsonl"
            metadata=${KAMON_METADATA_PATH:-metadata/records.jsonl}
        fi
        uv run kamonbench summarize-predictions "$predictions" \
            --metadata-path "$metadata" --split test \
            --bootstrap-samples 10000 --bootstrap-seed 20260428 \
            --output "reports/${model}-${labels}-metrics.json" || exit $?
    done
done
