#!/bin/sh

script_dir=$(cd "$(dirname "$0")" && pwd)

sh "$script_dir/run_vgg_pipeline.sh" || exit $?

sh "$script_dir/run_vgg_nomask_pipeline.sh" || exit $?

sh "$script_dir/run_vit_pipeline.sh"
