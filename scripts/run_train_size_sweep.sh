#!/bin/sh

script_dir=$(cd "$(dirname "$0")" && pwd)
code_root=$(cd "$script_dir/.." && pwd)
cd "$code_root" || exit 1
exec env PYTHONUNBUFFERED=1 uv run python "$script_dir/run_train_size_sweep.py" "$@"
