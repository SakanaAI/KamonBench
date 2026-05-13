#!/bin/sh

script_dir=$(cd "$(dirname "$0")" && pwd)
code_root=$(cd "$script_dir/.." && pwd)
python_exe="$code_root/.venv/bin/python"

if [ ! -x "$python_exe" ]; then
    echo "Missing Python environment: $python_exe" >&2
    exit 1
fi

exec env PYTHONUNBUFFERED=1 "$python_exe" "$script_dir/run_train_size_sweep.py" "$@"
