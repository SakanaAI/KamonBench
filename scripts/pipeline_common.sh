# Common pipeline helpers. Sourced by the per-baseline scripts.
# The caller is expected to set $pipeline_repo_root before sourcing this file.

pipeline_setup() {
    cd "$pipeline_repo_root" || return 1

    mkdir -p logs reports

    pipeline_base_metadata=${KAMON_METADATA_PATH:-metadata/records.jsonl}
    pipeline_program_metadata=${KAMON_PROGRAM_METADATA_PATH:-metadata/program.json}
}

has_best_checkpoint() {
    _dir=$1
    _match=$(find "$_dir" -maxdepth 1 -name 'checkpoint_best_*.pt' -print -quit 2>/dev/null)
    [ -n "$_match" ]
}

run_and_log() {
    _name=$1
    shift
    echo "==> $_name"
    _status_file=$(mktemp)
    {
        env PYTHONUNBUFFERED=1 "$@" 2>&1
        echo $? > "$_status_file"
    } | tee "logs/$_name.log"
    _cmd_status=$(cat "$_status_file")
    rm -f "$_status_file"
    if [ "$_cmd_status" -ne 0 ]; then
        echo "FAILED: $_name" >&2
        return "$_cmd_status"
    fi
}

ensure_base_metadata() {
    if [ -f "$pipeline_base_metadata" ]; then
        return 0
    fi
    echo "Missing metadata: $pipeline_base_metadata" >&2
    return 1
}

ensure_program_metadata() {
    if [ -f "$pipeline_program_metadata" ]; then
        return 0
    fi
    echo "Missing program-label metadata: $pipeline_program_metadata" >&2
    return 1
}

ensure_train() {
    _name=$1
    _checkpoint_dir=$2
    _final_predictions=$3
    shift 3
    if [ -f "$_final_predictions" ] && has_best_checkpoint "$_checkpoint_dir"; then
        echo "==> skip $_name"
        return 0
    fi
    run_and_log "$_name" "$@"
}

ensure_file_step() {
    _name=$1
    _output_path=$2
    shift 2
    if [ -f "$_output_path" ]; then
        echo "==> skip $_name"
        return 0
    fi
    run_and_log "$_name" "$@"
}
