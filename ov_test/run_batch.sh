#!/bin/bash
# run_batch.sh - 批量运行 RAG benchmark 实验
# 用法: bash run_batch.sh
# 并行组内的实验同时启动，等待全部完成后再执行下一组。
# 注意：必须用 bash 执行，不要用 sh。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/batch_logs"
mkdir -p "$LOG_DIR"

run_experiment() {
    pipeline="$1"
    config="$2"
    step="$3"
    skip_ingest="$4"

    skip_flag=""
    if [ "$skip_ingest" = "true" ]; then
        skip_flag="--skip-ingest"
    fi

    runner="run.py"
    if [ "$pipeline" = "per_question" ]; then
        runner="run_per_question.py"
    fi

    echo "[Batch] Pipeline: $pipeline | Config: $config | Step: $step | Skip Ingest: $skip_ingest"
    uv run python "$runner" --config "$config" --step "$step" $skip_flag
}

# 启动单个后台实验，PID 追加到 PID_FILE
start_one() {
    pipeline="$1"
    config="$2"
    step="$3"
    skip_ingest="$4"

    skip_flag=""
    if [ "$skip_ingest" = "true" ]; then
        skip_flag="--skip-ingest"
    fi

    runner="run.py"
    if [ "$pipeline" = "per_question" ]; then
        runner="run_per_question.py"
    fi

    log_name=$(echo "$config" | sed 's|/|_|g; s|\.yaml||')
    log_file="$LOG_DIR/${log_name}.log"

    echo "[Parallel] Starting: $pipeline $config $step -> $log_file"
    nohup uv run python "$SCRIPT_DIR/$runner" --config "$config" --step "$step" $skip_flag \
        > "$log_file" 2>&1 &
    echo $! >> "$PID_FILE"
}

# 等待 PID_FILE 中所有进程完成
wait_all() {
    failed=0
    total=0
    while read -r pid; do
        total=$((total + 1))
        if ! wait "$pid" 2>/dev/null; then
            echo "[Parallel] PID $pid failed"
            failed=$((failed + 1))
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"

    if [ "$failed" -gt 0 ]; then
        echo "[Parallel] $failed / $total experiment(s) failed. Check logs in $LOG_DIR"
    else
        echo "[Parallel] All $total experiments completed successfully."
    fi
}

# ==========================================
# 实验组合（取消注释需要运行的行）
# ==========================================

# --- PageIndex Per-Question Pipeline (并行) ---
PID_FILE=$(mktemp)
start_one per_question config_per_question_pageindex/locomo_config.yaml all false
start_one per_question config_per_question_pageindex/hotpot_config.yaml all false
start_one per_question config_per_question_pageindex/qasper_config.yaml all false
start_one per_question config_per_question_pageindex/clapnq_config.yaml all false
start_one per_question config_per_question_pageindex/finance_config.yaml all false
start_one per_question config_per_question_pageindex/syllabusqa_config.yaml all false
wait_all

# --- PageIndex Global Pipeline (并行) ---
# PID_FILE=$(mktemp)
# start_one global config_pageindex/locomo_config.yaml gen false
# start_one global config_pageindex/hotpot_config.yaml gen false
# start_one global config_pageindex/qasper_config.yaml gen false
# start_one global config_pageindex/clapnq_config.yaml gen false
# start_one global config_pageindex/finance_config.yaml gen false
# start_one global config_pageindex/syllabusqa_config.yaml gen false
# wait_all

# --- Viking Per-Question Pipeline (并行) ---
# PID_FILE=$(mktemp)
# start_one per_question config_per_question/locomo_config.yaml all false
# start_one per_question config_per_question/hotpot.yaml all false
# start_one per_question config_per_question/qasper_config.yaml all false
# start_one per_question config_per_question/clapnq_config.yaml all false
# start_one per_question config_per_question/finance_config.yaml all false
# start_one per_question config_per_question/syllabusqa_config.yaml all false
# wait_all

# --- Viking Global Pipeline (并行) ---
# PID_FILE=$(mktemp)
# start_one global config/locomo_config.yaml gen false
# start_one global config/hotpot.yaml gen false
# start_one global config/qasper_config.yaml gen false
# start_one global config/config_clapnq.yaml gen false
# start_one global config/config_finance.yaml gen false
# wait_all

# --- 单独执行（同步）---
# run_experiment global config_pageindex/locomo_config.yaml eval true
# run_experiment global config_pageindex/locomo_config.yaml del false

echo "[Batch] All experiments completed."
