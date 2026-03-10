#!/bin/bash
# run_batch.sh - 批量运行 RAG benchmark 实验
# 用法: bash run_batch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_experiment() {
    local pipeline="$1"    # global / per_question
    local config="$2"      # config file path (relative to SCRIPT_DIR)
    local step="$3"        # all / gen / eval / del
    local skip_ingest="$4" # true / false

    local skip_flag=""
    if [ "$skip_ingest" = "true" ]; then
        skip_flag="--skip-ingest"
    fi

    local runner="run.py"
    if [ "$pipeline" = "per_question" ]; then
        runner="run_per_question.py"
    fi

    echo "=========================================="
    echo "[Batch] Pipeline: $pipeline | Config: $config | Step: $step | Skip Ingest: $skip_ingest"
    echo "=========================================="
    uv run python "$runner" --config "$config" --step "$step" $skip_flag
}

# ==========================================
# 实验组合（取消注释需要运行的行）
# ==========================================

# --- PageIndex Global Pipeline ---
# run_experiment global config_pageindex/locomo_config.yaml gen false
# run_experiment global config_pageindex/hotpot_config.yaml gen false
# run_experiment global config_pageindex/qasper_config.yaml gen false
# run_experiment global config_pageindex/clapnq_config.yaml gen false
# run_experiment global config_pageindex/finance_config.yaml gen false
# run_experiment global config_pageindex/syllabusqa_config.yaml gen false

# --- PageIndex Per-Question Pipeline ---
# run_experiment per_question config_per_question_pageindex/locomo_config.yaml gen false
# run_experiment per_question config_per_question_pageindex/hotpot_config.yaml gen false
# run_experiment per_question config_per_question_pageindex/qasper_config.yaml gen false
# run_experiment per_question config_per_question_pageindex/clapnq_config.yaml gen false
# run_experiment per_question config_per_question_pageindex/finance_config.yaml gen false
# run_experiment per_question config_per_question_pageindex/syllabusqa_config.yaml gen false

# --- Viking Global Pipeline ---
run_experiment global config/locomo_config.yaml del false
# run_experiment global config/hotpot.yaml gen false
# run_experiment global config/qasper_config.yaml gen false
# run_experiment global config/config_clapnq.yaml gen false
# run_experiment global config/config_finance.yaml gen false

# --- Viking Per-Question Pipeline ---
# run_experiment per_question config_per_question/locomo_config.yaml gen false
# run_experiment per_question config_per_question/hotpot.yaml gen false
# run_experiment per_question config_per_question/qasper_config.yaml gen false
# run_experiment per_question config_per_question/clapnq_config.yaml gen false
# run_experiment per_question config_per_question/finance_config.yaml gen false
# run_experiment per_question config_per_question/syllabusqa_config.yaml gen false

# --- Evaluation Only ---
# run_experiment global config_pageindex/locomo_config.yaml eval true
# run_experiment per_question config_per_question_pageindex/locomo_config.yaml eval true

# --- Deletion ---
# run_experiment global config_pageindex/locomo_config.yaml del false

echo "[Batch] All experiments completed."
