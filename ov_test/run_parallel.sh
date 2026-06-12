#!/bin/bash
# run_parallel.sh - 进程级并行运行 DeepRead benchmark
#
# 用法:
#   bash run_parallel.sh <配置文件> <并行数> [step] [--resume]
#
# 示例:
#   bash run_parallel.sh config_deepread_global/qasper.yaml 7 geneval
#   bash run_parallel.sh config_deepread_global/locomo.yaml 1 geneval
#   bash run_parallel.sh config_deepread_global/qasper.yaml 7 gen --resume
#
# 流程:
#   1. 根据原始配置文件 + 并行数，自动生成 N 个 part 配置文件
#   2. 先单独跑一次 ingest（所有part共享store）
#   3. 并行启动 N 个 part 进程
#   4. 等待所有part完成后合并结果
#   5. 清理临时 part 配置文件

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PIDS=()

cleanup() {
    echo ""
    echo "[Interrupt] Ctrl+C detected, killing all child processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null
            echo "  Killed PID $pid"
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null
            echo "  Force killed PID $pid"
        fi
    done
    echo "[Interrupt] All child processes terminated."
    exit 130
}

trap cleanup SIGINT SIGTERM

CONFIG="$1"
NUM_PARTS="${2:-1}"
STEP="${3:-geneval}"
EXTRA_ARGS="${@:4}"

LOG_DIR="$SCRIPT_DIR/parallel_logs"
PART_CONFIG_DIR="$SCRIPT_DIR/.parallel_configs"
mkdir -p "$LOG_DIR"
mkdir -p "$PART_CONFIG_DIR"

if [ -z "$CONFIG" ]; then
    echo "Usage: bash run_parallel.sh <config> [num_parts] [step] [--resume]"
    echo "  config:    原始配置文件路径 (如 config_deepread_global/qasper.yaml)"
    echo "  num_parts: 并行进程数 (default: 1)"
    echo "  step:      执行阶段 (geneval, gen, eval, all) (default: geneval)"
    echo "  --resume:  断点恢复"
    echo ""
    echo "Examples:"
    echo "  bash run_parallel.sh config_deepread_global/qasper.yaml 7 geneval"
    echo "  bash run_parallel.sh config_deepread_global/locomo.yaml 1 geneval"
    echo "  bash run_parallel.sh config_deepread_global/qasper.yaml 7 gen --resume"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "[Error] Config not found: $CONFIG"
    exit 1
fi

CONFIG_BASENAME=$(basename "$CONFIG" .yaml)
CONFIG_DIR=$(dirname "$CONFIG")

echo "=========================================="
echo " Config:    $CONFIG"
echo " Num Parts: $NUM_PARTS"
echo " Step:      $STEP"
echo " Extra:     $EXTRA_ARGS"
echo "=========================================="

# --- 单进程模式：直接跑 ---
if [ "$NUM_PARTS" -le 1 ]; then
    echo "[Info] Single process mode..."
    uv run python run.py --config "$CONFIG" --step "$STEP" $EXTRA_ARGS
    echo "[Done] Completed."
    exit 0
fi

# --- 多进程模式 ---

# Step 1: 自动生成 part 配置文件
echo "[Step 1] Generating $NUM_PARTS part config files..."

# 用 Python 从原始 YAML 生成 part 配置，添加 worker_id/num_workers 和修改 output_dir
uv run python -c "
import yaml, sys, os

config_path = '$CONFIG'
num_parts = $NUM_PARTS
config_dir = '$PART_CONFIG_DIR'
config_basename = '$CONFIG_BASENAME'

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

for i in range(num_parts):
    part_config = yaml.safe_load(yaml.dump(config))

    # 添加/修改 worker_id 和 num_workers
    if 'execution' not in part_config:
        part_config['execution'] = {}
    part_config['execution']['worker_id'] = i
    part_config['execution']['num_workers'] = num_parts

    # 修改 output_dir，加上 _partN 后缀
    if 'paths' in part_config and 'output_dir' in part_config['paths']:
        original = part_config['paths']['output_dir']
        part_config['paths']['output_dir'] = original + '_part' + str(i)

    part_path = os.path.join(config_dir, f'{config_basename}_part{i}.yaml')
    with open(part_path, 'w', encoding='utf-8') as f:
        yaml.dump(part_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f'  Generated: {part_path}')
"

echo "[Step 1] Done."

# Step 2: Ingestion（仅 all/ingest 步骤时执行）
if [ "$STEP" = "all" ] || [ "$STEP" = "ingest" ]; then
    if echo "$EXTRA_ARGS" | grep -q "\-\-resume"; then
        echo "[Step 2] Resume mode, skipping ingestion."
    else
        echo "[Step 2] Running ingestion..."
        uv run python run.py --config "$CONFIG" --step ingest
        echo "[Step 2] Ingestion done."
    fi
else
    echo "[Step 2] Step=$STEP, skipping ingestion."
fi

# Step 3: 并行启动所有 part
echo "[Step 3] Launching $NUM_PARTS parts in parallel..."

# 解析 output_dir 用于判断 part 是否已完成
DEEPREAD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE=$(uv run python -c "
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    c = yaml.safe_load(f)
print(c.get('paths', {}).get('output_dir', ''))
")
DATASET_NAME=$(uv run python -c "
import yaml
with open('$CONFIG', 'r', encoding='utf-8') as f:
    c = yaml.safe_load(f)
print(c.get('dataset_name', 'Unknown'))
")
OUTPUT_BASE="${OUTPUT_BASE/\{dataset_name\}/$DATASET_NAME}"
if [[ "$OUTPUT_BASE" != /* ]]; then
    OUTPUT_BASE="$DEEPREAD_ROOT/$OUTPUT_BASE"
fi

SKIPPED=0
for i in $(seq 0 $((NUM_PARTS - 1))); do
    PART_CONFIG="$PART_CONFIG_DIR/${CONFIG_BASENAME}_part${i}.yaml"
    LOG_FILE="$LOG_DIR/${CONFIG_BASENAME}_part${i}.log"
    PART_DIR="${OUTPUT_BASE}_part${i}"

    # --resume 时跳过已完成的 part（无 checkpoint 且有结果文件）
    if echo "$EXTRA_ARGS" | grep -q "\-\-resume"; then
        if [ -d "$PART_DIR" ] && [ ! -f "$PART_DIR/benchmark_checkpoint.json" ]; then
            if [ -f "$PART_DIR/qa_eval_detailed_results.json" ] || [ -f "$PART_DIR/generated_answers.json" ]; then
                echo "[Part $i] Already completed, skipping. ($PART_DIR)"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi
    fi

    echo "[Part $i] Starting -> $LOG_FILE"
    nohup uv run python run.py --config "$PART_CONFIG" --step "$STEP" $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    PART_INDICES+=($i)
done

if [ "$SKIPPED" -gt 0 ]; then
    echo "[Step 3] Skipped $SKIPPED already completed part(s)."
fi

echo "[Step 3] All $NUM_PARTS parts launched. PIDs: ${PIDS[*]}"

# Step 4: 等待所有 part 完成
echo "[Step 4] Waiting for all parts to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait "$PID" 2>/dev/null; then
        echo "[Part $i] PID $PID completed successfully."
    else
        echo "[Part $i] PID $PID FAILED! Check $LOG_DIR/${CONFIG_BASENAME}_part${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[Error] $FAILED / $NUM_PARTS part(s) failed."
    exit 1
else
    echo "[Done] All $NUM_PARTS parts completed successfully."
fi

# Step 5: 合并结果
MERGE_SCRIPT="src/others/merge_muti_process_results.py"

if [ -f "$MERGE_SCRIPT" ]; then
    echo "[Step 5] Merging results..."

    PART_DIRS=""
    for i in $(seq 0 $((NUM_PARTS - 1))); do
        PART_DIR="${OUTPUT_BASE}_part${i}"
        if [ -d "$PART_DIR" ]; then
            PART_DIRS="$PART_DIRS $PART_DIR"
        fi
    done

    if [ -n "$PART_DIRS" ]; then
        uv run python "$MERGE_SCRIPT" --target-dirs $PART_DIRS --output-dir "$OUTPUT_BASE"
        echo "[Step 5] Results merged to $OUTPUT_BASE"
    else
        echo "[Step 5] No part output dirs found, skipping merge."
    fi
else
    echo "[Step 5] Merge script not found, skipping."
fi

# Step 6: 清理临时 part 配置文件
echo "[Step 6] Cleaning up temporary part configs..."
for i in $(seq 0 $((NUM_PARTS - 1))); do
    PART_CONFIG="$PART_CONFIG_DIR/${CONFIG_BASENAME}_part${i}.yaml"
    rm -f "$PART_CONFIG"
done
echo "[Step 6] Cleanup done."

echo "=========================================="
echo " All done!"
echo "=========================================="
