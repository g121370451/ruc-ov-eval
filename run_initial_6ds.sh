#!/usr/bin/env bash
# initial 版六数据集基线：数据集间进程并行，模型（阶段）间串行接续。
# 结果按模型分目录：
#   Output/<dataset>/deepread_global_initial_ds_NNNN/
#   Output/<dataset>/deepread_global_initial_doubao_NNNN/
#
# 前提：
#   1. 本仓库目录与 Data/、Output/ 同级（run.py 以父目录为 workspace 根）；
#   2. 各数据集 Data/<DS>/DeepRead/global/store_index 已存在（skip_ingestion 复用）；
#   3. ov_test/.env 已按 .env.initial_6ds.example 填好；
#   4. 依赖环境已就绪（uv sync 或服务器既有 venv，用 PYTHON_BIN 指定）。
#
# 用法（参数里 ds/doubao 识别为阶段，其余识别为数据集，顺序随意）：
#   ./run_initial_6ds.sh                        # ds → doubao，六数据集各自并行
#   ./run_initial_6ds.sh ds doubao financebench # 两个模型接续，只跑 financebench
#   ./run_initial_6ds.sh doubao                 # 只跑 doubao 阶段，六数据集并行
#   ./run_initial_6ds.sh financebench qasper    # 两个模型接续，只跑这两个数据集
# 后台长跑：
#   nohup ./run_initial_6ds.sh > run_initial_6ds.out 2>&1 &
#   tail -f run_initial_6ds.out                   # 总进度
#   tail -f ../Output/logs/initial_6ds_*.log      # 单数据集详细日志
#
# 注意：六数据集并行时，对中转站的并发 ≈ 各配置 max_workers 之和（约 32 路），
# 如中转站 QPS 吃紧，请分批传数据集参数（例：先 clapnq financebench hotpotqa，
# 完成后再 locomo qasper syllabusqa）。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR/ov_test"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/ov_test/.env}"
ALL_DATASETS=(clapnq financebench hotpotqa locomo qasper syllabusqa)
LOG_DIR="$WORKSPACE_ROOT/Output/logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$ENV_FILE" ]; then
    echo "[FATAL] 缺少 $ENV_FILE（按 .env.initial_6ds.example 填写）" >&2
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

# ---- 参数解析：ds/doubao 为阶段，其余为数据集 ----
PHASES=()
DATASETS=()
is_dataset() {
    local x="$1" d
    for d in "${ALL_DATASETS[@]}"; do [ "$x" = "$d" ] && return 0; done
    return 1
}
for arg in "$@"; do
    case "$arg" in
        ds|doubao) PHASES+=("$arg");;
        *) if is_dataset "$arg"; then DATASETS+=("$arg"); else
               echo "[FATAL] 无法识别的参数: ${arg}（阶段只能是 ds/doubao，数据集只能是 ${ALL_DATASETS[*]}）" >&2
               exit 2
           fi;;
    esac
done
[ ${#PHASES[@]} -eq 0 ] && PHASES=(ds doubao)
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=("${ALL_DATASETS[@]}")
# 阶段去重并保持 ds 先、doubao 后
SORTED_PHASES=()
for p in ds doubao; do
    for q in "${PHASES[@]}"; do [ "$p" = "$q" ] && SORTED_PHASES+=("$p") && break; done
done

SUMMARY=()

run_one() {
    local tag="$1" ds="$2"
    local cfg="config_initial_6ds/${ds}.yaml"
    local log="$LOG_DIR/initial_6ds_${tag}_${ds}.log"
    if [ ! -f "$cfg" ]; then
        echo "[SKIP] $cfg 不存在"
        echo "$tag/$ds: MISSING_CONFIG" >"$log.rc"
        return
    fi
    echo "===== [$tag] $ds 开始 $(date '+%F %T') ====="
    "$PYTHON_BIN" run.py --config "$cfg" --step all >>"$log" 2>&1
    local rc=$?
    echo "===== [$tag] $ds 结束 exit=$rc $(date '+%F %T') ====="
    echo "$tag/$ds: exit=$rc ($log)" >"$log.rc"
}

run_phase() {
    local tag="$1"; shift
    local dss=("$@")
    case "$tag" in
        ds)
            export LLM_MODEL="$DS_MODEL" LLM_BASE_URL="$DS_BASE_URL" LLM_API_KEY="$DS_API_KEY"
            ;;
        doubao)
            export LLM_MODEL="$DOUBAO_MODEL" LLM_BASE_URL="$DOUBAO_BASE_URL" LLM_API_KEY="$DOUBAO_API_KEY"
            ;;
    esac
    export MODEL_TAG="$tag"
    if [ -z "${LLM_MODEL:-}" ] || [ -z "${LLM_BASE_URL:-}" ] || [ -z "${LLM_API_KEY:-}" ]; then
        echo "[FATAL] 阶段 ${tag} 的模型变量为空，请检查 .env" >&2
        exit 1
    fi
    echo "######## 阶段 ${tag}（LLM_MODEL=${LLM_MODEL}）：${#dss[@]} 个数据集并行 ########"

    local pids=() names=()
    for ds in "${dss[@]}"; do
        run_one "$tag" "$ds" &
        pids+=($!)
        names+=("$ds")
    done
    local i rc
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        rc=$?
        if [ -f "$LOG_DIR/initial_6ds_${tag}_${names[$i]}.log.rc" ]; then
            SUMMARY+=("$(cat "$LOG_DIR/initial_6ds_${tag}_${names[$i]}.log.rc")")
        else
            SUMMARY+=("$tag/${names[$i]}: runner_exit=$rc")
        fi
    done
    echo "######## 阶段 ${tag} 全部结束 $(date '+%F %T') ########"
}

for phase in "${SORTED_PHASES[@]}"; do
    run_phase "$phase" "${DATASETS[@]}"
done

echo
echo "================ 汇总 ================"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
