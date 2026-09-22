#!/usr/bin/env bash
# initial 版六数据集基线：先跑 ds 答题模型，再跑 doubao；评审固定 JUDGE_*（默认 ds），
# embedding 固定 EMBEDDING_*（旧模型）。结果按模型分目录：
#   Output/<dataset>/deepread_global_initial_ds_NNNN/
#   Output/<dataset>/deepread_global_initial_doubao_NNNN/
#
# 前提：
#   1. 本仓库目录与 Data/、Output/ 同级（run.py 以父目录为 workspace 根）；
#   2. 各数据集 Data/<DS>/DeepRead/global/store_index 已存在（skip_ingestion 复用）；
#   3. ov_test/.env 已按 .env.initial_6ds.example 填好；
#   4. 依赖环境已就绪（uv sync 或服务器既有 venv，用 PYTHON_BIN 指定）。
#
# 用法：
#   ./run_initial_6ds.sh                    # ds 六数据集 → doubao 六数据集
#   ./run_initial_6ds.sh ds                 # 只跑 ds 阶段
#   ./run_initial_6ds.sh doubao qasper      # 只跑 doubao 阶段的 qasper
# 后台长跑：
#   nohup ./run_initial_6ds.sh > run_initial_6ds.out 2>&1 &
#   tail -f run_initial_6ds.out            # 总进度
#   tail -f ../Output/logs/initial_6ds_*.log  # 单数据集详细日志
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR/ov_test"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/ov_test/.env}"
DATASETS=(clapnq financebench hotpotqa locomo qasper syllabusqa)
LOG_DIR="$WORKSPACE_ROOT/Output/logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$ENV_FILE" ]; then
    echo "[FATAL] 缺少 $ENV_FILE（按 .env.initial_6ds.example 填写）" >&2
    exit 1
fi
set -a; source "$ENV_FILE"; set +a

SUMMARY=()

run_one() {
    local tag="$1" ds="$2"
    local cfg="config_initial_6ds/${ds}.yaml"
    local log="$LOG_DIR/initial_6ds_${tag}_${ds}.log"
    if [ ! -f "$cfg" ]; then
        echo "[SKIP] $cfg 不存在"
        SUMMARY+=("$tag/$ds: MISSING_CONFIG")
        return
    fi
    echo "===== [$tag] $ds 开始 $(date '+%F %T') ====="
    "$PYTHON_BIN" run.py --config "$cfg" --step all >>"$log" 2>&1
    local rc=$?
    echo "===== [$tag] $ds 结束 exit=$rc $(date '+%F %T') 日志: $log ====="
    SUMMARY+=("$tag/$ds: exit=$rc")
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
        *) echo "[FATAL] 未知阶段: $tag" >&2; exit 2;;
    esac
    export MODEL_TAG="$tag"
    if [ -z "${LLM_MODEL:-}" ] || [ -z "${LLM_BASE_URL:-}" ] || [ -z "${LLM_API_KEY:-}" ]; then
        echo "[FATAL] 阶段 $tag 的模型变量为空，请检查 .env" >&2
        exit 1
    fi
    echo "######## 阶段 $tag（LLM_MODEL=$LLM_MODEL）########"
    for ds in "${dss[@]}"; do
        run_one "$tag" "$ds"
    done
}

PHASES=()
if [ $# -ge 1 ]; then PHASES=("$1"); else PHASES=(ds doubao); fi
if [ $# -ge 2 ]; then shift; DATASETS=("$@"); fi

for phase in "${PHASES[@]}"; do
    run_phase "$phase" "${DATASETS[@]}"
done

echo
echo "================ 汇总 ================"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
