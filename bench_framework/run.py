"""
bench_framework 统一入口。

- 从 bench_framework 导入 Pipeline / LLMClient 等
- 根据 vector_store 配置自动选择 ov.conf 或 pageindex.conf
- vector_store / recall / adapter 均通过 config 动态加载
- bench_framework 完全自包含，不依赖外部模块
"""
import os
import sys
import yaml
import importlib
from argparse import ArgumentParser

# ==========================================
# 1. 环境初始化
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
PROJECT_ROOT = WORKSPACE_ROOT

# 将仓库根目录加入 sys.path，以便导入 bench_framework.*
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench_framework.pipeline import BenchmarkPipeline
from bench_framework.core.logger import setup_logging
from bench_framework.core.llm_client import LLMClientWrapper


# ==========================================
# 2. 辅助函数
# ==========================================

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str, base_path: str) -> str:
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_path, path_str))


def _auto_detect_conf(config: dict) -> str | None:
    """根据 vector_store 配置自动选择 ov.conf 或 pageindex.conf"""
    vs_class = config.get("vector_store", {}).get("class_name", "")
    if "Viking" in vs_class:
        candidate = os.path.join(SCRIPT_DIR, "ov.conf")
    else:
        candidate = os.path.join(SCRIPT_DIR, "pageindex.conf")
    return candidate if os.path.exists(candidate) else None


# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = ArgumentParser(description="bench_framework — Unified RAG Benchmark Runner")
    default_config = os.path.join(SCRIPT_DIR, "config/pageindex/locomo.yaml")

    parser.add_argument("--config", default=default_config,
                        help=f"Path to config YAML. Default: {default_config}")
    parser.add_argument("--step", choices=["all", "gen", "eval", "del"], default="all",
                        help="Execution step")
    args = parser.parse_args()

    # --- A. 加载 Config ---
    config_path = os.path.abspath(args.config)
    print(f"[Init] Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        return

    # --- B. 自动设置 conf 环境变量 ---
    conf_file = _auto_detect_conf(config)
    if conf_file:
        os.environ["OPENVIKING_CONFIG_FILE"] = conf_file
        print(f"[Init] Using conf: {conf_file}")

    # --- C. 路径修正 ---
    print(f"[Init] Resolving paths relative to: {PROJECT_ROOT}")
    dataset_name = config.get("dataset_name", "UnknownDataset")
    for key in ["raw_data", "output_dir", "vector_store", "log_file", "doc_output_dir"]:
        if key in config.get("paths", {}):
            rendered = config["paths"][key].format(dataset_name=dataset_name)
            config["paths"][key] = resolve_path(rendered, PROJECT_ROOT)

    # --- D. 初始化组件 ---
    try:
        logger = setup_logging(config["paths"]["log_file"])
        logger.info(">>> Benchmark Session Started")

        # 1. Adapter
        adapter_cfg = config.get("adapter", {})
        a_module = adapter_cfg.get("module", "bench_framework.adapters.locomo_adapter")
        a_class = adapter_cfg.get("class_name", "LocomoAdapter")
        logger.info(f"Loading Adapter: {a_class} from {a_module}")
        mod = importlib.import_module(a_module)
        adapter = getattr(mod, a_class)(raw_file_path=config["paths"]["raw_data"])

        # 2. Vector Store
        vs_cfg = config.get("vector_store", {})
        vs_module = vs_cfg.get("module", "bench_framework.stores.pageindex_store")
        vs_class = vs_cfg.get("class_name", "PageIndexStoreWrapper")
        logger.info(f"Loading VectorStore: {vs_class} from {vs_module}")
        vs_mod = importlib.import_module(vs_module)
        vector_store = getattr(vs_mod, vs_class)(store_path=config["paths"]["vector_store"],doc_output_dir = config["paths"]["doc_output_dir"])

        # 3. Recall Strategy (可选)
        recall_strategy = None
        rc_cfg = config.get("recall", {})
        if rc_cfg.get("module") and rc_cfg.get("class_name"):
            logger.info(f"Loading RecallStrategy: {rc_cfg['class_name']} from {rc_cfg['module']}")
            try:
                rc_mod = importlib.import_module(rc_cfg["module"])
                recall_strategy = getattr(rc_mod, rc_cfg["class_name"])()
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to load recall strategy: {e}. Using fallback.")

        # 4. LLM Client
        api_key = os.environ.get(
            config["llm"].get("api_key_env_var", ""),
            config["llm"].get("api_key"),
        )
        if not api_key:
            logger.warning("No API Key found!")
        llm_client = LLMClientWrapper(config=config["llm"], api_key=api_key)

        # 5. Pipeline
        pipeline = BenchmarkPipeline(
            config=config,
            adapter=adapter,
            vector_db=vector_store,
            llm=llm_client,
            recall_strategy=recall_strategy,
        )

        # --- E. 执行 ---
        if args.step in ["all", "gen"]:
            logger.info("Stage: Generation")
            pipeline.run_generation()
        if args.step in ["all", "eval"]:
            logger.info("Stage: Evaluation")
            pipeline.run_evaluation()
        if args.step in ["del"]:
            logger.info("Stage: Deletion")
            pipeline.run_deletion()

        logger.info("Benchmark finished successfully.")

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted by user.")
    except Exception as e:
        if "logger" in locals():
            logger.exception("Fatal error")
        print(f"\n[Fatal Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
