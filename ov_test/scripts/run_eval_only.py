#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已有 generated_answers.json 执行 eval 评测（不初始化向量库）。

用法：
    cd ov_test/
    uv run python scripts/run_eval_only.py --config config_kimi/financebench_sample20_eval.yaml

说明：
    原项目 run.py 在 --step eval 时仍会初始化向量库并自动递增 output_dir 编号，
    与 kimi_prompt_rag.py 生成的固定输出目录不兼容。此脚本直接复用 BenchmarkPipeline.run_evaluation()
    的逻辑，但跳过入库/检索/向量库初始化。
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # ov_test/
WORKSPACE_ROOT = REPO_ROOT.parent.parent  # 仓库父目录（Data/、Output/ 所在目录）
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import BenchmarkPipeline
from src.core.llm_client import LLMClientWrapper
from src.core.logger import setup_logging
import importlib


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_env_vars(obj):
    """递归替换配置中的 ${VAR} 引用为环境变量值"""
    if isinstance(obj, str):
        def _replace(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                raise ValueError(f"环境变量 {var_name} 未设置，请检查 .env 文件")
            return value
        return re.sub(r"\$\{(\w+)\}", _replace, obj)
    elif isinstance(obj, dict):
        return {k: resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_env_vars(item) for item in obj]
    return obj


def resolve_path(path_str: str, base_path: str) -> str:
    if not path_str or os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_path, path_str))


class DummyVectorStore:
    """eval 阶段不需要向量库，仅作为占位对象传入 BenchmarkPipeline。"""
    pass


def main():
    parser = argparse.ArgumentParser(description="Eval-only runner for existing generated_answers.json")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    print(f"[Init] Loading configuration from: {config_path}")

    config = load_config(config_path)
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
    config = resolve_env_vars(config)

    dataset_name = config.get("dataset_name", "UnknownDataset")
    for key in ["raw_data", "output_dir", "vector_store", "doc_output_dir"]:
        if key in config.get("paths", {}):
            original = config["paths"][key]
            rendered = original.format(dataset_name=dataset_name)
            config["paths"][key] = resolve_path(rendered, WORKSPACE_ROOT)

    output_dir = config["paths"]["output_dir"]
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory not found: {output_dir}\n"
            f"请先运行 kimi_prompt_rag.py 生成 generated_answers.json"
        )

    generated_file = os.path.join(output_dir, "generated_answers.json")
    if not os.path.exists(generated_file):
        raise FileNotFoundError(f"generated_answers.json not found: {generated_file}")

    # 初始化 logger
    log_file = config["paths"].get("log_file", os.path.join(output_dir, "benchmark.log"))
    logger = setup_logging(log_file)
    logger.info(">>> Eval-only Session Started")

    # 加载 adapter
    adapter_cfg = config.get("adapter", {})
    module_path = adapter_cfg.get("module", "src.adapters.finance_bench_adapter")
    class_name = adapter_cfg.get("class_name", "FinanceBenchAdapter")
    mod = importlib.import_module(module_path)
    AdapterClass = getattr(mod, class_name)
    adapter = AdapterClass(raw_file_path=config["paths"]["raw_data"])

    # 初始化 LLM client
    llm_cfg = config.get("llm", {})
    llm_client = LLMClientWrapper(llm_cfg, api_key=llm_cfg.get("api_key", ""))

    # 创建 pipeline（使用 DummyVectorStore，eval 阶段不会访问）
    pipeline = BenchmarkPipeline(
        config=config,
        adapter=adapter,
        vector_db=DummyVectorStore(),
        llm=llm_client,
    )

    logger.info(f"Evaluating answers from: {generated_file}")
    pipeline.run_evaluation()
    logger.info("Benchmark evaluation finished successfully.")
    print(f"[Done] Evaluation results saved to: {output_dir}")


if __name__ == "__main__":
    main()
