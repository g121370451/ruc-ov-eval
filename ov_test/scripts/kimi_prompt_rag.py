#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过调用本机 kimi CLI 的 -p 模式，直接在 FinanceBench markdown 目录上问答，
并输出与原项目 eval 流程兼容的 generated_answers.json。

核心思路：不额外实现 RAG，而是把 markdown 目录作为 kimi 的工作目录，
让 kimi -p 自己利用项目上下文/文件检索能力回答问题。

用法（在 ov_test/ 目录下执行）：
    uv run python scripts/kimi_prompt_rag.py \
        --data-jsonl ../Data/FinanceBench_sample20/data/financebench_open_source.jsonl \
        --markdown-dir ../Data/FinanceBench_sample20/markdown \
        --output-dir ../Output/FinanceBench_sample20/kimi_prompt_experiment_0001

可选参数：
    --max-queries N       只处理前 N 个问题（调试用）
    --max-workers N       并发数（默认 1，避免 kimi 过载）
    --model MODEL         指定 kimi 模型别名（默认使用 config.toml 中的 default_model）
    --yolo                自动放行 kimi 的所有 action（默认已启用）
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.finance_bench_adapter import FinanceBenchAdapter
from src.core.logger import get_logger

logger = get_logger()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


FINANCEBENCH_PROMPT = """You are a financial analyst. Answer the following question based on the financial markdown documents available in the current directory. Focus on the document named {doc_name}.md if it is present.

If the answer involves a numerical value, include the unit (e.g., USD millions, %, etc.).
If the provided documents do not contain sufficient information to answer the question, respond with 'Insufficient information'.

Question: {question}
Answer:"""


def run_kimi_prompt(
    prompt: str,
    markdown_dir: Path,
    model: str = None,
    yolo: bool = True,
    timeout: int = 300,
) -> str:
    """
    调用本机 kimi CLI 的 -p 模式，返回 stdout 文本。
    """
    cmd = ["kimi", "-p", prompt]
    if yolo:
        cmd.append("-y")
    if model:
        cmd.extend(["-m", model])

    logger.debug(f"Running: {' '.join(cmd)} in {markdown_dir}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(markdown_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or "(no stderr)"
            logger.error(f"kimi -p failed: {err}")
            return f"ERROR: kimi -p exited with {result.returncode}: {err}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"kimi -p timed out after {timeout}s")
        return "ERROR: timeout"
    except Exception as e:
        logger.error(f"kimi -p exception: {e}")
        return f"ERROR: {e}"


def make_cache_key(doc_name: str, question: str, prompt_template: str) -> str:
    """基于 doc_name + question + prompt 模板生成稳定缓存 key。"""
    raw = f"{doc_name}::{question}::{prompt_template}"
    import hashlib
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(cache_file: Path) -> Dict[str, str]:
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
    return {}


def save_cache(cache_file: Path, cache: Dict[str, str]):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def process_one(
    task: Dict[str, Any],
    markdown_dir: Path,
    model: str,
    yolo: bool,
    cache: Dict[str, str],
    cache_file: Path,
    timeout: int,
) -> Dict[str, Any]:
    qa = task["qa"]
    doc_name = task["sample_id"]
    question = qa.question

    prompt = FINANCEBENCH_PROMPT.format(doc_name=doc_name, question=question)
    cache_key = make_cache_key(doc_name, question, FINANCEBENCH_PROMPT)

    if cache_key in cache:
        answer = cache[cache_key]
        logger.info(f"[Query-{task['id']}] cache hit")
    else:
        t0 = time.time()
        answer = run_kimi_prompt(prompt, markdown_dir, model=model, yolo=yolo, timeout=timeout)
        latency = time.time() - t0
        logger.info(f"[Query-{task['id']}] kimi latency: {latency:.2f}s")
        cache[cache_key] = answer
        save_cache(cache_file, cache)

    return {
        "_global_index": task["id"],
        "sample_id": doc_name,
        "question": question,
        "gold_answers": qa.gold_answers,
        "category": str(qa.category),
        "evidence": qa.evidence,
        # eval 流程只需要 llm.final_answer；其余字段为占位，可选
        "llm": {
            "final_answer": answer,
            "not_mentioned_reason": "",
        },
        "metrics": {
            "Recall": 0.0,  # kimi 内部检索结果不可见，设为 0
        },
        "retrieval": {
            "latency_sec": 0.0,
            "uris": [],
            "recall_texts": [],
            "prompt_texts": [],
            "sql_queries": [],
        },
        "token_usage": {
            "total_input_tokens": 0,
            "llm_output_tokens": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="使用 kimi -p 模式生成 FinanceBench 答案")
    parser.add_argument("--data-jsonl", required=True, help="FinanceBench JSONL 文件路径")
    parser.add_argument("--markdown-dir", required=True, help="Markdown 文档目录（kimi 工作目录）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--max-queries", type=int, default=None, help="最多处理的问题数")
    parser.add_argument("--max-workers", type=int, default=1, help="kimi -p 并发数")
    parser.add_argument("--model", default=None, help="kimi 模型别名")
    parser.add_argument("--yolo", action="store_true", default=True, help="自动放行 kimi action")
    parser.add_argument("--no-yolo", dest="yolo", action="store_false", help="不自动放行 kimi action")
    parser.add_argument("--timeout", type=int, default=300, help="单个 kimi -p 超时时间（秒）")
    args = parser.parse_args()

    data_jsonl = resolve_path(args.data_jsonl)
    markdown_dir = resolve_path(args.markdown_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not markdown_dir.exists():
        raise FileNotFoundError(f"Markdown directory not found: {markdown_dir}")

    cache_file = output_dir / ".kimi_prompt_cache.json"
    cache = load_cache(cache_file)

    # 加载数据
    adapter = FinanceBenchAdapter(str(data_jsonl))
    samples = adapter.load_and_transform()

    tasks = []
    global_idx = 0
    for sample in samples:
        for qa in sample.qa_pairs:
            if args.max_queries is not None and global_idx >= args.max_queries:
                break
            tasks.append({"id": global_idx, "sample_id": sample.sample_id, "qa": qa})
            global_idx += 1
        if args.max_queries is not None and global_idx >= args.max_queries:
            break

    logger.info(f"Processing {len(tasks)} questions from {markdown_dir} ...")

    results_map: Dict[int, Dict[str, Any]] = {}
    if args.max_workers <= 1:
        for task in tqdm(tasks, desc="kimi -p"):
            res = process_one(
                task, markdown_dir, args.model, args.yolo, cache, cache_file, args.timeout
            )
            results_map[res["_global_index"]] = res
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_one,
                    task,
                    markdown_dir,
                    args.model,
                    args.yolo,
                    cache,
                    cache_file,
                    args.timeout,
                ): task
                for task in tasks
            }
            pbar = tqdm(total=len(tasks), desc="kimi -p")
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results_map[res["_global_index"]] = res
                except Exception as e:
                    logger.error(f"Failed for task {task['id']}: {e}")
                pbar.update(1)
            pbar.close()

    sorted_results = [results_map[i] for i in sorted(results_map.keys())]
    save_data = {
        "summary": {
            "dataset": output_dir.name,
            "total_queries": len(sorted_results),
        },
        "results": sorted_results,
    }

    generated_file = output_dir / "generated_answers.json"
    with open(generated_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved generated_answers.json -> {generated_file}")
    logger.info(
        f"Successful answers: {sum(1 for r in sorted_results if not r['llm']['final_answer'].startswith('ERROR:'))}/{len(sorted_results)}"
    )


if __name__ == "__main__":
    main()
