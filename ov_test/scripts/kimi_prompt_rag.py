#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通过调用本机 agent CLI 的 -p 模式，直接在 FinanceBench markdown 目录上问答，
并输出与原项目 eval 流程兼容的 generated_answers.json。

支持两种 agent：
  - kimi: kimi --output-format stream-json -p
  - mc:   mc --code -p

核心思路：不额外实现 RAG，而是把 markdown 目录作为 agent 的工作目录，
让 agent 自己利用项目上下文/文件检索能力回答问题。

用法（在 ov_test/ 目录下执行，相对路径按项目约定解析到仓库父目录）：
    uv run python scripts/kimi_prompt_rag.py \
        --agent kimi \
        --data-jsonl Data/FinanceBench_sample20/data/financebench_open_source.jsonl \
        --markdown-dir Data/FinanceBench_sample20/markdown \
        --output-dir Output/FinanceBench_sample20/kimi_prompt_experiment_0001

使用 mc agent：
    uv run python scripts/kimi_prompt_rag.py \
        --agent mc \
        --data-jsonl Data/FinanceBench_sample20/data/financebench_open_source.jsonl \
        --markdown-dir Data/FinanceBench_sample20/markdown \
        --output-dir Output/FinanceBench_sample20/mc_prompt_experiment_0001
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
OV_TEST_ROOT = SCRIPT_DIR.parent  # ov_test/
# 项目约定：配置中的相对路径基于仓库父目录（即 Data/、Output/ 所在目录）解析
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(OV_TEST_ROOT))

from src.adapters.finance_bench_adapter import FinanceBenchAdapter
from src.core.logger import get_logger

logger = get_logger()


# 默认重试/退避参数
DEFAULT_MAX_RETRIES = 50
DEFAULT_BASE_DELAY = 10.0   # 初始退避秒数
DEFAULT_MAX_DELAY = 1800.0  # 最大退避秒数（30 分钟）
DEFAULT_JITTER = 5.0        # 随机抖动秒数


def resolve_path(path_str: str) -> Path:
    """按项目约定解析路径：相对路径基于仓库父目录（PROJECT_ROOT）。"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def compute_backoff_delay(attempt: int, base_delay: float, max_delay: float, jitter: float) -> float:
    """指数退避 + 随机抖动。attempt 从 0 开始计数。"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter > 0:
        delay += random.uniform(0, jitter)
    return delay


FINANCEBENCH_PROMPT = """Based on the financial markdown documents available in the current directory, answer the following question accurately and concisely.
If the answer involves a numerical value, include the unit (e.g., USD millions, %, etc.).
If the provided documents do not contain sufficient information to answer the question, respond with 'Insufficient information'.

Question: {question}
Answer:"""


def parse_kimi_stream_json(stdout: str) -> str:
    """
    解析 kimi --output-format stream-json 的输出，提取最终 assistant 答案。

    stream-json 每行是一个 JSON 对象，常见 role：
      - assistant + tool_calls：推理/工具调用意图
      - assistant（无 tool_calls）：最终回答
      - tool：工具返回结果
      - meta：session 恢复提示等无关信息

    返回最后一个不带 tool_calls 的 assistant content。
    """
    final_contents = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("role") == "assistant":
            content = obj.get("content", "").strip()
            if content and "tool_calls" not in obj:
                final_contents.append(content)
    if final_contents:
        return final_contents[-1]
    return ""


def run_agent_prompt(
    prompt: str,
    markdown_dir: Path,
    agent: str = "kimi",
    model: str = None,
    timeout: int = 300,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
) -> str:
    """
    调用本机 agent CLI 回答问题，返回最终答案。

    支持两种 agent：
      - kimi: kimi --output-format stream-json -p
      - mc:   mc --code -p

    对 RPM 限制、网络抖动、额度耗尽等失败情况进行指数退避重试，最长退避 30 分钟。
    """
    if agent == "kimi":
        # --output-format 必须放在 -p 之前，否则 kimi CLI 解析会出错
        cmd = ["kimi", "--output-format", "stream-json", "-p", prompt]
        if model:
            cmd.extend(["-m", model])
        parse_func = parse_kimi_stream_json
    elif agent == "mc":
        cmd = ["mc", "--code", "-p", prompt]
        # mc 暂不支持通过 -m 传模型，忽略 model 参数
        parse_func = lambda stdout: stdout.strip()
    else:
        raise ValueError(f"Unknown agent: {agent}. Supported: kimi, mc")

    logger.debug(f"Running: {' '.join(cmd)} in {markdown_dir}")

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                cwd=str(markdown_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                parsed = parse_func(result.stdout)
                if parsed:
                    return parsed
                raw = result.stdout.strip()
                if raw:
                    return raw
                last_error = f"ERROR: {agent} returned empty response"
            else:
                err = result.stderr.strip() or "(no stderr)"
                last_error = f"ERROR: {agent} exited with {result.returncode}: {err}"

        except subprocess.TimeoutExpired:
            last_error = f"ERROR: timeout after {timeout}s"
        except Exception as e:
            last_error = f"ERROR: {e}"

        # 最后一次尝试不再 sleep
        if attempt >= max_retries:
            break

        delay = compute_backoff_delay(attempt, base_delay, max_delay, jitter)
        logger.warning(
            f"{agent} failed (attempt {attempt + 1}/{max_retries + 1}), "
            f"retrying in {delay:.1f}s... | {last_error[:120]}"
        )
        time.sleep(delay)

    logger.error(f"{agent} exhausted all {max_retries + 1} attempts: {last_error}")
    return last_error


# 保留旧函数名作为别名，避免外部调用被破坏（实际不再使用）
run_kimi_prompt = run_agent_prompt


def make_cache_key(doc_name: str, question: str, prompt_template: str) -> str:
    """基于 doc_name + question + prompt 模板生成稳定缓存 key。"""
    raw = f"{doc_name}::{question}::{prompt_template}"
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
    agent: str,
    model: str,
    yolo: bool,
    cache: Dict[str, str],
    cache_file: Path,
    timeout: int,
    ignore_cache: bool,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    jitter: float,
) -> Dict[str, Any]:
    qa = task["qa"]
    doc_name = task["sample_id"]
    question = qa.question

    prompt = FINANCEBENCH_PROMPT.format(question=question)
    cache_key = make_cache_key(doc_name, question, FINANCEBENCH_PROMPT)

    if not ignore_cache and cache_key in cache:
        answer = cache[cache_key]
        logger.info(f"[Query-{task['id']}] cache hit")
    else:
        t0 = time.time()
        answer = run_agent_prompt(
            prompt,
            markdown_dir,
            agent=agent,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
        )
        latency = time.time() - t0
        logger.info(f"[Query-{task['id']}] {agent} latency: {latency:.2f}s")
        # 只有成功回答才写入缓存；ERROR 结果不缓存，便于额度恢复后重试
        if not answer.startswith("ERROR:"):
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
    parser = argparse.ArgumentParser(description="使用 agent CLI 的 -p 模式生成 FinanceBench 答案")
    parser.add_argument("--agent", choices=["kimi", "mc"], default="kimi",
                        help="使用的 agent CLI：kimi 或 mc（默认 kimi）")
    parser.add_argument("--data-jsonl", required=True, help="FinanceBench JSONL 文件路径")
    parser.add_argument("--markdown-dir", required=True, help="Markdown 文档目录（agent 工作目录）")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--max-queries", type=int, default=None, help="最多处理的问题数")
    parser.add_argument("--max-workers", type=int, default=1, help="agent -p 并发数")
    parser.add_argument("--model", default=None, help="kimi 模型别名（仅 kimi agent 有效）")
    parser.add_argument("--yolo", action="store_true", default=True, help="自动放行 kimi action（仅 kimi 有效）")
    parser.add_argument("--no-yolo", dest="yolo", action="store_false", help="不自动放行 kimi action")
    parser.add_argument("--timeout", type=int, default=300, help="单个 agent -p 超时时间（秒）")
    parser.add_argument("--ignore-cache", action="store_true", help="忽略已有缓存，强制重新调用 agent")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"失败重试次数上限（默认 {DEFAULT_MAX_RETRIES}）")
    parser.add_argument("--base-delay", type=float, default=DEFAULT_BASE_DELAY,
                        help=f"指数退避初始秒数（默认 {DEFAULT_BASE_DELAY}）")
    parser.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY,
                        help=f"指数退避最大秒数（默认 {DEFAULT_MAX_DELAY}，即 30 分钟）")
    parser.add_argument("--jitter", type=float, default=DEFAULT_JITTER,
                        help=f"随机抖动秒数（默认 {DEFAULT_JITTER}）")
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
    logger.info(f"Agent: {args.agent}")
    logger.info(
        f"Retry policy: max_retries={args.max_retries}, base_delay={args.base_delay}s, "
        f"max_delay={args.max_delay}s, jitter={args.jitter}s"
    )

    results_map: Dict[int, Dict[str, Any]] = {}
    if args.max_workers <= 1:
        for task in tqdm(tasks, desc=f"{args.agent} -p"):
            res = process_one(
                task, markdown_dir, args.agent, args.model, args.yolo, cache, cache_file,
                args.timeout, args.ignore_cache, args.max_retries, args.base_delay,
                args.max_delay, args.jitter,
            )
            results_map[res["_global_index"]] = res
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    process_one,
                    task,
                    markdown_dir,
                    args.agent,
                    args.model,
                    args.yolo,
                    cache,
                    cache_file,
                    args.timeout,
                    args.ignore_cache,
                    args.max_retries,
                    args.base_delay,
                    args.max_delay,
                    args.jitter,
                ): task
                for task in tasks
            }
            pbar = tqdm(total=len(tasks), desc=f"{args.agent} -p")
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
