#!/usr/bin/env python3
"""Evaluate missing QA records and patch benchmark result files.

This is an operational helper for cases where the normal eval stage is stuck
near the end. It uses the same config/model with a shorter equivalent judge
prompt, then writes the missing records back into the standard result files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.checkpoint import CheckpointManager  # noqa: E402
from src.core.llm_client import LLMClientWrapper  # noqa: E402
from src.core.metrics import MetricsCalculator  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = config.get("dataset_name", "UnknownDataset")
    retrieval_topk = config.get("execution", {}).get("retrieval_topk", 5)
    format_vars = {"dataset_name": dataset_name, "retrieval_topk": retrieval_topk}
    for key in ("dataset_path", "output_dir", "vector_store", "log_file", "doc_output_dir"):
        if key in config.get("paths", {}):
            rendered = config["paths"][key].format(**format_vars)
            config["paths"][key] = str((ROOT / rendered).resolve()) if not os.path.isabs(rendered) else rendered
    return config


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, path.with_suffix(path.suffix + f".bak_{stamp}"))


def compact_llm_grader(
    llm_client: Any,
    question: str,
    golds: list[str],
    answer: str,
) -> dict[str, Any]:
    gold_text = " | ".join(str(g) for g in golds)
    system_prompt = (
        "You are a strict but fair answer evaluator. "
        "Do not brainstorm. Do not rewrite the answer. "
        "Return only compact JSON."
    )
    user_prompt = f"""
Grade the Generated Answer against the Gold Answer for the Question.

Use this 0-4 scale:
4 = fully correct; same core answer and key facts.
3 = mostly correct; minor imprecision or minor missing detail.
2 = partially correct; correct topic or conclusion but missing/using a wrong key fact, calculation, or rationale.
1 = weakly related but mostly wrong.
0 = wrong or contradictory.

Important:
- Judge the final answer, not writing style.
- If the conclusion is correct but the required calculation/rationale differs from the Gold Answer, use 2 or 3, not 4.
- Keep reasoning to one short sentence.

Question: {question}
Gold Answer: {gold_text}
Generated Answer: {answer}

Return exactly this JSON shape:
{{"score": 0, "reasoning": "one short sentence"}}
""".strip()

    response = llm_client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = response.content if response and hasattr(response, "content") else ""
    text = str(content).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(text[start : end + 1])
        else:
            raise

    score = int(parsed.get("score", 0))
    score = max(0, min(4, score))
    return {
        "score": score,
        "reasoning": parsed.get("reasoning", text),
        "prompt_type": "Compact_Generic_0-4",
    }


def build_eval_record(item: dict[str, Any], llm_client: LLMClientWrapper, config: dict[str, Any]) -> dict[str, Any]:
    ans = item["llm"]["final_answer"]
    golds = item["gold_answers"]
    f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)

    eval_res = compact_llm_grader(
        llm_client.llm,
        item["question"],
        golds,
        ans,
    )

    if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
        f1 = 1.0
        eval_res["score"] = 4.0
        eval_res["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
        eval_res["prompt_type"] = "Heuristic_Refusal_Check"

    item = json.loads(json.dumps(item, ensure_ascii=False))
    item.setdefault("metrics", {})
    item["metrics"].update({"F1": f1, "Accuracy": eval_res["score"]})
    item["llm_evaluation"] = {
        "prompt_used": eval_res["prompt_type"],
        "reasoning": eval_res["reasoning"],
        "normalized_score": eval_res["score"],
    }
    return item


def update_report(report_path: Path, config: dict[str, Any], eval_records: list[dict[str, Any]]) -> None:
    report = load_json(report_path) if report_path.exists() else {}
    total = len(eval_records)
    if total == 0:
        return
    report.update(
        {
            "Dataset": config.get("dataset_name", "Unknown_Dataset"),
            "Total Queries Evaluated": total,
            "Performance Metrics": {
                "Average F1 Score": sum(r["metrics"]["F1"] for r in eval_records) / total,
                "Average Recall": sum(r["metrics"]["Recall"] for r in eval_records) / total,
                "Average Accuracy (Hit 0-4)": sum(r["metrics"]["Accuracy"] for r in eval_records) / total,
                "Average Accuracy (normalization)": (
                    sum(r["metrics"]["Accuracy"] for r in eval_records) / total
                )
                / 4,
            },
        }
    )
    save_json(report_path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and patch missing benchmark QA results.")
    parser.add_argument("--config", required=True, help="Benchmark config path.")
    parser.add_argument("--ids", nargs="*", type=int, help="Specific _global_index values to evaluate.")
    parser.add_argument("--dry-run", action="store_true", help="Print missing IDs without calling the model.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = resolve_config_paths(load_config(Path(args.config).resolve()))

    output_dir = Path(config["paths"]["output_dir"])
    generated_path = output_dir / "generated_answers.json"
    eval_path = output_dir / "qa_eval_detailed_results.json"
    report_path = output_dir / "benchmark_metrics_report.json"
    checkpoint_path = output_dir / "benchmark_checkpoint.json"

    generated = load_json(generated_path)["results"]
    eval_data = load_json(eval_path) if eval_path.exists() else {"results": []}
    eval_map = {r["_global_index"]: r for r in eval_data.get("results", [])}
    generated_map = {r["_global_index"]: r for r in generated}

    if args.ids:
        target_ids = args.ids
    elif checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        completed = set(checkpoint.get("execution_state", {}).get("completed_tasks", []))
        target_ids = sorted(set(generated_map) - completed)
    else:
        target_ids = sorted(set(generated_map) - set(eval_map))

    target_ids = [idx for idx in target_ids if idx not in eval_map]
    print(f"Output dir: {output_dir}")
    print(f"Target IDs: {target_ids}")

    if args.dry_run:
        for idx in target_ids:
            item = generated_map[idx]
            print(f"\nID {idx}")
            print(f"Question: {item.get('question')}")
            print(f"Answer: {item.get('llm', {}).get('final_answer')}")
            print(f"Gold: {item.get('gold_answers')}")
        return 0

    if not target_ids:
        print("No missing records to evaluate.")
        return 0

    llm_client = LLMClientWrapper(config["llm"], config["llm"]["api_key"])
    for idx in target_ids:
        print(f"\nEvaluating ID {idx} ...", flush=True)
        eval_map[idx] = build_eval_record(generated_map[idx], llm_client, config)
        print(
            f"Done ID {idx}: score={eval_map[idx]['metrics']['Accuracy']}, "
            f"f1={eval_map[idx]['metrics']['F1']:.6f}",
            flush=True,
        )

        # Persist after every successful ID so work is not lost if the next one stalls.
        eval_records = [eval_map[idx] for idx in sorted(eval_map)]
        for path in (eval_path, report_path, checkpoint_path):
            backup(path)
        save_json(eval_path, {"results": eval_records})
        update_report(report_path, config, eval_records)

        completed = set(eval_map)
        if len(completed) == len(generated_map):
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        else:
            manager = CheckpointManager(str(output_dir), config)
            manager.update_completed_tasks("evaluation", completed, len(generated_map))

    print("\nPatch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
