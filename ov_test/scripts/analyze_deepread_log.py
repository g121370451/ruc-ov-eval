#!/usr/bin/env python3
"""
分析 deepread_run.log，统计每个 query 的实际迭代轮次。
支持输出到控制台和 JSON 文件。
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path


def analyze(log_path: str):
    per_query = defaultdict(lambda: {"max_round": 0, "terminated_by": None})

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = rec.get("event")
            qid = rec.get("query_id")
            if not qid:
                continue

            if event in ("llm_request", "llm_response", "llm_thinking_only", "llm_empty_message"):
                round_num = rec.get("round", 0)
                per_query[qid]["max_round"] = max(per_query[qid]["max_round"], round_num)

            if event == "final_answer":
                per_query[qid]["terminated_by"] = "final_answer"
            elif event == "max_rounds_reached":
                per_query[qid]["terminated_by"] = "max_rounds_reached"

    rounds = [info["max_round"] for info in per_query.values()]
    if not rounds:
        return {}

    total = len(rounds)
    avg = sum(rounds) / total
    max_r = max(rounds)
    min_r = min(rounds)
    sorted_rounds = sorted(rounds)
    median = sorted_rounds[total // 2] if total % 2 else (sorted_rounds[total // 2 - 1] + sorted_rounds[total // 2]) / 2

    dist = defaultdict(int)
    for r in rounds:
        dist[r] += 1

    return {
        "total_queries": total,
        "average_rounds": round(avg, 2),
        "median_rounds": median,
        "max_rounds": max_r,
        "min_rounds": min_r,
        "distribution": dict(sorted(dist.items())),
        "per_query": [
            {
                "query_id": qid,
                "rounds": info["max_round"],
                "terminated_by": info["terminated_by"] or "unknown"
            }
            for qid, info in sorted(per_query.items())
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze DeepRead iteration rounds from deepread_run.log")
    parser.add_argument("log_path", help="Path to deepread_run.log")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    args = parser.parse_args()

    stats = analyze(args.log_path)
    if not stats:
        print("No valid records found.")
        return

    print("=== DeepRead Iteration Statistics ===")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Avg Rounds:    {stats['average_rounds']}")
    print(f"Median Rounds: {stats['median_rounds']}")
    print(f"Max Rounds:    {stats['max_rounds']}")
    print(f"Min Rounds:    {stats['min_rounds']}")
    print("Distribution:")
    for r, c in stats["distribution"].items():
        pct = c / stats["total_queries"] * 100
        print(f"  {r:2d} round(s): {c:3d} ({pct:5.1f}%)")

    out_path = args.output or Path(args.log_path).with_name("deepread_iteration_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
