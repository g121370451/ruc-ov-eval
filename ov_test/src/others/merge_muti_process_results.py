'''
merge muti_process experiment results

Each results include:
    - generated_answers.json                (retrieval + LLM answers)
    - qa_eval_detailed_results.json         (per-query eval scores, optional)
    - benchmark_metrics_report.json         (summary metrics, optional)

results are identified by _global_index, so results are non-overlapping.

recommended Usage:
    # Glob pattern
    uv run python ov_test/src/others/merge_muti_process_results.py \
        --target-dirs /Users/zhangqianyi/Desktop/ruc-ov/Output/FinanceBench_sample20/deepread_global_experiment_part*_0001
'''

import argparse
import glob
import json
import sys
import re
from pathlib import Path

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def merge_results(target_dirs: list[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_results: list[dict] = []
    eval_results: list[dict] = []
    dataset_name = "Unknown"
    shard_reports: list[dict] = []

    for shard_dir in target_dirs:
        gen_path = shard_dir / "generated_answers.json"
        eval_path = shard_dir / "qa_eval_detailed_results.json"
        report_path = shard_dir / "benchmark_metrics_report.json"

        if not gen_path.exists():
            print(f"Warning: Missing generated_answers.json in {shard_dir}, skipping this shard.", file=sys.stderr)
            continue

        gen_data = load_json(gen_path)
        dataset_name = gen_data.get("summary", {}).get("dataset", dataset_name)
        gen_results.extend(gen_data.get("results", []))

        if eval_path.exists():
            eval_data = load_json(eval_path)
            eval_results.extend(eval_data.get("results", []))

        if report_path.exists():
            report_data = load_json(report_path)
            shard_reports.append(report_data)
    
    if not gen_results:
        print("Error: No generated answers found in any shard. Exiting.", file=sys.stderr)
        sys.exit(1)

    # check for duplicate _global_index
    indexs = [item.get("_global_index") for item in gen_results]
    duplicates = sorted(set([x for x in indexs if indexs.count(x) > 1]))
    if duplicates:
        print(f"Warning: Found duplicate _global_index in generated answers: {duplicates[:20]}", file=sys.stderr)

    gen_results.sort(key=lambda x: x.get("_global_index", 0))
    total = len(gen_results)

    # merge generated_answers.json
    gen_out = {
        "summary": {
            "dataset": dataset_name,
            "total_queries": total,
            "merged_from": [str(d) for d in target_dirs],
        },
        "results": gen_results
    }

    with (output_dir / "generated_answers.json").open("w", encoding="utf-8") as f:
        json.dump(gen_out, f, indent=2, ensure_ascii=False)
    
    # merge qa_eval_detailed_results.json if available
    has_eval = len(eval_results) > 0
    if has_eval:
        eval_results.sort(key=lambda x: x.get("_global_index", 0))
        eval_out_path = output_dir / "qa_eval_detailed_results.json"
        with eval_out_path.open("w", encoding="utf-8") as f:
            json.dump({"results": eval_results}, f, indent=2, ensure_ascii=False)

    # merge benchmark_metrics_report.json if available
    report: dict = {}

    # reserve the first ingestion imformation
    for r in shard_reports:
        if "Insertion Efficiency (Total Dataset)" in r:
            report["Insertion Efficiency (Total Dataset)"] = r["Insertion Efficiency (Total Dataset)"]
            break
    
    # re-cal Query Efficiency
    if total > 0:
        report["Dataset"] = dataset_name
        report["Total Queries Generated"] = total
        report["Query Efficiency (Average Per Query)"] = {
            "Average Retrieval Time (s)": sum(
                r["retrieval"]["latency_sec"] for r in gen_results
            ) / total,
            "Average Input Tokens": sum(
                r["token_usage"]["total_input_tokens"] for r in gen_results
            ) / total,
            "Average Output Tokens": sum(
                r["token_usage"]["llm_output_tokens"] for r in gen_results
            ) / total,
        }

    if has_eval and total > 0:
        report["Total Queries Evaluated"] = total
        report["Performance Metrics"] = {
            "Average F1 Score": sum(r["metrics"]["F1"] for r in eval_results) / total,
            "Average Recall": sum(r["metrics"]["Recall"] for r in eval_results) / total,
            "Average Accuracy (Hit  0-4 )": sum(r["metrics"]["Accuracy"] for r in eval_results) / total,
            "Average Accuracy (normalization)": sum(r["metrics"]["Accuracy"] for r in eval_results) / total / 4,
        }
    
    report_out_path = output_dir / "benchmark_metrics_report.json"
    with report_out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Merged results saved to {output_dir}")


def infer_output_dir(shard_dirs: list[Path]) -> Path:
    first_dir = shard_dirs[0]
    parent = first_dir.parent
    basename = first_dir.name

    m = re.match(r"^(.*)_part\d+(_\d+)?$", basename)
    if m:
        prefix = m.group(1)
        suffix = m.group(2) or ""
        return parent / f"{prefix}_merged{suffix}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Merge multi-process experiment results")

    parser.add_argument("--target-dirs", type=str, nargs="+", required=True,
                        help="List of target directories containing experiment results to merge. Can use glob patterns.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save merged results. Will be created if it doesn't exist.")
    args = parser.parse_args()

    shard_dirs: list[Path] = []
    for pattern in args.target_dirs:
        matched_dirs = glob.glob(pattern)
        if not matched_dirs:
            print(f"Warning: No directories matched pattern '{pattern}'", file=sys.stderr)
        else:
            shard_dirs.extend([Path(d) for d in matched_dirs])

    if not shard_dirs:
        print("Error: No valid directories found to merge. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_dirs)} directories to merge:")
    for d in shard_dirs:
        print(f"  - {d}")

    if not args.output_dir:
        args.output_dir = str(infer_output_dir(shard_dirs))
        if not args.output_dir:
            print("Error: Could not infer output directory name. Please specify --output-dir explicitly.", file=sys.stderr)
            sys.exit(1)

    merge_results(shard_dirs, Path(args.output_dir))


if __name__ == "__main__":
    main()