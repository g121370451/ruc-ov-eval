#!/usr/bin/env python3

import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _sanitize_for_path(text: str, max_length: int = 50) -> str:
    safe = re.sub(
        r"[^\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u3400-\u4dbf\U00020000-\U0002a6df\s-]",
        "",
        text,
    )
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("_")
    if not safe:
        return "section"
    if len(safe) > max_length:
        hash_suffix = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"{safe[: max_length - 9]}_{hash_suffix}"
    return safe


def _load_results(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("results") or []


def _check_uri_allowed(dataset: str, sample_id: str, uri: str) -> Tuple[bool, str]:
    if dataset == "FinanceBench":
        prefix = "viking://resources/pdfs/" + str(sample_id)
        return (uri == prefix or uri.startswith(prefix + "/")), "not_under_pdf_sample_dir"

    if dataset == "Locomo":
        base_ok = uri.startswith("viking://resources/Locomo_processed_docs/")
        if not base_ok:
            return False, "not_under_locomo_base"
        raw_token = str(sample_id) + "_doc"
        token = _sanitize_for_path(raw_token)
        return (raw_token in uri or token in uri), "missing_sample_doc_dir"

    if dataset == "Qasper":
        base_ok = uri.startswith("viking://resources/Qasper_processed_docs/")
        if not base_ok:
            return False, "not_under_qasper_base"
        raw_token = str(sample_id) + "_doc"
        token = _sanitize_for_path(raw_token)
        return (raw_token in uri or token in uri), "missing_sample_doc_dir"

    if dataset == "SyllabusQA":
        base_ok = uri.startswith("viking://resources/SyllabusQA_processed_docs/")
        if not base_ok:
            return False, "not_under_syllabus_base"
        raw_token = str(sample_id) + "_doc"
        token = _sanitize_for_path(raw_token)
        return (raw_token in uri or token in uri), "missing_sample_doc_dir"

    if dataset == "ClapNQ":
        return uri.startswith("viking://resources/ClapNQ_processed_docs/"), "not_under_clapnq_base"

    if dataset == "HotpotQA":
        return uri.startswith("viking://resources/HotpotQA_processed_docs/"), "not_under_hotpot_base"

    return True, ""


def check_experiment(output_root: Path, experiment_name: str, datasets: List[str]) -> Dict[str, dict]:
    reports: Dict[str, dict] = {}

    for ds in datasets:
        result_path = output_root / ds / experiment_name / "generated_answers.json"
        if not result_path.exists():
            reports[ds] = {"missing": True, "file": str(result_path)}
            continue

        results = _load_results(result_path)

        total = 0
        bad = 0
        bad_reasons = Counter()
        examples = []

        for item in results:
            sample_id = item.get("sample_id", "")
            uris = (((item or {}).get("retrieval") or {}).get("uris") or [])
            for uri in uris:
                total += 1
                ok, reason = _check_uri_allowed(ds, sample_id, uri)
                if ok:
                    continue
                bad += 1
                bad_reasons[reason] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "_global_index": item.get("_global_index"),
                            "sample_id": sample_id,
                            "reason": reason,
                            "uri": uri,
                        }
                    )

        reports[ds] = {
            "missing": False,
            "file": str(result_path),
            "queries": len(results),
            "total_uris": total,
            "bad_uris": bad,
            "bad_reasons": dict(bad_reasons.most_common()),
            "examples": examples,
        }

    return reports


def main() -> int:
    root = Path("/Users/bytedance/PR/OpenViking/benchmark/RAG")
    output_root = root / "Output"
    experiment = "experiment_test_top_5_1_8_per_query"
    datasets = ["ClapNQ", "FinanceBench", "HotpotQA", "Locomo", "Qasper", "SyllabusQA"]

    reports = check_experiment(output_root, experiment, datasets)
    for ds in datasets:
        rep = reports[ds]
        print(f"== {ds} ==")
        print("file:", rep.get("file"))
        if rep.get("missing"):
            print("missing result file")
            print()
            continue
        print("queries:", rep["queries"], "total_uris:", rep["total_uris"], "bad_uris:", rep["bad_uris"])
        if rep["bad_uris"]:
            print("bad_reasons:", rep["bad_reasons"])
            for ex in rep["examples"][:10]:
                print("  ex:", ex)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
