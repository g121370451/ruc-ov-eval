#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple


L2_MD_SKIP = {".abstract.md", ".overview.md"}
SENT_END_RE = re.compile(r"[.!?。！？]")
ONLY_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
MD_NOISE_RE = re.compile(r"^\s*(#{1,6}\s+|\*\s+|\-\s+|\d+\.\s+|>\s+)", re.M)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

BAD_START_RE = re.compile(r"^[\s\"'\(\[\{\*\-\d#>]*([a-z,;:\)\]\}])")
BAD_END_ALNUM_RE = re.compile(r"([A-Za-z0-9])\s*$")

END_OK_CHARS = set(".!?。！？…)]}\"”’")


@dataclass(frozen=True)
class ShardMeta:
    path: str
    rel_path: str
    words: int
    chars: int
    sentences: int
    first_nonempty: str
    last_nonempty: str
    head: str
    tail: str


def iter_l2_md_files(resources_dir: Path) -> Iterable[Path]:
    for root, _, files in os.walk(resources_dir):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            if fn in L2_MD_SKIP:
                continue
            yield Path(root) / fn


def safe_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def summarize_text(raw: str) -> Tuple[List[str], ShardMeta]:
    stripped = raw.strip("\n\r\t ")
    if not stripped:
        meta = ShardMeta(
            path="",
            rel_path="",
            words=0,
            chars=0,
            sentences=0,
            first_nonempty="",
            last_nonempty="",
            head="",
            tail="",
        )
        return ["empty"], meta

    lines = [ln.rstrip() for ln in raw.splitlines()]
    nonempty = [ln for ln in lines if ln.strip()]

    title_only = False
    if 1 <= len(nonempty) <= 2 and all(ONLY_HEADING_RE.match(ln) for ln in nonempty):
        title_only = True

    word_count = len(WORD_RE.findall(raw))
    char_count = len(raw)

    denoised = MD_NOISE_RE.sub("", raw)
    sentence_count = len(SENT_END_RE.findall(denoised))

    bad_start = bool(BAD_START_RE.match(stripped))

    tail_ctx = stripped[-300:]
    bad_end = False
    if stripped and stripped[-1] not in END_OK_CHARS:
        if stripped[-1] in ",;:([{-/\\":
            bad_end = True
        elif BAD_END_ALNUM_RE.search(tail_ctx) and not SENT_END_RE.search(tail_ctx[-200:]):
            bad_end = True

    very_short = (word_count < 30) or (char_count < 200)
    one_sentence = (sentence_count <= 1 and word_count < 80)

    tags: List[str] = []
    if title_only:
        tags.append("title_only")
    if very_short:
        tags.append("very_short")
    if one_sentence:
        tags.append("one_sentence")
    if bad_start:
        tags.append("bad_start")
    if bad_end:
        tags.append("bad_end")

    head = stripped[:140].replace("\n", "\\n")
    tail = stripped[-140:].replace("\n", "\\n")
    meta = ShardMeta(
        path="",
        rel_path="",
        words=word_count,
        chars=char_count,
        sentences=sentence_count,
        first_nonempty=(nonempty[0] if nonempty else "")[:120],
        last_nonempty=(nonempty[-1] if nonempty else "")[:120],
        head=head,
        tail=tail,
    )
    return (tags or ["ok"]), meta


def scan_dataset(
    dataset: str,
    storage_root: Path,
    examples_per_type: int,
    anomalies_fp: Optional[object],
) -> Tuple[int, Dict[str, int], Dict[str, List[ShardMeta]]]:
    resources_dir = (
        storage_root
        / dataset
        / f"{dataset}_viking_store_index"
        / "viking"
        / "default"
        / "resources"
    )

    counts: DefaultDict[str, int] = defaultdict(int)
    examples: DefaultDict[str, List[ShardMeta]] = defaultdict(list)

    total = 0
    for path in iter_l2_md_files(resources_dir):
        total += 1
        raw = safe_read_text(path)
        if raw is None:
            counts["read_error"] += 1
            continue

        tags, meta = summarize_text(raw)
        rel_path = str(path).replace(str(storage_root) + os.sep, "")
        meta = ShardMeta(
            **{
                **asdict(meta),
                "path": str(path),
                "rel_path": rel_path,
            }
        )

        for t in tags:
            counts[t] += 1
        if tags != ["ok"] and anomalies_fp is not None:
            anomalies_fp.write(
                json.dumps(
                    {"dataset": dataset, "tags": tags, **asdict(meta)},
                    ensure_ascii=False,
                )
                + "\n"
            )
        for t in tags:
            if t != "ok" and len(examples[t]) < examples_per_type:
                examples[t].append(meta)

    return total, dict(counts), dict(examples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path(__file__).parent.parent / "ov_storage",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="ClapNQ,FinanceBench,HotpotQA,Locomo,Qasper,SyllabusQA",
    )
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    storage_root: Path = args.storage_root.resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    report = {}
    anomalies_fp = None
    out_path: Optional[Path] = None

    print("L2 shard sanity check (heuristics)")
    print(f"storage_root: {storage_root}")
    print()

    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anomalies_fp = out_path.open("w", encoding="utf-8")

    for ds in datasets:
        total, counts, examples = scan_dataset(ds, storage_root, args.examples, anomalies_fp)
        report[ds] = {"total": total, "counts": counts}

        print(f"== {ds} ==")
        print(f"total: {total}")
        bad = {k: v for k, v in counts.items() if k != "ok" and v}
        if not bad:
            print("no issues flagged")
            print()
            continue

        for k in sorted(bad, key=lambda x: (-bad[x], x)):
            print(f"{k}: {bad[k]}")

        for k in sorted(bad, key=lambda x: (-bad[x], x)):
            print(f"-- examples: {k} --")
            for ex in examples.get(k, []):
                print(
                    f"{ex.rel_path} | words={ex.words} chars={ex.chars} sents={ex.sentences}"
                )
                print(f"  first: {ex.first_nonempty}")
                print(f"  last : {ex.last_nonempty}")
                print(f"  head : {ex.head}")
                print(f"  tail : {ex.tail}")
            print()
        print()

    if anomalies_fp is not None:
        anomalies_fp.close()

    if out_path is not None:
        summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {out_path}")
        print(f"wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
