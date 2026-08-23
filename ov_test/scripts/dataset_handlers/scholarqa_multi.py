"""Download and prepare the valid ScholarQA-Multi benchmark subset.

ScholarQA-Multi contains expert answers grounded in short citation contexts. Seven
of the 108 upstream records contain citation indices outside their ``ctxs`` lists.
This handler rejects those records and materializes the remaining citation contexts
as a shared corpus of merged TXT documents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm import tqdm


SCHOLARQA_REVISION = "95e6fc52b0a8a0ce0a74956029991e3bb00c38b9"
SCHOLARQA_URL = (
    "https://raw.githubusercontent.com/AkariAsai/ScholarQABench/"
    f"{SCHOLARQA_REVISION}/data/scholarqa_multi/human_answers.json"
)
SCHOLARQA_SHA256 = "83af055a691a8a0be078cb2a02813193a3fc2966091b28bd51824b43049fffea"

EXPECTED_TOTAL_QAS = 108
EXPECTED_VALID_QAS = 101
EXPECTED_DOCUMENTS = 413
EXPECTED_SUBJECT_COUNTS = {
    "bio": 19,
    "biophysics": 9,
    "cs_hci": 9,
    "cs_nlp": 29,
    "photonics": 29,
    "physics": 6,
}
EXPECTED_INVALID_QA_IDS = {
    "benjamin_bio_4",
    "bohao_cs_10",
    "jacqueline_cs_7",
    "jacqueline_cs_8",
    "jacqueline_cs_9",
    "weijia_cs_2",
    "yanyu_photonics_10",
}

_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SEMANTIC_SCHOLAR_URL_RE = re.compile(
    r"^https://www\.semanticscholar\.org/paper/([0-9a-fA-F]{40})/?$"
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"ScholarQA-Multi file must be a JSON list of objects: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _citation_indices(answer: str) -> list[int]:
    return [
        int(index)
        for group in _CITATION_GROUP_RE.findall(answer)
        for index in re.findall(r"\d+", group)
    ]


def _validate_record(record: dict[str, Any]) -> None:
    qa_id = str(record.get("id") or "").strip()
    question = record.get("input")
    answer = record.get("output")
    subject = str(record.get("subject") or "").strip()
    contexts = record.get("ctxs")
    if not qa_id or not isinstance(question, str) or not question.strip():
        raise ValueError(f"ScholarQA-Multi record has an empty ID or question: {qa_id!r}")
    if not isinstance(answer, str) or not answer.strip() or not subject:
        raise ValueError(f"ScholarQA-Multi record has an empty answer or subject: {qa_id}")
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"ScholarQA-Multi record has no citation contexts: {qa_id}")
    for context_index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise ValueError(f"ScholarQA-Multi {qa_id} ctxs[{context_index}] is invalid")
        title = context.get("title")
        text = context.get("text")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"ScholarQA-Multi {qa_id} ctxs[{context_index}] has no title")
        if not isinstance(text, str) or not text.strip():
            if not isinstance(text, float) or not math.isnan(text):
                raise ValueError(
                    f"ScholarQA-Multi {qa_id} ctxs[{context_index}] has invalid text"
                )
        url = context.get("url")
        if url and not _SEMANTIC_SCHOLAR_URL_RE.fullmatch(str(url).strip()):
            raise ValueError(
                f"ScholarQA-Multi {qa_id} ctxs[{context_index}] has an unexpected URL: {url}"
            )


def select_valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_TOTAL_QAS:
        raise ValueError(
            f"Unexpected ScholarQA-Multi snapshot size: {len(records)} != {EXPECTED_TOTAL_QAS}"
        )
    ids = [str(record.get("id") or "").strip() for record in records]
    if any(not qa_id for qa_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("ScholarQA-Multi QA IDs must be non-empty and unique")

    invalid_ids: set[str] = set()
    valid_records: list[dict[str, Any]] = []
    subject_counts: dict[str, int] = {}
    for record in records:
        _validate_record(record)
        qa_id = str(record["id"]).strip()
        contexts = record["ctxs"]
        invalid_indices = {
            index
            for index in _citation_indices(record["output"])
            if index >= len(contexts)
        }
        if invalid_indices:
            invalid_ids.add(qa_id)
            continue
        valid_records.append(record)
        subject = str(record["subject"]).strip()
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    if invalid_ids != EXPECTED_INVALID_QA_IDS:
        raise ValueError(
            "ScholarQA-Multi invalid-citation records changed: "
            f"{sorted(invalid_ids)} != {sorted(EXPECTED_INVALID_QA_IDS)}"
        )
    if len(valid_records) != EXPECTED_VALID_QAS:
        raise ValueError(
            f"Unexpected valid QA count: {len(valid_records)} != {EXPECTED_VALID_QAS}"
        )
    if subject_counts != EXPECTED_SUBJECT_COUNTS:
        raise ValueError(
            f"Unexpected ScholarQA-Multi subject counts: {subject_counts}"
        )
    return valid_records


def _normalize_missing_context_texts(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert the two upstream ``NaN`` context texts to standard JSON nulls."""
    normalized: list[dict[str, Any]] = []
    for record in records:
        copied_record = dict(record)
        copied_contexts: list[dict[str, Any]] = []
        for context in record["ctxs"]:
            copied_context = dict(context)
            text = copied_context.get("text")
            if isinstance(text, float) and math.isnan(text):
                copied_context["text"] = None
            copied_contexts.append(copied_context)
        copied_record["ctxs"] = copied_contexts
        normalized.append(copied_record)
    return normalized


def _normalized_title(title: str) -> str:
    return " ".join(title.casefold().split())


def _source_key(context: dict[str, Any]) -> tuple[str, str]:
    url = str(context.get("url") or "").strip()
    if url:
        match = _SEMANTIC_SCHOLAR_URL_RE.fullmatch(url)
        if not match:
            raise ValueError(f"Unexpected ScholarQA-Multi source URL: {url}")
        return "semantic_scholar", match.group(1).lower()
    return "title", _normalized_title(str(context["title"]))


def _author_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for author in value:
        if isinstance(author, dict):
            name = str(author.get("name") or "").strip()
        else:
            name = str(author).strip()
        if name and name not in names:
            names.append(name)
    return names


def _document_id(source_type: str, source_value: str) -> str:
    if source_type == "semantic_scholar":
        return f"s2_{source_value}"
    digest = hashlib.sha256(source_value.encode("utf-8")).hexdigest()[:20]
    return f"title_{digest}"


def collect_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    for record in records:
        qa_id = str(record["id"]).strip()
        subject = str(record["subject"]).strip()
        for context in record["ctxs"]:
            source_type, source_value = _source_key(context)
            entry = merged.setdefault(
                (source_type, source_value),
                {
                    "id": _document_id(source_type, source_value),
                    "source_type": source_type,
                    "paper_id": source_value if source_type == "semantic_scholar" else None,
                    "source_url": str(context.get("url") or "").strip(),
                    "title": str(context["title"]).strip(),
                    "alternate_titles": [],
                    "year": context.get("year"),
                    "authors": _author_names(context.get("authors")),
                    "subjects": [],
                    "qa_ids": [],
                    "excerpts": [],
                },
            )
            title = str(context["title"]).strip()
            if title != entry["title"] and title not in entry["alternate_titles"]:
                entry["alternate_titles"].append(title)
            if subject not in entry["subjects"]:
                entry["subjects"].append(subject)
            if qa_id not in entry["qa_ids"]:
                entry["qa_ids"].append(qa_id)
            raw_excerpt = context["text"]
            excerpt = raw_excerpt.strip() if isinstance(raw_excerpt, str) else ""
            if excerpt and excerpt not in entry["excerpts"]:
                entry["excerpts"].append(excerpt)

    documents = list(merged.values())
    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Unexpected merged ScholarQA-Multi document count: "
            f"{len(documents)} != {EXPECTED_DOCUMENTS}"
        )
    ids = [entry["id"] for entry in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("ScholarQA-Multi merged document IDs are not unique")
    return documents


def _render_document(entry: dict[str, Any]) -> str:
    authors = ", ".join(entry["authors"]) if entry["authors"] else "Not provided"
    year = entry["year"] if entry["year"] not in (None, "") else "Not provided"
    source_url = entry["source_url"] or "Not provided"
    lines = [
        f"# {entry['title']}",
        "",
        f"- Authors: {authors}",
        f"- Year: {year}",
        f"- Source URL: {source_url}",
        f"- ScholarQA-Multi subjects: {', '.join(entry['subjects'])}",
    ]
    if entry["alternate_titles"]:
        lines.append(f"- Alternate titles: {'; '.join(entry['alternate_titles'])}")
    for index, excerpt in enumerate(entry["excerpts"], 1):
        lines.extend(["", f"## Reference excerpt {index}", "", excerpt])
    return "\n".join(lines).strip() + "\n"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _download_source(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".json.part")
    try:
        with requests.get(SCHOLARQA_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                temp_path.open("wb") as handle,
                tqdm(
                    desc="Downloading ScholarQA-Multi",
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        if _sha256_file(temp_path) != SCHOLARQA_SHA256:
            raise ValueError("Downloaded ScholarQA-Multi file checksum mismatch")
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_dataset(source_path: Path, dataset_dir: Path) -> None:
    records = load_json(source_path)
    valid_records = select_valid_records(records)
    documents = collect_documents(valid_records)
    build_dir = dataset_dir.parent / f".{dataset_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        _link_or_copy(source_path, build_dir / "human_answers.json")
        write_json(
            build_dir / "scholarqa_multi_valid_101.json",
            _normalize_missing_context_texts(valid_records),
        )
        manifest: list[dict[str, Any]] = []
        for document in documents:
            relative_path = f"reference_documents/{document['id']}.txt"
            destination = build_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(_render_document(document), encoding="utf-8")
            manifest.append(
                {
                    key: value
                    for key, value in document.items()
                    if key != "excerpts"
                }
                | {
                    "relative_path": relative_path,
                    "excerpt_count": len(document["excerpts"]),
                    "size_bytes": destination.stat().st_size,
                }
            )
        write_jsonl(build_dir / "documents.jsonl", manifest)
        write_json(
            build_dir / "dataset_info.json",
            {
                "dataset": "ScholarQA-Multi",
                "revision": SCHOLARQA_REVISION,
                "source_sha256": SCHOLARQA_SHA256,
                "license": "ODC-BY",
                "selection": "all_records_without_out_of_range_citation_indices",
                "original_question_count": len(records),
                "valid_question_count": len(valid_records),
                "excluded_question_ids": sorted(EXPECTED_INVALID_QA_IDS),
                "subject_counts": EXPECTED_SUBJECT_COUNTS,
                "document_count": len(manifest),
                "corpus_format": "merged_official_citation_contexts_as_txt",
            },
        )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        build_dir.replace(dataset_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def verify_scholarqa_multi_download(dataset_dir: Path) -> bool:
    try:
        source_records = load_json(dataset_dir / "human_answers.json")
        valid_records = load_json(dataset_dir / "scholarqa_multi_valid_101.json")
        manifest = load_jsonl(dataset_dir / "documents.jsonl")
        with (dataset_dir / "dataset_info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        selected = select_valid_records(source_records)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        _sha256_file(dataset_dir / "human_answers.json") != SCHOLARQA_SHA256
        or info.get("revision") != SCHOLARQA_REVISION
        or info.get("valid_question_count") != EXPECTED_VALID_QAS
        or info.get("document_count") != EXPECTED_DOCUMENTS
        or valid_records != _normalize_missing_context_texts(selected)
        or len(manifest) != EXPECTED_DOCUMENTS
    ):
        return False
    ids = [entry.get("id") for entry in manifest]
    paths = [entry.get("relative_path") for entry in manifest]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        return False
    for entry in manifest:
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            return False
        path = dataset_dir / relative_path
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            return False
    return True


def download_scholarqa_multi_valid_101(
    output_dir: Path,
    dataset_name: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download and materialize the fixed valid ScholarQA-Multi scope."""
    dataset_dir = output_dir / dataset_name
    if not force and verify_scholarqa_multi_download(dataset_dir):
        print(f"{dataset_name} already exists and passed verification")
        return True
    shared_dir = output_dir / "ScholarQABench"
    source_path = shared_dir / f"human_answers-{SCHOLARQA_REVISION}.json"
    if force or not source_path.is_file() or _sha256_file(source_path) != SCHOLARQA_SHA256:
        _download_source(source_path)
    _write_dataset(source_path, dataset_dir)
    if verify and not verify_scholarqa_multi_download(dataset_dir):
        return False
    print(
        f"✓ {dataset_name}: {EXPECTED_VALID_QAS} valid QA and "
        f"{EXPECTED_DOCUMENTS} merged reference TXT documents ready at {dataset_dir}"
    )
    return True


def prepare_scholarqa_multi_valid_101(
    input_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Copy the verified fixed scope into the prepared dataset directory."""
    if not verify_scholarqa_multi_download(input_dir):
        raise ValueError(f"ScholarQAMultiValid101 failed verification: {input_dir}")
    manifest = load_jsonl(input_dir / "documents.jsonl")
    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        for filename in (
            "scholarqa_multi_valid_101.json",
            "documents.jsonl",
            "dataset_info.json",
        ):
            _link_or_copy(input_dir / filename, build_dir / filename)
        for entry in manifest:
            _link_or_copy(
                input_dir / entry["relative_path"],
                build_dir / entry["relative_path"],
            )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        build_dir.replace(output_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)
    return {
        "dataset": "ScholarQAMultiValid101",
        "revision": SCHOLARQA_REVISION,
        "original_total_qas": EXPECTED_TOTAL_QAS,
        "sampled_total_qas": EXPECTED_VALID_QAS,
        "excluded_qa_ids": sorted(EXPECTED_INVALID_QA_IDS),
        "sampled_num_docs": len(manifest),
        "subjects": EXPECTED_SUBJECT_COUNTS,
        "is_full": False,
    }
