"""Download and prepare the fixed EnterpriseRAG-Bench selected-80 scope.

The scope contains every Project Related, Conflicting Info, and Completeness
question from the official v1.0.0 release.  Only the physical TXT documents
referenced by those questions are extracted from the full document archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests
from tqdm import tqdm


ENTERPRISE_RAG_VERSION = "v1.0.0"
RELEASE_ROOT = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/"
    f"{ENTERPRISE_RAG_VERSION}"
)
QUESTIONS_URL = f"{RELEASE_ROOT}/questions.jsonl"
DOCUMENTS_URL = f"{RELEASE_ROOT}/all_documents.zip"
QUESTIONS_SHA256 = "f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905"
DOCUMENTS_SHA256 = "9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd"
QUESTIONS_SIZE_BYTES = 764_927
DOCUMENTS_SIZE_BYTES = 1_256_181_062

TARGET_CATEGORIES = (
    "project_related",
    "conflicting_info",
    "completeness",
)
EXPECTED_CATEGORY_COUNTS = {
    "project_related": 40,
    "conflicting_info": 20,
    "completeness": 20,
}
EXPECTED_TOTAL_QAS = 500
EXPECTED_SELECTED_QAS = 80
EXPECTED_LOGICAL_DOCUMENTS = 322
EXPECTED_PHYSICAL_DOCUMENTS = 323
EXPECTED_DOCUMENT_OCCURRENCES = 339
CONFLICT_QUESTION_ID = "qst_0413"
CONFLICT_DOCUMENT_ID = "dsid_6df52fdb96ae4edcb76464738bca3340"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_matches(path: Path, size_bytes: int, sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == size_bytes
        and sha256_file(path) == sha256
    )


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
    return float(2 ** attempt)


def _download_artifact(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    """Download an immutable release asset, resuming a partial file when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    session = requests.Session()
    session.headers["User-Agent"] = (
        "ov-llm-wiki EnterpriseRAG-Bench downloader "
        "(https://github.com/sponge225/ov-llm-wiki)"
    )
    last_error: Exception | None = None

    for attempt in range(3):
        response: requests.Response | None = None
        try:
            current_size = partial.stat().st_size if partial.is_file() else 0
            if current_size > expected_size:
                partial.unlink()
                current_size = 0
            if current_size == expected_size:
                if sha256_file(partial) == expected_sha256:
                    partial.replace(destination)
                    return
                partial.unlink()
                current_size = 0
            headers = {"Range": f"bytes={current_size}-"} if current_size else {}
            response = session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            )
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
            response.raise_for_status()

            resumed = current_size > 0 and response.status_code == 206
            if not resumed:
                current_size = 0
            mode = "ab" if resumed else "wb"
            with (
                partial.open(mode) as handle,
                tqdm(
                    desc=f"Downloading {label}",
                    total=expected_size,
                    initial=current_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))

            if partial.stat().st_size != expected_size:
                raise ValueError(
                    f"{label} size mismatch: {partial.stat().st_size} != {expected_size}"
                )
            if sha256_file(partial) != expected_sha256:
                partial.unlink()
                raise ValueError(f"{label} checksum mismatch")
            partial.replace(destination)
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            status_code = response.status_code if response is not None else None
            retryable = (
                isinstance(exc, (OSError, ValueError))
                or status_code in _RETRYABLE_STATUS_CODES
                or status_code is None
            )
            if attempt == 2 or not retryable:
                break
            time.sleep(_retry_delay(response, attempt))
        finally:
            if response is not None:
                response.close()
    raise RuntimeError(f"Failed to download {label}: {last_error}")


def select_questions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_TOTAL_QAS:
        raise ValueError(
            f"Unexpected EnterpriseRAG-Bench QA count: {len(records)}"
        )
    question_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question_id = str(record.get("question_id") or "").strip()
        category = str(record.get("question_type") or "").strip()
        question = str(record.get("question") or "").strip()
        gold_answer = str(record.get("gold_answer") or "").strip()
        source_types = record.get("source_types")
        expected_doc_ids = record.get("expected_doc_ids")
        answer_facts = record.get("answer_facts")
        if (
            not question_id
            or question_id in question_ids
            or not category
            or not question
            or not gold_answer
            or not isinstance(source_types, list)
            or not isinstance(expected_doc_ids, list)
            or not isinstance(answer_facts, list)
            or not answer_facts
            or any(not isinstance(value, str) or not value.strip() for value in answer_facts)
        ):
            raise ValueError(f"Invalid EnterpriseRAG-Bench QA at row {index}")
        question_ids.add(question_id)
        if category in TARGET_CATEGORIES:
            if (
                not source_types
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in source_types
                )
                or not expected_doc_ids
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in expected_doc_ids
                )
            ):
                raise ValueError(
                    f"Selected EnterpriseRAG-Bench QA has no valid sources at row {index}"
                )
            selected.append(record)

    counts = Counter(str(record["question_type"]) for record in selected)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS or len(selected) != EXPECTED_SELECTED_QAS:
        raise ValueError(f"Unexpected selected category counts: {dict(counts)}")
    logical_ids = {
        document_id
        for record in selected
        for document_id in record["expected_doc_ids"]
    }
    occurrences = sum(len(record["expected_doc_ids"]) for record in selected)
    if (
        len(logical_ids) != EXPECTED_LOGICAL_DOCUMENTS
        or occurrences != EXPECTED_DOCUMENT_OCCURRENCES
    ):
        raise ValueError(
            "Unexpected EnterpriseRAG-Bench selected document coverage: "
            f"{len(logical_ids)} logical IDs, {occurrences} occurrences"
        )
    repeated_rows = [
        record
        for record in selected
        if len(record["expected_doc_ids"]) != len(set(record["expected_doc_ids"]))
    ]
    if (
        len(repeated_rows) != 1
        or repeated_rows[0]["question_id"] != CONFLICT_QUESTION_ID
        or repeated_rows[0]["expected_doc_ids"]
        != [CONFLICT_DOCUMENT_ID, CONFLICT_DOCUMENT_ID]
    ):
        raise ValueError("The intentional duplicate conflict document changed upstream")
    return selected


def _archive_document_id(info: zipfile.ZipInfo) -> str | None:
    path = PurePosixPath(info.filename)
    if info.is_dir() or path.suffix.casefold() != ".txt":
        return None
    filename = path.name
    document_id, separator, _ = filename.partition("__")
    if not separator or not document_id.startswith("dsid_"):
        return None
    return document_id


def _safe_archive_path(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
        raise ValueError(f"Unsafe path in EnterpriseRAG-Bench archive: {info.filename}")
    return path


def _physical_id(logical_id: str, archive_path: str) -> str:
    suffix = hashlib.sha256(archive_path.encode("utf-8")).hexdigest()[:12]
    return f"{logical_id}__{suffix}"


def _extract_selected_documents(
    archive_path: Path,
    selected_questions: list[dict[str, Any]],
    destination_root: Path,
) -> list[dict[str, Any]]:
    logical_ids = {
        document_id
        for record in selected_questions
        for document_id in record["expected_doc_ids"]
    }
    matches: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            document_id = _archive_document_id(info)
            if document_id in logical_ids:
                _safe_archive_path(info)
                matches[document_id].append(info)

        missing = sorted(logical_ids - set(matches))
        unexpected_multiples = {
            document_id: len(infos)
            for document_id, infos in matches.items()
            if len(infos) != (2 if document_id == CONFLICT_DOCUMENT_ID else 1)
        }
        if missing or unexpected_multiples:
            raise ValueError(
                "EnterpriseRAG-Bench archive does not match selected QA: "
                f"missing={missing[:10]} multiples={unexpected_multiples}"
            )

        selected_infos = sorted(
            (info for infos in matches.values() for info in infos),
            key=lambda info: info.filename,
        )
        if len(selected_infos) != EXPECTED_PHYSICAL_DOCUMENTS:
            raise ValueError(
                f"Expected {EXPECTED_PHYSICAL_DOCUMENTS} physical TXT files, "
                f"got {len(selected_infos)}"
            )

        manifest: list[dict[str, Any]] = []
        for info in tqdm(selected_infos, desc="Extracting selected documents", unit="doc"):
            logical_id = _archive_document_id(info)
            if logical_id is None:
                raise ValueError(f"Invalid selected archive entry: {info.filename}")
            archive_member = _safe_archive_path(info)
            relative_path = PurePosixPath("reference_documents") / archive_member
            destination = destination_root.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info, "r") as source, destination.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
            if destination.stat().st_size != info.file_size:
                raise ValueError(f"Extracted TXT size mismatch: {info.filename}")

            path_parts = archive_member.parts
            source_type = path_parts[0] if len(path_parts) > 1 else "unknown"
            title_slug = archive_member.stem.partition("__")[2]
            manifest.append(
                {
                    "id": _physical_id(logical_id, info.filename),
                    "logical_doc_id": logical_id,
                    "source_type": source_type,
                    "title_slug": title_slug,
                    "archive_path": info.filename,
                    "relative_path": relative_path.as_posix(),
                    "size_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": digest.hexdigest(),
                }
            )
    return manifest


def _write_dataset(
    questions_path: Path,
    archive_path: Path,
    dataset_dir: Path,
) -> None:
    records = load_jsonl(questions_path)
    selected = select_questions(records)
    build_dir = dataset_dir.parent / f".{dataset_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        shutil.copy2(questions_path, build_dir / "questions.jsonl")
        write_jsonl(build_dir / "questions_selected_80.jsonl", selected)
        manifest = _extract_selected_documents(archive_path, selected, build_dir)
        write_jsonl(build_dir / "documents.jsonl", manifest)
        physical_counts = Counter(
            str(entry["logical_doc_id"]) for entry in manifest
        )
        write_json(
            build_dir / "dataset_info.json",
            {
                "dataset": "EnterpriseRAG-Bench",
                "release": ENTERPRISE_RAG_VERSION,
                "license": "MIT",
                "selection": list(TARGET_CATEGORIES),
                "question_count": len(selected),
                "category_counts": EXPECTED_CATEGORY_COUNTS,
                "logical_document_count": len(physical_counts),
                "physical_document_count": len(manifest),
                "document_occurrence_count": EXPECTED_DOCUMENT_OCCURRENCES,
                "intentional_duplicate": {
                    "question_id": CONFLICT_QUESTION_ID,
                    "logical_doc_id": CONFLICT_DOCUMENT_ID,
                    "physical_document_count": physical_counts[CONFLICT_DOCUMENT_ID],
                },
                "questions_sha256": QUESTIONS_SHA256,
                "documents_archive_sha256": DOCUMENTS_SHA256,
                "documents_archive_size_bytes": DOCUMENTS_SIZE_BYTES,
                "selected_text_bytes": sum(entry["size_bytes"] for entry in manifest),
            },
        )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        build_dir.replace(dataset_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def verify_enterprise_rag_bench_download(dataset_dir: Path) -> bool:
    try:
        questions_path = dataset_dir / "questions.jsonl"
        selected_path = dataset_dir / "questions_selected_80.jsonl"
        source_records = load_jsonl(questions_path)
        selected_records = load_jsonl(selected_path)
        manifest = load_jsonl(dataset_dir / "documents.jsonl")
        with (dataset_dir / "dataset_info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        expected_selected = select_questions(source_records)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    ids = [str(entry.get("id") or "") for entry in manifest]
    paths = [str(entry.get("relative_path") or "") for entry in manifest]
    logical_counts = Counter(str(entry.get("logical_doc_id") or "") for entry in manifest)
    if (
        not _artifact_matches(questions_path, QUESTIONS_SIZE_BYTES, QUESTIONS_SHA256)
        or selected_records != expected_selected
        or len(manifest) != EXPECTED_PHYSICAL_DOCUMENTS
        or len(set(logical_counts)) != EXPECTED_LOGICAL_DOCUMENTS
        or logical_counts.get(CONFLICT_DOCUMENT_ID) != 2
        or any(
            count != (2 if document_id == CONFLICT_DOCUMENT_ID else 1)
            for document_id, count in logical_counts.items()
        )
        or len(ids) != len(set(ids))
        or len(paths) != len(set(paths))
        or info.get("release") != ENTERPRISE_RAG_VERSION
        or info.get("question_count") != EXPECTED_SELECTED_QAS
        or info.get("physical_document_count") != EXPECTED_PHYSICAL_DOCUMENTS
    ):
        return False

    for entry in manifest:
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            return False
        path = dataset_dir / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("size_bytes")
            or sha256_file(path) != entry.get("sha256")
        ):
            return False
    return True


def download_enterprise_rag_bench_selected_80(
    output_dir: Path,
    dataset_name: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download the release and materialize the fixed selected-80 scope."""
    dataset_dir = output_dir / dataset_name
    if not force and verify_enterprise_rag_bench_download(dataset_dir):
        print(f"{dataset_name} already exists and passed verification")
        return True

    cache_dir = output_dir / "EnterpriseRAGBench" / ENTERPRISE_RAG_VERSION
    questions_cache = cache_dir / "questions.jsonl"
    archive_cache = cache_dir / "all_documents.zip"
    if force and questions_cache.exists():
        questions_cache.unlink()
    if force and archive_cache.exists():
        archive_cache.unlink()
    if force:
        for cache_path in (questions_cache, archive_cache):
            partial = cache_path.with_name(cache_path.name + ".part")
            if partial.exists():
                partial.unlink()
    if not _artifact_matches(questions_cache, QUESTIONS_SIZE_BYTES, QUESTIONS_SHA256):
        _download_artifact(
            QUESTIONS_URL,
            questions_cache,
            expected_size=QUESTIONS_SIZE_BYTES,
            expected_sha256=QUESTIONS_SHA256,
            label="EnterpriseRAG-Bench questions.jsonl",
        )
    if not _artifact_matches(archive_cache, DOCUMENTS_SIZE_BYTES, DOCUMENTS_SHA256):
        _download_artifact(
            DOCUMENTS_URL,
            archive_cache,
            expected_size=DOCUMENTS_SIZE_BYTES,
            expected_sha256=DOCUMENTS_SHA256,
            label="EnterpriseRAG-Bench all_documents.zip",
        )

    _write_dataset(questions_cache, archive_cache, dataset_dir)
    if verify and not verify_enterprise_rag_bench_download(dataset_dir):
        return False
    print(
        f"✓ {dataset_name}: {EXPECTED_SELECTED_QAS} QA and "
        f"{EXPECTED_PHYSICAL_DOCUMENTS} selected TXT documents ready at {dataset_dir}"
    )
    return True


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


def prepare_enterprise_rag_bench_selected_80(
    input_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Copy the verified selected scope into the prepared dataset directory."""
    if not verify_enterprise_rag_bench_download(input_dir):
        raise ValueError(
            f"EnterpriseRAGBenchSelected80 failed verification: {input_dir}"
        )
    manifest = load_jsonl(input_dir / "documents.jsonl")
    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        for filename in (
            "questions_selected_80.jsonl",
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
        "dataset": "EnterpriseRAGBenchSelected80",
        "release": ENTERPRISE_RAG_VERSION,
        "categories": list(TARGET_CATEGORIES),
        "category_counts": EXPECTED_CATEGORY_COUNTS,
        "sampled_total_qas": EXPECTED_SELECTED_QAS,
        "logical_document_count": EXPECTED_LOGICAL_DOCUMENTS,
        "sampled_num_docs": len(manifest),
        "is_full": False,
    }
