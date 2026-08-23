"""Download and prepare the fixed MuDABench Simple and Complex scopes.

Both QA scopes contain 166 released rows and reference the same complete set of
589 financial PDFs. The PDFs are downloaded once into a shared cache and then
hard-linked into each scope-specific raw/prepared dataset when possible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests
from tqdm import tqdm


MUDABENCH_REVISION = "af2360876c0b8789e2ca1af9d648f9370eb52600"
MUDABENCH_RESOLVE_ROOT = (
    "https://huggingface.co/datasets/Zhanli-Li/MuDABench/resolve/"
    f"{MUDABENCH_REVISION}/data"
)
MUDABENCH_TREE_URL = (
    "https://huggingface.co/api/datasets/Zhanli-Li/MuDABench/tree/"
    f"{MUDABENCH_REVISION}/data/pdf"
)

SCOPES = ("simple", "complex")
EXPECTED_QAS = 166
EXPECTED_UNIQUE_QUESTION_IDS = 164
EXPECTED_DOCUMENTS = 589
EXPECTED_DOCUMENT_OCCURRENCES = 2457
EXPECTED_DOCUMENT_GROUPS = 154
EXPECTED_TOTAL_PDF_BYTES = 4_057_308_684
EXPECTED_QA_SHA256 = {
    "simple": "194ace4fd7cfd811f8df9a2efb88c48916281669ac48561093822f593d2e747d",
    "complex": "39a7cb280773c840fbe2c29d66d6a92d03e07c81a7f1a86f24ab6a9266965b9f",
}
EXPECTED_DUPLICATE_ROWS = {
    "simple": [[140, 159], [141, 160]],
    "complex": [[140, 159], [141, 160]],
}
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"MuDABench file must be a JSON list of objects: {path}")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pdf_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _validate_uuid(value: Any, *, location: str) -> str:
    document_id = str(value or "").strip()
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError(f"Invalid MuDABench document ID at {location}: {value}") from exc
    if str(parsed) != document_id:
        raise ValueError(f"Non-canonical MuDABench document ID at {location}: {value}")
    return document_id


def validate_qa_records(
    records: list[dict[str, Any]], scope: str
) -> list[str]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown MuDABench scope: {scope}")
    if len(records) != EXPECTED_QAS:
        raise ValueError(
            f"Unexpected MuDABench {scope} QA count: {len(records)} != {EXPECTED_QAS}"
        )

    document_ids: set[str] = set()
    document_occurrences = 0
    document_groups: set[tuple[str, ...]] = set()
    rows_by_question_id: dict[str, list[int]] = defaultdict(list)
    for row_index, record in enumerate(records):
        location = f"{scope}[{row_index}]"
        question_id = str(record.get("question_id") or "").strip()
        question = record.get("question")
        final_answer = record.get("final_answer")
        source_answer = record.get("source_answer")
        metadata = record.get("metadata")
        if not question_id or not isinstance(question, str) or not question.strip():
            raise ValueError(f"MuDABench record has no question ID or question: {location}")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError(f"MuDABench record has an empty final answer: {location}")
        if not isinstance(source_answer, list) or not source_answer or any(
            not isinstance(item, str) or not item.strip() for item in source_answer
        ):
            raise ValueError(f"MuDABench record has invalid source_answer: {location}")
        if not isinstance(metadata, list) or not metadata:
            raise ValueError(f"MuDABench record has no document metadata: {location}")

        row_document_ids: list[str] = []
        for metadata_index, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise ValueError(
                    f"MuDABench metadata is not an object: {location}[{metadata_index}]"
                )
            document_id = _validate_uuid(
                item.get("id"), location=f"{location}.metadata[{metadata_index}]"
            )
            row_document_ids.append(document_id)
        if len(row_document_ids) != len(set(row_document_ids)):
            raise ValueError(f"MuDABench record repeats a document ID: {location}")

        rows_by_question_id[question_id].append(row_index)
        document_ids.update(row_document_ids)
        document_occurrences += len(row_document_ids)
        document_groups.add(tuple(sorted(row_document_ids)))

    duplicate_rows = sorted(
        indices for indices in rows_by_question_id.values() if len(indices) > 1
    )
    if duplicate_rows != EXPECTED_DUPLICATE_ROWS[scope]:
        raise ValueError(
            f"Unexpected MuDABench {scope} duplicate rows: {duplicate_rows}"
        )
    for first_index, second_index in duplicate_rows:
        if records[first_index] != records[second_index]:
            raise ValueError(
                f"MuDABench {scope} duplicate question IDs do not contain identical rows"
            )
    if len(rows_by_question_id) != EXPECTED_UNIQUE_QUESTION_IDS:
        raise ValueError(
            f"Unexpected MuDABench {scope} unique question ID count: "
            f"{len(rows_by_question_id)}"
        )
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Unexpected MuDABench {scope} document count: {len(document_ids)}"
        )
    if document_occurrences != EXPECTED_DOCUMENT_OCCURRENCES:
        raise ValueError(
            f"Unexpected MuDABench {scope} document occurrence count: "
            f"{document_occurrences}"
        )
    if len(document_groups) != EXPECTED_DOCUMENT_GROUPS:
        raise ValueError(
            f"Unexpected MuDABench {scope} document-group count: {len(document_groups)}"
        )
    return sorted(document_ids)


def _download_json(scope: str, destination: Path) -> None:
    url = f"{MUDABENCH_RESOLVE_ROOT}/{scope}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".json.part")
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        handle.write(chunk)
        if sha256_file(temp_path) != EXPECTED_QA_SHA256[scope]:
            raise ValueError(f"Downloaded MuDABench {scope} JSON checksum mismatch")
        validate_qa_records(load_json(temp_path), scope)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _fetch_document_manifest(expected_ids: list[str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    next_url: str | None = MUDABENCH_TREE_URL
    params: dict[str, Any] | None = {
        "recursive": "false",
        "expand": "true",
        "limit": 50,
    }
    visited_urls: set[str] = set()
    while next_url:
        if next_url in visited_urls or len(visited_urls) >= 20:
            raise ValueError("Invalid MuDABench Hugging Face tree pagination")
        visited_urls.add(next_url)
        response = requests.get(next_url, params=params, timeout=(30, 180))
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list) or not all(
            isinstance(value, dict) for value in page
        ):
            raise ValueError("Unexpected MuDABench Hugging Face tree response")
        values.extend(page)
        next_url = response.links.get("next", {}).get("url")
        params = None

    entries: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or value.get("type") != "file":
            continue
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value.endswith(".pdf"):
            continue
        source_path = PurePosixPath(path_value)
        document_id = _validate_uuid(source_path.stem, location=path_value)
        lfs = value.get("lfs")
        if not isinstance(lfs, dict):
            raise ValueError(f"MuDABench PDF has no LFS metadata: {path_value}")
        sha256 = str(lfs.get("oid") or "").strip()
        size_bytes = lfs.get("size")
        if len(sha256) != 64 or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError(f"Invalid MuDABench PDF metadata: {path_value}")
        entries.append(
            {
                "id": document_id,
                "filename": source_path.name,
                "source_path": path_value,
                "pdf_url": (
                    "https://huggingface.co/datasets/Zhanli-Li/MuDABench/resolve/"
                    f"{MUDABENCH_REVISION}/{path_value}"
                ),
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )

    entries.sort(key=lambda entry: entry["id"])
    manifest_ids = [entry["id"] for entry in entries]
    if manifest_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(manifest_ids))
        extra = sorted(set(manifest_ids) - set(expected_ids))
        raise ValueError(
            f"MuDABench PDF tree does not match QA metadata; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    if len(entries) != EXPECTED_DOCUMENTS:
        raise ValueError(f"Unexpected MuDABench PDF tree size: {len(entries)}")
    total_bytes = sum(entry["size_bytes"] for entry in entries)
    if total_bytes != EXPECTED_TOTAL_PDF_BYTES:
        raise ValueError(f"Unexpected MuDABench PDF byte total: {total_bytes}")
    return entries


def _pdf_matches_entry(path: Path, entry: dict[str, Any]) -> bool:
    return bool(
        is_pdf_file(path)
        and path.stat().st_size == entry.get("size_bytes")
        and sha256_file(path) == entry.get("sha256")
    )


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            return max(float(retry_after), 1.0)
    return float(2**attempt)


def _download_pdf(
    session: requests.Session, entry: dict[str, Any], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".pdf.part")
    last_error: Exception | None = None
    for attempt in range(3):
        response: requests.Response | None = None
        try:
            response = session.get(entry["pdf_url"], stream=True, timeout=(30, 300))
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
            if not _pdf_matches_entry(temp_path, entry):
                raise ValueError(
                    f"Downloaded MuDABench PDF failed validation: {entry['id']}"
                )
            temp_path.replace(destination)
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink()
            status_code = response.status_code if response is not None else None
            retryable = status_code in _RETRYABLE_STATUS_CODES or status_code is None
            if attempt == 2 or not retryable:
                break
            time.sleep(_retry_delay(response, attempt))
        finally:
            if response is not None:
                response.close()
    raise RuntimeError(f"Failed to download MuDABench PDF {entry['id']}: {last_error}")


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


def _validate_pdf_directory(pdf_dir: Path, entries: list[dict[str, Any]]) -> bool:
    if not pdf_dir.is_dir():
        return False
    expected_names = {entry["filename"] for entry in entries}
    actual_names = {path.name for path in pdf_dir.glob("*.pdf")}
    if actual_names != expected_names:
        return False
    return all(_pdf_matches_entry(pdf_dir / entry["filename"], entry) for entry in entries)


def verify_mudabench_download(dataset_dir: Path, scope: str) -> bool:
    if scope not in SCOPES:
        return False
    try:
        qa_path = dataset_dir / f"{scope}.json"
        records = load_json(qa_path)
        expected_ids = validate_qa_records(records, scope)
        entries = load_jsonl(dataset_dir / "documents.jsonl")
        with (dataset_dir / "dataset_info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    manifest_ids = [str(entry.get("id") or "") for entry in entries]
    if (
        sha256_file(qa_path) != EXPECTED_QA_SHA256[scope]
        or info.get("revision") != MUDABENCH_REVISION
        or info.get("scope") != scope
        or info.get("qa_count") != EXPECTED_QAS
        or info.get("document_count") != EXPECTED_DOCUMENTS
        or manifest_ids != expected_ids
    ):
        return False
    return _validate_pdf_directory(dataset_dir / "pdfs", entries)


def _write_scope_dataset(
    qa_cache: Path,
    entries: list[dict[str, Any]],
    pdf_cache: Path,
    dataset_dir: Path,
    scope: str,
) -> None:
    build_dir = dataset_dir.parent / f".{dataset_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        _link_or_copy(qa_cache, build_dir / f"{scope}.json")
        write_jsonl(build_dir / "documents.jsonl", entries)
        write_json(
            build_dir / "dataset_info.json",
            {
                "dataset": "MuDABench",
                "revision": MUDABENCH_REVISION,
                "scope": scope,
                "qa_count": EXPECTED_QAS,
                "unique_question_id_count": EXPECTED_UNIQUE_QUESTION_IDS,
                "exact_duplicate_rows": EXPECTED_DUPLICATE_ROWS[scope],
                "document_count": EXPECTED_DOCUMENTS,
                "document_occurrence_count": EXPECTED_DOCUMENT_OCCURRENCES,
                "document_group_count": EXPECTED_DOCUMENT_GROUPS,
                "total_pdf_bytes": EXPECTED_TOTAL_PDF_BYTES,
            },
        )
        for entry in entries:
            _link_or_copy(
                pdf_cache / entry["filename"],
                build_dir / "pdfs" / entry["filename"],
            )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        build_dir.replace(dataset_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def download_mudabench(
    output_dir: Path,
    dataset_name: str,
    scope: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download one QA scope with the complete shared 589-PDF corpus."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown MuDABench scope: {scope}")
    dataset_dir = output_dir / dataset_name
    if not force and verify_mudabench_download(dataset_dir, scope):
        print(f"{dataset_name} already exists and passed verification")
        return True

    shared_dir = output_dir / "MuDABench"
    qa_cache = shared_dir / f"{scope}.json"
    pdf_cache = shared_dir / "pdf_cache"
    if (
        force
        or not qa_cache.is_file()
        or sha256_file(qa_cache) != EXPECTED_QA_SHA256[scope]
    ):
        print(f"Downloading MuDABench {scope}.json...")
        _download_json(scope, qa_cache)

    records = load_json(qa_cache)
    expected_ids = validate_qa_records(records, scope)
    entries = _fetch_document_manifest(expected_ids)

    session = requests.Session()
    session.headers["User-Agent"] = (
        "ov-llm-wiki MuDABench downloader "
        "(https://github.com/sponge225/ov-llm-wiki)"
    )
    with tqdm(total=len(entries), desc="Preparing MuDABench PDFs", unit="pdf") as progress:
        for index, entry in enumerate(entries, 1):
            cached_pdf = pdf_cache / entry["filename"]
            if force or not _pdf_matches_entry(cached_pdf, entry):
                print(f"[{index}/{len(entries)}] Downloading {entry['filename']}")
                _download_pdf(session, entry, cached_pdf)
            progress.update(1)

    _write_scope_dataset(qa_cache, entries, pdf_cache, dataset_dir, scope)
    if verify and not verify_mudabench_download(dataset_dir, scope):
        return False
    print(
        f"✓ {dataset_name}: {EXPECTED_QAS} QA and the complete "
        f"{EXPECTED_DOCUMENTS}-PDF corpus ready at {dataset_dir}"
    )
    return True


def prepare_mudabench(
    input_dir: Path, output_dir: Path, *, scope: str
) -> dict[str, Any]:
    """Copy one verified fixed scope into the prepared dataset directory."""
    if not verify_mudabench_download(input_dir, scope):
        raise ValueError(f"MuDABench {scope} download failed verification: {input_dir}")
    entries = load_jsonl(input_dir / "documents.jsonl")
    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        for filename in (f"{scope}.json", "documents.jsonl", "dataset_info.json"):
            _link_or_copy(input_dir / filename, build_dir / filename)
        for entry in entries:
            _link_or_copy(
                input_dir / "pdfs" / entry["filename"],
                build_dir / "pdfs" / entry["filename"],
            )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        build_dir.replace(output_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)

    return {
        "dataset": f"MuDABench{scope.title()}",
        "revision": MUDABENCH_REVISION,
        "scope": scope,
        "sampled_total_qas": EXPECTED_QAS,
        "unique_question_id_count": EXPECTED_UNIQUE_QUESTION_IDS,
        "exact_duplicate_rows": EXPECTED_DUPLICATE_ROWS[scope],
        "sampled_num_docs": len(entries),
        "is_full": True,
    }
