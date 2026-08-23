"""Download and prepare the first 100 MDA-QA examples.

The public MDA-QA file contains 6,804 multi-document QA records. This module
creates a fixed benchmark subset consisting of records 0 through 99 and the
143 unique arXiv papers referenced by their ``support`` fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


MDAQA_URL = (
    "https://huggingface.co/datasets/YeloDriver/MDAQA/resolve/"
    "7c4a4c374e3ff8298e9694648e0d793197a30814/"
    "MDA-QA.json"
)
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"

EXPECTED_TOTAL_QAS = 6804
EXPECTED_SUBSET_QAS = 100
EXPECTED_DOCUMENTS = 143
ARXIV_REQUEST_INTERVAL_SECONDS = 1.0
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"MDA-QA file must contain a JSON list of objects: {path}")
    return data


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def select_first_100(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(records) != EXPECTED_TOTAL_QAS:
        raise ValueError(
            f"Unexpected MDA-QA snapshot size: {len(records)} != {EXPECTED_TOTAL_QAS}"
        )
    selected = records[:EXPECTED_SUBSET_QAS]
    actual_ids = [record.get("id") for record in selected]
    expected_ids = list(range(EXPECTED_SUBSET_QAS))
    if actual_ids != expected_ids:
        raise ValueError(
            "MDA-QA first 100 records no longer correspond to integer IDs 0 through 99"
        )

    for record in selected:
        qa_id = record["id"]
        question = record.get("question")
        answer = record.get("answer")
        support = record.get("support")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"MDA-QA record {qa_id} has an empty question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"MDA-QA record {qa_id} has an empty answer")
        if not isinstance(support, list) or len(support) < 2:
            raise ValueError(f"MDA-QA record {qa_id} has invalid support")
        if len(support) != len(set(support)):
            raise ValueError(f"MDA-QA record {qa_id} has duplicate support IDs")
        invalid_ids = [paper_id for paper_id in support if not _is_arxiv_id(paper_id)]
        if invalid_ids:
            raise ValueError(
                f"MDA-QA record {qa_id} has invalid arXiv IDs: {invalid_ids}"
            )
    return selected


def _is_arxiv_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ARXIV_ID_RE.fullmatch(value.strip()))


def collect_paper_ids(records: list[dict[str, Any]]) -> list[str]:
    paper_ids = sorted(
        {str(paper_id).strip() for record in records for paper_id in record["support"]}
    )
    if len(paper_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"Unexpected MDA-QA first-100 document count: "
            f"{len(paper_ids)} != {EXPECTED_DOCUMENTS}"
        )
    return paper_ids


def is_pdf_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_json(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(MDAQA_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                temp_path.open("wb") as handle,
                tqdm(
                    desc="Downloading MDA-QA.json",
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        load_json_records(temp_path)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            return max(float(retry_after), 1.0)
    return float(2**attempt)


def _download_arxiv_pdf(
    session: requests.Session, paper_id: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".pdf.part")
    url = ARXIV_PDF_URL.format(paper_id=paper_id)
    last_error: Exception | None = None

    for attempt in range(3):
        response: requests.Response | None = None
        try:
            response = session.get(url, stream=True, timeout=(30, 180))
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
            if not is_pdf_file(temp_path):
                raise ValueError(f"arXiv returned non-PDF content for {paper_id}")
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

    raise RuntimeError(f"Failed to download arXiv PDF {paper_id}: {last_error}")


def _link_or_copy(source: Path, destination: Path, *, pdf: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            pdf
            and is_pdf_file(destination)
            and destination.stat().st_size == source.stat().st_size
        ):
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _expected_pdf_names(entries: list[dict[str, Any]]) -> set[str]:
    return {f"{entry['id']}.pdf" for entry in entries}


def _validate_pdf_directory(pdf_dir: Path, entries: list[dict[str, Any]]) -> bool:
    if not pdf_dir.is_dir():
        return False
    expected_names = _expected_pdf_names(entries)
    actual_names = {path.name for path in pdf_dir.glob("*.pdf")}
    if actual_names != expected_names:
        return False
    for entry in entries:
        pdf_path = pdf_dir / f"{entry['id']}.pdf"
        if not is_pdf_file(pdf_path):
            return False
        expected_hash = str(entry.get("sha256") or "")
        if expected_hash and sha256_file(pdf_path) != expected_hash:
            return False
    return True


def verify_mdaqa_download(dataset_dir: Path) -> bool:
    qa_path = dataset_dir / "MDA-QA.json"
    manifest_path = dataset_dir / "documents.jsonl"
    if not qa_path.is_file() or not manifest_path.is_file():
        return False
    try:
        selected = select_first_100(load_json_records(qa_path))
        expected_ids = collect_paper_ids(selected)
        entries = load_jsonl(manifest_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    manifest_ids = [str(entry.get("id") or "") for entry in entries]
    if manifest_ids != expected_ids:
        return False
    return _validate_pdf_directory(dataset_dir / "pdfs", entries)


def download_mdaqa_first_100(
    output_dir: Path,
    dataset_name: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download the fixed first-100 MDA-QA subset and its 143 arXiv PDFs."""
    shared_dir = output_dir / "MDAQA"
    qa_cache = shared_dir / "MDA-QA.json"
    pdf_cache = shared_dir / "pdf_cache"
    dataset_dir = output_dir / dataset_name

    if dataset_dir.exists() and not force and verify_mdaqa_download(dataset_dir):
        print(f"{dataset_name} already exists and passed verification")
        return True

    if force or not qa_cache.is_file():
        print("Downloading MDA-QA.json...")
        _download_json(qa_cache)

    records = load_json_records(qa_cache)
    selected = select_first_100(records)
    paper_ids = collect_paper_ids(selected)

    session = requests.Session()
    session.headers["User-Agent"] = (
        "ov-llm-wiki MDA-QA downloader "
        "(https://github.com/sponge225/ov-llm-wiki)"
    )
    downloaded_any = False
    for index, paper_id in enumerate(paper_ids, 1):
        cached_pdf = pdf_cache / f"{paper_id}.pdf"
        if not force and is_pdf_file(cached_pdf):
            continue
        if downloaded_any:
            time.sleep(ARXIV_REQUEST_INTERVAL_SECONDS)
        print(f"[{index}/{len(paper_ids)}] Downloading arXiv paper {paper_id}")
        _download_arxiv_pdf(session, paper_id, cached_pdf)
        downloaded_any = True

    entries = []
    for paper_id in paper_ids:
        cached_pdf = pdf_cache / f"{paper_id}.pdf"
        if not is_pdf_file(cached_pdf):
            raise FileNotFoundError(f"Missing or invalid cached PDF: {cached_pdf}")
        entries.append(
            {
                "id": paper_id,
                "pdf_url": ARXIV_PDF_URL.format(paper_id=paper_id),
                "sha256": sha256_file(cached_pdf),
            }
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(qa_cache, dataset_dir / "MDA-QA.json")
    write_jsonl(dataset_dir / "documents.jsonl", entries)
    for entry in entries:
        paper_id = entry["id"]
        _link_or_copy(
            pdf_cache / f"{paper_id}.pdf",
            dataset_dir / "pdfs" / f"{paper_id}.pdf",
            pdf=True,
        )

    if verify and not verify_mdaqa_download(dataset_dir):
        return False
    print(f"✓ {dataset_name}: 100 QA and {len(entries)} PDFs ready at {dataset_dir}")
    return True


def prepare_mdaqa_first_100(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Create the prepared fixed subset consumed by ``MDAQAAdapter``."""
    qa_path = input_dir / "MDA-QA.json"
    manifest_path = input_dir / "documents.jsonl"
    source_pdf_dir = input_dir / "pdfs"
    if not qa_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete MDAQAFirst100 download at {input_dir}")

    records = load_json_records(qa_path)
    selected = select_first_100(records)
    expected_ids = collect_paper_ids(selected)
    entries = load_jsonl(manifest_path)
    manifest_ids = [str(entry.get("id") or "") for entry in entries]
    if manifest_ids != expected_ids:
        raise ValueError(
            "MDAQAFirst100 manifest does not match the selected QA support IDs"
        )
    if not _validate_pdf_directory(source_pdf_dir, entries):
        raise ValueError(
            f"MDAQAFirst100 PDF directory failed verification: {source_pdf_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "mdaqa_first_100.json", selected)
    write_jsonl(output_dir / "documents.jsonl", entries)
    for entry in entries:
        paper_id = entry["id"]
        _link_or_copy(
            source_pdf_dir / f"{paper_id}.pdf",
            output_dir / "pdfs" / f"{paper_id}.pdf",
            pdf=True,
        )

    return {
        "dataset": "MDAQAFirst100",
        "original_total_qas": len(records),
        "sampled_total_qas": len(selected),
        "sampled_num_docs": len(entries),
        "selection": "first_100_in_source_order",
        "qa_id_range": [0, 99],
        "is_full": False,
    }
