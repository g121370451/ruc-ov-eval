"""PaperScope Summary download and preparation helpers.

The public dataset contains 600 summary records. Some records have an
``Error generating answer`` payload instead of a usable gold answer. This
module prepares two reproducible document scopes:

* ``valid``: papers referenced by usable summary records (57 papers)
* ``all``: papers referenced by every summary record (93 papers)

PDFs are cached once and hard-linked into each prepared dataset when possible.
"""

from __future__ import annotations

import difflib
import getpass
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import requests
from tqdm import tqdm


SUMMARY_URL = (
    "https://huggingface.co/datasets/Youxll/PaperScope/resolve/main/"
    "summary_corpus_1.jsonl"
)
CORPUS_URL = (
    "https://huggingface.co/datasets/Youxll/PaperScope/resolve/main/"
    "corpus_1.jsonl"
)
OPENREVIEW_LOGIN_URL = "https://api2.openreview.net/login"
OPENREVIEW_PDF_URL = "https://openreview.net/pdf?id={paper_id}"
OPENREVIEW_ATTACHMENT_URL = (
    "https://api2.openreview.net/attachment?id={paper_id}&name=pdf"
)

PROMPT_TYPES = ("trend", "gap", "results_comparison")
EXPECTED_RECORDS = 600
EXPECTED_VALID_RECORDS = 352
EXPECTED_DOCUMENT_COUNTS = {"valid": 57, "all": 93}
_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_valid_summary_record(record: dict[str, Any]) -> bool:
    answer = record.get("answer")
    return bool(
        isinstance(answer, str)
        and answer.strip()
        and not answer.lstrip().lower().startswith("error generating answer")
    )


def placeholder_title_from_url(url: str) -> str | None:
    prefix = "placeholder_link_for_"
    suffix = ".pdf"
    if not url.startswith(prefix) or not url.endswith(suffix):
        return None
    return url[len(prefix) : -len(suffix)].replace("_", " ").strip()


def _resolve_placeholder_title(
    title: str, title_catalog: dict[str, str]
) -> str:
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, title.casefold(), candidate.casefold()).ratio(), candidate)
            for candidate in title_catalog
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < 0.85:
        raise ValueError(f"Cannot resolve placeholder PaperScope title: {title}")
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.1:
        raise ValueError(f"Ambiguous placeholder PaperScope title: {title}")
    return title_catalog[scored[0][1]]


def paper_id_from_url(
    url: str, title_catalog: dict[str, str] | None = None
) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("PaperScope record contains an empty PDF URL")
    query = parse_qs(urlparse(url).query)
    paper_id = str((query.get("id") or [""])[0]).strip()
    if not paper_id:
        placeholder_title = placeholder_title_from_url(url)
        if placeholder_title and title_catalog:
            paper_id = _resolve_placeholder_title(placeholder_title, title_catalog)
    if not paper_id or not _PAPER_ID_RE.fullmatch(paper_id):
        raise ValueError(f"Cannot extract a safe OpenReview paper ID from: {url}")
    return paper_id


def collect_document_entries(
    records: Iterable[dict[str, Any]], *, valid_only: bool,
    title_catalog: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    links_by_id: dict[str, str] = {}
    for record in records:
        if valid_only and not is_valid_summary_record(record):
            continue
        links = record.get("pdf_links")
        if not isinstance(links, list):
            raise ValueError("PaperScope summary record has invalid pdf_links")
        for link in links:
            paper_id = paper_id_from_url(link, title_catalog)
            links_by_id.setdefault(paper_id, link)
    return [
        {"id": paper_id, "pdf_link": links_by_id[paper_id]}
        for paper_id in sorted(links_by_id)
    ]


def is_pdf_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                temp_path.open("wb") as handle,
                tqdm(
                    desc=f"Downloading {destination.name}",
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
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _authenticate_openreview(session: requests.Session) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "OpenReview authentication is required, but input is not interactive"
        )
    print("OpenReview requires a temporary login before downloading PDFs.")
    email = input("OpenReview email: ").strip()
    password = getpass.getpass("OpenReview password (input hidden): ")
    response = session.post(
        OPENREVIEW_LOGIN_URL,
        json={"id": email, "password": password},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenReview login failed (HTTP {response.status_code})")
    token = str(response.json().get("token") or "").strip()
    if not token:
        raise RuntimeError("OpenReview login response did not contain a token")
    session.headers["Authorization"] = f"Bearer {token}"


def _save_pdf_response(response: requests.Response, destination: Path) -> bool:
    if response.status_code >= 400:
        return False
    content = response.content
    if not content.startswith(b"%PDF-"):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".pdf.part")
    try:
        temp_path.write_bytes(content)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return True


def _download_openreview_pdf(
    session: requests.Session,
    paper_id: str,
    destination: Path,
    *,
    authenticated: bool,
) -> bool:
    urls = [OPENREVIEW_PDF_URL.format(paper_id=paper_id)]
    if authenticated:
        urls.append(OPENREVIEW_ATTACHMENT_URL.format(paper_id=paper_id))
    for url in urls:
        response = session.get(url, timeout=90)
        if _save_pdf_response(response, destination):
            return True
    return False


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if is_pdf_file(destination) and destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def verify_paperscope_download(dataset_dir: Path, document_scope: str) -> bool:
    summary_path = dataset_dir / "summary_corpus_1.jsonl"
    manifest_path = dataset_dir / "documents.jsonl"
    pdf_dir = dataset_dir / "pdfs"
    if not summary_path.is_file() or not manifest_path.is_file() or not pdf_dir.is_dir():
        return False
    try:
        entries = load_jsonl(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = EXPECTED_DOCUMENT_COUNTS[document_scope]
    if len(entries) != expected:
        return False
    return all(is_pdf_file(pdf_dir / f"{entry.get('id', '')}.pdf") for entry in entries)


def download_paperscope_summary(
    output_dir: Path,
    dataset_name: str,
    document_scope: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download one PaperScope document scope into ``raw_data``.

    A shared cache under ``raw_data/PaperScopeSummary`` lets a later 93-paper
    preparation reuse the 57 PDFs already downloaded.
    """
    if document_scope not in EXPECTED_DOCUMENT_COUNTS:
        raise ValueError(f"Unknown PaperScope document scope: {document_scope}")

    shared_dir = output_dir / "PaperScopeSummary"
    summary_cache = shared_dir / "summary_corpus_1.jsonl"
    corpus_cache = shared_dir / "corpus_1.jsonl"
    pdf_cache = shared_dir / "pdf_cache"
    dataset_dir = output_dir / dataset_name

    if dataset_dir.exists() and not force and verify_paperscope_download(
        dataset_dir, document_scope
    ):
        print(f"{dataset_name} already exists and passed verification")
        return True

    if force or not summary_cache.is_file():
        print("Downloading PaperScope summary_corpus_1.jsonl...")
        _download_file(SUMMARY_URL, summary_cache)

    records = load_jsonl(summary_cache)
    valid_records = [record for record in records if is_valid_summary_record(record)]
    if len(records) != EXPECTED_RECORDS or len(valid_records) != EXPECTED_VALID_RECORDS:
        raise ValueError(
            "Unexpected PaperScope snapshot: "
            f"records={len(records)} valid={len(valid_records)}; "
            f"expected {EXPECTED_RECORDS}/{EXPECTED_VALID_RECORDS}"
        )

    title_catalog = None
    if document_scope == "all":
        if force or not corpus_cache.is_file():
            print("Downloading PaperScope corpus_1.jsonl to resolve placeholder links...")
            _download_file(CORPUS_URL, corpus_cache)
        corpus_records = load_jsonl(corpus_cache)
        title_catalog = {
            str(record.get("title") or "").strip(): str(record.get("id") or "").strip()
            for record in corpus_records
            if record.get("title") and record.get("id")
        }

    entries = collect_document_entries(
        records,
        valid_only=(document_scope == "valid"),
        title_catalog=title_catalog,
    )
    expected_documents = EXPECTED_DOCUMENT_COUNTS[document_scope]
    if len(entries) != expected_documents:
        raise ValueError(
            f"Unexpected PaperScope {document_scope} document count: "
            f"{len(entries)} != {expected_documents}"
        )

    session = requests.Session()
    session.headers["User-Agent"] = "ov-llm-wiki PaperScope downloader"
    authenticated = False
    for index, entry in enumerate(entries, 1):
        paper_id = entry["id"]
        cached_pdf = pdf_cache / f"{paper_id}.pdf"
        if not force and is_pdf_file(cached_pdf):
            continue
        print(f"[{index}/{len(entries)}] Downloading OpenReview paper {paper_id}")
        if not _download_openreview_pdf(
            session, paper_id, cached_pdf, authenticated=authenticated
        ):
            if not authenticated:
                _authenticate_openreview(session)
                authenticated = True
            if not _download_openreview_pdf(
                session, paper_id, cached_pdf, authenticated=True
            ):
                raise RuntimeError(f"Failed to download a valid PDF for {paper_id}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _link_or_copy(summary_cache, dataset_dir / "summary_corpus_1.jsonl")
    write_jsonl(dataset_dir / "documents.jsonl", entries)
    for entry in entries:
        paper_id = entry["id"]
        _link_or_copy(
            pdf_cache / f"{paper_id}.pdf",
            dataset_dir / "pdfs" / f"{paper_id}.pdf",
        )

    if verify and not verify_paperscope_download(dataset_dir, document_scope):
        return False
    print(f"✓ {dataset_name}: {len(entries)} PDFs ready at {dataset_dir}")
    return True


def prepare_paperscope_summary(
    input_dir: Path,
    output_dir: Path,
    *,
    document_scope: str,
) -> dict[str, Any]:
    """Create type-specific, valid-QA files for benchmark execution."""
    summary_path = input_dir / "summary_corpus_1.jsonl"
    manifest_path = input_dir / "documents.jsonl"
    source_pdf_dir = input_dir / "pdfs"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete PaperScope download at {input_dir}")

    records = load_jsonl(summary_path)
    valid_records = [record for record in records if is_valid_summary_record(record)]
    by_type = {
        prompt_type: [
            record
            for record in valid_records
            if record.get("prompt_type") == prompt_type
        ]
        for prompt_type in PROMPT_TYPES
    }
    expected_type_counts = {"trend": 117, "gap": 119, "results_comparison": 116}
    actual_type_counts = {key: len(value) for key, value in by_type.items()}
    if actual_type_counts != expected_type_counts:
        raise ValueError(
            f"Unexpected valid QA counts: {actual_type_counts} != {expected_type_counts}"
        )

    entries = load_jsonl(manifest_path)
    expected_documents = EXPECTED_DOCUMENT_COUNTS[document_scope]
    if len(entries) != expected_documents:
        raise ValueError(
            f"Unexpected document manifest size: {len(entries)} != {expected_documents}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for prompt_type, typed_records in by_type.items():
        write_jsonl(output_dir / f"summary_{prompt_type}.jsonl", typed_records)
    write_jsonl(output_dir / "documents.jsonl", entries)
    for entry in entries:
        paper_id = str(entry["id"])
        source = source_pdf_dir / f"{paper_id}.pdf"
        if not is_pdf_file(source):
            raise FileNotFoundError(f"Missing or invalid PDF: {source}")
        _link_or_copy(source, output_dir / "pdfs" / source.name)

    return {
        "dataset": f"PaperScopeSummary{expected_documents}",
        "document_scope": document_scope,
        "original_total_qas": len(records),
        "valid_total_qas": len(valid_records),
        "valid_qas_by_type": actual_type_counts,
        "sampled_num_docs": len(entries),
        "is_full": True,
    }
