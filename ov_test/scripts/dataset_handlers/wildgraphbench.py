"""Download and prepare the WildGraphBench Summary benchmark scopes.

The upstream repository contains twelve domains. This module exposes two fixed
benchmark scopes while preserving the dataset's nested reference-page paths:

* ``all``: all reference pages and all Summary questions
* ``health``: Health reference pages and Health Summary questions
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests
from tqdm import tqdm


WILDGRAPHBENCH_REVISION = "c334bc806511e029285d208c24c2a8901ed5cf2c"
WILDGRAPHBENCH_ARCHIVE_URL = (
    "https://codeload.github.com/BstWPY/WildGraphBench/zip/"
    f"{WILDGRAPHBENCH_REVISION}"
)

DOMAINS = (
    "culture",
    "geography",
    "health",
    "history",
    "human_activities",
    "mathematics",
    "nature",
    "people",
    "philosophy",
    "religion",
    "society",
    "technology",
)
EXPECTED_COUNTS = {
    "all": {"documents": 3894, "questions": 339, "domains": 12},
    "health": {"documents": 509, "questions": 55, "domains": 1},
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _is_summary_question(record: dict[str, Any]) -> bool:
    question_types = record.get("question_type")
    if isinstance(question_types, str):
        question_types = [question_types]
    return isinstance(question_types, list) and any(
        isinstance(value, str) and value.strip().casefold() == "summary"
        for value in question_types
    )


def _validate_summary_question(record: dict[str, Any], *, location: str) -> None:
    question = record.get("question")
    gold_statements = record.get("gold_statements")
    ref_urls = record.get("ref_urls")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"WildGraphBench Summary question is empty: {location}")
    if not isinstance(gold_statements, list) or not gold_statements or any(
        not isinstance(statement, str) or not statement.strip()
        for statement in gold_statements
    ):
        raise ValueError(f"WildGraphBench gold_statements are invalid: {location}")
    if not isinstance(ref_urls, list) or not ref_urls or any(
        not isinstance(url, str) or not url.strip() for url in ref_urls
    ):
        raise ValueError(f"WildGraphBench ref_urls are invalid: {location}")


def _document_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
    return f"wildgraphbench_{digest}"


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


def _download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".zip.part")
    try:
        with requests.get(
            WILDGRAPHBENCH_ARCHIVE_URL,
            stream=True,
            timeout=(30, 300),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                temp_path.open("wb") as handle,
                tqdm(
                    desc="Downloading WildGraphBench",
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
        with zipfile.ZipFile(temp_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Downloaded WildGraphBench archive failed integrity check")
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _extract_snapshot(archive_path: Path, snapshot_dir: Path) -> None:
    build_dir = snapshot_dir.parent / f".{snapshot_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe ZIP member path: {member.filename}")
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise ValueError(f"Unsafe ZIP symlink: {member.filename}")
                archive.extract(member, build_dir)

        roots = [path for path in build_dir.iterdir() if path.is_dir()]
        if len(roots) != 1 or not (roots[0] / "QA").is_dir() or not (
            roots[0] / "corpus"
        ).is_dir():
            raise ValueError("Unexpected WildGraphBench archive layout")
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        roots[0].replace(snapshot_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def _ensure_snapshot(shared_dir: Path, *, force: bool) -> Path:
    archive_path = shared_dir / f"WildGraphBench-{WILDGRAPHBENCH_REVISION}.zip"
    snapshot_dir = shared_dir / f"WildGraphBench-{WILDGRAPHBENCH_REVISION}"
    if force or not archive_path.is_file():
        _download_archive(archive_path)
    if force or not (snapshot_dir / "QA").is_dir() or not (
        snapshot_dir / "corpus"
    ).is_dir():
        _extract_snapshot(archive_path, snapshot_dir)
    return snapshot_dir


def _scope_domains(scope: str) -> tuple[str, ...]:
    if scope == "all":
        return DOMAINS
    if scope == "health":
        return ("health",)
    raise ValueError(f"Unknown WildGraphBench scope: {scope}")


def _collect_scope(
    snapshot_dir: Path, scope: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []

    for domain in _scope_domains(scope):
        domain_corpus_dir = snapshot_dir / "corpus" / domain
        topic_dirs = sorted(path for path in domain_corpus_dir.iterdir() if path.is_dir())
        if len(topic_dirs) != 1:
            raise ValueError(
                f"Expected one WildGraphBench topic under {domain}, got {len(topic_dirs)}"
            )
        topic = topic_dirs[0].name
        reference_dir = topic_dirs[0] / "reference_pages"
        if not reference_dir.is_dir():
            raise FileNotFoundError(f"Missing reference_pages directory: {reference_dir}")

        for source_path in sorted(reference_dir.rglob("*.txt")):
            if not source_path.is_file() or source_path.stat().st_size == 0:
                raise ValueError(f"Empty WildGraphBench reference page: {source_path}")
            page_relative = source_path.relative_to(reference_dir).as_posix()
            prepared_relative = (
                Path("reference_pages") / domain / topic / Path(page_relative)
            ).as_posix()
            documents.append(
                {
                    "id": _document_id(prepared_relative),
                    "domain": domain,
                    "topic": topic,
                    "relative_path": prepared_relative,
                    "source_relative_path": source_path.relative_to(snapshot_dir).as_posix(),
                    "size_bytes": source_path.stat().st_size,
                }
            )

        question_path = snapshot_dir / "QA" / domain / "questions.jsonl"
        for row_index, record in enumerate(load_jsonl(question_path)):
            if not _is_summary_question(record):
                continue
            location = f"{domain}/questions.jsonl:{row_index + 1}"
            _validate_summary_question(record, location=location)
            prepared = dict(record)
            prepared.update(
                {
                    "domain": domain,
                    "topic": topic,
                    "source_row_index": row_index,
                }
            )
            questions.append(prepared)

    expected = EXPECTED_COUNTS[scope]
    if len(documents) != expected["documents"] or len(questions) != expected["questions"]:
        raise ValueError(
            f"Unexpected WildGraphBench {scope} counts: "
            f"documents={len(documents)}, questions={len(questions)}; "
            f"expected {expected['documents']}/{expected['questions']}"
        )
    document_ids = [entry["id"] for entry in documents]
    relative_paths = [entry["relative_path"] for entry in documents]
    if len(document_ids) != len(set(document_ids)) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise ValueError("WildGraphBench document manifest contains duplicate entries")
    return documents, questions


def _write_scope_dataset(
    snapshot_dir: Path, dataset_dir: Path, scope: str
) -> None:
    documents, questions = _collect_scope(snapshot_dir, scope)
    build_dir = dataset_dir.parent / f".{dataset_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        write_jsonl(build_dir / "documents.jsonl", documents)
        write_jsonl(build_dir / "summary_questions.jsonl", questions)
        write_json(
            build_dir / "dataset_info.json",
            {
                "dataset": "WildGraphBench",
                "revision": WILDGRAPHBENCH_REVISION,
                "scope": scope,
                "domains": list(_scope_domains(scope)),
                "document_count": len(documents),
                "summary_question_count": len(questions),
            },
        )
        for entry in documents:
            source = snapshot_dir / entry["source_relative_path"]
            destination = build_dir / entry["relative_path"]
            _link_or_copy(source, destination)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        build_dir.replace(dataset_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)


def verify_wildgraphbench_download(dataset_dir: Path, scope: str) -> bool:
    if scope not in EXPECTED_COUNTS:
        return False
    try:
        documents = load_jsonl(dataset_dir / "documents.jsonl")
        questions = load_jsonl(dataset_dir / "summary_questions.jsonl")
        with (dataset_dir / "dataset_info.json").open("r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return False

    expected = EXPECTED_COUNTS[scope]
    if (
        info.get("revision") != WILDGRAPHBENCH_REVISION
        or info.get("scope") != scope
        or len(documents) != expected["documents"]
        or len(questions) != expected["questions"]
    ):
        return False
    for entry in documents:
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            return False
        path = dataset_dir / relative_path
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            return False
    try:
        for index, question in enumerate(questions):
            if not _is_summary_question(question):
                return False
            _validate_summary_question(question, location=f"prepared:{index}")
    except ValueError:
        return False
    return True


def download_wildgraphbench_summary(
    output_dir: Path,
    dataset_name: str,
    scope: str,
    *,
    force: bool = False,
    verify: bool = True,
) -> bool:
    """Download one fixed WildGraphBench Summary scope into ``raw_data``."""
    if scope not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown WildGraphBench scope: {scope}")
    dataset_dir = output_dir / dataset_name
    if not force and verify_wildgraphbench_download(dataset_dir, scope):
        print(f"{dataset_name} already exists and passed verification")
        return True

    snapshot_dir = _ensure_snapshot(output_dir / "WildGraphBench", force=force)
    _write_scope_dataset(snapshot_dir, dataset_dir, scope)
    if verify and not verify_wildgraphbench_download(dataset_dir, scope):
        return False
    expected = EXPECTED_COUNTS[scope]
    print(
        f"✓ {dataset_name}: {expected['questions']} Summary QA and "
        f"{expected['documents']} reference pages ready at {dataset_dir}"
    )
    return True


def prepare_wildgraphbench_summary(
    input_dir: Path, output_dir: Path, *, scope: str
) -> dict[str, Any]:
    """Copy one verified fixed scope into the benchmark dataset directory."""
    if not verify_wildgraphbench_download(input_dir, scope):
        raise ValueError(f"WildGraphBench {scope} download failed verification: {input_dir}")
    documents = load_jsonl(input_dir / "documents.jsonl")
    questions = load_jsonl(input_dir / "summary_questions.jsonl")

    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    try:
        for filename in ("documents.jsonl", "summary_questions.jsonl", "dataset_info.json"):
            _link_or_copy(input_dir / filename, build_dir / filename)
        for entry in documents:
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

    expected = EXPECTED_COUNTS[scope]
    return {
        "dataset": "WildGraphBenchSummary",
        "revision": WILDGRAPHBENCH_REVISION,
        "scope": scope,
        "domains": list(_scope_domains(scope)),
        "sampled_num_docs": len(documents),
        "sampled_total_qas": len(questions),
        "is_full": True,
        "expected_counts": expected,
    }
