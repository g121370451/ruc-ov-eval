"""Adapter for the fixed MuDABench Simple and Complex QA scopes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


SCOPES = ("simple", "complex")
EXPECTED_QAS = 166
EXPECTED_UNIQUE_QUESTION_IDS = 164
EXPECTED_DOCUMENTS = 589
EXPECTED_DOCUMENT_GROUPS = 154
EXPECTED_DUPLICATE_ROWS = [[140, 159], [141, 160]]


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"MuDABench file must be a JSON list of objects: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _is_pdf(destination) and destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _canonical_uuid(value: Any, *, location: str) -> str:
    document_id = str(value or "").strip()
    try:
        parsed = uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError(f"Invalid MuDABench document ID at {location}: {value}") from exc
    if str(parsed) != document_id:
        raise ValueError(f"Non-canonical MuDABench document ID at {location}: {value}")
    return document_id


class MuDABenchAdapter(BaseAdapter):
    """Load either QA scope while staging the complete shared PDF corpus."""

    def _qa_path(self) -> Path:
        path = Path(self.raw_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"MuDABench QA file not found: {path}")
        return path

    def _dataset_info(self) -> dict[str, Any]:
        info_path = self._qa_path().parent / "dataset_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"MuDABench dataset info not found: {info_path}")
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        scope = info.get("scope")
        if scope not in SCOPES:
            raise ValueError(f"Unsupported MuDABench scope: {scope}")
        if self._qa_path().name != f"{scope}.json":
            raise ValueError(
                f"MuDABench QA filename and dataset scope disagree: {self._qa_path()}"
            )
        return info

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest_path = self._qa_path().parent / "documents.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MuDABench manifest not found: {manifest_path}")
        entries = _load_jsonl(manifest_path)
        if len(entries) != EXPECTED_DOCUMENTS:
            raise ValueError(
                f"MuDABench manifest must contain {EXPECTED_DOCUMENTS} PDFs, "
                f"got {len(entries)}"
            )
        ids = [
            _canonical_uuid(entry.get("id"), location=f"manifest[{index}]")
            for index, entry in enumerate(entries)
        ]
        if ids != sorted(set(ids)):
            raise ValueError("MuDABench manifest IDs must be unique and sorted")
        filenames = [str(entry.get("filename") or "").strip() for entry in entries]
        if filenames != [f"{document_id}.pdf" for document_id in ids]:
            raise ValueError("MuDABench manifest filenames do not match document IDs")
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        entries = self._manifest_entries()
        source_pdf_dir = self._qa_path().parent / "pdfs"
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_names = {entry["filename"] for entry in entries}
        extra_pdfs = sorted(
            path.name
            for path in destination_dir.glob("*.pdf")
            if path.name not in expected_names
        )
        if extra_pdfs:
            preview = extra_pdfs[:10]
            suffix = "..." if len(extra_pdfs) > 10 else ""
            raise RuntimeError(
                "Document staging directory contains PDFs outside the MuDABench "
                f"manifest: {preview}{suffix}. Use the shared MuDABench doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            source = source_pdf_dir / entry["filename"]
            if not _is_pdf(source):
                raise FileNotFoundError(f"Missing or invalid MuDABench PDF: {source}")
            if source.stat().st_size != entry.get("size_bytes"):
                raise ValueError(f"MuDABench PDF size mismatch: {source}")
            destination = destination_dir / entry["filename"]
            _link_or_copy(source, destination)
            documents.append(
                StandardDoc(sample_id=str(entry["id"]), doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[MuDABench adapter] prepared the complete {len(documents)}-PDF corpus"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_json(self._qa_path())
        scope = str(self._dataset_info()["scope"])
        if len(records) != EXPECTED_QAS:
            raise ValueError(
                f"MuDABench {scope} must contain {EXPECTED_QAS} QA, got {len(records)}"
            )

        manifest_ids = {str(entry["id"]) for entry in self._manifest_entries()}
        grouped: "OrderedDict[tuple[str, ...], list[StandardQA]]" = OrderedDict()
        rows_by_question_id: dict[str, list[int]] = {}
        for row_index, record in enumerate(records):
            question_id = str(record.get("question_id") or "").strip()
            question = str(record.get("question") or "").strip()
            final_answer = str(record.get("final_answer") or "").strip()
            source_answer = record.get("source_answer")
            metadata = record.get("metadata")
            if (
                not question_id
                or not question
                or not final_answer
                or not isinstance(source_answer, list)
                or not source_answer
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in source_answer
                )
                or not isinstance(metadata, list)
                or not metadata
            ):
                raise ValueError(f"Incomplete MuDABench {scope} QA at row {row_index}")

            document_ids = []
            for metadata_index, item in enumerate(metadata):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Invalid MuDABench metadata at row {row_index}, "
                        f"index {metadata_index}"
                    )
                document_ids.append(
                    _canonical_uuid(
                        item.get("id"),
                        location=f"{scope}[{row_index}].metadata[{metadata_index}]",
                    )
                )
            if len(document_ids) != len(set(document_ids)):
                raise ValueError(f"Duplicate document IDs in MuDABench row {row_index}")
            missing_ids = sorted(set(document_ids) - manifest_ids)
            if missing_ids:
                raise ValueError(
                    f"MuDABench row {row_index} references IDs outside the manifest: "
                    f"{missing_ids}"
                )

            qa = StandardQA(
                question=question,
                gold_answers=[final_answer],
                evidence=[item.strip() for item in source_answer],
                category=scope,
                metadata={
                    "source_row_index": row_index,
                    "question_id": question_id,
                    "scope": scope,
                    "document_ids": list(document_ids),
                    "source_answer": list(source_answer),
                    "document_metadata": metadata,
                },
            )
            document_key = tuple(sorted(document_ids))
            grouped.setdefault(document_key, []).append(qa)
            rows_by_question_id.setdefault(question_id, []).append(row_index)

        duplicate_rows = sorted(
            indices for indices in rows_by_question_id.values() if len(indices) > 1
        )
        if (
            len(rows_by_question_id) != EXPECTED_UNIQUE_QUESTION_IDS
            or duplicate_rows != EXPECTED_DUPLICATE_ROWS
        ):
            raise ValueError(
                f"Unexpected MuDABench {scope} question IDs or duplicate rows"
            )
        for first_index, second_index in duplicate_rows:
            if records[first_index] != records[second_index]:
                raise ValueError(
                    f"MuDABench {scope} duplicate question IDs are not identical"
                )
        if len(grouped) != EXPECTED_DOCUMENT_GROUPS:
            raise ValueError(
                f"Unexpected MuDABench {scope} document-group count: {len(grouped)}"
            )

        samples: list[StandardSample] = []
        for document_ids, qa_pairs in grouped.items():
            digest = hashlib.sha256(
                "\n".join(document_ids).encode("utf-8")
            ).hexdigest()[:12]
            samples.append(
                StandardSample(
                    sample_id=f"mudabench_{digest}",
                    qa_pairs=qa_pairs,
                    metadata={
                        "scope": scope,
                        "document_ids": list(document_ids),
                    },
                )
            )

        self.logger.info(
            f"[MuDABench adapter] loaded {len(records)} {scope} QA records "
            f"in {len(samples)} document groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = (
            "Answer the multi-document financial analysis question using only the "
            "retrieved report content below. Compare the relevant companies, years, "
            "metrics, and document types carefully. Preserve units and numerical "
            "precision, and do not invent unsupported values.\n\n"
            f"Retrieved financial report content:\n{context}\n\n"
            f"Question:\n{qa.question}\n\n"
            "Answer:"
        )
        return prompt, {
            "category": qa.category,
            "question_id": qa.metadata.get("question_id"),
        }
