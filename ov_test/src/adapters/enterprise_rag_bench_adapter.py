"""Adapter for the fixed EnterpriseRAG-Bench selected-80 scope."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


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
EXPECTED_QAS = 80
EXPECTED_LOGICAL_DOCUMENTS = 322
EXPECTED_PHYSICAL_DOCUMENTS = 323
CONFLICT_DOCUMENT_ID = "dsid_6df52fdb96ae4edcb76464738bca3340"


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


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("EnterpriseRAG-Bench manifest contains an empty relative_path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe EnterpriseRAG-Bench relative_path: {value}")
    return path


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


class EnterpriseRAGBenchAdapter(BaseAdapter):
    """Load three QA categories and stage their shared selected TXT corpus."""

    def _qa_path(self) -> Path:
        path = Path(self.raw_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"EnterpriseRAG-Bench QA file not found: {path}")
        return path

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest_path = self._qa_path().parent / "documents.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"EnterpriseRAG-Bench document manifest not found: {manifest_path}"
            )
        entries = _load_jsonl(manifest_path)
        if len(entries) != EXPECTED_PHYSICAL_DOCUMENTS:
            raise ValueError(
                f"EnterpriseRAG-Bench manifest must contain "
                f"{EXPECTED_PHYSICAL_DOCUMENTS} documents, got {len(entries)}"
            )
        physical_ids = [str(entry.get("id") or "").strip() for entry in entries]
        logical_ids = [
            str(entry.get("logical_doc_id") or "").strip() for entry in entries
        ]
        paths = [_safe_relative_path(entry.get("relative_path")) for entry in entries]
        if (
            any(not value for value in physical_ids)
            or len(physical_ids) != len(set(physical_ids))
            or any(not value for value in logical_ids)
            or len(set(logical_ids)) != EXPECTED_LOGICAL_DOCUMENTS
            or len(paths) != len(set(paths))
        ):
            raise ValueError("Invalid EnterpriseRAG-Bench document manifest")
        counts = Counter(logical_ids)
        if counts.get(CONFLICT_DOCUMENT_ID) != 2 or any(
            count != (2 if document_id == CONFLICT_DOCUMENT_ID else 1)
            for document_id, count in counts.items()
        ):
            raise ValueError(
                "EnterpriseRAG-Bench physical/logical document mapping changed"
            )
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        dataset_dir = self._qa_path().parent
        entries = self._manifest_entries()
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_paths = {
            _safe_relative_path(entry["relative_path"]).relative_to(
                "reference_documents"
            )
            for entry in entries
        }
        actual_paths = {
            path.relative_to(destination_dir)
            for path in destination_dir.rglob("*.txt")
            if path.is_file()
        }
        extra_paths = sorted(actual_paths - expected_paths)
        if extra_paths:
            preview = [path.as_posix() for path in extra_paths[:10]]
            suffix = "..." if len(extra_paths) > 10 else ""
            raise RuntimeError(
                "Document staging directory contains TXT files outside the "
                f"EnterpriseRAG-Bench manifest: {preview}{suffix}. Use a clean, "
                "dataset-specific doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            manifest_path = _safe_relative_path(entry["relative_path"])
            source = dataset_dir / manifest_path
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Missing or empty EnterpriseRAG-Bench TXT: {source}"
                )
            if source.stat().st_size != entry.get("size_bytes"):
                raise ValueError(f"EnterpriseRAG-Bench TXT size mismatch: {source}")
            staged_path = manifest_path.relative_to("reference_documents")
            destination = destination_dir / staged_path
            _link_or_copy(source, destination)
            documents.append(
                StandardDoc(sample_id=str(entry["id"]), doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[EnterpriseRAG-Bench adapter] prepared {len(documents)} selected TXT documents"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_jsonl(self._qa_path())
        if len(records) != EXPECTED_QAS:
            raise ValueError(
                f"EnterpriseRAG-Bench selected scope must contain {EXPECTED_QAS} QA, "
                f"got {len(records)}"
            )

        entries = self._manifest_entries()
        physical_by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            physical_by_logical[str(entry["logical_doc_id"])].append(entry)
        for values in physical_by_logical.values():
            values.sort(key=lambda entry: str(entry["archive_path"]))

        grouped: "OrderedDict[str, list[StandardQA]]" = OrderedDict(
            (category, []) for category in TARGET_CATEGORIES
        )
        seen_question_ids: set[str] = set()
        for qa_index, record in enumerate(records):
            question_id = str(record.get("question_id") or "").strip()
            category = str(record.get("question_type") or "").strip()
            question = str(record.get("question") or "").strip()
            gold_answer = str(record.get("gold_answer") or "").strip()
            source_types = record.get("source_types")
            expected_doc_ids = record.get("expected_doc_ids")
            answer_facts = record.get("answer_facts")
            if (
                not question_id
                or question_id in seen_question_ids
                or category not in TARGET_CATEGORIES
                or not question
                or not gold_answer
                or not isinstance(source_types, list)
                or not source_types
                or not isinstance(expected_doc_ids, list)
                or not expected_doc_ids
                or not isinstance(answer_facts, list)
                or not answer_facts
            ):
                raise ValueError(
                    f"Incomplete EnterpriseRAG-Bench QA at index {qa_index}"
                )
            seen_question_ids.add(question_id)

            occurrence_index: dict[str, int] = defaultdict(int)
            expected_physical_documents: list[dict[str, Any]] = []
            for logical_id in expected_doc_ids:
                logical_id = str(logical_id)
                candidates = physical_by_logical.get(logical_id, [])
                index = occurrence_index[logical_id]
                if index >= len(candidates):
                    raise ValueError(
                        f"EnterpriseRAG-Bench {question_id} references unavailable "
                        f"document occurrence: {logical_id} occurrence {index + 1}"
                    )
                entry = candidates[index]
                occurrence_index[logical_id] += 1
                expected_physical_documents.append(
                    {
                        "id": entry["id"],
                        "logical_doc_id": logical_id,
                        "source_type": entry.get("source_type"),
                        "archive_path": entry.get("archive_path"),
                    }
                )

            grouped[category].append(
                StandardQA(
                    question=question,
                    gold_answers=[gold_answer],
                    evidence=[str(fact).strip() for fact in answer_facts],
                    category=category,
                    metadata={
                        "qa_index": qa_index,
                        "question_id": question_id,
                        "question_type": category,
                        "source_types": list(source_types),
                        "expected_doc_ids": list(expected_doc_ids),
                        "expected_physical_documents": expected_physical_documents,
                        "answer_facts": list(answer_facts),
                    },
                )
            )

        actual_counts = {
            category: len(qa_pairs) for category, qa_pairs in grouped.items()
        }
        if actual_counts != EXPECTED_CATEGORY_COUNTS:
            raise ValueError(
                f"Unexpected EnterpriseRAG-Bench category counts: {actual_counts}"
            )

        samples = [
            StandardSample(
                sample_id=f"enterprise_rag_bench_{category}",
                qa_pairs=qa_pairs,
                metadata={"category": category, "shared_corpus": True},
            )
            for category, qa_pairs in grouped.items()
        ]
        self.logger.info(
            f"[EnterpriseRAG-Bench adapter] loaded {len(records)} QA records "
            f"in {len(samples)} category groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = (
            "Answer the enterprise multi-document question using only the retrieved "
            "content below. Synthesize all relevant sources, reconcile conflicting "
            "statements explicitly when present, and include every requested item or "
            "fact. Do not invent unsupported details.\n\n"
            f"Retrieved enterprise content:\n{context}\n\n"
            f"Question:\n{qa.question}\n\n"
            "Answer:"
        )
        return prompt, {
            "category": qa.category,
            "question_id": qa.metadata.get("question_id"),
        }
