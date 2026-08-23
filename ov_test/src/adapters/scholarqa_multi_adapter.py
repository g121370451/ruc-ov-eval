"""Adapter for the valid 101-question ScholarQA-Multi benchmark scope."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


EXPECTED_QAS = 101
EXPECTED_DOCUMENTS = 413
EXPECTED_SUBJECT_COUNTS = {
    "bio": 19,
    "biophysics": 9,
    "cs_hci": 9,
    "cs_nlp": 29,
    "photonics": 29,
    "physics": 6,
}
_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"ScholarQA-Multi file must be a JSON list of objects: {path}")
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


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ScholarQA-Multi manifest contains an empty relative_path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ScholarQA-Multi relative_path: {value}")
    return path


def _citation_indices(answer: str) -> list[int]:
    return [
        int(index)
        for group in _CITATION_GROUP_RE.findall(answer)
        for index in re.findall(r"\d+", group)
    ]


def _augmented_gold(answer: str, contexts: list[dict[str, Any]]) -> str:
    reference_lines = []
    for index, context in enumerate(contexts):
        title = str(context.get("title") or "").strip()
        if not title:
            raise ValueError(f"ScholarQA-Multi citation context {index} has no title")
        reference_lines.append(f"[{index}] {title}")
    return (
        answer.strip()
        + "\n\nReference key for resolving citation labels only.\n"
        + "The generated answer does not need to reproduce this reference list.\n\n"
        + "\n".join(reference_lines)
    )


class ScholarQAMultiAdapter(BaseAdapter):
    """Load valid expert QA and stage the merged official citation snippets."""

    def _qa_path(self) -> Path:
        path = Path(self.raw_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"ScholarQA-Multi QA file not found: {path}")
        return path

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest_path = self._qa_path().parent / "documents.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"ScholarQA-Multi document manifest not found: {manifest_path}"
            )
        entries = _load_jsonl(manifest_path)
        if len(entries) != EXPECTED_DOCUMENTS:
            raise ValueError(
                f"ScholarQA-Multi manifest must contain {EXPECTED_DOCUMENTS} documents, "
                f"got {len(entries)}"
            )
        ids = [str(entry.get("id") or "").strip() for entry in entries]
        if any(not document_id for document_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("ScholarQA-Multi manifest IDs must be non-empty and unique")
        paths = [_safe_relative_path(entry.get("relative_path")) for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("ScholarQA-Multi manifest document paths are not unique")
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        dataset_dir = self._qa_path().parent
        entries = self._manifest_entries()
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_paths = {
            _safe_relative_path(entry["relative_path"]).relative_to("reference_documents")
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
                f"ScholarQA-Multi manifest: {preview}{suffix}. Use a clean, "
                "dataset-specific doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            manifest_path = _safe_relative_path(entry["relative_path"])
            source = dataset_dir / manifest_path
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty ScholarQA-Multi TXT: {source}")
            if source.stat().st_size != entry.get("size_bytes"):
                raise ValueError(f"ScholarQA-Multi TXT size mismatch: {source}")
            staged_path = manifest_path.relative_to("reference_documents")
            destination = destination_dir / staged_path
            _link_or_copy(source, destination)
            documents.append(
                StandardDoc(sample_id=str(entry["id"]), doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[ScholarQA-Multi adapter] prepared {len(documents)} merged reference documents"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_json(self._qa_path())
        if len(records) != EXPECTED_QAS:
            raise ValueError(
                f"ScholarQA-Multi prepared subset must contain {EXPECTED_QAS} QA, "
                f"got {len(records)}"
            )

        grouped: "OrderedDict[str, list[StandardQA]]" = OrderedDict()
        seen_ids: set[str] = set()
        for qa_index, record in enumerate(records):
            qa_id = str(record.get("id") or "").strip()
            question = str(record.get("input") or "").strip()
            raw_answer = str(record.get("output") or "").strip()
            subject = str(record.get("subject") or "").strip()
            contexts = record.get("ctxs")
            if (
                not qa_id
                or qa_id in seen_ids
                or not question
                or not raw_answer
                or subject not in EXPECTED_SUBJECT_COUNTS
                or not isinstance(contexts, list)
                or not contexts
            ):
                raise ValueError(f"Incomplete ScholarQA-Multi QA at index {qa_index}")
            seen_ids.add(qa_id)
            invalid_indices = sorted(
                {index for index in _citation_indices(raw_answer) if index >= len(contexts)}
            )
            if invalid_indices:
                raise ValueError(
                    f"ScholarQA-Multi {qa_id} contains out-of-range citations: "
                    f"{invalid_indices} for {len(contexts)} contexts"
                )
            evidence = []
            citation_map: dict[str, str] = {}
            for context_index, context in enumerate(contexts):
                if not isinstance(context, dict):
                    raise ValueError(f"ScholarQA-Multi {qa_id} has an invalid context")
                raw_text = context.get("text")
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                title = str(context.get("title") or "").strip()
                if not title:
                    raise ValueError(f"ScholarQA-Multi {qa_id} has a context without title")
                if text:
                    evidence.append(text)
                citation_map[str(context_index)] = title

            qa = StandardQA(
                question=question,
                gold_answers=[_augmented_gold(raw_answer, contexts)],
                evidence=evidence,
                category=subject,
                metadata={
                    "qa_index": qa_index,
                    "qa_id": qa_id,
                    "subject": subject,
                    "annotator": record.get("annotator"),
                    "raw_expert_answer": raw_answer,
                    "citation_map": citation_map,
                    "ctxs": contexts,
                },
            )
            grouped.setdefault(subject, []).append(qa)

        actual_counts = {subject: len(qa_pairs) for subject, qa_pairs in grouped.items()}
        if actual_counts != EXPECTED_SUBJECT_COUNTS:
            raise ValueError(f"Unexpected ScholarQA-Multi subject counts: {actual_counts}")

        samples = [
            StandardSample(
                sample_id=f"scholarqa_multi_{subject}",
                qa_pairs=qa_pairs,
                metadata={"subject": subject, "shared_corpus": True},
            )
            for subject, qa_pairs in grouped.items()
        ]
        self.logger.info(
            f"[ScholarQA-Multi adapter] loaded {len(records)} QA records "
            f"in {len(samples)} subject groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = (
            "Answer the scientific synthesis question using only the retrieved "
            "ScholarQA-Multi reference content below. Integrate relevant evidence "
            "across sources, preserve important qualifications, and do not invent "
            "unsupported claims or citation labels.\n\n"
            f"Retrieved reference content:\n{context}\n\n"
            f"Question:\n{qa.question}\n\n"
            "Answer:"
        )
        return prompt, {"category": qa.category, "qa_id": qa.metadata.get("qa_id")}
