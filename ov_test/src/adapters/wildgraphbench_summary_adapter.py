"""Adapter for the fixed WildGraphBench Summary scopes."""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


EXPECTED_COUNTS = {
    "all": {"documents": 3894, "questions": 339},
    "health": {"documents": 509, "questions": 55},
}


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


def _text_to_markdown(source: Path, destination: Path) -> None:
    """Stage Markdown-formatted WildGraphBench TXT without changing its content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("WildGraphBench manifest contains an empty relative_path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe WildGraphBench relative_path: {value}")
    return path


class WildGraphBenchSummaryAdapter(BaseAdapter):
    """Load either the all-domain or Health-only Summary benchmark scope."""

    def _dataset_dir(self) -> Path:
        qa_path = Path(self.raw_file_path)
        if not qa_path.is_file():
            raise FileNotFoundError(f"WildGraphBench QA file not found: {qa_path}")
        return qa_path.parent

    def _dataset_info(self) -> dict[str, Any]:
        info_path = self._dataset_dir() / "dataset_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"WildGraphBench dataset info not found: {info_path}")
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        scope = info.get("scope")
        if scope not in EXPECTED_COUNTS:
            raise ValueError(f"Unsupported WildGraphBench scope: {scope}")
        return info

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest_path = self._dataset_dir() / "documents.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"WildGraphBench document manifest not found: {manifest_path}"
            )
        entries = _load_jsonl(manifest_path)
        scope = self._dataset_info()["scope"]
        expected = EXPECTED_COUNTS[scope]["documents"]
        if len(entries) != expected:
            raise ValueError(
                f"WildGraphBench {scope} manifest must contain {expected} documents, "
                f"got {len(entries)}"
            )
        document_ids = [str(entry.get("id") or "").strip() for entry in entries]
        if any(not document_id for document_id in document_ids):
            raise ValueError("WildGraphBench manifest contains an empty document ID")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("WildGraphBench manifest contains duplicate document IDs")
        relative_paths = [_safe_relative_path(entry.get("relative_path")) for entry in entries]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("WildGraphBench manifest contains duplicate document paths")
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        dataset_dir = self._dataset_dir()
        entries = self._manifest_entries()
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_paths = {
            _safe_relative_path(entry["relative_path"])
            .relative_to("reference_pages")
            .with_suffix(".md")
            for entry in entries
        }
        actual_paths = {
            path.relative_to(destination_dir)
            for path in destination_dir.rglob("*.md")
            if path.is_file()
        }
        extra_paths = sorted(actual_paths - expected_paths)
        if extra_paths:
            preview = [path.as_posix() for path in extra_paths[:10]]
            suffix = "..." if len(extra_paths) > 10 else ""
            raise RuntimeError(
                "Document staging directory contains Markdown files outside the "
                f"WildGraphBench manifest: {preview}{suffix}. Use a clean, "
                "scope-specific doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            manifest_path = _safe_relative_path(entry["relative_path"])
            source = dataset_dir / manifest_path
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty WildGraphBench TXT: {source}")
            expected_size = entry.get("size_bytes")
            if expected_size is not None and source.stat().st_size != expected_size:
                raise ValueError(f"WildGraphBench TXT size mismatch: {source}")
            staged_path = manifest_path.relative_to("reference_pages").with_suffix(".md")
            destination = destination_dir / staged_path
            _text_to_markdown(source, destination)
            documents.append(
                StandardDoc(sample_id=str(entry["id"]), doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[WildGraphBench Summary adapter] prepared {len(documents)} reference pages"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_jsonl(Path(self.raw_file_path))
        scope = self._dataset_info()["scope"]
        expected = EXPECTED_COUNTS[scope]["questions"]
        if len(records) != expected:
            raise ValueError(
                f"WildGraphBench {scope} QA file must contain {expected} questions, "
                f"got {len(records)}"
            )

        grouped: "OrderedDict[str, list[StandardQA]]" = OrderedDict()
        topics_by_domain: dict[str, str] = {}
        for qa_index, record in enumerate(records):
            question = record.get("question")
            gold_statements = record.get("gold_statements")
            ref_urls = record.get("ref_urls")
            domain = str(record.get("domain") or "").strip()
            topic = str(record.get("topic") or "").strip()
            question_types = record.get("question_type")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(gold_statements, list)
                or not gold_statements
                or any(
                    not isinstance(statement, str) or not statement.strip()
                    for statement in gold_statements
                )
                or not isinstance(ref_urls, list)
                or not ref_urls
                or not domain
                or not topic
                or not isinstance(question_types, list)
                or "summary" not in question_types
            ):
                raise ValueError(f"Incomplete WildGraphBench Summary QA at index {qa_index}")

            # The generic evaluator expects complete alternative answers. Joining the
            # statements makes the existing metrics assess coverage of the full gold
            # summary instead of treating any single statement as sufficient.
            combined_gold = "\n".join(
                f"- {statement.strip()}" for statement in gold_statements
            )
            qa = StandardQA(
                question=question.strip(),
                gold_answers=[combined_gold],
                evidence=[],
                category="summary",
                metadata={
                    "qa_index": qa_index,
                    "domain": domain,
                    "topic": topic,
                    "question_type": list(question_types),
                    "gold_statements": list(gold_statements),
                    "ref_urls": list(ref_urls),
                    "source_row_index": record.get("source_row_index"),
                },
            )
            grouped.setdefault(domain, []).append(qa)
            topics_by_domain[domain] = topic

        samples = [
            StandardSample(
                sample_id=f"wildgraphbench_summary_{domain}",
                qa_pairs=qa_pairs,
                metadata={
                    "domain": domain,
                    "topic": topics_by_domain[domain],
                    "scope": scope,
                },
            )
            for domain, qa_pairs in grouped.items()
        ]
        self.logger.info(
            f"[WildGraphBench Summary adapter] loaded {len(records)} QA records "
            f"in {len(samples)} domain groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = (
            "Answer the question with a comprehensive factual summary using only "
            "the retrieved reference-page content below. Cover the distinct relevant "
            "facts supported by the sources, reconcile complementary information, "
            "and do not invent unsupported details.\n\n"
            f"Retrieved reference-page content:\n{context}\n\n"
            f"Question:\n{qa.question}\n\n"
            "Answer:"
        )
        return prompt, {"category": qa.category, "domain": qa.metadata.get("domain")}
