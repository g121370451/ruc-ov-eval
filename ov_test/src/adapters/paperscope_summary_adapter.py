"""Adapter for the valid QA subset of PaperScope Summary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, List
from urllib.parse import parse_qs, urlparse

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


PROMPT_TYPES = {"trend", "gap", "results_comparison"}

CATEGORY_INSTRUCTIONS = {
    "trend": (
        "Synthesize the development trend across the papers. Distinguish the "
        "chronology, shared direction, and meaningful methodological changes."
    ),
    "gap": (
        "Analyze research limitations supported by the papers. Do not treat missing "
        "repository files, summaries, or images as defects of the research methods."
    ),
    "results_comparison": (
        "Compare reported experimental results across the papers. Use exact numbers "
        "only when they are present in the provided context; do not invent metrics."
    ),
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


def _is_valid_record(record: dict[str, Any]) -> bool:
    answer = record.get("answer")
    return bool(
        isinstance(answer, str)
        and answer.strip()
        and not answer.lstrip().lower().startswith("error generating answer")
    )


def _paper_id(url: str) -> str:
    query = parse_qs(urlparse(str(url)).query)
    value = str((query.get("id") or [""])[0]).strip()
    if not value:
        raise ValueError(f"Cannot extract OpenReview paper ID from: {url}")
    return value


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


class PaperScopeSummaryAdapter(BaseAdapter):
    """Load one prepared PaperScope Summary prompt type.

    ``raw_file_path`` points to one of ``summary_trend.jsonl``,
    ``summary_gap.jsonl`` or ``summary_results_comparison.jsonl``. The sibling
    ``documents.jsonl`` controls whether 57 or 93 PDFs are staged for ingestion.
    """

    def _qa_path(self) -> Path:
        path = Path(self.raw_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PaperScope QA file not found: {path}")
        return path

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest = self._qa_path().parent / "documents.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"PaperScope document manifest not found: {manifest}")
        entries = _load_jsonl(manifest)
        ids = [str(entry.get("id") or "").strip() for entry in entries]
        if any(not paper_id for paper_id in ids):
            raise ValueError(f"PaperScope document manifest contains an empty ID: {manifest}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"PaperScope document manifest contains duplicate IDs: {manifest}")
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        entries = self._manifest_entries()
        source_pdf_dir = self._qa_path().parent / "pdfs"
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_names = {f"{str(entry['id'])}.pdf" for entry in entries}
        extra_pdfs = sorted(
            path.name for path in destination_dir.glob("*.pdf") if path.name not in expected_names
        )
        if extra_pdfs:
            raise RuntimeError(
                f"Document staging directory contains PDFs outside the manifest: {extra_pdfs}. "
                "Use a clean, scope-specific doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            paper_id = str(entry["id"])
            source = source_pdf_dir / f"{paper_id}.pdf"
            if not _is_pdf(source):
                raise FileNotFoundError(f"Missing or invalid PaperScope PDF: {source}")
            destination = destination_dir / source.name
            _link_or_copy(source, destination)
            documents.append(
                StandardDoc(sample_id=paper_id, doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[PaperScope Summary adapter] prepared {len(documents)} documents from manifest"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_jsonl(self._qa_path())
        grouped: "OrderedDict[tuple[str, ...], list[StandardQA]]" = OrderedDict()

        for row_index, record in enumerate(records):
            if not _is_valid_record(record):
                continue
            prompt_type = str(record.get("prompt_type") or "").strip()
            if prompt_type not in PROMPT_TYPES:
                raise ValueError(
                    f"Unsupported PaperScope prompt_type at row {row_index}: {prompt_type}"
                )
            question = str(record.get("question") or "").strip()
            answer = str(record.get("answer") or "").strip()
            links = record.get("pdf_links")
            if not question or not answer or not isinstance(links, list) or not links:
                raise ValueError(f"Incomplete PaperScope QA record at row {row_index}")
            paper_ids = tuple(sorted(_paper_id(link) for link in links))
            if len(paper_ids) != int(record.get("num_papers", len(paper_ids))):
                raise ValueError(f"num_papers mismatch at PaperScope row {row_index}")

            qa = StandardQA(
                question=question,
                gold_answers=[answer],
                evidence=[],
                category=prompt_type,
                metadata={
                    "source_row_index": row_index,
                    "prompt_type": prompt_type,
                    "session": record.get("session"),
                    "num_papers": record.get("num_papers"),
                    "common_entities": record.get("common_entities", []),
                    "pdf_links": links,
                    "paper_ids": list(paper_ids),
                    "original_question": record.get("original_question", ""),
                },
            )
            grouped.setdefault(paper_ids, []).append(qa)

        samples: list[StandardSample] = []
        for paper_ids, qa_pairs in grouped.items():
            digest = hashlib.sha256("\n".join(paper_ids).encode("utf-8")).hexdigest()[:12]
            samples.append(
                StandardSample(
                    sample_id=f"paperscope_{digest}",
                    qa_pairs=qa_pairs,
                    metadata={"paper_ids": list(paper_ids)},
                )
            )

        self.logger.info(
            f"[PaperScope Summary adapter] loaded {sum(len(s.qa_pairs) for s in samples)} "
            f"valid QA records in {len(samples)} paper groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        instruction = CATEGORY_INSTRUCTIONS.get(str(qa.category), "Synthesize the papers faithfully.")
        prompt = f"""Use only the retrieved paper content below to answer the multi-paper question.

{instruction}

Retrieved paper content:
{context}

Question:
{qa.question}

Answer:"""
        return prompt, {"prompt_type": qa.category}
