"""Adapter for the fixed first-100 MDA-QA benchmark subset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


EXPECTED_QAS = 100
EXPECTED_DOCUMENTS = 143
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"MDA-QA file must contain a JSON list of objects: {path}")
    return data


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


class MDAQAAdapter(BaseAdapter):
    """Load 100 multi-document QA records and stage their 143 arXiv PDFs."""

    def _qa_path(self) -> Path:
        path = Path(self.raw_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"MDA-QA file not found: {path}")
        return path

    def _manifest_entries(self) -> list[dict[str, Any]]:
        manifest = self._qa_path().parent / "documents.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"MDA-QA document manifest not found: {manifest}")
        entries = _load_jsonl(manifest)
        ids = [str(entry.get("id") or "").strip() for entry in entries]
        if len(ids) != EXPECTED_DOCUMENTS:
            raise ValueError(
                f"MDA-QA manifest must contain {EXPECTED_DOCUMENTS} documents, got {len(ids)}"
            )
        if any(not _ARXIV_ID_RE.fullmatch(paper_id) for paper_id in ids):
            raise ValueError(f"MDA-QA manifest contains an invalid arXiv ID: {manifest}")
        if ids != sorted(set(ids)):
            raise ValueError(f"MDA-QA manifest IDs must be unique and sorted: {manifest}")
        return entries

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        entries = self._manifest_entries()
        source_pdf_dir = self._qa_path().parent / "pdfs"
        destination_dir = Path(doc_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_names = {f"{entry['id']}.pdf" for entry in entries}
        extra_pdfs = sorted(
            path.name
            for path in destination_dir.glob("*.pdf")
            if path.name not in expected_names
        )
        if extra_pdfs:
            raise RuntimeError(
                f"Document staging directory contains PDFs outside the MDA-QA manifest: "
                f"{extra_pdfs}. Use a clean, dataset-specific doc_output_dir."
            )

        documents: list[StandardDoc] = []
        for entry in entries:
            paper_id = str(entry["id"])
            source = source_pdf_dir / f"{paper_id}.pdf"
            if not _is_pdf(source):
                raise FileNotFoundError(f"Missing or invalid MDA-QA PDF: {source}")
            destination = destination_dir / source.name
            _link_or_copy(source, destination)
            documents.append(
                StandardDoc(sample_id=paper_id, doc_paths=[str(destination)])
            )

        self.logger.info(
            f"[MDA-QA adapter] prepared {len(documents)} documents from manifest"
        )
        return documents

    def load_and_transform(self) -> List[StandardSample]:
        records = _load_json(self._qa_path())
        if len(records) != EXPECTED_QAS:
            raise ValueError(
                f"MDA-QA prepared subset must contain {EXPECTED_QAS} QA, got {len(records)}"
            )
        expected_ids = list(range(EXPECTED_QAS))
        if [record.get("id") for record in records] != expected_ids:
            raise ValueError("MDA-QA prepared subset must contain IDs 0 through 99 in order")

        manifest_ids = {str(entry["id"]) for entry in self._manifest_entries()}
        grouped: "OrderedDict[tuple[str, ...], list[StandardQA]]" = OrderedDict()
        for record in records:
            qa_id = int(record["id"])
            question = str(record.get("question") or "").strip()
            answer = str(record.get("answer") or "").strip()
            support = record.get("support")
            if (
                not question
                or not answer
                or not isinstance(support, list)
                or len(support) < 2
            ):
                raise ValueError(f"Incomplete MDA-QA record: {qa_id}")
            paper_ids = tuple(sorted(str(paper_id).strip() for paper_id in support))
            if len(paper_ids) != len(set(paper_ids)):
                raise ValueError(f"Duplicate support IDs in MDA-QA record: {qa_id}")
            missing_ids = sorted(set(paper_ids) - manifest_ids)
            if missing_ids:
                raise ValueError(
                    f"MDA-QA record {qa_id} references papers outside the manifest: {missing_ids}"
                )

            qa = StandardQA(
                question=question,
                gold_answers=[answer],
                evidence=[],
                category="multi_document",
                metadata={
                    "qa_id": qa_id,
                    "support_paper_ids": list(paper_ids),
                    "num_support_papers": len(paper_ids),
                },
            )
            grouped.setdefault(paper_ids, []).append(qa)

        samples: list[StandardSample] = []
        for paper_ids, qa_pairs in grouped.items():
            digest = hashlib.sha256("\n".join(paper_ids).encode("utf-8")).hexdigest()[:12]
            samples.append(
                StandardSample(
                    sample_id=f"mdaqa_{digest}",
                    qa_pairs=qa_pairs,
                    metadata={"support_paper_ids": list(paper_ids)},
                )
            )

        self.logger.info(
            f"[MDA-QA adapter] loaded {sum(len(sample.qa_pairs) for sample in samples)} "
            f"QA records in {len(samples)} support groups"
        )
        return samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        prompt = (
            "Answer the multi-document scientific question using only the retrieved "
            "paper content below.\n\n"
            "Synthesize complementary or contrasting evidence across the relevant "
            "papers. Do not invent methods, results, comparisons, or numerical values "
            "that are absent from the retrieved content.\n\n"
            f"Retrieved paper content:\n{context}\n\n"
            f"Question:\n{qa.question}\n\n"
            "Answer:"
        )
        return prompt, {"category": qa.category}
