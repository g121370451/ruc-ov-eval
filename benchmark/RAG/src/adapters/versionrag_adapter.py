# src/adapters/versionrag_adapter.py
"""
VersionRAG dataset adapter.
"""

import csv
import json
import os
import shutil
from typing import Any, Dict, List

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample

QA_PROMPT = """Based on the provided context, answer the following question accurately and concisely.
Use the exact wording from the context whenever possible.

Question: {}
Answer:"""

MISSING_RULE = "If the provided context does not contain sufficient information to answer the question, respond with 'Not mentioned'."


class VersionRAGAdapter(BaseAdapter):
    """
    Adapter for VersionRAG.

    Expected layout relative to raw_file_path:
      data/raw/
      data/test/evaluation_set.csv
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        test_dir = os.path.dirname(self.raw_file_path)
        self.raw_doc_dir = os.path.normpath(os.path.join(test_dir, "..", "raw"))
        self.dataset_root = os.path.normpath(os.path.join(test_dir, "..", ".."))
        self.qa_doc_mapping_path = os.path.join(self.dataset_root, "qa_doc_mapping.json")
        self._qa_doc_mapping: List[Dict[str, Any]] = []
        self._source_to_processed: Dict[str, str] = {}

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        if not os.path.exists(self.raw_doc_dir):
            raise FileNotFoundError(f"Raw document directory not found: {self.raw_doc_dir}")

        os.makedirs(doc_dir, exist_ok=True)

        docs: List[StandardDoc] = []
        processed_by_source: Dict[str, str] = {}
        for filename in sorted(os.listdir(self.raw_doc_dir)):
            raw_path = os.path.join(self.raw_doc_dir, filename)
            if not os.path.isfile(raw_path):
                continue

            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower == ".md":
                dst_path = os.path.join(doc_dir, filename)
                shutil.copy2(raw_path, dst_path)
                docs.append(StandardDoc(sample_id=name, doc_path=dst_path))
                processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = dst_path

            elif ext_lower == ".pdf":
                md_filename = f"{name}.md"
                md_path = os.path.join(doc_dir, md_filename)
                existing_md_path = self._find_existing_processed_md(doc_dir, name)
                if existing_md_path:
                    docs.append(StandardDoc(sample_id=name, doc_path=existing_md_path))
                    processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = existing_md_path
                elif self._has_pymupdf():
                    if not os.path.exists(md_path):
                        self._pdf_to_markdown(raw_path, md_path)
                    docs.append(StandardDoc(sample_id=name, doc_path=md_path))
                    processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = md_path
                else:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    docs.append(StandardDoc(sample_id=name, doc_path=dst_path))
                    processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = dst_path
                    self.logger.warning(f"[VersionRAG] PyMuPDF not available; copied PDF directly: {filename}")

        self._source_to_processed = processed_by_source
        self._qa_doc_mapping = self._load_qa_doc_mapping()
        if self._qa_doc_mapping:
            self.logger.info(f"[VersionRAG] Loaded QA-doc mapping: {len(self._qa_doc_mapping)} questions")

        md_count = len([d for d in docs if d.doc_path.endswith(".md")])
        self.logger.info(f"[VersionRAG] Prepared {len(docs)} documents for ingestion ({md_count} md)")
        return docs

    @staticmethod
    def _has_pymupdf() -> bool:
        try:
            import fitz  # noqa: F401
            return True
        except ImportError:
            return False

    def _pdf_to_markdown(self, pdf_path: str, md_path: str):
        import fitz

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        with open(md_path, "w", encoding="utf-8") as f:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                f.write(f"## Page {page_num}\n\n")
                f.write(text)
                f.write("\n\n")
        doc.close()
        self.logger.info(f"[VersionRAG] Converted PDF -> Markdown: {os.path.basename(pdf_path)} ({page_count} pages)")

    def _find_existing_processed_md(self, doc_dir: str, raw_stem: str) -> str:
        """Find a preprocessed Markdown file for a raw PDF stem, tolerating filename sanitization differences."""
        direct_path = os.path.join(doc_dir, f"{raw_stem}.md")
        if os.path.exists(direct_path):
            return direct_path
        if not os.path.isdir(doc_dir):
            return ""

        def normalize_name(text: str) -> str:
            import re

            return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()

        target = normalize_name(raw_stem)
        for filename in os.listdir(doc_dir):
            if not filename.lower().endswith(".md"):
                continue
            stem, _ = os.path.splitext(filename)
            if normalize_name(stem) == target:
                return os.path.join(doc_dir, filename)
        return ""

    def load_and_transform(self) -> List[StandardSample]:
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Evaluation set not found: {self.raw_file_path}")

        groups: Dict[str, List[Dict[str, str]]] = {}
        all_rows: List[Dict[str, str]] = []
        with open(self.raw_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for local_idx, row in enumerate(reader):
                row["_local_index"] = str(local_idx)
                q_type = row.get("Type", "Unknown").strip()
                groups.setdefault(q_type, []).append(row)
                all_rows.append(row)

        if self._qa_doc_mapping:
            mapping_by_question = {
                item.get("question", "").strip(): item
                for item in self._qa_doc_mapping
                if item.get("question")
            }
            samples: List[StandardSample] = []
            for local_idx, row in enumerate(all_rows):
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                q_type = row.get("Type", "Unknown").strip()
                mapping_item = mapping_by_question.get(question, {})
                source_files = mapping_item.get("source_files", [])
                target_doc_paths = self._resolve_source_files(source_files)
                sample_id = self._mapping_sample_id(mapping_item, local_idx)
                qa = StandardQA(
                    question=question,
                    gold_answers=[answer] if answer else [],
                    evidence=[],
                    category=q_type,
                    metadata={
                        "row_index": mapping_item.get("row_index", local_idx),
                        "local_index": local_idx,
                        "source_files": source_files,
                        "target_doc_paths": target_doc_paths,
                    },
                )
                samples.append(StandardSample(
                    sample_id=sample_id,
                    qa_pairs=[qa],
                    metadata={"question_type": q_type, "num_questions": 1, "source_files": source_files},
                ))

            self.logger.info(f"[VersionRAG] Loaded {len(samples)} mapped per-query samples")
            return samples

        samples: List[StandardSample] = []
        for q_type, rows in groups.items():
            qa_pairs = []
            for idx, row in enumerate(rows):
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                qa_pairs.append(StandardQA(
                    question=question,
                    gold_answers=[answer] if answer else [],
                    evidence=[],
                    category=q_type,
                    metadata={"row_index": idx},
                ))

            samples.append(StandardSample(
                sample_id=self._slugify(q_type),
                qa_pairs=qa_pairs,
                metadata={"question_type": q_type, "num_questions": len(qa_pairs)},
            ))

        total_q = sum(len(s.qa_pairs) for s in samples)
        self.logger.info(f"[VersionRAG] Loaded {total_q} questions across {len(samples)} type groups")
        return samples

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{MISSING_RULE}\n\n{QA_PROMPT.format(qa.question)}"
        meta = {
            "question_type": qa.category,
        }
        return full_prompt, meta

    @staticmethod
    def _slugify(text: str) -> str:
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        return text

    def _load_qa_doc_mapping(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.qa_doc_mapping_path):
            return []
        with open(self.qa_doc_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            self.logger.warning(f"[VersionRAG] Invalid qa_doc_mapping format: {self.qa_doc_mapping_path}")
            return []
        return data

    @staticmethod
    def _normalize_source_file(source_file: str) -> str:
        return os.path.normpath(str(source_file)).replace("\\", "/")

    @staticmethod
    def _mapping_sample_id(mapping_item: Dict[str, Any], fallback_idx: int) -> str:
        return f"qa_{mapping_item.get('row_index', fallback_idx)}"

    def _resolve_source_files(self, source_files: List[str]) -> List[str]:
        doc_paths = []
        for source_file in source_files:
            normalized = self._normalize_source_file(source_file)
            processed_path = self._source_to_processed.get(normalized)
            if processed_path:
                doc_paths.append(processed_path)
            else:
                self.logger.warning(f"[VersionRAG] Mapped source file not found: {source_file}")
        return doc_paths
