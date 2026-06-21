# src/adapters/versionrag_adapter.py
"""
VersionRAG 数据集适配器。

VersionRAG 是面向版本化文档集合的检索增强问答数据集。原始文档混合
Markdown 与 PDF，QA 数据以 CSV 形式存储，无显式文档-问题映射。
"""

import csv
import os
import re
import shutil
from typing import Any, Dict, List

from .base import (
    BaseAdapter,
    EVIDENCE_BASED_ASSESSMENT_INSTRUCTION,
    StandardDoc,
    StandardQA,
    StandardSample,
)


QA_PROMPT = """Based on the provided context, answer the following question accurately and concisely.
Use the exact wording from the context whenever possible.

Question: {}"""

ASSESSMENT_INSTRUCTION = EVIDENCE_BASED_ASSESSMENT_INSTRUCTION


class VersionRAGAdapter(BaseAdapter):
    """
    VersionRAG 数据集适配器。

    目录结构约定：
      {versionrag_dir}/
        data/raw/
        data/test/evaluation_set.csv
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        test_dir = os.path.dirname(self.raw_file_path)
        self.raw_doc_dir = os.path.normpath(os.path.join(test_dir, "..", "raw"))

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        if not os.path.exists(self.raw_doc_dir):
            raise FileNotFoundError(f"Raw document directory not found: {self.raw_doc_dir}")

        os.makedirs(doc_dir, exist_ok=True)

        all_doc_paths: List[str] = []
        for filename in sorted(os.listdir(self.raw_doc_dir)):
            raw_path = os.path.join(self.raw_doc_dir, filename)
            if not os.path.isfile(raw_path):
                continue

            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()
            if ext_lower == ".md":
                dst_path = os.path.join(doc_dir, filename)
                shutil.copy2(raw_path, dst_path)
                all_doc_paths.append(dst_path)
            elif ext_lower == ".pdf":
                md_path = os.path.join(doc_dir, f"{name}.md")
                if not os.path.exists(md_path):
                    self._pdf_to_markdown(raw_path, md_path)
                all_doc_paths.append(md_path)

        docs = [
            StandardDoc(sample_id=sample_id, doc_paths=list(all_doc_paths))
            for sample_id in self._sample_ids()
        ]

        self.logger.info(
            f"[VersionRAG] Prepared {len(docs)} doc groups with {len(all_doc_paths)} files for ingestion"
        )
        return docs

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

    def load_and_transform(self) -> List[StandardSample]:
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Evaluation set not found: {self.raw_file_path}")

        groups: Dict[str, List[Dict[str, str]]] = {}
        with open(self.raw_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_type = row.get("Type", "Unknown").strip()
                groups.setdefault(q_type, []).append(row)

        samples: List[StandardSample] = []
        for q_type, rows in groups.items():
            qa_pairs = []
            for idx, row in enumerate(rows):
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                qa_pairs.append(
                    StandardQA(
                        question=question,
                        gold_answers=[answer] if answer else [],
                        evidence=[],
                        category=q_type,
                        metadata={"row_index": idx},
                    )
                )

            sample_id = self._slugify(q_type)
            samples.append(
                StandardSample(
                    sample_id=sample_id,
                    qa_pairs=qa_pairs,
                    metadata={"question_type": q_type, "num_questions": len(qa_pairs)},
                )
            )

        total_q = sum(len(sample.qa_pairs) for sample in samples)
        self.logger.info(f"[VersionRAG] Loaded {total_q} questions across {len(samples)} type groups")
        return samples

    def _sample_ids(self) -> List[str]:
        if not os.path.exists(self.raw_file_path):
            return []
        sample_ids = []
        seen = set()
        with open(self.raw_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_type = row.get("Type", "Unknown").strip()
                sample_id = self._slugify(q_type)
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                sample_ids.append(sample_id)
        return sample_ids

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{ASSESSMENT_INSTRUCTION}\n\n{QA_PROMPT.format(qa.question)}"
        return full_prompt, {"question_type": qa.category}

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        return text
