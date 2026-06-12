# src/adapters/versionrag_adapter.py
"""
VersionRAG 数据集适配器

VersionRAG 是一个面向版本化文档集合的检索增强问答数据集。
原始文档混合了 Markdown 与 PDF 两种格式（Node.js / Bootstrap 为 Markdown，
Apache Spark 为 PDF），QA 数据以 CSV 形式存储，无显式文档-问题映射。

本适配器在 data_prepare 阶段会提前将所有 PDF 转换为 Markdown，
以确保下游各 Store Wrapper（尤其是 OpenViking / PageIndex）获得最佳解析效果。
"""

import csv
import json
import os
import shutil
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

QA_PROMPT = """Based on the provided context, answer the following question accurately and concisely.
Use the exact wording from the context whenever possible.

Question: {}
Answer:"""

MISSING_RULE = "If the provided context does not contain sufficient information to answer the question, respond with 'Not mentioned'."


class VersionRAGAdapter(BaseAdapter):
    """
    VersionRAG 数据集适配器。

    目录结构约定（相对于 raw_file_path）：
      {versionrag_dir}/
        data/raw/          # 原始文档（.md + .pdf 混合）
        data/test/evaluation_set.csv   # QA 文件
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        # raw_file_path 指向 data/test/evaluation_set.csv
        test_dir = os.path.dirname(self.raw_file_path)
        self.raw_doc_dir = os.path.join(test_dir, "..", "raw")
        self.raw_doc_dir = os.path.normpath(self.raw_doc_dir)
        self.dataset_root = os.path.normpath(os.path.join(test_dir, "..", ".."))
        self.qa_doc_mapping_path = os.path.join(self.dataset_root, "qa_doc_mapping.json")
        self._use_qa_doc_mapping = False
        self._qa_doc_mapping: List[Dict[str, Any]] = []

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        准备入库文档列表。

        处理策略：
        1. 扫描 data/raw/ 下的所有文件；
        2. Markdown 文件直接复制到 doc_dir；
        3. PDF 文件使用 pymupdf (fitz) 提取文本并保存为同名 .md；
        4. 返回所有文档路径列表。
        """
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
                # 直接复制 Markdown 文件
                dst_path = os.path.join(doc_dir, filename)
                shutil.copy2(raw_path, dst_path)
                docs.append(StandardDoc(sample_id=name, doc_paths=[dst_path]))
                processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = dst_path

            elif ext_lower == ".pdf":
                # PDF -> Markdown 转换；若当前环境缺少 PyMuPDF，则保留 PDF 原文件交给后端解析。
                md_filename = f"{name}.md"
                md_path = os.path.join(doc_dir, md_filename)
                if self._has_pymupdf():
                    if not os.path.exists(md_path):
                        self._pdf_to_markdown(raw_path, md_path)
                    docs.append(StandardDoc(sample_id=name, doc_paths=[md_path]))
                    processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = md_path
                else:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    docs.append(StandardDoc(sample_id=name, doc_paths=[dst_path]))
                    processed_by_source[self._normalize_source_file(f"data/raw/{filename}")] = dst_path
                    self.logger.warning(f"[VersionRAG] PyMuPDF not available; copied PDF directly: {filename}")

        self._qa_doc_mapping = self._load_qa_doc_mapping()
        self._use_qa_doc_mapping = (
            bool(self._qa_doc_mapping)
            and "per_question" in os.path.normpath(doc_dir).replace("\\", "/").lower()
        )
        if self._use_qa_doc_mapping:
            mapped_docs = self._build_mapped_docs(processed_by_source)
            self.logger.info(
                f"[VersionRAG] Loaded QA-doc mapping: {len(mapped_docs)} per-question document groups"
            )
            return mapped_docs

        self.logger.info(f"[VersionRAG] Prepared {len(docs)} documents for ingestion ({len([d for d in docs if d.doc_paths[0].endswith('.md')])} md)")
        return docs

    @staticmethod
    def _has_pymupdf() -> bool:
        try:
            import fitz  # noqa: F401
            return True
        except ImportError:
            return False

    def _pdf_to_markdown(self, pdf_path: str, md_path: str):
        """
        使用 pymupdf (fitz) 从数字原生 PDF 提取文本并保存为 Markdown。
        每页以 `## Page N` 作为标题，保留段落换行。
        """
        import fitz  # pymupdf

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
        """
        解析 evaluation_set.csv，按问题类型（Type）分组为 StandardSample。

        CSV 格式：Type, Question, Answer
        - 无显式 evidence 字段，因此 recall 将自然为 0；
        - category 使用 Type 字段；
        - 同类型的问题聚合到同一个 sample 中。
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Evaluation set not found: {self.raw_file_path}")

        rows_by_type: Dict[str, List[Dict[str, str]]] = {}
        all_rows: List[Dict[str, str]] = []
        with open(self.raw_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for local_idx, row in enumerate(reader):
                row["_local_index"] = str(local_idx)
                q_type = row.get("Type", "Unknown").strip()
                rows_by_type.setdefault(q_type, []).append(row)
                all_rows.append(row)

        if self._use_qa_doc_mapping:
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
                sample_id = self._mapping_sample_id(mapping_item, local_idx)
                qa = StandardQA(
                    question=question,
                    gold_answers=[answer] if answer else [],
                    evidence=[],
                    category=q_type,
                    metadata={
                        "row_index": mapping_item.get("row_index", local_idx),
                        "local_index": local_idx,
                        "source_files": mapping_item.get("source_files", []),
                    },
                )
                samples.append(StandardSample(
                    sample_id=sample_id,
                    qa_pairs=[qa],
                    metadata={
                        "question_type": q_type,
                        "num_questions": 1,
                        "source_files": mapping_item.get("source_files", []),
                    },
                ))

            self.logger.info(f"[VersionRAG] Loaded {len(samples)} mapped per-question samples")
            return samples

        samples: List[StandardSample] = []
        for q_type, rows in rows_by_type.items():
            qa_pairs = []
            for idx, row in enumerate(rows):
                question = row.get("Question", "").strip()
                answer = row.get("Answer", "").strip()
                qa_pairs.append(StandardQA(
                    question=question,
                    gold_answers=[answer] if answer else [],
                    evidence=[],  # VersionRAG 未提供标注 evidence
                    category=q_type,
                    metadata={"row_index": idx},
                ))

            # sample_id 使用 slugified 的 Type 名
            sample_id = self._slugify(q_type)
            samples.append(StandardSample(
                sample_id=sample_id,
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
        """将类型名称转换为合法的 sample_id（去除特殊字符、空格替换为下划线）。"""
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
        return os.path.normpath(source_file).replace("\\", "/")

    @staticmethod
    def _mapping_sample_id(mapping_item: Dict[str, Any], fallback_idx: int) -> str:
        row_index = mapping_item.get("row_index", fallback_idx)
        return f"qa_{row_index}"

    def _build_mapped_docs(self, processed_by_source: Dict[str, str]) -> List[StandardDoc]:
        mapped_docs: List[StandardDoc] = []
        for idx, item in enumerate(self._qa_doc_mapping):
            doc_paths = []
            for source_file in item.get("source_files", []):
                normalized = self._normalize_source_file(source_file)
                processed_path = processed_by_source.get(normalized)
                if processed_path:
                    doc_paths.append(processed_path)
                else:
                    self.logger.warning(f"[VersionRAG] Mapped source file not found: {source_file}")
            if not doc_paths:
                continue
            mapped_docs.append(StandardDoc(
                sample_id=self._mapping_sample_id(item, idx),
                doc_paths=doc_paths,
                metadata={
                    "row_index": item.get("row_index", idx),
                    "source_files": item.get("source_files", []),
                },
            ))
        return mapped_docs
