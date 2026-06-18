# src/adapters/versionrag_adapter.py
"""
VersionRAG 数据集适配器

VersionRAG 是一个面向版本化文档集合的检索增强问答数据集。
原始文档混合了 Markdown 与 PDF 两种格式（Node.js / Bootstrap 为 Markdown，
Apache Spark 为 PDF），QA 数据以 CSV 形式存储，无显式文档-问题映射。

本适配器在 data_prepare 阶段按目标 store 准备文档；MoDora 路径会保留
Markdown/PDF 原始输入，统一交给 MoDora store 物化为 PDF。
"""

import csv
import os
import shutil
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA, EVIDENCE_BASED_ASSESSMENT_INSTRUCTION

QA_PROMPT = """Based on the provided context, answer the following question accurately and concisely.
Use the exact wording from the context whenever possible.

Question: {}"""

ASSESSMENT_INSTRUCTION = EVIDENCE_BASED_ASSESSMENT_INSTRUCTION


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

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        准备入库文档列表。

        处理策略：
        1. 扫描 data/raw/ 下的所有文件；
        2. MoDora: Markdown/PDF 原样复制，统一由 MoDora store 物化为 PDF；
        3. 返回所有文档路径列表。
        """
        if not os.path.exists(self.raw_doc_dir):
            raise FileNotFoundError(f"Raw document directory not found: {self.raw_doc_dir}")

        os.makedirs(doc_dir, exist_ok=True)

        docs: List[StandardDoc] = []
        all_doc_paths: List[str] = []
        target_store = str(getattr(self, "target_store_type", "") or "").lower()
        use_modora_docs = target_store == "modora"
        for filename in sorted(os.listdir(self.raw_doc_dir)):
            raw_path = os.path.join(self.raw_doc_dir, filename)
            if not os.path.isfile(raw_path):
                continue

            name, ext = os.path.splitext(filename)
            ext_lower = ext.lower()

            if ext_lower == ".md":
                if use_modora_docs:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    all_doc_paths.append(dst_path)
                else:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    all_doc_paths.append(dst_path)

            elif ext_lower == ".pdf":
                if use_modora_docs:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    all_doc_paths.append(dst_path)
                else:
                    md_filename = f"{name}.md"
                    md_path = os.path.join(doc_dir, md_filename)
                    if not os.path.exists(md_path):
                        self._pdf_to_markdown(raw_path, md_path)
                    all_doc_paths.append(md_path)

        for sample_id in self._sample_ids():
            docs.append(StandardDoc(sample_id=sample_id, doc_paths=list(all_doc_paths)))

        file_count = len(all_doc_paths)
        self.logger.info(f"[VersionRAG] Prepared {len(docs)} doc groups with {file_count} files for ingestion")
        return docs

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
                if sample_id not in seen:
                    seen.add(sample_id)
                    sample_ids.append(sample_id)
        return sample_ids

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{ASSESSMENT_INSTRUCTION}\n\n{QA_PROMPT.format(qa.question)}"
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
