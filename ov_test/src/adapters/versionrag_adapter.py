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
import hashlib
import json
import os
import re
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
        self.data_dir = os.path.normpath(os.path.join(test_dir, ".."))
        self.dataset_root = os.path.normpath(os.path.join(self.data_dir, ".."))
        self.raw_doc_dir = os.path.join(test_dir, "..", "raw")
        self.raw_doc_dir = os.path.normpath(self.raw_doc_dir)
        self.mapping_path = os.path.join(self.dataset_root, "qa_doc_mapping.json")
        self.grouping_mode = "type"
        self._prepared_docs_by_sample_id: Dict[str, List[str]] = {}
        self._mapping_rows: List[Dict[str, Any]] | None = None

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

        if self._use_qa_doc_mapping():
            return self._data_prepare_from_mapping(doc_dir)

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

    def _data_prepare_from_mapping(self, doc_dir: str) -> List[StandardDoc]:
        mapping_rows = self._load_mapping_rows()
        prepared_by_source: Dict[str, str] = {}
        docs_by_sample_id: Dict[str, List[str]] = {}

        for item in mapping_rows:
            source_files = self._source_files_for_mapping_item(item)
            sample_id = self._sample_id_for_source_files(source_files)
            if sample_id in docs_by_sample_id:
                continue

            prepared_paths: List[str] = []
            for source_file in source_files:
                raw_path = self._resolve_source_file(source_file)
                if not os.path.exists(raw_path):
                    raise FileNotFoundError(f"Mapped source file not found: {raw_path}")

                if source_file not in prepared_by_source:
                    dst_path = os.path.join(doc_dir, os.path.basename(raw_path))
                    shutil.copy2(raw_path, dst_path)
                    prepared_by_source[source_file] = dst_path
                prepared_paths.append(prepared_by_source[source_file])

            docs_by_sample_id[sample_id] = prepared_paths

        self._prepared_docs_by_sample_id = docs_by_sample_id
        docs = [
            StandardDoc(sample_id=sample_id, doc_paths=doc_paths)
            for sample_id, doc_paths in docs_by_sample_id.items()
        ]
        self.logger.info(
            f"[VersionRAG] Prepared {len(docs)} qa_doc_mapping doc groups "
            f"with {len(prepared_by_source)} files for ingestion"
        )
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

        if self._use_qa_doc_mapping():
            return self._load_and_transform_from_mapping()

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

    def _load_and_transform_from_mapping(self) -> List[StandardSample]:
        rows = self._read_eval_rows()
        mapping_rows = self._load_mapping_rows()
        mapping_by_index = {
            int(item["row_index"]): item
            for item in mapping_rows
            if "row_index" in item
        }

        samples: List[StandardSample] = []
        for row_index, row in enumerate(rows):
            mapping_item = mapping_by_index.get(row_index)
            if mapping_item is None:
                raise KeyError(f"qa_doc_mapping missing row_index={row_index}")

            q_type = row.get("Type", "Unknown").strip()
            question = row.get("Question", "").strip()
            answer = row.get("Answer", "").strip()
            source_files = self._source_files_for_mapping_item(mapping_item)
            sample_id = self._sample_id_for_source_files(source_files)

            qa = StandardQA(
                question=question,
                gold_answers=[answer] if answer else [],
                evidence=[],
                category=q_type,
                metadata={
                    "row_index": row_index,
                    "source_files": source_files,
                    "mapping_type": mapping_item.get("type"),
                },
            )
            samples.append(StandardSample(
                sample_id=sample_id,
                qa_pairs=[qa],
                metadata={
                    "row_index": row_index,
                    "question_type": q_type,
                    "source_files": source_files,
                },
            ))

        self.logger.info(
            f"[VersionRAG] Loaded {len(samples)} questions with qa_doc_mapping groups"
        )
        return samples

    def _use_qa_doc_mapping(self) -> bool:
        mode = str(getattr(self, "grouping_mode", "") or "").lower()
        return mode in {"qa_doc_mapping", "doc_mapping", "per_doc", "doc_paths"}

    def _read_eval_rows(self) -> List[Dict[str, str]]:
        with open(self.raw_file_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _load_mapping_rows(self) -> List[Dict[str, Any]]:
        if self._mapping_rows is not None:
            return self._mapping_rows
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"qa_doc_mapping not found: {self.mapping_path}")
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"qa_doc_mapping must be a list, got: {type(data).__name__}")
        self._mapping_rows = data
        return data

    @staticmethod
    def _source_files_for_mapping_item(item: Dict[str, Any]) -> List[str]:
        source_files = item.get("source_files") or []
        if isinstance(source_files, str):
            source_files = [source_files]
        source_files = [str(path) for path in source_files if path]
        if not source_files:
            raise ValueError(f"qa_doc_mapping item has no source_files: {item}")
        return source_files

    def _resolve_source_file(self, source_file: str) -> str:
        if os.path.isabs(source_file):
            return source_file
        return os.path.normpath(os.path.join(self.dataset_root, source_file))

    def _sample_id_for_source_files(self, source_files: List[str]) -> str:
        if len(source_files) == 1:
            name = os.path.splitext(os.path.basename(source_files[0]))[0]
            return self._slugify(name)
        raw = "|".join(sorted(source_files))
        prefix = self._slugify(os.path.splitext(os.path.basename(source_files[0]))[0])[:80]
        suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}__{suffix}"

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
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[-\s]+", "_", text)
        return text
