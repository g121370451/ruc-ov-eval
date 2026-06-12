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
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple

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

        # 数据集根目录（{versionrag_dir}），用于解析 qa_doc_mapping.json
        self.dataset_root = os.path.normpath(os.path.join(test_dir, "..", ".."))
        mapping_path = os.path.join(self.dataset_root, "qa_doc_mapping.json")
        self.qa_doc_mapping_path: Optional[str] = mapping_path if os.path.isfile(mapping_path) else None

        # 默认保持 global 实验的旧行为；per-question pipeline 会显式打开该开关。
        self.use_qa_doc_mapping = False

        # 文档基名 -> 入库后 markdown 路径，由 data_prepare 填充
        self._doc_basename_to_path: Dict[str, str] = {}

    def enable_qa_doc_mapping(self, enabled: bool = True):
        """显式启用/关闭基于 qa_doc_mapping.json 的 per-question 组织方式。"""
        self.use_qa_doc_mapping = enabled

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        准备入库文档列表。

        处理策略：
        1. 扫描 data/raw/ 下的所有文件；
        2. Markdown 文件直接复制到 doc_dir；
        3. PDF 文件使用 pymupdf (fitz) 提取文本并保存为同名 .md；
        4. 仅当 per-question pipeline 显式启用 qa_doc_mapping 时，才按每条 QA
           输出 StandardDoc，sample_id 形如 ``qa_{row_index}``；
        5. 否则按原有“每文档一个 sample”行为返回。
        """
        if not os.path.exists(self.raw_doc_dir):
            raise FileNotFoundError(f"Raw document directory not found: {self.raw_doc_dir}")

        os.makedirs(doc_dir, exist_ok=True)

        per_doc_entries: List[StandardDoc] = []
        self._doc_basename_to_path = {}
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
                per_doc_entries.append(StandardDoc(sample_id=name, doc_paths=[dst_path]))
                self._doc_basename_to_path[name] = dst_path

            elif ext_lower == ".pdf":
                # PDF -> Markdown 转换；若当前环境缺少 PyMuPDF，则保留 PDF 原文件交给后端解析。
                md_filename = f"{name}.md"
                md_path = os.path.join(doc_dir, md_filename)
                if self._has_pymupdf():
                    if not os.path.exists(md_path):
                        self._pdf_to_markdown(raw_path, md_path)
                    per_doc_entries.append(StandardDoc(sample_id=name, doc_paths=[md_path]))
                    self._doc_basename_to_path[name] = md_path
                else:
                    dst_path = os.path.join(doc_dir, filename)
                    shutil.copy2(raw_path, dst_path)
                    per_doc_entries.append(StandardDoc(sample_id=name, doc_paths=[dst_path]))
                    self._doc_basename_to_path[name] = dst_path
                    self.logger.warning(f"[VersionRAG] PyMuPDF not available; copied PDF directly: {filename}")

        if not self.use_qa_doc_mapping or self.qa_doc_mapping_path is None:
            self.logger.info(
                f"[VersionRAG] Prepared {len(per_doc_entries)} documents for ingestion "
                f"({len([d for d in per_doc_entries if d.doc_paths[0].endswith('.md')])} md)"
            )
            return per_doc_entries

        # Per-question 模式：基于 qa_doc_mapping.json 构造 StandardDoc
        mapping = self._load_qa_doc_mapping()
        docs: List[StandardDoc] = []
        missing: List[str] = []
        for entry in mapping:
            qa_id = self._qa_id_for(entry)
            doc_paths = self._resolve_source_files(entry.get("source_files", []), missing)
            if not doc_paths:
                continue
            docs.append(StandardDoc(sample_id=qa_id, doc_paths=doc_paths))

        if missing:
            uniq_missing = sorted(set(missing))
            self.logger.warning(
                f"[VersionRAG] {len(uniq_missing)} mapping doc(s) not found, sample: {uniq_missing[:3]}"
            )

        self.logger.info(
            f"[VersionRAG] Prepared {len(docs)} per-question doc bindings over "
            f"{len(self._doc_basename_to_path)} unique documents"
        )
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
        解析 evaluation_set.csv，生成 StandardSample。

        若 per-question pipeline 显式启用了 ``qa_doc_mapping.json``：
          - 每条 QA 单独成一个 sample，``sample_id = qa_{row_index}``，与
            ``data_prepare`` 输出对齐，使 per-question pipeline 能按问题级别建库；
          - QA 优先使用 mapping 中的 ``question``/``answer``/``type``；
          - metadata 中保留 ``source_files`` 与解析后的 ``doc_paths``。

        否则保留旧行为：按问题类型聚合。
        """
        if self.use_qa_doc_mapping and self.qa_doc_mapping_path is not None:
            return self._load_per_question_samples()
        return self._load_grouped_samples()

    def _load_per_question_samples(self) -> List[StandardSample]:
        mapping = self._load_qa_doc_mapping()
        # 若 data_prepare 尚未运行，则尝试推导 doc_dir：默认与 sql_agent_global 配置一致
        # 但我们更稳妥的做法是依赖 data_prepare 已填充的映射；如果为空则给出警告
        if not self._doc_basename_to_path:
            self.logger.warning(
                "[VersionRAG] _doc_basename_to_path is empty; load_and_transform may be called "
                "before data_prepare. doc_paths metadata will rely on raw filenames."
            )

        samples: List[StandardSample] = []
        skipped: List[str] = []
        for entry in mapping:
            question = (entry.get("question") or "").strip()
            answer = (entry.get("answer") or "").strip()
            q_type = (entry.get("type") or "Unknown").strip()
            row_index = entry.get("row_index")

            doc_paths = self._resolve_source_files(entry.get("source_files", []), skipped)
            if not doc_paths:
                continue

            qa_id = self._qa_id_for(entry)
            qa = StandardQA(
                question=question,
                gold_answers=[answer] if answer else [],
                evidence=[],
                category=q_type,
                metadata={
                    "row_index": row_index,
                    "question_type": q_type,
                    "source_files": list(entry.get("source_files", [])),
                    "doc_paths": doc_paths,
                },
            )
            samples.append(StandardSample(
                sample_id=qa_id,
                qa_pairs=[qa],
                metadata={
                    "question_type": q_type,
                    "row_index": row_index,
                    "source_files": list(entry.get("source_files", [])),
                    "doc_paths": doc_paths,
                    "num_questions": 1,
                },
            ))

        self.logger.info(
            f"[VersionRAG] Loaded {len(samples)} per-question samples "
            f"(mapping entries={len(mapping)}, skipped={len(skipped)})"
        )
        return samples

    def _load_grouped_samples(self) -> List[StandardSample]:
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

    # ------------------------------------------------------------------
    # qa_doc_mapping.json 相关辅助方法
    # ------------------------------------------------------------------

    def _load_qa_doc_mapping(self) -> List[Dict[str, Any]]:
        """读取 qa_doc_mapping.json，结果按 row_index 排序。"""
        assert self.qa_doc_mapping_path is not None
        with open(self.qa_doc_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"qa_doc_mapping.json must be a list, got {type(data).__name__}"
            )
        # 按 row_index 排序，保证可复现性
        try:
            data.sort(key=lambda d: int(d.get("row_index", 0)))
        except Exception:
            pass
        return data

    @staticmethod
    def _qa_id_for(entry: Dict[str, Any]) -> str:
        """根据 mapping entry 生成 sample_id（与 data_prepare 一致）。"""
        row_index = entry.get("row_index")
        if row_index is None:
            # 退化方案：以 question 哈希
            import hashlib
            q = (entry.get("question") or "").strip()
            return f"qa_{hashlib.sha1(q.encode('utf-8')).hexdigest()[:12]}"
        return f"qa_{int(row_index)}"

    def _resolve_source_files(
        self,
        source_files: List[str],
        missing_out: Optional[List[str]] = None,
    ) -> List[str]:
        """将 mapping 中的 source_files 解析为 data_prepare 后的 markdown 文件路径。"""
        resolved: List[str] = []
        for sf in source_files:
            if not sf:
                continue
            base = os.path.splitext(os.path.basename(sf))[0]
            # 优先复用 data_prepare 填好的映射
            mapped = self._doc_basename_to_path.get(base)
            if mapped and os.path.isfile(mapped):
                resolved.append(mapped)
                continue
            # 退化：尝试常见的 doc_output_dir 推断（默认 sql_agent/processed_docs）
            candidate_md = os.path.join(self.dataset_root, "sql_agent", "processed_docs", f"{base}.md")
            if os.path.isfile(candidate_md):
                resolved.append(candidate_md)
                self._doc_basename_to_path.setdefault(base, candidate_md)
                continue
            # 最后退化：原始 raw 路径
            raw_candidate = os.path.normpath(os.path.join(self.dataset_root, sf))
            if os.path.isfile(raw_candidate):
                resolved.append(raw_candidate)
                continue
            if missing_out is not None:
                missing_out.append(sf)
        # 去重并保持顺序
        seen = set()
        unique: List[str] = []
        for p in resolved:
            if p in seen:
                continue
            seen.add(p)
            unique.append(p)
        return unique
