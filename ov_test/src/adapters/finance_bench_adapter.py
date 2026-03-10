# src/adapters/finance_bench_adapter.py
"""
FinanceBench 数据集适配器

FinanceBench 是一个金融领域的问答数据集，文档为 SEC 财报 PDF。
数据格式：JSONL，每行包含 question、answer、doc_name、evidence 等字段。
evidence 中的 evidence_text 用于 recall 计算。
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

QA_PROMPT = """Based on the financial document excerpts above, answer the following question accurately and concisely.
If the answer involves a numerical value, include the unit (e.g., USD millions, %, etc.).

Question: {}
Answer:"""

MISSING_RULE = "If the provided context does not contain sufficient information to answer the question, respond with 'Insufficient information'."


class FinanceBenchAdapter(BaseAdapter):
    """
    FinanceBench 数据集适配器。
    处理金融领域的问答数据，文档为 SEC 财报 PDF。
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        data_dir = os.path.dirname(self.raw_file_path)
        self.pdf_dir = os.path.join(os.path.dirname(data_dir), "pdfs")

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        准备入库文档列表。仅入库 JSONL 中引用的文档。
        """
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")

        # 收集 JSONL 中引用的文档名
        doc_names = set()
        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    doc_names.add(json.loads(line)["doc_name"])

        docs: List[StandardDoc] = []
        for doc_name in sorted(doc_names):
            pdf_path = os.path.join(self.pdf_dir, f"{doc_name}.pdf")
            if not os.path.exists(pdf_path):
                self.logger.warning(f"PDF not found: {pdf_path}, skipping")
                continue
            docs.append(StandardDoc(sample_id=doc_name, doc_paths=[pdf_path]))

        self.logger.info(f"[FinanceBench] Prepared {len(docs)} documents for ingestion (referenced only)")
        return docs

    def load_and_transform(self) -> List[StandardSample]:
        """
        解析 JSONL 问题文件，按 doc_name 分组为 StandardSample。
        evidence 使用每条 evidence 中的 evidence_text 字段。
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        groups: Dict[str, List[Dict]] = defaultdict(list)
        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                groups[item["doc_name"]].append(item)

        samples: List[StandardSample] = []
        for doc_name, items in groups.items():
            qa_pairs = []
            for item in items:
                # 提取 evidence_text 用于 recall 计算
                evidence_texts = [
                    ev["evidence_text"]
                    for ev in item.get("evidence", [])
                    if ev.get("evidence_text")
                ]

                qa_pairs.append(StandardQA(
                    question=item["question"],
                    gold_answers=[item["answer"]],
                    evidence=evidence_texts,
                    category=item.get("question_type"),
                    metadata={
                        "financebench_id": item.get("financebench_id"),
                        "question_reasoning": item.get("question_reasoning"),
                        "justification": item.get("justification", ""),
                        "company": item.get("company"),
                    }
                ))

            samples.append(StandardSample(
                sample_id=doc_name,
                qa_pairs=qa_pairs,
            ))

        self.logger.info(
            f"[FinanceBench] Loaded {sum(len(s.qa_pairs) for s in samples)} questions "
            f"across {len(samples)} documents"
        )
        return samples

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{MISSING_RULE}\n\n{QA_PROMPT.format(qa.question)}"
        meta = {
            "question_type": qa.category,
            "financebench_id": qa.metadata.get("financebench_id"),
        }
        return full_prompt, meta
