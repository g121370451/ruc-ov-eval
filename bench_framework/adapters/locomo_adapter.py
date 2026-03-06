import json
import os
from typing import Dict, List, Tuple

from bench_framework.adapters.base import BaseAdapter, StandardDoc, StandardSample, StandardQA
from bench_framework.types import Metadata

QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."

class LocomoAdapter(BaseAdapter):
    """
    专门用于处理 LocoMo 数据集的适配器。
    将 Session 格式的 JSON 转换为带有时间信息的 Markdown。
    """
    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        res: List[StandardDoc] = []

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset = [data] if isinstance(data, dict) else data
        os.makedirs(doc_dir, exist_ok=True)
        for item in dataset:
            sample_id = item.get("sample_id", "unknown")
            doc_content = self._convert_conversation_to_markdown(sample_id, item.get("conversation", {}))

            try:
                doc_path = os.path.join(doc_dir, f"{sample_id}_doc.md")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                res.append(StandardDoc(sample_id, doc_path))
            except Exception as e:
                self.logger.error(f"[locomo adapter] doc:{sample_id} prepare error {e}")
                raise e
        return res

    def load_and_transform(self) -> List[StandardSample]:
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset = [data] if isinstance(data, dict) else data

        standard_samples = []

        for item in dataset:
            sample_id = item.get("sample_id", "unknown")

            qa_pairs = []
            for q in item.get("qa", []):
                if str(q.get("category")) == "5":
                    continue
                raw_ans = q.get("answer")

                if isinstance(raw_ans, list):
                    golds = raw_ans
                elif raw_ans is None or raw_ans == "":
                    golds = ["Not mentioned"]
                else:
                    golds = [raw_ans]

                qa_pairs.append(StandardQA(
                    question=q["question"],
                    gold_answers=[str(g) for g in golds],
                    evidence=q.get("evidence", []),
                    category=q.get("category"),
                    metadata={"original_id": q.get("id")}
                ))

            standard_samples.append(StandardSample(
                sample_id=sample_id,
                qa_pairs=qa_pairs
            ))

        return standard_samples

    def _convert_conversation_to_markdown(self, sample_id: str, conv: dict) -> str:
        md_lines = [f"# Chat History: {sample_id}"]

        session_idx = 1
        while f"session_{session_idx}" in conv:
            s_key = f"session_{session_idx}"
            dt_key = f"session_{session_idx}_date_time"
            sum_key = f"session_{session_idx}_summary"

            md_lines.append(f"\n## Session {session_idx}")

            session_dt = conv.get(dt_key)
            if session_dt:
                md_lines.append(f"DATE: {session_dt}")

            session_sum = conv.get(sum_key)
            if session_sum:
                md_lines.append(f"SUMMARY: {session_sum}")

            for turn in conv[s_key]:
                spk = turn.get("speaker", "Unknown")
                txt = turn.get("text", "")

                raw_id = turn.get("dia_id") or turn.get("id")
                suffix = f" [{raw_id}]" if raw_id else ""

                md_lines.append(f"**{spk}**: {txt}{suffix}")

            session_idx += 1

        return "\n".join(md_lines)

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> Tuple[str, Metadata]:
        category = str(qa.category)
        eff_q = qa.question

        if category == "2":
            eff_q += " Use DATE of CONVERSATION to answer with an approximate date."

        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{MISSING_RULE}\n\n{QA_PROMPT.format(eff_q)}"

        meta: Metadata = {"category": category}
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Metadata) -> str:
        return raw_answer.strip()