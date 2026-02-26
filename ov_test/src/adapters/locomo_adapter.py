# src/adapters/locomo_adapter.py
import json
import os
from typing import List, Dict, Any

from .base import BaseAdapter, StandardSample, StandardQA

class LocomoAdapter(BaseAdapter):
    """
    专门用于处理 LocoMo 数据集的适配器。
    将 Session 格式的 JSON 转换为带有时间信息的 Markdown。
    """

    def load_and_transform(self) -> List[StandardSample]:
        """
        加载原始 JSON 数据并转换为标准化的 StandardSample 对象列表。
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容 dataset 是列表或单字典的情况
            dataset = [data] if isinstance(data, dict) else data

        standard_samples = []

        for item in dataset:
            sample_id = item.get("sample_id", "unknown")
            
            # 1. 转换文档内容
            doc_content = self._convert_conversation_to_markdown(sample_id, item.get("conversation", {}))
            
            # 2. 转换 QA 对
            qa_pairs = []
            for q in item.get("qa", []):
                raw_ans = q.get("answer")
                
                # --- 确保 golds 始终是可迭代的列表，即使原答案是 int 或 float ---
                if isinstance(raw_ans, list):
                    golds = raw_ans
                elif raw_ans is None or raw_ans == "":
                    golds = ["Not mentioned"]
                else:
                    # 将单值（str, int, float 等）包装在列表中
                    golds = [raw_ans]
                
                qa_pairs.append(StandardQA(
                    question=q["question"],
                    # 确保将列表中的每个元素都转为字符串
                    gold_answers=[str(g) for g in golds],
                    evidence=q.get("evidence", []),
                    category=q.get("category"),
                    metadata={"original_id": q.get("id")}
                ))

            standard_samples.append(StandardSample(
                sample_id=sample_id,
                doc_content=doc_content,
                qa_pairs=qa_pairs
            ))

        return standard_samples

    def _convert_conversation_to_markdown(self, sample_id: str, conv: Dict[str, Any]) -> str:
        """
        将 LocoMo 的 session 结构转换为扁平的 Markdown 字符串。
       
        """
        md_lines = [f"# Chat History: {sample_id}"]

        session_idx = 1
        # 循环查找 session_1, session_2 ... 直到找不到
        while f"session_{session_idx}" in conv:
            s_key = f"session_{session_idx}"
            dt_key = f"session_{session_idx}_date_time"
            sum_key = f"session_{session_idx}_summary"

            md_lines.append(f"\n## Session {session_idx}")

            # 添加日期 (让 LLM 能解析相对日期)
            session_dt = conv.get(dt_key)
            if session_dt:
                md_lines.append(f"DATE: {session_dt}")

            # 添加 session 摘要
            session_sum = conv.get(sum_key)
            if session_sum:
                md_lines.append(f"SUMMARY: {session_sum}")

            # 遍历并添加具体对话内容
            for turn in conv[s_key]:
                spk = turn.get("speaker", "Unknown")
                txt = turn.get("text", "")
                
                # 保留原始 dia_id 以支持证据回溯
                raw_id = turn.get("dia_id") or turn.get("id")
                suffix = f" [{raw_id}]" if raw_id else ""

                md_lines.append(f"**{spk}**: {txt}{suffix}")

            session_idx += 1

        return "\n".join(md_lines)