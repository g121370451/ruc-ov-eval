"""
OpenViking 文本匹配 Recall 策略。

OpenViking 可以取出原文，直接做子串匹配 / token 覆盖统计。
preprocess 不需要额外处理，只保留原始文本。
"""
from typing import List

from bench_framework.types import EvidenceLocation, RetrievalInfo
from bench_framework.recall.base import BaseRecallStrategy


class TextRecallStrategy(BaseRecallStrategy):
    """
    文本匹配策略（适用于 OpenViking + 任意数据集）。

    匹配逻辑：
    1. 严格子串匹配
    2. 长度阻断（短文本不进入软匹配）
    3. 80% token 覆盖软匹配
    """

    def __init__(self, soft_threshold: float = 0.8, min_soft_match_tokens: int = 4):
        self.soft_threshold = soft_threshold
        self.min_soft_match_tokens = min_soft_match_tokens

    def preprocess(self, evidence: List[str], **kwargs) -> List[EvidenceLocation]:
        """OpenViking 无需预处理，直接保留原始文本"""
        return [EvidenceLocation(raw_evidence=ev) for ev in evidence]

    def compute_recall(
        self,
        evidence_locations: List[EvidenceLocation],
        retrieval_info: RetrievalInfo,
    ) -> float:
        if not evidence_locations:
            return 0.0

        combined = " ".join(retrieval_info.retrieved_texts)
        norm_combined = self._normalize(combined)
        ret_tokens = set(norm_combined.split())

        hit_count = 0
        for loc in evidence_locations:
            ev = loc.raw_evidence
            # 严格子串匹配
            if ev in combined:
                hit_count += 1
                continue

            norm_ev = self._normalize(ev)
            ev_tokens = set(norm_ev.split())
            if not ev_tokens:
                continue
            # 长度阻断
            if len(ev_tokens) < self.min_soft_match_tokens:
                continue
            # 80% token 覆盖
            coverage = len(ev_tokens & ret_tokens) / len(ev_tokens)
            if coverage >= self.soft_threshold:
                hit_count += 1

        return hit_count / len(evidence_locations)
