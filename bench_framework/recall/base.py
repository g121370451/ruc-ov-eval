"""
Recall 策略基类。

Recall 的计算同时取决于数据集和 RAG 方法，分为两步：
1. preprocess: 将原始 evidence 转化为结构化的 EvidenceLocation
2. compute_recall: 将 EvidenceLocation 与 RetrievalInfo 比对，计算召回率
"""
import re
import string
from abc import ABC, abstractmethod
from typing import List

from bench_framework.types import EvidenceLocation, RetrievalInfo


class BaseRecallStrategy(ABC):
    """Recall 策略基类"""

    @abstractmethod
    def preprocess(self, evidence: List[str], **kwargs) -> List[EvidenceLocation]:
        """
        将原始 evidence 列表转化为结构化的 EvidenceLocation 列表。

        不同数据集 + RAG 方法的组合需要不同的预处理逻辑：
        - OpenViking + 任意数据集: 直接保留原始文本，无需额外处理
        - PageIndex + Locomo: 将 "D1:2" 转化为 doc_name + line_start/line_end

        Args:
            evidence: 原始证据列表（来自 QA 对）
            **kwargs: 预处理所需的额外上下文（如 doc_dir, sample_id 等）
        """
        pass

    @abstractmethod
    def compute_recall(
        self,
        evidence_locations: List[EvidenceLocation],
        retrieval_info: RetrievalInfo,
    ) -> float:
        """
        计算召回率。

        Args:
            evidence_locations: preprocess 输出的结构化证据位置
            retrieval_info: 检索阶段的结果信息
        Returns:
            float: 召回率 0.0 ~ 1.0
        """
        pass

    @staticmethod
    def _normalize(text: str) -> str:
        """标准化文本：去标点、转小写、去冠词"""
        s = str(text).replace(",", "")
        s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())
