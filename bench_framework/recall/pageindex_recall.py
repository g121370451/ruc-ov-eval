"""
PageIndex 行号匹配 Recall 策略。

PageIndex 没有原文，只有入库时生成的切片摘要和行号。
需要将 evidence 预处理为 doc_name + line_range，再与检索到的 node_ranges 比对。

使用方式：
    继承此类并实现 preprocess 方法，将数据集特定的 evidence 格式
    转化为 EvidenceLocation（doc_name + line_start/line_end）。
"""
from typing import List

from bench_framework.types import EvidenceLocation, RetrievalInfo
from bench_framework.recall.base import BaseRecallStrategy


class PageIndexRecallStrategy(BaseRecallStrategy):
    """
    PageIndex 行号匹配策略。

    compute_recall 判断逻辑：
    对每条 evidence，检查其 line_range 是否落在某个检索到的 node_range 内。
    如果 evidence 的行范围与某个 node 的行范围有交集，则视为命中。

    preprocess 需要子类实现，将 evidence 转化为 doc_name + line_start/line_end。
    例如 Locomo 的 "D1:2" 需要：
    1. 在 markdown 文件中找到 [D1:2] 所在行号
    2. 填入 EvidenceLocation(doc_name=..., line_start=行号, line_end=行号)
    """

    def preprocess(self, evidence: List[str], **kwargs) -> List[EvidenceLocation]:
        """
        将 evidence 转化为 EvidenceLocation。

        子类必须实现此方法。kwargs 中可传入：
        - doc_dir: str        生成的 markdown 文档目录
        - sample_id: str      样本 ID（用于定位对应的文档文件）
        - raw_data_path: str  原始数据文件路径

        Returns:
            List[EvidenceLocation]: 每条 evidence 对应的文档位置信息
        """
        raise NotImplementedError(
            "PageIndexRecallStrategy.preprocess() 需要子类实现，"
            "将 evidence 转化为 doc_name + line_start/line_end"
        )

    def compute_recall(
        self,
        evidence_locations: List[EvidenceLocation],
        retrieval_info: RetrievalInfo,
    ) -> float:
        if not evidence_locations:
            return 0.0

        node_ranges = retrieval_info.node_ranges
        hit_count = 0

        for loc in evidence_locations:
            if loc.line_start < 0:
                # 预处理未能定位，跳过
                continue
            for nr in node_ranges:
                # doc_name 匹配 + 行范围有交集
                if loc.doc_name and loc.doc_name != nr.doc_name:
                    continue
                ev_end = loc.line_end if loc.line_end >= 0 else loc.line_start
                nr_end = nr.line_end if nr.line_end >= 0 else float("inf")
                if loc.line_start <= nr_end and ev_end >= nr.line_start:
                    hit_count += 1
                    break

        return hit_count / len(evidence_locations)
