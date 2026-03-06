"""
bench_framework 类型定义模块。

为 pipeline 各阶段的返回值、检索结果、recall 预处理结果等提供明确的类型。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# ============================================================
# 通用类型别名
# ============================================================

MetadataValue = Union[str, int, float, bool, None, list, dict]
"""元数据值类型：适配器 / 数据集特定的灵活键值对"""

Metadata = Dict[str, MetadataValue]


# ============================================================
# 基础类型
# ============================================================

@dataclass
class TokenUsage:
    """Token 消耗统计"""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class IngestStats:
    """入库阶段统计"""
    time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# ============================================================
# 检索结果类型
# ============================================================

@dataclass
class SearchResource:
    """检索返回的单条资源"""
    uri: str
    level: int = 2
    abstract: str = ""
    overview: str = ""


@dataclass
class SearchResult:
    """检索返回的结果集"""
    resources: List[SearchResource] = field(default_factory=list)


# ============================================================
# Recall 相关类型
# ============================================================

@dataclass
class EvidenceLocation:
    """
    预处理后的证据位置信息。

    由 RecallStrategy.preprocess() 生成，用于与检索结果比对。
    - 对于 PageIndex: doc_name + line_start/line_end 表示证据在 markdown 中的行范围
    - 对于 OpenViking: raw_evidence 保留原始文本用于子串/token 匹配
    """
    raw_evidence: str
    doc_name: str = ""
    line_start: int = -1
    line_end: int = -1


@dataclass
class NodeRange:
    """
    PageIndex 检索到的节点范围信息。

    表示一个切片在 markdown 文档中的行范围。
    """
    node_id: str
    doc_name: str
    line_start: int
    line_end: int  # -1 表示到文档末尾


@dataclass
class RetrievalInfo:
    """
    检索阶段的结果信息，持久化到 generated_answers.json 供 eval 阶段使用。

    不同 RAG 方法填充不同字段：
    - OpenViking: 填充 retrieved_texts（原文文本）
    - PageIndex: 填充 node_ranges（节点行范围列表）
    """
    uris: List[str] = field(default_factory=list)
    latency_sec: float = 0.0
    # OpenViking: 检索到的原文文本
    retrieved_texts: List[str] = field(default_factory=list)
    # PageIndex: 检索到的节点范围
    node_ranges: List[NodeRange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "uris": self.uris,
            "latency_sec": self.latency_sec,
            "retrieved_texts": self.retrieved_texts,
            "node_ranges": [
                {"node_id": n.node_id, "doc_name": n.doc_name,
                 "line_start": n.line_start, "line_end": n.line_end}
                for n in self.node_ranges
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievalInfo":
        return cls(
            uris=d.get("uris", []),
            latency_sec=d.get("latency_sec", 0.0),
            retrieved_texts=d.get("retrieved_texts", []),
            node_ranges=[
                NodeRange(**nr) for nr in d.get("node_ranges", [])
            ],
        )


# ============================================================
# 评测结果类型
# ============================================================

@dataclass
class LLMEvaluation:
    """LLM 裁判的评测结果"""
    prompt_used: str = ""
    reasoning: str = ""
    normalized_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_used": self.prompt_used,
            "reasoning": self.reasoning,
            "normalized_score": self.normalized_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LLMEvaluation":
        return cls(
            prompt_used=d.get("prompt_used", ""),
            reasoning=d.get("reasoning", ""),
            normalized_score=d.get("normalized_score", 0.0),
        )


# ============================================================
# Pipeline 阶段返回值
# ============================================================

@dataclass
class GenerationRecord:
    """单条 QA 的生成阶段结果（写入 generated_answers.json）"""
    global_index: int
    sample_id: str
    question: str
    gold_answers: List[str]
    category: str
    evidence: List[str]
    retrieval: RetrievalInfo
    final_answer: str
    token_usage: TokenUsage
    metrics: Dict[str, float] = field(default_factory=dict)
    llm_evaluation: Optional[LLMEvaluation] = None

    def to_dict(self) -> dict:
        d = {
            "_global_index": self.global_index,
            "sample_id": self.sample_id,
            "question": self.question,
            "gold_answers": self.gold_answers,
            "category": self.category,
            "evidence": self.evidence,
            "retrieval": self.retrieval.to_dict(),
            "llm": {"final_answer": self.final_answer},
            "metrics": self.metrics,
            "token_usage": {
                "total_input_tokens": self.token_usage.input_tokens,
                "llm_output_tokens": self.token_usage.output_tokens,
            },
        }
        if self.llm_evaluation is not None:
            d["llm_evaluation"] = self.llm_evaluation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationRecord":
        llm_eval_raw = d.get("llm_evaluation")
        return cls(
            global_index=d["_global_index"],
            sample_id=d["sample_id"],
            question=d["question"],
            gold_answers=d["gold_answers"],
            category=d.get("category", ""),
            evidence=d.get("evidence", []),
            retrieval=RetrievalInfo.from_dict(d.get("retrieval", {})),
            final_answer=d["llm"]["final_answer"],
            token_usage=TokenUsage(
                input_tokens=d.get("token_usage", {}).get("total_input_tokens", 0),
                output_tokens=d.get("token_usage", {}).get("llm_output_tokens", 0),
            ),
            metrics=d.get("metrics", {}),
            llm_evaluation=LLMEvaluation.from_dict(llm_eval_raw) if llm_eval_raw else None,
        )
