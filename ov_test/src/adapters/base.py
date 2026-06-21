from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re
from typing import List, Dict, Any, Union, Optional

from src.core.logger import get_logger

EVIDENCE_BASED_ASSESSMENT_INSTRUCTION = """IMPORTANT: Answer strictly based on the provided context above. Do NOT use external knowledge or information not present in the context.

Before answering, audit the provided context against the question. Provide a concise evidence analysis that can be checked:
- Point 1: Quote the exact sentence(s) from the context that directly answer the question, focusing on content that matches the question's key terms.
- Point 2: Identify any additional evidence, constraints, dates, entities, numbers, or multi-hop links needed for the answer.
- Point 3: State whether any key information is missing or conflicting.

If the context is INSUFFICIENT (missing key facts, conflicting information, or would require guessing), set "sufficient" to false, list the missing information, and set "answer" to "Not mentioned". Do NOT guess or fabricate.

If the context is SUFFICIENT, set "sufficient" to true and provide a complete answer in the "answer" field. Include all relevant details (dates, ranges, names) rather than oversimplifying.

Respond ONLY as a JSON object in the following format:
{
  "sufficient": true/false,
  "evidence_analysis": [
    "Point 1: [Quote] ...",
    "Point 2: ...",
    "Point 3: ..."
  ],
  "missing_info": [],
  "answer": "<final answer or Not mentioned>",
  "reasoning": "<one short sentence summarizing why the answer is supported or why it is insufficient>"
}"""


@dataclass
class StandardQA:
    """标准化的单个问答对"""
    question: str
    gold_answers: List[str]
    evidence: List[str] = field(default_factory=list)
    category: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StandardSample:
    """标准化的样本（包含文档内容和对应的 QA 列表）"""
    sample_id: str
    qa_pairs: List[StandardQA]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StandardDoc:
    """标准化的 sample_id 与文档路径映射，doc_paths 支持一对多（如 HotpotQA 多跳）"""
    sample_id: str
    doc_paths: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAdapter(ABC):
    """所有数据集适配器的基类"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.logger = get_logger()

    @abstractmethod
    def data_prepare(self, doc_dir:str) -> List[StandardDoc]:
        """_summary_
            数据预处理。
            1. 将数据集的数据格式转化为对ov友好的格式
            2. 返回转化后(或不转化)的文件地址
        Returns:
            List[StandardDoc]: 预计输入到ov的文件地址数组
        """
        pass
    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """
        读取原始文件并转换为标准格式列表。
        必须由子类实现。
        """
        pass
    
    @abstractmethod
    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        根据检索到的上下文和 QA 对，构建最终发给 LLM 的 Prompt。
        返回:
            - full_prompt (str): 完整的 Prompt 字符串
            - meta (Dict): 传递给后处理函数的元数据（如选择题的选项映射）
        """
        pass

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        对大模型的原始输出进行后处理。

        evidence-based prompt 会要求模型返回 JSON，这里优先提取 JSON 中的 answer
        字段；非 JSON 输出保持原有 strip 行为。
        """
        text = raw_answer.strip()
        parsed = self._try_parse_json_object(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer = parsed.get("answer")
            if answer is None:
                return "Not mentioned"
            return str(answer).strip()
        return text

    @staticmethod
    def _try_parse_json_object(text: str):
        if not text:
            return None

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        for candidate in (cleaned,):
            try:
                return json.loads(candidate)
            except Exception:
                pass

        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
