"""
bench_framework 数据集适配器基类。

定义标准化的数据结构和适配器接口，所有数据集适配器必须继承 BaseAdapter。
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from bench_framework.types import Metadata


@dataclass
class StandardQA:
    """标准化的单个问答对"""
    question: str
    gold_answers: List[str]
    evidence: List[str] = field(default_factory=list)
    category: Optional[Union[int, str]] = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class StandardSample:
    """标准化的样本（包含文档内容和对应的 QA 列表）"""
    sample_id: str
    qa_pairs: List[StandardQA]
    metadata: Metadata = field(default_factory=dict)


@dataclass
class StandardDoc:
    """标准化的 sample_id 与 doc_path 映射结构"""
    sample_id: str
    doc_path: str


class BaseAdapter(ABC):
    """所有数据集适配器的基类"""

    def __init__(self, raw_file_path: str, logger: Optional[logging.Logger] = None):
        self.raw_file_path = raw_file_path
        self.logger = logger or logging.getLogger("Benchmark")

    @abstractmethod
    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        数据预处理。
        1. 将数据集的数据格式转化为对向量存储友好的格式
        2. 返回转化后(或不转化)的文件地址
        """
        pass

    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """读取原始文件并转换为标准格式列表。"""
        pass

    @abstractmethod
    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> Tuple[str, Metadata]:
        """
        根据检索到的上下文和 QA 对，构建最终发给 LLM 的 Prompt。
        返回:
            - full_prompt (str): 完整的 Prompt 字符串
            - meta (Metadata): 传递给后处理函数的元数据
        """
        pass

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Metadata) -> str:
        """对大模型的原始输出进行后处理（默认只去除首尾空格）。"""
        return raw_answer.strip()
