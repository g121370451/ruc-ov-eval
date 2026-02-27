from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

from src.core.logger import get_logger

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
    """标准化的sampleid 与 doc_path映射结构"""
    sample_id:str
    doc_path:str

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
        对大模型的原始输出进行后处理（默认实现为只去除首尾空格）。
        """
        return raw_answer.strip()