from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional

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
    doc_content: str        # 转换后的 Markdown 文本，用于入库
    qa_pairs: List[StandardQA]
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseAdapter(ABC):
    """所有数据集适配器的基类"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path

    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """
        读取原始文件并转换为标准格式列表。
        必须由子类实现。
        """
        pass