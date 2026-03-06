"""
bench_framework 向量存储基类。

定义统一的向量存储接口标准，所有向量存储实现必须继承 VectorStoreBase。
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from bench_framework.adapters.base import StandardDoc
from bench_framework.core.monitor import BenchmarkMonitor
from bench_framework.types import IngestStats, SearchResult


class VectorStoreBase(ABC):
    """向量存储基类，定义统一的接口标准"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        pass

    @abstractmethod
    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: int = 10,
        monitor: Optional[BenchmarkMonitor] = None,
    ) -> IngestStats:
        """入库文档"""
        pass

    @abstractmethod
    def retrieve(self, query: str, topk: int = 10, target_uri: Optional[str] = None) -> SearchResult:
        """检索文档"""
        pass

    @abstractmethod
    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空库"""
        pass
