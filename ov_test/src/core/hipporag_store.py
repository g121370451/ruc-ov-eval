import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.adapters.base import StandardDoc
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class HippoRAGResource:
    """HippoRAG 检索返回的单个资源"""
    uri: str
    content: str = ""
    score: float = 0.0


@dataclass
class HippoRAGResult:
    """HippoRAG 检索返回结果，与其他 Store 的 retrieve 返回格式对齐"""
    resources: List[HippoRAGResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0


class HippoRAGStoreWrapper:
    """HippoRAG 向量存储包装器，接口与 VikingStoreWrapper / PageIndexStoreWrapper 对齐。

    内部持有一个 HippoRAG 实例，将 ingest/retrieve/clear 映射到
    HippoRAG 的 index/retrieve/delete。
    """

    def __init__(self, store_path: str, hipporag_config: Optional[dict] = None):
        self.store_path = store_path
        self.logger = logger
        os.makedirs(store_path, exist_ok=True)
        # 构建 HippoRAG BaseConfig
        from hipporag.utils.config_utils import BaseConfig
        config_kwargs = {
            "save_dir": store_path,
            "use_langchain": True,
        }
        if hipporag_config:
            config_kwargs.update(hipporag_config)

        # 提取 store wrapper 自用参数，不传给 BaseConfig
        chunk_size = int(config_kwargs.pop('chunk_size', 512))
        chunk_overlap = int(config_kwargs.pop('chunk_overlap', 50))

        self.hippo_config = BaseConfig(**config_kwargs)

        # 延迟初始化 HippoRAG 实例（首次 ingest/retrieve 时创建）
        self._hippo = None

        # 文本分片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self.logger.info(f"HippoRAG text splitter: chunk_size={chunk_size}, overlap={chunk_overlap}")

        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed: {e}")
            self.enc = None

    def _ensure_hippo(self):
        if self._hippo is None:
            from hipporag import HippoRAG
            self._hippo = HippoRAG(global_config=self.hippo_config)
        return self._hippo

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def _read_document(self, doc_path: str) -> str:
        """读取文档内容，支持 txt/md/pdf 格式"""
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == '.pdf':
            from pdfminer.high_level import extract_text
            from markdownify import markdownify
            raw_text = extract_text(doc_path)
            content = markdownify(raw_text).strip()
        else:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        return content

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor=None) -> dict:
        """入库：读取文档内容，调用 HippoRAG.index()"""
        start_time = time.time()
        hippo = self._ensure_hippo()

        # 入库前重置 token 计数
        llm = hippo.llm_model
        if hasattr(llm, 'reset_token_usage'):
            llm.reset_token_usage()

        # 收集所有文档内容
        raw_docs = []
        for sample in samples:
            for doc_path in sample.doc_paths:
                try:
                    content = self._read_document(doc_path)
                    if content:
                        raw_docs.append(content)
                except Exception as e:
                    self.logger.error(f"Failed to read {doc_path}: {e}")

        # 分片
        docs = []
        for doc in raw_docs:
            chunks = self.text_splitter.split_text(doc)
            docs.extend(chunks)
        self.logger.info(f"Split {len(raw_docs)} docs into {len(docs)} chunks")

        if docs:
            hippo.index(docs)

        # 读取 token 消耗
        input_tokens = 0
        output_tokens = 0
        if hasattr(llm, 'get_token_usage'):
            usage = llm.get_token_usage()
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        return {
            "time": time.time() - start_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def retrieve(self, query: str, topk: int = 10, target_uri: str = None) -> HippoRAGResult:
        """检索：调用 HippoRAG.retrieve()"""
        hippo = self._ensure_hippo()

        try:
            results = hippo.retrieve([query], num_to_retrieve=topk)
            # retrieve 返回 List[QuerySolution] 或 (List[QuerySolution], dict)
            if isinstance(results, tuple):
                results = results[0]

            resources = []
            if results and len(results) > 0:
                qs = results[0]
                for i, doc in enumerate(qs.docs):
                    score = qs.doc_scores[i] if i < len(qs.doc_scores) else 0.0
                    resources.append(HippoRAGResource(
                        uri=f"chunk_{i}",
                        content=doc,
                        score=float(score),
                    ))
            return HippoRAGResult(resources=resources)
        except Exception as e:
            self.logger.error(f"HippoRAG retrieve failed: {e}")
            return HippoRAGResult()

    def process_retrieval_results(self, search_res: HippoRAGResult):
        """从检索结果中提取 retrieved_texts / context_blocks / retrieved_uris"""
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []
        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            retrieved_texts.append(r.content)
            context_blocks.append(r.content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    def clear(self) -> None:
        """清空库：通过 HippoRAG.delete 接口删除所有已入库的文档"""
        hippo = self._ensure_hippo()
        all_chunks = list(hippo.chunk_embedding_store.get_all_texts())
        if all_chunks:
            self.logger.info(f"Deleting {len(all_chunks)} chunks via HippoRAG.delete()")
            hippo.delete(all_chunks)
        else:
            self.logger.info("No chunks to delete")
        self._hippo = None

    def close(self):
        """释放资源"""
        self._hippo = None
