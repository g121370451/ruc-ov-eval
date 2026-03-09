import os
import json
import time
import asyncio
import shutil
import logging
import tiktoken
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from functools import partial
from tqdm import tqdm

from src.adapters.base import StandardDoc

logger = logging.getLogger(__name__)


@dataclass
class LightRAGResource:
    uri: str
    content: str = ""
    score: float = 0.0


@dataclass
class LightRAGResult:
    resources: List[LightRAGResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0


class LightRAGStoreWrapper:
    """LightRAG store wrapper, interface aligned with VikingStoreWrapper / PageIndexStoreWrapper"""

    def __init__(self, store_path: str, config_path: str = None):
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)
        self.config_path = config_path

        # Load config
        self.conf = self._load_config(config_path)

        # Set API key for OpenAI-compatible calls
        os.environ["OPENAI_API_KEY"] = self.conf.get("api_key", "")

        # Tiktoken for count_tokens
        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.enc = None

        # Build LLM and embedding functions
        self._llm_func = self._build_llm_func()
        self._embed_func = self._build_embed_func()

        # Internal doc content cache: doc_id -> content string
        self._doc_cache: Dict[str, str] = {}
        # Track indexed doc_ids
        self._indexed_doc_ids: set = set()

        # Create LightRAG instance
        from lightrag import LightRAG
        working_dir = self.conf.get("working_dir") or self.store_path
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=self._llm_func,
            llm_model_name=self.conf.get("llm_model_name", "doubao-seed-1-8-251228"),
            embedding_func=self._embed_func,
        )
        # Initialize storages synchronously
        self._run_async(self.rag.initialize_storages())

    def _load_config(self, config_path: str) -> dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if config_path:
            logger.error(f"LightRAG config file not found: {config_path}")
            print(f"[Error] LightRAG config file not found: {config_path}")
        return {}

    def _run_async(self, coro):
        """Run an async coroutine synchronously"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def _build_llm_func(self):
        """Build async LLM function for LightRAG"""
        from lightrag.llm.openai import openai_complete_if_cache
        model = self.conf.get("llm_model_name", "doubao-seed-1-8-251228")
        base_url = self.conf.get("llm_base_url", "https://ark.cn-beijing.volces.com/api/v3")
        api_key = self.conf.get("api_key", "")

        async def llm_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_func

    def _build_embed_func(self):
        """Build embedding function for LightRAG"""
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        model = self.conf.get("embedding_model_name", "doubao-embedding-vision-250615")
        base_url = self.conf.get("embedding_base_url", "https://ark.cn-beijing.volces.com/api/v3")
        api_key = self.conf.get("api_key", "")
        dim = self.conf.get("embedding_dim", 2048)
        max_tokens = self.conf.get("max_token_size", 8192)

        @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=max_tokens, model_name=model)
        async def embed_func(texts: list[str], **kwargs) -> np.ndarray:
            return await openai_embed.func(
                texts,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        return embed_func

    # ---- Public interface methods ----

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor=None) -> dict:
        """Ingest documents into LightRAG"""
        start_time = time.time()

        for sample in tqdm(samples, desc="LightRAG Ingesting"):
            if monitor:
                monitor.worker_start()
            try:
                doc_id = os.path.basename(sample.doc_path)
                with open(sample.doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not content.strip():
                    logger.warning(f"Empty document: {sample.doc_path}")
                    if monitor:
                        monitor.worker_end(success=False)
                    continue

                self._doc_cache[doc_id] = content
                self._indexed_doc_ids.add(doc_id)
                self._run_async(self.rag.ainsert(content, ids=[doc_id], file_paths=[sample.doc_path]))

                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                logger.error(f"Failed to ingest {sample.doc_path}: {e}")
                if monitor:
                    monitor.worker_end(success=False)

        return {
            "time": time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def retrieve(self, query: str, topk: int = 5, target_uri: str = None) -> LightRAGResult:
        """Retrieve relevant chunks using LightRAG's query_data API"""
        from lightrag import QueryParam
        mode = self.conf.get("query_mode", "hybrid")
        top_k = self.conf.get("top_k", 60)

        try:
            result = self._run_async(
                self.rag.aquery_data(
                    query,
                    param=QueryParam(mode=mode, top_k=top_k),
                )
            )
        except Exception as e:
            logger.error(f"LightRAG retrieve failed: {e}")
            return LightRAGResult()

        # Extract chunks from result
        resources = []
        chunks = []
        if isinstance(result, dict):
            data = result.get("data", {})
            chunks = data.get("chunks", [])

        for i, chunk in enumerate(chunks[:topk]):
            content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            doc_id = ""
            if isinstance(chunk, dict):
                doc_id = chunk.get("file_path", "") or chunk.get("chunk_id", f"chunk_{i}")
            resources.append(LightRAGResource(
                uri=doc_id,
                content=content,
                score=1.0 - i * 0.01,
            ))

        return LightRAGResult(resources=resources)

    def process_retrieval_results(self, search_res: LightRAGResult):
        """Extract retrieved_texts / context_blocks / retrieved_uris from results"""
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []
        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            retrieved_texts.append(r.content)
            context_blocks.append(r.content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    def build_uri_map(self, doc_info: List[StandardDoc]) -> Dict[str, list]:
        """Build sample_id -> [doc_id] mapping"""
        uri_map = {}
        for doc in doc_info:
            doc_id = os.path.basename(doc.doc_path)
            if doc_id in self._indexed_doc_ids:
                uri_map.setdefault(doc.sample_id, []).append(doc_id)
            else:
                logger.warning(f"Doc not indexed in LightRAG: {doc_id} (sample_id={doc.sample_id})")
        return uri_map

    def read_resource(self, uri: str) -> str:
        """Read document content from cache"""
        return self._doc_cache.get(uri, "")

    def clear(self) -> None:
        """Clear all LightRAG data"""
        self._doc_cache.clear()
        self._indexed_doc_ids.clear()
        try:
            self._run_async(self.rag.finalize_storages())
        except Exception:
            pass
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path, ignore_errors=True)
            os.makedirs(self.store_path, exist_ok=True)
