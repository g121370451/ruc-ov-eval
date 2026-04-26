"""
KohakuRAG vector store wrapper for ruc-ov-eval benchmark pipeline.

Each sample gets its own SQLite database at:
    {store_path}/{sample_id}.db

Ingestion:
    PDF -> pdf_to_document_payload (pypdf, page-level sections) -> DocumentIndexer -> KVaultNodeStore

Retrieval:
    KVaultNodeStore.search -> ContextSnippets -> retrieved_texts + context_blocks

Embedding:
    Volcengine doubao API (same as DeepRead), implements KohakuRAG's EmbeddingModel protocol.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from src.adapters.base import StandardDoc
from src.core.logger import get_logger
from src.core.monitor import BenchmarkMonitor

# KohakuRAG imports (installed as editable package via `uv add --editable ./KohakuRAG`)
from kohakurag.datastore import KVaultNodeStore
from kohakurag.indexer import DocumentIndexer
from kohakurag.parsers import markdown_to_payload, payload_to_dict
from kohakurag.pdf_utils import pdf_to_document_payload
from kohakurag.pipeline import RAGPipeline
from kohakurag.types import NodeKind

# DeepRead imports (DeepRead/ lives directly under REPO_ROOT; add to path if needed)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from DeepRead.utils import VolcengineEmbedder, embedding_token_tracker, token_tracker

# Query Planner - wraps our LLM client to satisfy KohakuRAG's QueryPlanner protocol
class LLMQueryPlanner:
    """LLM-backed planner that proposes follow-up retrieval queries.

    Mirrors the implementation in KohakuRAG/scripts/wattbot_answer.py,
    adapted to use a langchain ChatModel instead of OpenAIChatModel.
    """

    def __init__(self, llm, max_queries: int = 3) -> None:
        self._llm = llm
        self._max_queries = max(1, max_queries)

    async def plan(self, question: str) -> Sequence[str]:
        """Generate multiple retrieval queries from a single question.

        Strategy:
        1. Always include the original question
        2. Ask LLM to generate paraphrases/entity-focused queries
        3. Fall back to simple reformulation if LLM fails
        """
        base = [question.strip()]
        prompt = f"""
You convert a question into targeted document search queries.
- The first retrieval query should remain the original question.
- Generate up to {self._max_queries - 1} additional short queries that highlight key entities, units, or paraphrases.
- Respond with JSON: {{"queries": ["query 1", "query 2"]}}
- Return an empty list if the question is already precise.

Question: {question.strip()}

JSON:
""".strip()

        # Ask LLM to generate query variations
        try:
            from langchain_core.messages import HumanMessage
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._llm.invoke([HumanMessage(content=prompt)])
            )
            raw = resp.content
            input_tokens = resp.usage_metadata['input_tokens']
            output_tokens = resp.usage_metadata['output_tokens']
            token_tracker.add(input_tokens, output_tokens)
            start = raw.index("{")
            end = raw.rindex("}") + 1
            extracted = raw[start:end]
            data = json.loads(extracted)
            items = data.get("queries")
            extra = [str(item).strip() for item in items or [] if str(item).strip()]
        except Exception:
            extra = []  # If LLM returns invalid JSON, just use original question

        # Deduplicate and enforce max_queries limit
        seen = {q.lower() for q in base if q}
        for query in extra:
            key = query.lower()
            if key in seen:
                continue
            base.append(query)
            seen.add(key)
            if len(base) >= self._max_queries:
                break

        # Fallback: add simple reformulation if LLM provided nothing useful
        if len(base) == 1:
            reformulation = question.strip().split("?", 1)[0].strip()
            if reformulation and reformulation.lower() not in seen:
                base.append(reformulation)
        return base


class VolcengineEmbeddingModel:
    """
    Wraps VolcengineEmbedder to satisfy KohakuRAG's EmbeddingModel protocol.

    embed(texts) -> np.ndarray of shape (len(texts), dimension)
    """

    def __init__(self, api_key: str, api_base: str, dimension: int = 2048):
        self._embedder = VolcengineEmbedder(
            model_name="doubao-embedding-vision-250615",
            api_key=api_key,
            api_base=api_base,
            input_type="multimodal",
            dimension=dimension,
        )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Async embed: runs synchronous Volcengine API calls in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_embed, list(texts))

    def _sync_embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._embedder.embed_batch(texts)
        return np.asarray(vectors, dtype=np.float32)


# Result types
@dataclass
class KohakuResource:
    uri: str
    content: str
    score: float = 0.0


@dataclass
class KohakuResult:
    resources: List[KohakuResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    retrieved_texts: List[str] = field(default_factory=list)


# Main wrapper
def _run_async(coro):
    """
    Run an async coroutine synchronously.

    Uses a fresh event loop to avoid conflicts with ThreadPoolExecutor
    (which may already have a running loop in the calling thread).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class KohakuStoreWrapper:
    """
    KohakuRAG vector store wrapper with the same interface as VikingStoreWrapper.

    store_path: directory where per-sample SQLite databases are stored.
    doc_output_dir: directory containing per-sample Markdown files
        (produced by adapter.data_prepare).
    """

    is_agent_mode: bool = False

    def __init__(
        self,
        store_path: str,
        doc_output_dir: str,
        api_key: str,
        api_base: str,
        embedding_dimension: int = 2048,
        top_k: int = 5,
        parent_depth: int = 1,
        child_depth: int = 0,
        deduplicate_retrieval: bool = True,
        rerank_strategy: Optional[str] = None,
        paragraph_embedding_mode: str = "averaged",
        per_sample_db: bool = True,
        planner_max_queries: int = 1,
        llm=None,
    ):
        self.store_path = store_path
        self.doc_output_dir = doc_output_dir
        self.logger = get_logger()
        self.top_k = top_k
        self._per_sample_db = per_sample_db

        os.makedirs(self.store_path, exist_ok=True)

        self._embedder = VolcengineEmbeddingModel(
            api_key=api_key,
            api_base=api_base,
            dimension=embedding_dimension,
        )
        self._parent_depth = parent_depth
        self._child_depth = child_depth
        self._deduplicate = deduplicate_retrieval
        self._rerank_strategy = rerank_strategy
        self._paragraph_embedding_mode = paragraph_embedding_mode
        self._planner = (
            LLMQueryPlanner(llm, max_queries=planner_max_queries)
            if llm is not None and planner_max_queries > 1
            else None
        )

        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed: {e}")
            self._enc = None

    @classmethod
    def from_config(cls, store_path: str, doc_output_dir: str, llm_cfg: dict, store_cfg: dict) -> "KohakuStoreWrapper":
        planner_max_queries = store_cfg.get("planner_max_queries", 1)
        llm = None
        if planner_max_queries > 1:
            from langchain_openai import ChatOpenAI
            import os as _os
            api_key = _os.environ.get(llm_cfg.get("api_key_env_var", ""), llm_cfg.get("api_key", ""))
            llm = ChatOpenAI(
                model=llm_cfg.get("model", ""),
                temperature=llm_cfg.get("temperature", 0.0),
                api_key=api_key,
                base_url=llm_cfg.get("base_url", ""),
            )
        return cls(
            store_path=store_path,
            doc_output_dir=doc_output_dir,
            api_key=llm_cfg.get("api_key", ""),
            api_base=llm_cfg.get("base_url", ""),
            embedding_dimension=store_cfg.get("embedding_dimension", 2048),
            top_k=store_cfg.get("retrieval_topk", 5),
            parent_depth=store_cfg.get("parent_depth", 1),
            child_depth=store_cfg.get("child_depth", 0),
            deduplicate_retrieval=store_cfg.get("deduplicate_retrieval", True),
            rerank_strategy=store_cfg.get("rerank_strategy", None),
            paragraph_embedding_mode=store_cfg.get("paragraph_embedding_mode", "averaged"),
            per_sample_db=store_cfg.get("per_sample_db", False),
            planner_max_queries=planner_max_queries,
            llm=llm,
        )

    def _db_path(self, sample_id: str) -> str:
        if self._per_sample_db:
            return os.path.join(self.store_path, f"{sample_id}.db")
        else:
            return os.path.join(self.store_path, "kohaku.db")

    def _table_prefix(self, sample_id: str) -> str:
        if self._per_sample_db:
            return sample_id
        else:
            return "kohaku"

    def count_tokens(self, text: str) -> int:
        if not text or not self._enc:
            return 0
        return len(self._enc.encode(str(text)))

    # TODO 无用，注释掉
    def build_uri_map(self, doc_info: List[StandardDoc]) -> Dict[str, list]:
        if self._per_sample_db:
            return {doc for doc in doc_info}
        else:
            # 单 DB 模式: 所有文档共享同一检索空间, target_uri 不限定范围
            return {doc.sample_id: [] for doc in doc_info}

    # Ingest
    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: int = 1,
        monitor: Optional[BenchmarkMonitor] = None,
    ) -> dict:
        start_time = time.time()
        embedding_token_tracker.reset()

        indexer = DocumentIndexer(
            embedding_model=self._embedder,
            paragraph_embedding_mode=self._paragraph_embedding_mode,
        )

        # 展开 doc_paths 并去重（保持顺序）
        seen = set()
        all_paths = []
        for sample in samples:
            for p in sample.doc_paths:
                if p not in seen:
                    seen.add(p)
                    all_paths.append(p)

        for path in tqdm(all_paths, desc="Ingesting Docs to KohakuRAG"):
            if monitor:
                monitor.worker_start()
            try:
                self._ingest_one(path, indexer)
                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                self.logger.error(f"Failed to ingest sample {path}: {e}")
                if monitor:
                    monitor.worker_end(success=False)
                    raise

        token_usage = embedding_token_tracker.get()
        return {
            "time": time.time() - start_time,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        }

    def _ingest_one(self, path: str, indexer: DocumentIndexer):
        sample_id = os.path.splitext(os.path.basename(path))[0]
        db_path = os.path.join(self.store_path, "kohaku.db")
        table_prefix = "kohaku"

        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Document path not found for sample '{sample_id}': {path}")

        from pathlib import Path
        ext = os.path.splitext(path)[1].lower()
        if ext in (".md", ".markdown"):
            with open(path, "r", encoding="utf-8") as f:
                md_text = f.read()
            payload = markdown_to_payload(
                document_id=sample_id,
                title=sample_id,
                markdown_text=md_text,
                metadata={"sample_id": sample_id},
            )
        else:
            payload = pdf_to_document_payload(
                Path(path),
                doc_id=sample_id,
                title=sample_id,
                metadata={"sample_id": sample_id},
            )

        # Persist parsed payload as JSON for inspection and future re-use
        # json_path = os.path.join(self.store_path, f"{sample_id}.json")
        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(payload_to_dict(payload), f, ensure_ascii=False)
        # self.logger.info(f"[{sample_id}] Payload saved to {json_path}")

        # Build hierarchical nodes + embed (async -> sync)
        nodes = _run_async(indexer.index(payload))

        # Persist to SQLite DB
        store = KVaultNodeStore(
            db_path,
            table_prefix=table_prefix,
            dimensions=self._embedder.dimension,
        )
        _run_async(store.upsert_nodes(nodes))

        self.logger.info(
            f"[{sample_id}] Indexed {len(nodes)} nodes -> {db_path} (table={table_prefix})"
        )

    # Retrieve
    def retrieve(
        self,
        query: str,
        topk: int = 5,
        target_uri: Optional[str] = None,
    ) -> KohakuResult:
        # 单 DB 模式: target_uri 不限定范围, 使用全局唯一 DB
        # 多 DB 模式: target_uri 即 sample_id, 指向对应的独立 DB
        # if self._per_sample_db:
        #     sample_id = target_uri
        #     if not sample_id:
        #         self.logger.error("retrieve() called without target_uri in per_sample_db mode.")
        #         return KohakuResult()
        # else:
        #     sample_id = target_uri  # 仅用于日志, 可为 None

        db_path = os.path.join(self.store_path, "kohaku.db")
        table_prefix = "kohaku"

        if not os.path.exists(db_path):
            self.logger.error(f"DB not found: {db_path}")
            return KohakuResult()

        embedding_token_tracker.reset()
        token_tracker.reset()

        store = KVaultNodeStore(db_path, table_prefix=table_prefix)  # dimensions auto-inferred from existing DB
        pipeline_kwargs = dict(
            store=store,
            embedder=self._embedder,
            top_k=topk,
            parent_depth=self._parent_depth,
            child_depth=self._child_depth,
            deduplicate_retrieval=self._deduplicate,
            rerank_strategy=self._rerank_strategy,
        )
        if self._planner is not None:
            pipeline_kwargs["planner"] = self._planner
        pipeline = RAGPipeline(**pipeline_kwargs)

        try:
            result = _run_async(pipeline.retrieve(query, top_k=topk))
        except Exception as e:
            self.logger.error(f"KohakuRAG retrieve failed for '{query}': {e}")
            return KohakuResult()

        embedding_token_usage = embedding_token_tracker.get()
        token_usage = token_tracker.get()

        retrieved_texts = [s.text for s in result.snippets]
        resources = [
            KohakuResource(
                uri=f"kohaku://{s.node_id}",
                content=s.text,
                score=s.score,
            )
            for s in result.snippets
        ]

        return KohakuResult(
            resources=resources,
            retrieve_input_tokens=embedding_token_usage["input_tokens"] + token_usage["input_tokens"],
            retrieve_output_tokens=embedding_token_usage["output_tokens"] + token_usage["output_tokens"],
            retrieved_texts=retrieved_texts,
        )

    def process_retrieval_results(self, search_res: KohakuResult):
        """
        Returns (retrieved_texts, context_blocks, retrieved_uris).

        retrieved_texts: snippet texts for recall calculation.
        context_blocks: formatted context for LLM prompt.
        retrieved_uris: list of resource URIs.
        """
        retrieved_texts = search_res.retrieved_texts
        context_blocks = [r.content for r in search_res.resources]
        retrieved_uris = [r.uri for r in search_res.resources]
        return retrieved_texts, context_blocks, retrieved_uris

    def clear(self):
        """Delete SQLite database(s)."""
        if not os.path.exists(self.store_path):
            return
        if self._per_sample_db:
            for fname in os.listdir(self.store_path):
                if fname.endswith(".db"):
                    fpath = os.path.join(self.store_path, fname)
                    try:
                        os.remove(fpath)
                        self.logger.info(f"Removed DB: {fpath}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {fpath}: {e}")
        else:
            fpath = self._db_path("")
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    self.logger.info(f"Removed DB: {fpath}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {fpath}: {e}")
