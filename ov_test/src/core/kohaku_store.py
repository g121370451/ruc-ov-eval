"""
KohakuRAG vector store wrapper for ruc-ov-eval benchmark pipeline.

Each sample gets its own SQLite database at:
    {store_path}/{sample_id}.db

Ingestion:
    PDF -> pdf_to_document_payload (pypdf, page-level sections) -> DocumentIndexer -> KVaultNodeStore

Retrieval:
    KVaultNodeStore.search -> ContextSnippets -> retrieved_texts + context_blocks

Embedding:
    Volcengine doubao API, implements KohakuRAG's EmbeddingModel protocol.
"""

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from src.adapters.base import StandardDoc
from src.core.logger import get_logger
from src.core.monitor import BenchmarkMonitor
from src.core.token_tracer_util import token_tracker
from src.core.doubao_embedding_util import VolcengineEmbedder, embedding_token_tracker

# KohakuRAG imports (installed as editable package via `uv add --editable ./KohakuRAG`)
from kohakurag.datastore import KVaultNodeStore
from kohakurag.indexer import DocumentIndexer
from kohakurag.parsers import markdown_to_payload, payload_to_dict
from kohakurag.pdf_utils import pdf_to_document_payload
from kohakurag.pipeline import RAGPipeline
from kohakurag.types import NodeKind

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
        # 用于 ingestion 阶段跨线程汇总 token 的局部 tracker
        self._ingest_tracker = None
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

        self._kv_store_cache: Optional[KVaultNodeStore] = None
        self._kv_store_lock = threading.Lock()

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

    def _get_kv_store(self) -> KVaultNodeStore:
        if self._kv_store_cache is not None:
            return self._kv_store_cache
        with self._kv_store_lock:
            if self._kv_store_cache is None:
                db_path = os.path.join(self.store_path, "kohaku.db")
                self._kv_store_cache = KVaultNodeStore(
                    db_path,
                    table_prefix="kohaku",
                )
                self.logger.info(f"KVaultNodeStore opened and cached: {db_path}")
        return self._kv_store_cache
    
    def invalidate_kv_store_cache(self):
        with self._kv_store_lock:
            self._kv_store_cache = None
    
    def count_tokens(self, text: str) -> int:
        if not text or not self._enc:
            return 0
        return len(self._enc.encode(str(text)))

    # Ingest
    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: int = 1,
        monitor: Optional[BenchmarkMonitor] = None,
    ) -> dict:
        from src.core.token_tracer_util import SimpleTokenTracker

        start_time = time.time()
        self._ingest_tracker = SimpleTokenTracker()
        # 临时替换底层 embedder 的 tracker，使 run_in_executor 中的线程也能汇总到同一计数器
        original_tracker = getattr(self._embedder._embedder, "tracker", None)
        self._embedder._embedder.tracker = self._ingest_tracker

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

        try:
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
        finally:
            self._embedder._embedder.tracker = original_tracker

        self.invalidate_kv_store_cache()

        token_usage = self._ingest_tracker.get()
        self._ingest_tracker = None
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

        if not os.path.exists(db_path):
            self.logger.error(f"DB not found: {db_path}")
            return KohakuResult()

        embedding_token_tracker.reset()
        token_tracker.reset()

        store = self._get_kv_store()
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

    def delete_document(self, document_id: str) -> bool:
        """Delete a single document and all its descendant nodes from the store.

        This removes entries from:
          - KVault metadata table
          - main vector table (``_vectors``)
          - optional full-paragraph vector table (``_para_full_vectors``)
          - optional BM25 text index (``_bm25``)

        The optional image-only vector table is currently not cleaned because
        ``VectorKVault`` does not expose a key/row iterator and the benchmark
        wrapper does not build image indexes.

        Args:
            document_id: The root ``node_id`` of the document to delete.

        Returns:
            True if at least one node was found and removed, False otherwise.
        """
        if not document_id:
            self.logger.warning("delete_document called with empty document_id")
            return False

        store = self._get_kv_store()
        doc_prefix = document_id + ":"

        def belongs(node_id: str) -> bool:
            return node_id == document_id or node_id.startswith(doc_prefix)

        found_any = False
        try:
            # ------------------------------------------------------------------
            # 1. Metadata + vector tables
            # ------------------------------------------------------------------
            # KVault.keys returns an iterator of bytes; collect matching records
            # first to avoid iterator invalidation while deleting.
            records_to_delete = []
            for raw_key in store._kv.keys(prefix=document_id, limit=10_000_000):
                key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                if key == store.META_KEY:
                    continue
                try:
                    record = store._kv[key]
                except KeyError:
                    continue
                node_id = record.get("node_id")
                if not belongs(node_id):
                    # Guard against prefix collisions (e.g. "doc1" vs "doc10").
                    continue
                records_to_delete.append((key, record))

            if records_to_delete:
                found_any = True

            for key, record in records_to_delete:
                vec_row_id = record.get("vector_row_id")
                if vec_row_id is not None:
                    try:
                        store._vectors.delete(int(vec_row_id))
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to delete vector row {vec_row_id} for {key}: {e}"
                        )

                if store._para_full_vectors is not None:
                    para_row_id = record.get("para_full_row_id")
                    if para_row_id is not None:
                        try:
                            store._para_full_vectors.delete(int(para_row_id))
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to delete para_full row {para_row_id} for {key}: {e}"
                            )

                try:
                    store._kv.delete(key)
                except Exception as e:
                    self.logger.warning(f"Failed to delete metadata key {key}: {e}")

            # ------------------------------------------------------------------
            # 2. BM25 index
            # ------------------------------------------------------------------
            if store._bm25 is not None:
                bm25_rows_to_delete = []
                for row_id in store._bm25.keys(limit=10_000_000):
                    try:
                        _, node_id = store._bm25.get_by_id(row_id)
                        if belongs(node_id):
                            bm25_rows_to_delete.append(row_id)
                    except Exception:
                        continue
                for row_id in bm25_rows_to_delete:
                    try:
                        store._bm25.delete(row_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete BM25 row {row_id}: {e}")

            # ------------------------------------------------------------------
            # 3. Image-only vector index (not used by the current wrapper)
            # ------------------------------------------------------------------
            if store._image_vectors is not None and store._image_vectors.count() > 0:
                self.logger.warning(
                    "Image-only vector index exists but per-document image deletion "
                    "is not implemented; image embeddings for the deleted document "
                    "may remain."
                )

        finally:
            # Force the wrapper to reopen the store on the next access so that
            # internal caches/views do not hold stale state.
            self.invalidate_kv_store_cache()

        if found_any:
            self.logger.info(f"Deleted document {document_id} from Kohaku store")
        else:
            self.logger.info(f"No nodes found for document_id={document_id}")

        return found_any

    def clear(self):
        """Delete all ingested documents one by one, but keep the SQLite DB file.

        This calls ``delete_document`` for every root document found in the
        metadata table. The total ``clear()`` time can be divided by the number
        of deleted documents to obtain a per-document deletion cost.

        The underlying ``kohaku.db`` file is intentionally retained so that its
        size can be inspected after deletion.
        """
        if not os.path.exists(self.store_path):
            return

        db_path = os.path.join(self.store_path, "kohaku.db")
        if not os.path.exists(db_path):
            return

        # ------------------------------------------------------------------
        # 1. Enumerate root documents (parent_id is None)
        # ------------------------------------------------------------------
        doc_ids: list[str] = []
        try:
            store = self._get_kv_store()
            for raw_key in store._kv.keys(limit=10_000_000):
                key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                if key == store.META_KEY:
                    continue
                try:
                    record = store._kv[key]
                except KeyError:
                    continue
                # Root document nodes have no parent
                if record.get("parent_id") is None:
                    node_id = record.get("node_id")
                    if node_id:
                        doc_ids.append(node_id)
        except Exception as e:
            self.logger.warning(f"clear(): failed to enumerate documents: {e}")

        self.logger.info(
            f"clear(): found {len(doc_ids)} documents to delete: {doc_ids}"
        )

        # ------------------------------------------------------------------
        # 2. Delete each document via delete_document
        # ------------------------------------------------------------------
        for doc_id in doc_ids:
            self.logger.info(f"clear(): deleting document {doc_id}")
            try:
                self.delete_document(doc_id)
            except Exception as e:
                self.logger.warning(f"clear(): failed to delete document {doc_id}: {e}")

        self.logger.info(
            f"clear(): finished deleting {len(doc_ids)} documents, DB kept at {db_path}"
        )

        # Keep the DB file for size inspection; only invalidate the in-memory cache.
        self.invalidate_kv_store_cache()
