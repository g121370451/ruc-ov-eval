"""Microsoft GraphRAG adapter for the benchmark store contract."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

from src.adapters.base import StandardDoc
from src.core.env_config import required_env
from src.core.logger import get_logger


_SUPPORTED_QUERY_MODES = {"basic", "local", "global", "drift"}
_SUPPORTED_INDEXING_METHODS = {"standard", "fast"}
_TABLES_BY_MODE = {
    "basic": ("text_units",),
    "local": (
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    ),
    "global": ("entities", "communities", "community_reports"),
    "drift": (
        "entities",
        "communities",
        "community_reports",
        "text_units",
        "relationships",
    ),
}


@dataclass(frozen=True)
class GraphRAGResource:
    """One actual context block used by a GraphRAG query."""

    uri: str
    content: str
    score: float = 0.0


@dataclass
class GraphRAGResult:
    """GraphRAG query result aligned with the benchmark retrieval contract."""

    resources: List[GraphRAGResource] = field(default_factory=list)
    answer: str = ""
    query_mode: str = "local"
    source_uris: List[str] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    llm_calls: int = 0
    llm_calls_categories: dict[str, int] = field(default_factory=dict)
    input_tokens_categories: dict[str, int] = field(default_factory=dict)
    output_tokens_categories: dict[str, int] = field(default_factory=dict)


class GraphRAGStoreWrapper:
    """Index and query documents with Microsoft GraphRAG.

    The wrapper deliberately calls GraphRAG's non-streaming search engines.  Their
    ``SearchResult`` contains all internal map/reduce/local/DRIFT token usage, while
    the public streaming helpers expose only the final text and context.
    """

    manifest_name = "_index_manifest.json"

    def __init__(
        self,
        store_path: str,
        llm_cfg: dict[str, Any],
        graphrag_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store_path = str(Path(store_path).resolve())
        self.root = Path(self.store_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.root / "output"
        self.cache_dir = self.root / "cache"
        self.logs_dir = self.root / "logs"
        self.vector_dir = self.output_dir / "lancedb"
        self.manifest_path = self.root / self.manifest_name
        self.logger = get_logger()
        self.options = dict(graphrag_config or {})

        self.query_mode = str(self.options.get("query_mode", "local")).lower()
        if self.query_mode not in _SUPPORTED_QUERY_MODES:
            raise ValueError(
                f"Unsupported GraphRAG query_mode={self.query_mode!r}; expected one of "
                f"{sorted(_SUPPORTED_QUERY_MODES)}"
            )

        self.indexing_method = str(
            self.options.get("indexing_method", "standard")
        ).lower()
        if self.indexing_method not in _SUPPORTED_INDEXING_METHODS:
            raise ValueError(
                f"Unsupported GraphRAG indexing_method={self.indexing_method!r}; "
                f"expected one of {sorted(_SUPPORTED_INDEXING_METHODS)}"
            )

        self.community_level = int(self.options.get("community_level", 2))
        self.dynamic_community_selection = bool(
            self.options.get("dynamic_community_selection", False)
        )
        self.response_type = str(
            self.options.get("response_type", "Multiple Paragraphs")
        )
        self.context_block_max_chars = int(
            self.options.get("context_block_max_chars", 12000)
        )

        self._tables: dict[str, pd.DataFrame] = {}
        self._tables_lock = threading.Lock()
        self._config = self._create_config(llm_cfg)
        completion = self._config.get_completion_model_config(
            "default_completion_model"
        )
        self._tokenizer_model_id = (
            f"{completion.model_provider}/{completion.model}"
        )
        _register_offline_chunk_tokenizer()

    @classmethod
    def from_config(
        cls,
        store_path: str,
        llm_cfg: dict[str, Any],
        store_cfg: dict[str, Any],
    ) -> "GraphRAGStoreWrapper":
        """Create a wrapper from the benchmark's ``store`` configuration."""

        options = store_cfg.get("graphrag", store_cfg.get("graphrag_config", {}))
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise TypeError("store.graphrag must be a mapping")
        return cls(store_path=store_path, llm_cfg=llm_cfg, graphrag_config=options)

    def _create_config(self, llm_cfg: dict[str, Any]):
        from graphrag.config.models.graph_rag_config import GraphRagConfig

        completion_options = dict(self.options.get("completion", {}))
        embedding_options = dict(self.options.get("embedding", {}))

        completion_model = str(
            completion_options.get("model")
            or llm_cfg.get("model")
            or required_env("VLM_MODEL")
        )
        completion_api_key = str(
            completion_options.get("api_key")
            or llm_cfg.get("api_key")
            or required_env("VLM_API_KEY")
        )
        completion_api_base = str(
            completion_options.get("api_base")
            or llm_cfg.get("base_url")
            or required_env("VLM_BASE_URL")
        )

        embedding_model = str(
            embedding_options.get("model") or required_env("EMBEDDING_MODEL")
        )
        embedding_api_key = str(
            embedding_options.get("api_key") or required_env("EMBEDDING_API_KEY")
        )
        embedding_api_base = str(
            embedding_options.get("api_base") or required_env("EMBEDDING_BASE_URL")
        )
        embedding_dimension = int(
            embedding_options.get("dimension") or required_env("EMBEDDING_DIMENSION")
        )

        completion_call_args = {"temperature": llm_cfg.get("temperature", 0)}
        completion_call_args.update(completion_options.get("call_args", {}))
        embedding_call_args = dict(embedding_options.get("call_args", {}))
        metrics = {"type": "default", "store": "memory", "writer": None}
        retry = {
            "type": "exponential_backoff",
            "max_retries": int(self.options.get("max_retries", 5)),
        }

        chunking = {
            "type": "tokens",
            "size": 1200,
            "overlap": 100,
            "encoding_model": "o200k_base",
        }
        chunking.update(self.options.get("chunking", {}))

        local_search = {
            "top_k_entities": int(self.options.get("local_top_k_entities", 10)),
            "top_k_relationships": int(
                self.options.get("local_top_k_relationships", 10)
            ),
            "max_context_tokens": int(self.options.get("max_context_tokens", 12000)),
        }
        local_search.update(self.options.get("local_search", {}))

        global_search = {
            "max_context_tokens": int(self.options.get("max_context_tokens", 12000)),
            "data_max_tokens": int(self.options.get("max_context_tokens", 12000)),
        }
        global_search.update(self.options.get("global_search", {}))

        drift_search = {
            "data_max_tokens": int(self.options.get("max_context_tokens", 12000)),
            "local_search_max_data_tokens": int(
                self.options.get("max_context_tokens", 12000)
            ),
        }
        drift_search.update(self.options.get("drift_search", {}))

        basic_search = {
            "k": int(self.options.get("basic_k", 10)),
            "max_context_tokens": int(self.options.get("max_context_tokens", 12000)),
        }
        basic_search.update(self.options.get("basic_search", {}))

        config_data: dict[str, Any] = {
            "completion_models": {
                "default_completion_model": {
                    "type": "litellm",
                    "model_provider": str(
                        completion_options.get("model_provider", "openai")
                    ),
                    "model": completion_model,
                    "api_key": completion_api_key,
                    "api_base": completion_api_base,
                    "call_args": completion_call_args,
                    "retry": retry,
                    "metrics": metrics,
                }
            },
            "embedding_models": {
                "default_embedding_model": {
                    "type": "litellm",
                    "model_provider": str(
                        embedding_options.get("model_provider", "openai")
                    ),
                    "model": embedding_model,
                    "api_key": embedding_api_key,
                    "api_base": embedding_api_base,
                    "call_args": embedding_call_args,
                    "retry": retry,
                    "metrics": metrics,
                }
            },
            "concurrent_requests": int(self.options.get("concurrent_requests", 8)),
            "async_mode": str(self.options.get("async_mode", "threaded")),
            "input_storage": {"type": "file", "base_dir": str(self.root / "input")},
            "output_storage": {"type": "file", "base_dir": str(self.output_dir)},
            "update_output_storage": {
                "type": "file",
                "base_dir": str(self.root / "update_output"),
            },
            "table_provider": {"type": "parquet"},
            "cache": {
                "type": "json" if self.options.get("cache_enabled", True) else "none",
                "storage": {"type": "file", "base_dir": str(self.cache_dir)},
            },
            "reporting": {"type": "file", "base_dir": str(self.logs_dir)},
            "vector_store": {
                "type": "lancedb",
                "db_uri": str(self.vector_dir),
                "vector_size": embedding_dimension,
            },
            "chunking": chunking,
            "local_search": local_search,
            "global_search": global_search,
            "drift_search": drift_search,
            "basic_search": basic_search,
        }

        for section in (
            "embed_text",
            "extract_graph",
            "summarize_descriptions",
            "extract_graph_nlp",
            "prune_graph",
            "cluster_graph",
            "extract_claims",
            "community_reports",
            "snapshots",
        ):
            if section in self.options:
                config_data[section] = self.options[section]

        return GraphRagConfig(**config_data)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        from litellm import encode

        return len(encode(model=self._tokenizer_model_id, text=str(text)))

    @staticmethod
    def _read_document(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8").strip()
        if suffix == ".pdf":
            import fitz

            with fitz.open(path) as document:
                return "\n\n".join(page.get_text() for page in document).strip()

        from markitdown import MarkItDown

        converted = MarkItDown().convert(str(path))
        return str(converted.text_content or "").strip()

    def _prepare_input_documents(
        self, samples: Iterable[StandardDoc]
    ) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for sample in samples:
            for raw_path in sample.doc_paths:
                path = Path(raw_path).resolve()
                path_key = os.path.normcase(str(path))
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                if not path.is_file():
                    raise FileNotFoundError(f"GraphRAG input document not found: {path}")
                content = self._read_document(path)
                if not content:
                    raise ValueError(f"GraphRAG input document has no extractable text: {path}")

                content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                document_id = hashlib.sha256(
                    f"{sample.sample_id}\0{path.name}\0{content_sha256}".encode("utf-8")
                ).hexdigest()
                creation_date = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
                raw_data = {
                    "sample_id": str(sample.sample_id),
                    "source_path": str(path),
                }
                rows.append(
                    {
                        "id": document_id,
                        "human_readable_id": len(rows),
                        "title": path.stem,
                        "text": content,
                        "creation_date": creation_date,
                        "raw_data": raw_data,
                    }
                )
                sources.append(
                    {
                        "id": document_id,
                        "sample_id": str(sample.sample_id),
                        "source_path": str(path),
                        "content_sha256": content_sha256,
                    }
                )

        if not rows:
            raise ValueError("GraphRAG ingestion received no documents")

        index_settings = self._public_index_settings()
        fingerprint_payload = {
            "graphrag_version": package_version("graphrag"),
            "indexing_method": self.indexing_method,
            "index_settings": index_settings,
            "sources": sources,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        return pd.DataFrame(rows), fingerprint, sources

    def _public_index_settings(self) -> dict[str, Any]:
        completion = self._config.get_completion_model_config(
            "default_completion_model"
        )
        embedding = self._config.get_embedding_model_config("default_embedding_model")
        return {
            "completion_model_provider": completion.model_provider,
            "completion_model": completion.model,
            "completion_api_base_sha256": hashlib.sha256(
                str(completion.api_base or "").encode("utf-8")
            ).hexdigest(),
            "completion_call_args": _redact_secrets(completion.call_args),
            "embedding_model_provider": embedding.model_provider,
            "embedding_model": embedding.model,
            "embedding_api_base_sha256": hashlib.sha256(
                str(embedding.api_base or "").encode("utf-8")
            ).hexdigest(),
            "embedding_call_args": _redact_secrets(embedding.call_args),
            "embedding_dimension": self._config.vector_store.vector_size,
            "tokenizer_backend": "litellm/openai/gpt-4o-mini",
            "chunking": self._config.chunking.model_dump(mode="json"),
            "extract_graph": self._config.extract_graph.model_dump(mode="json"),
            "extract_graph_nlp": self._config.extract_graph_nlp.model_dump(
                mode="json"
            ),
            "prune_graph": self._config.prune_graph.model_dump(mode="json"),
            "summarize_descriptions": self._config.summarize_descriptions.model_dump(
                mode="json"
            ),
            "cluster_graph": self._config.cluster_graph.model_dump(mode="json"),
            "community_reports": self._config.community_reports.model_dump(mode="json"),
            "embed_text": self._config.embed_text.model_dump(mode="json"),
        }

    def _load_manifest(self) -> Optional[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid GraphRAG index manifest: {self.manifest_path}"
            ) from exc

    def _required_output_paths(self, mode: Optional[str] = None) -> list[Path]:
        selected_mode = mode or self.query_mode
        return [
            self.output_dir / f"{table}.parquet"
            for table in _TABLES_BY_MODE[selected_mode]
        ]

    @staticmethod
    def _collect_model_metrics() -> dict[str, int]:
        """Aggregate GraphRAG's in-memory completion and embedding metrics."""

        try:
            from graphrag_llm.completion.completion_factory import completion_factory
            from graphrag_llm.embedding.embedding_factory import embedding_factory
        except ImportError:
            return {"prompt_tokens": 0, "completion_tokens": 0}

        totals = {"prompt_tokens": 0, "completion_tokens": 0}
        seen_stores: set[int] = set()
        for factory in (completion_factory, embedding_factory):
            services = getattr(factory, "_initialized_services", {})
            for service in services.values():
                store = getattr(service, "metrics_store", None)
                if store is None or id(store) in seen_stores:
                    continue
                seen_stores.add(id(store))
                try:
                    metrics = store.get_metrics()
                except Exception:
                    continue
                totals["prompt_tokens"] += int(metrics.get("prompt_tokens", 0))
                totals["completion_tokens"] += int(
                    metrics.get("completion_tokens", 0)
                )
        return totals

    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: int = 1,
        monitor=None,
        checkpoint_manager=None,
    ) -> dict[str, Any]:
        """Build a GraphRAG index from standardized document paths."""

        del checkpoint_manager
        start_time = time.time()
        documents, fingerprint, sources = self._prepare_input_documents(samples)
        existing = self._load_manifest()
        if existing and existing.get("complete"):
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Existing GraphRAG index was built from different documents or "
                    "index settings. Use a clean vector_store directory."
                )
            missing = [path for path in self._required_output_paths() if not path.is_file()]
            if not missing:
                self.logger.info("Reusing complete GraphRAG index at %s", self.store_path)
                return dict(
                    existing.get(
                        "ingestion_metrics",
                        {"time": 0.0, "input_tokens": 0, "output_tokens": 0},
                    )
                )

        self.logger.info(
            "Building GraphRAG index: documents=%d, method=%s, internal_concurrency=%d "
            "(benchmark ingest_workers=%d is not used)",
            len(documents),
            self.indexing_method,
            self._config.concurrent_requests,
            max_workers,
        )
        if monitor:
            monitor.worker_start()

        before_metrics = self._collect_model_metrics()
        success = False
        try:
            from graphrag import api

            results = _run_async(
                api.build_index(
                    config=self._config,
                    method=self.indexing_method,
                    input_documents=documents,
                )
            )
            failures = [result for result in results if result.error is not None]
            if failures:
                details = "; ".join(
                    f"{item.workflow}: {item.error}" for item in failures
                )
                raise RuntimeError(f"GraphRAG indexing failed: {details}")

            missing = [path for path in self._required_output_paths() if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "GraphRAG index completed without required output tables: "
                    + ", ".join(str(path) for path in missing)
                )
            success = True
        finally:
            _release_graphrag_file_handlers(self.root)
            if monitor:
                monitor.worker_end(success=success)

        after_metrics = self._collect_model_metrics()
        ingestion_metrics = {
            "time": time.time() - start_time,
            "input_tokens": max(
                0,
                after_metrics["prompt_tokens"] - before_metrics["prompt_tokens"],
            ),
            "output_tokens": max(
                0,
                after_metrics["completion_tokens"]
                - before_metrics["completion_tokens"],
            ),
        }
        manifest = {
            "complete": True,
            "fingerprint": fingerprint,
            "graphrag_version": package_version("graphrag"),
            "indexing_method": self.indexing_method,
            "index_settings": self._public_index_settings(),
            "document_count": len(documents),
            "sources": sources,
            "ingestion_metrics": ingestion_metrics,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with self._tables_lock:
            self._tables.clear()
        return ingestion_metrics

    def _load_table(self, name: str, required: bool = True) -> Optional[pd.DataFrame]:
        with self._tables_lock:
            if name in self._tables:
                return self._tables[name]
            path = self.output_dir / f"{name}.parquet"
            if not path.is_file():
                if required:
                    raise FileNotFoundError(f"GraphRAG index table not found: {path}")
                return None
            table = pd.read_parquet(path)
            self._tables[name] = table
            return table

    def _create_query_engine(self, topk: int):
        from graphrag.config.embeddings import (
            community_full_content_embedding,
            entity_description_embedding,
            text_unit_text_embedding,
        )
        from graphrag.query.factory import (
            get_basic_search_engine,
            get_drift_search_engine,
            get_global_search_engine,
            get_local_search_engine,
        )
        from graphrag.query.indexer_adapters import (
            read_indexer_communities,
            read_indexer_covariates,
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_report_embeddings,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.utils.api import get_embedding_store, load_search_prompt

        config = self._config.model_copy(deep=True)
        if self.query_mode == "basic":
            config.basic_search.k = topk
            text_units = self._load_table("text_units")
            embedding_store = get_embedding_store(
                config=config.vector_store,
                embedding_name=text_unit_text_embedding,
            )
            return get_basic_search_engine(
                config=config,
                text_units=read_indexer_text_units(text_units),
                text_unit_embeddings=embedding_store,
                response_type=self.response_type,
                system_prompt=load_search_prompt(config.basic_search.prompt),
            )

        entities = self._load_table("entities")
        communities = self._load_table("communities")
        reports_df = self._load_table("community_reports")

        if self.query_mode == "global":
            return get_global_search_engine(
                config=config,
                reports=read_indexer_reports(
                    reports_df,
                    communities,
                    community_level=self.community_level,
                    dynamic_community_selection=self.dynamic_community_selection,
                ),
                entities=read_indexer_entities(
                    entities, communities, community_level=self.community_level
                ),
                communities=read_indexer_communities(communities, reports_df),
                response_type=self.response_type,
                dynamic_community_selection=self.dynamic_community_selection,
                map_system_prompt=load_search_prompt(config.global_search.map_prompt),
                reduce_system_prompt=load_search_prompt(
                    config.global_search.reduce_prompt
                ),
                general_knowledge_inclusion_prompt=load_search_prompt(
                    config.global_search.knowledge_prompt
                ),
            )

        text_units = self._load_table("text_units")
        relationships = self._load_table("relationships")
        entity_embedding_store = get_embedding_store(
            config=config.vector_store,
            embedding_name=entity_description_embedding,
        )
        graph_entities = read_indexer_entities(
            entities, communities, community_level=self.community_level
        )
        graph_reports = read_indexer_reports(
            reports_df, communities, community_level=self.community_level
        )

        if self.query_mode == "local":
            covariates_df = self._load_table("covariates", required=False)
            covariates = (
                read_indexer_covariates(covariates_df)
                if covariates_df is not None
                else []
            )
            return get_local_search_engine(
                config=config,
                reports=graph_reports,
                text_units=read_indexer_text_units(text_units),
                entities=graph_entities,
                relationships=read_indexer_relationships(relationships),
                covariates={"claims": covariates},
                description_embedding_store=entity_embedding_store,
                response_type=self.response_type,
                system_prompt=load_search_prompt(config.local_search.prompt),
            )

        report_embedding_store = get_embedding_store(
            config=config.vector_store,
            embedding_name=community_full_content_embedding,
        )
        read_indexer_report_embeddings(graph_reports, report_embedding_store)
        return get_drift_search_engine(
            config=config,
            reports=graph_reports,
            text_units=read_indexer_text_units(text_units),
            entities=graph_entities,
            relationships=read_indexer_relationships(relationships),
            description_embedding_store=entity_embedding_store,
            response_type=self.response_type,
            local_system_prompt=load_search_prompt(config.drift_search.prompt),
            reduce_system_prompt=load_search_prompt(config.drift_search.reduce_prompt),
        )

    def retrieve(
        self, query: str, topk: int = 10, target_uri: Optional[str] = None
    ) -> GraphRAGResult:
        """Run the configured end-to-end GraphRAG query strategy."""

        del target_uri
        if not query.strip():
            raise ValueError("GraphRAG query cannot be empty")
        manifest = self._load_manifest()
        if not manifest or not manifest.get("complete"):
            raise RuntimeError(
                f"GraphRAG index is missing or incomplete at {self.store_path}; run ingestion first"
            )
        missing = [path for path in self._required_output_paths() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "GraphRAG index is missing required tables: "
                + ", ".join(str(path) for path in missing)
            )

        engine = self._create_query_engine(topk=max(1, int(topk)))
        result = _run_async(engine.search(query=query))
        answer = _response_to_text(result.response)
        context_blocks = _flatten_context_text(result.context_text)
        if not context_blocks:
            context_blocks = _context_data_to_text(result.context_data)
        input_categories = dict(result.prompt_tokens_categories or {})
        input_adjustments = self._query_input_token_adjustments(
            query=query,
            context_data=result.context_data,
            llm_calls_categories=dict(result.llm_calls_categories or {}),
        )
        input_categories.update(input_adjustments)

        resources = [
            GraphRAGResource(
                uri=f"graphrag://{self.query_mode}/context/{index}",
                content=block,
            )
            for index, block in enumerate(context_blocks)
        ]
        source_uris = _context_source_uris(result.context_data, self.query_mode)
        return GraphRAGResult(
            resources=resources,
            answer=answer,
            query_mode=self.query_mode,
            source_uris=source_uris,
            retrieve_input_tokens=int(result.prompt_tokens)
            + sum(input_adjustments.values()),
            retrieve_output_tokens=int(result.output_tokens),
            llm_calls=int(result.llm_calls),
            llm_calls_categories=dict(result.llm_calls_categories or {}),
            input_tokens_categories=input_categories,
            output_tokens_categories=dict(result.output_tokens_categories or {}),
        )

    def _query_input_token_adjustments(
        self,
        query: str,
        context_data: Any,
        llm_calls_categories: dict[str, int],
    ) -> dict[str, int]:
        """Add query tokens omitted by GraphRAG 3.0.9's SearchResult counters.

        GraphRAG counts rendered system prompts, but Basic/Local/Global pass the
        user query as a separate message and do not include it in ``prompt_tokens``.
        Embedding inputs are likewise absent.  DRIFT renders most queries directly
        into its prompts; only its local action messages and embedding inputs need
        the adjustment below.
        """

        query_tokens = self.count_tokens(query)
        if self.query_mode in {"basic", "local"}:
            return {
                "query_embedding": query_tokens,
                "user_query_messages": query_tokens,
            }
        if self.query_mode == "global":
            message_calls = int(llm_calls_categories.get("map", 0)) + int(
                llm_calls_categories.get("reduce", 0)
            )
            return {"user_query_messages": query_tokens * message_calls}

        action_queries: list[str] = []
        if isinstance(context_data, dict):
            action_queries = [
                str(value)
                for value in context_data.keys()
                if str(value).strip()
            ]
        action_tokens = sum(self.count_tokens(value) for value in action_queries)
        return {
            "query_embedding": query_tokens + action_tokens,
            "user_query_messages": action_tokens,
        }

    def process_retrieval_results(self, search_res: GraphRAGResult):
        """Expose actual GraphRAG context for recall and result inspection."""

        retrieved_texts = [resource.content for resource in search_res.resources]
        context_blocks = [
            resource.content[: self.context_block_max_chars]
            for resource in search_res.resources
        ]
        retrieved_uris = search_res.source_uris or [
            resource.uri for resource in search_res.resources
        ]
        return retrieved_texts, context_blocks, retrieved_uris

    def get_final_answer(self, search_result: GraphRAGResult) -> str:
        """Return the answer generated internally by GraphRAG."""

        return search_result.answer

    def clear(self) -> None:
        """Remove this wrapper's complete local index and recreate its root."""

        with self._tables_lock:
            self._tables.clear()
        _release_graphrag_file_handlers(self.root)
        gc.collect()
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Release cached in-memory tables."""

        with self._tables_lock:
            self._tables.clear()


class _OfflineChunkTokenizer:
    """GraphRAG tokenizer backed by LiteLLM's bundled model tokenizer data."""

    model_id = "openai/gpt-4o-mini"

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        from litellm import encode

        return encode(model=self.model_id, text=text)

    def decode(self, tokens: list[int]) -> str:
        from litellm import decode

        return decode(model=self.model_id, tokens=tokens)

    def num_tokens(self, text: str) -> int:
        return len(self.encode(text))


def _register_offline_chunk_tokenizer() -> None:
    """Avoid tiktoken's first-run download in GraphRAG's chunking workflow.

    GraphRAG requests the ``tiktoken`` tokenizer strategy by encoding name.  The
    normal implementation downloads its BPE table from an OpenAI blob endpoint on
    first use.  LiteLLM already ships/locates the model tokenizer data used by the
    query engines, so registering this compatible strategy keeps indexing offline
    and deterministic on Windows build hosts as well.
    """

    from graphrag_llm.config import TokenizerType
    from graphrag_llm.tokenizer import register_tokenizer

    register_tokenizer(
        TokenizerType.Tiktoken,
        _OfflineChunkTokenizer,
        scope="singleton",
    )


def _release_graphrag_file_handlers(root: Path) -> None:
    """Close GraphRAG file handlers rooted in this store on Windows.

    GraphRAG configures process-global loggers during ``build_index``.  Their file
    handlers otherwise keep ``indexing-engine.log`` open, which prevents the
    benchmark deletion/rebuild stages from removing a store on Windows.
    """

    root = root.resolve()
    loggers: list[logging.Logger] = [logging.getLogger()]
    for candidate in logging.Logger.manager.loggerDict.values():
        if isinstance(candidate, logging.Logger):
            loggers.append(candidate)

    seen_handlers: set[int] = set()
    for logger in loggers:
        for handler in list(logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            base_filename = getattr(handler, "baseFilename", None)
            if not base_filename:
                continue
            try:
                Path(base_filename).resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            logger.removeHandler(handler)
            if id(handler) not in seen_handlers:
                seen_handlers.add(id(handler))
                handler.close()


def _run_async(coroutine):
    """Run one coroutine from normal code, including inside an active event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:  # propagate the original exception to caller
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    return json.dumps(response, ensure_ascii=False, indent=2)


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, nested in value.items():
            normalized = str(key).lower()
            if any(
                marker in normalized
                for marker in ("api_key", "token", "authorization", "password")
            ):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_secrets(nested)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)
    return value


def _flatten_context_text(value: Any) -> list[str]:
    blocks: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                blocks.append(text)
            return
        if isinstance(item, pd.DataFrame):
            blocks.extend(_dataframe_to_text(item))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        text = str(item).strip()
        if text:
            blocks.append(text)

    visit(value)
    return list(dict.fromkeys(blocks))


def _context_data_to_text(context_data: Any) -> list[str]:
    return _flatten_context_text(context_data)


def _dataframe_to_text(frame: pd.DataFrame) -> list[str]:
    preferred_columns = (
        "text",
        "content",
        "full_content",
        "summary",
        "description",
        "title",
    )
    columns = [column for column in preferred_columns if column in frame.columns]
    if not columns:
        return [frame.to_csv(index=False).strip()] if not frame.empty else []

    rows: list[str] = []
    for _, row in frame.iterrows():
        parts = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            text = str(value).strip()
            if text:
                parts.append(text)
        if parts:
            rows.append("\n".join(parts))
    return rows


def _context_source_uris(context_data: Any, mode: str) -> list[str]:
    uris: list[str] = []

    def visit(name: str, item: Any) -> None:
        if isinstance(item, pd.DataFrame):
            id_columns = (
                "id",
                "human_readable_id",
                "community",
                "title",
            )
            id_column = next(
                (column for column in id_columns if column in item.columns), None
            )
            if id_column is None:
                return
            for value in item[id_column].tolist():
                if value is not None and str(value).strip():
                    uris.append(f"graphrag://{mode}/{name}/{value}")
        elif isinstance(item, dict):
            for child_name, nested in item.items():
                visit(str(child_name), nested)
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(f"{name}/{index}", nested)

    if isinstance(context_data, dict):
        for name, value in context_data.items():
            visit(str(name), value)
    else:
        visit("context", context_data)
    return list(dict.fromkeys(uris))
