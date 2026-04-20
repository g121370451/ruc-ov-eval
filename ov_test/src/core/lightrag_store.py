import asyncio
import contextvars
import hashlib
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.adapters.base import StandardDoc
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class LightRAGResource:
    """LightRAG 检索结果中的统一资源结构。"""

    uri: str
    content: str = ""
    score: float = 0.0
    file_path: str = ""
    chunk_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LightRAGResult:
    """LightRAG 检索结果，与其他 Store wrapper 保持一致。"""

    resources: List[LightRAGResource] = field(default_factory=list)
    lightrag_context: str = ""
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    native_generation_used: bool = False
    native_final_answer: str = ""
    native_input_tokens: int = 0
    native_output_tokens: int = 0
    raw_result: Dict[str, Any] = field(default_factory=dict)


class ScopedTokenTracker:
    """按作用域记录 token 用量，避免并发检索时统计互相污染。"""

    def __init__(self):
        self._usage_by_scope: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self._scope_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "lightrag_scope", default=None
        )

    def set_scope(self, scope: str):
        return self._scope_var.set(scope)

    def reset_scope(self, token) -> None:
        self._scope_var.reset(token)

    def reset(self) -> None:
        with self._lock:
            self._usage_by_scope.clear()

    def add_usage(self, token_counts: Dict[str, int]) -> None:
        scope = self._scope_var.get()
        if scope is None:
            return

        prompt_tokens = int(token_counts.get("prompt_tokens", 0) or 0)
        completion_tokens = int(token_counts.get("completion_tokens", 0) or 0)
        total_tokens = token_counts.get("total_tokens")
        total_tokens = (
            int(total_tokens)
            if total_tokens is not None
            else prompt_tokens + completion_tokens
        )

        with self._lock:
            usage = self._usage_by_scope.setdefault(
                scope,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
            )
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += total_tokens
            usage["call_count"] += 1

    def get_usage(self, scope: Optional[str] = None) -> Dict[str, int]:
        scope = scope or self._scope_var.get()
        if scope is None:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            }

        with self._lock:
            usage = self._usage_by_scope.get(scope, {})
            return {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "call_count": int(usage.get("call_count", 0)),
            }

    def __deepcopy__(self, memo):
        # LightRAG 在初始化时会对配置做 dataclasses.asdict()，其中会 deep copy 回调对象。
        # token tracker 内部带有线程锁，不适合被深拷贝；这里显式复用同一实例即可。
        memo[id(self)] = self
        return self


def _ensure_vendored_lightrag():
    """强制使用仓库内 vendored LightRAG，避免误用环境中的同名安装。"""

    repo_root = Path(__file__).resolve().parents[3]
    vendored_root = repo_root / "lightrag"
    vendored_package = vendored_root / "lightrag"

    if not vendored_package.exists():
        raise ImportError(f"Vendored LightRAG package not found: {vendored_package}")

    vendored_root_str = str(vendored_root)
    if vendored_root_str not in sys.path:
        sys.path.insert(0, vendored_root_str)
    else:
        sys.path.remove(vendored_root_str)
        sys.path.insert(0, vendored_root_str)

    loaded_module = sys.modules.get("lightrag")
    if loaded_module is not None:
        module_file = getattr(loaded_module, "__file__", "") or ""
        module_paths = [str(p) for p in getattr(loaded_module, "__path__", [])]
        expected_prefix = str(vendored_root.resolve())
        if module_file and not Path(module_file).resolve().is_relative_to(
            vendored_root.resolve()
        ):
            for key in [
                k for k in sys.modules if k == "lightrag" or k.startswith("lightrag.")
            ]:
                sys.modules.pop(key, None)
        elif module_paths and any(
            not Path(p).resolve().is_relative_to(vendored_root.resolve())
            for p in module_paths
        ):
            for key in [
                k for k in sys.modules if k == "lightrag" or k.startswith("lightrag.")
            ]:
                sys.modules.pop(key, None)

    from lightrag import LightRAG, QueryParam  # type: ignore
    from lightrag.utils import EmbeddingFunc  # type: ignore

    module_file = getattr(sys.modules["lightrag"], "__file__", "") or ""
    if module_file and not Path(module_file).resolve().is_relative_to(
        vendored_root.resolve()
    ):
        raise ImportError(
            f"Imported LightRAG from unexpected location: {module_file}, expected under {vendored_root}"
        )

    return LightRAG, QueryParam, EmbeddingFunc


def _ensure_vendored_openviking_cli():
    """确保复用仓库内 OpenViking 的 rerank 实现，避免环境中的其他安装副本。"""

    repo_root = Path(__file__).resolve().parents[3]
    vendored_root = repo_root / "OpenViking"
    vendored_package = vendored_root / "openviking_cli"

    if not vendored_package.exists():
        raise ImportError(
            f"Vendored OpenViking CLI package not found: {vendored_package}"
        )

    vendored_root_str = str(vendored_root)
    if vendored_root_str not in sys.path:
        sys.path.insert(0, vendored_root_str)
    else:
        sys.path.remove(vendored_root_str)
        sys.path.insert(0, vendored_root_str)

    loaded_module = sys.modules.get("openviking_cli")
    if loaded_module is not None:
        module_file = getattr(loaded_module, "__file__", "") or ""
        module_paths = [str(p) for p in getattr(loaded_module, "__path__", [])]
        if module_file and not Path(module_file).resolve().is_relative_to(
            vendored_root.resolve()
        ):
            for key in [
                k
                for k in sys.modules
                if k == "openviking_cli" or k.startswith("openviking_cli.")
            ]:
                sys.modules.pop(key, None)
        elif module_paths and any(
            not Path(p).resolve().is_relative_to(vendored_root.resolve())
            for p in module_paths
        ):
            for key in [
                k
                for k in sys.modules
                if k == "openviking_cli" or k.startswith("openviking_cli.")
            ]:
                sys.modules.pop(key, None)

    from openviking_cli.utils.rerank import RerankClient  # type: ignore

    module_file = getattr(sys.modules["openviking_cli"], "__file__", "") or ""
    if module_file and not Path(module_file).resolve().is_relative_to(
        vendored_root.resolve()
    ):
        raise ImportError(
            f"Imported OpenViking CLI from unexpected location: {module_file}, expected under {vendored_root}"
        )

    return RerankClient


class LightRAGStoreWrapper:
    """LightRAG 向量/图检索包装器，统一对齐 benchmark store 接口。"""

    def __init__(self, store_path: str, lightrag_config: Optional[dict] = None):
        self.store_path = store_path
        self.logger = logger
        os.makedirs(store_path, exist_ok=True)

        self.config = dict(lightrag_config or {})
        self.query_mode = self.config.get("query_mode", "mix")
        self.enable_rerank = self._coerce_optional_bool(
            self.config.get("enable_rerank")
        )
        self.rerank_ak = self.config.get("rerank_ak", "")
        self.rerank_sk = self.config.get("rerank_sk", "")
        self.rerank_ak_env = self.config.get("rerank_ak_env", "")
        self.rerank_sk_env = self.config.get("rerank_sk_env", "")
        self.rerank_host = self.config.get(
            "rerank_host", "api-vikingdb.vikingdb.cn-beijing.volces.com"
        )
        self.rerank_model_name = self.config.get(
            "rerank_model_name", "doubao-seed-rerank"
        )
        self.rerank_model_version = self.config.get("rerank_model_version", "251028")
        self.rerank_threshold = self._coerce_optional_float(
            self.config.get("rerank_threshold")
        )
        self.embedding_max_token_size = self._coerce_optional_int(
            self.config.get("embedding_max_token_size")
        )
        self.embedding_batch_num = self._coerce_optional_int(
            self.config.get("embedding_batch_num")
        )
        self.embedding_func_max_async = self._coerce_optional_int(
            self.config.get("embedding_func_max_async")
        )
        self.chunk_token_size = self._coerce_optional_int(
            self.config.get("chunk_token_size")
        )
        self.chunk_overlap_token_size = self._coerce_optional_int(
            self.config.get("chunk_overlap_token_size")
        )
        self.llm_model = self.config.get("llm_model", "")
        self.llm_base_url = self.config.get("llm_base_url", "")
        self.llm_api_key = self.config.get("llm_api_key", "")
        self.llm_api_key_env = self.config.get("llm_api_key_env", "")
        self.embedding_model_name = self.config.get("embedding_model_name", "")
        self.embedding_base_url = self.config.get("embedding_base_url", "")
        self.embedding_api_key = self.config.get("embedding_api_key", "")
        self.embedding_api_key_env = self.config.get("embedding_api_key_env", "")
        self.enable_llm_cache = self._coerce_optional_bool(
            self.config.get("enable_llm_cache")
        )
        self.use_native_answer_generation = bool(
            self.config.get("use_native_answer_generation", False)
        )
        self.max_parallel_insert = self._coerce_optional_int(
            self.config.get("max_parallel_insert")
        )
        self.workspace = hashlib.sha1(
            os.path.abspath(store_path).encode("utf-8")
        ).hexdigest()[:16]

        self._rag = None
        self._rag_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._token_tracker = ScopedTokenTracker()
        self._embedding_dim = None
        self._closed = False
        self._rerank_warning_emitted = False

        self.LightRAG, self.QueryParam, self.EmbeddingFunc = _ensure_vendored_lightrag()
        self.RerankClient = _ensure_vendored_openviking_cli()

        try:
            import tiktoken

            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed: {e}")
            self.enc = None

    def __deepcopy__(self, memo):
        # LightRAG.__post_init__ 会 deep copy llm/embedding 回调。
        # 这些回调绑定到当前 wrapper，而 wrapper 含有 Lock/RLock，不能被正常 deep copy。
        # wrapper 本身不是值对象，复用当前实例即可。
        memo[id(self)] = self
        return self

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def _get_llm_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_api_key_env:
            return os.environ.get(self.llm_api_key_env, "")
        return ""

    def _get_embedding_api_key(self) -> str:
        if self.embedding_api_key:
            return self.embedding_api_key
        if self.embedding_api_key_env:
            return os.environ.get(self.embedding_api_key_env, "")
        return ""

    def _get_rerank_ak(self) -> str:
        if self.rerank_ak:
            return self.rerank_ak
        if self.rerank_ak_env:
            return os.environ.get(self.rerank_ak_env, "")
        return ""

    def _get_rerank_sk(self) -> str:
        if self.rerank_sk:
            return self.rerank_sk
        if self.rerank_sk_env:
            return os.environ.get(self.rerank_sk_env, "")
        return ""

    def _has_rerank_backend(self) -> bool:
        return bool(self._get_rerank_ak() and self._get_rerank_sk())

    def _should_enable_rerank(self) -> bool:
        rerank_available = self._has_rerank_backend()
        effective_enable_rerank = self.enable_rerank
        if effective_enable_rerank is None:
            effective_enable_rerank = rerank_available
        elif effective_enable_rerank and not rerank_available:
            if not self._rerank_warning_emitted:
                self.logger.warning(
                    "LightRAG rerank is enabled in config but rerank AK/SK are missing; rerank will be disabled."
                )
                self._rerank_warning_emitted = True
            effective_enable_rerank = False
        return bool(effective_enable_rerank)

    async def _ark_multimodal_embed(self, texts: List[str]) -> np.ndarray:
        from volcenginesdkarkruntime import Ark

        api_key = self._get_embedding_api_key()
        client = Ark(api_key=api_key, base_url=self.embedding_base_url)
        embeddings = []
        for text in texts:
            normalized = text if text and text.strip() else " "
            response = client.multimodal_embeddings.create(
                model=self.embedding_model_name,
                input=[{"type": "text", "text": normalized}],
            )
            self._token_tracker.add_usage(
                {
                    "prompt_tokens": getattr(
                        getattr(response, "usage", None), "prompt_tokens", 0
                    ),
                    "total_tokens": getattr(
                        getattr(response, "usage", None), "total_tokens", 0
                    ),
                }
            )
            embeddings.append(response.data.embedding)
        return np.array(embeddings, dtype=np.float32)

    async def _get_embedding_dim(self) -> int:
        if self._embedding_dim is None:
            probe = await self._ark_multimodal_embed(["dimension probe"])
            self._embedding_dim = int(probe.shape[1])
        return self._embedding_dim

    async def _llm_model_func(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Optional[list[dict[str, Any]]] = None,
        token_tracker=None,
        **kwargs,
    ) -> str:
        from lightrag.llm.openai import openai_complete_if_cache  # type: ignore

        tracker = token_tracker or self._token_tracker
        return await openai_complete_if_cache(
            model=self.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=self.llm_base_url,
            api_key=self._get_llm_api_key(),
            token_tracker=tracker,
            **kwargs,
        )

    async def _rerank_model_func(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        ak = self._get_rerank_ak()
        sk = self._get_rerank_sk()
        if not ak or not sk:
            return []

        def _run_rerank() -> List[Dict[str, Any]]:
            client = self.RerankClient(
                ak=ak,
                sk=sk,
                host=self.rerank_host,
                model_name=self.rerank_model_name,
                model_version=self.rerank_model_version,
            )
            scores = client.rerank_batch(query=query, documents=documents)
            results = [
                {"index": idx, "relevance_score": float(score or 0.0)}
                for idx, score in enumerate(scores)
            ]
            results.sort(key=lambda item: item["relevance_score"], reverse=True)
            if top_n is not None:
                return results[:top_n]
            return results

        return await asyncio.to_thread(_run_rerank)

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF: pdfplumber -> pypdf -> docling fallback chain."""
        # Priority 1: pdfplumber
        try:
            import pdfplumber

            self.logger.info("Attempting to extract text using pdfplumber")
            pages_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
            content = "\n\n".join(pages_text)
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)
        # Priority 2: docling
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            content = result.document.export_to_markdown()
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning(
                "docling failed for %s: %s, falling back", pdf_path, exc
            )

        # Priority 3: pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                reader.decrypt("")
            content = ""
            for page in reader.pages:
                content += (page.extract_text() or "") + "\n"
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pypdf failed for %s: %s, falling back", pdf_path, exc)

        self.logger.error(
            "Cannot extract text from %s. "
            "Install one of: pip install 'docling>=2' / pip install pypdf / pip install pdfplumber",
            pdf_path,
        )
        return ""

    def _read_document(self, doc_path: str) -> str:
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == ".pdf":
            try:
                import docling  # noqa: F401  # type: ignore[import-not-found]
            except ImportError:
                docling = None

            if docling is not None:
                from docling.document_converter import DocumentConverter  # type: ignore

                converter = DocumentConverter()
                result = converter.convert(Path(doc_path))
                return result.document.export_to_markdown().strip()

            from pypdf import PdfReader  # type: ignore

            pdf_password = os.environ.get("PDF_DECRYPT_PASSWORD")
            with open(doc_path, "rb") as f:
                pdf_file = BytesIO(f.read())

            reader = PdfReader(pdf_file)
            if reader.is_encrypted:
                decrypt_result = reader.decrypt(pdf_password or "")
                if decrypt_result == 0:
                    if pdf_password:
                        raise Exception("Incorrect PDF password")
                    raise Exception("PDF is encrypted but no password provided")

            content = ""
            for page in reader.pages:
                extracted = page.extract_text() or ""
                content += extracted + "\n"
            return content.strip()

        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _run_async(self, coro):
        if self._closed:
            raise RuntimeError("LightRAGStoreWrapper is already closed")
        result_container = {}
        error_container = {}

        def _runner():
            try:
                result_container["value"] = asyncio.run(coro)
            except Exception as exc:
                error_container["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_container:
            raise error_container["error"]
        return result_container.get("value")

    def _shutdown_loop(self) -> None:
        if self._closed:
            return
        self._closed = True

    def _make_scope(self, prefix: str) -> str:
        return f"{prefix}:{time.time_ns()}:{threading.get_ident()}"

    def _get_token_delta(
        self, before: Dict[str, int], after: Dict[str, int]
    ) -> Dict[str, int]:
        return {
            "prompt_tokens": after.get("prompt_tokens", 0)
            - before.get("prompt_tokens", 0),
            "completion_tokens": after.get("completion_tokens", 0)
            - before.get("completion_tokens", 0),
            "total_tokens": after.get("total_tokens", 0)
            - before.get("total_tokens", 0),
            "call_count": after.get("call_count", 0) - before.get("call_count", 0),
        }

    async def _ensure_rag_async(self):
        if self._rag is not None:
            return self._rag
        embedding_dim = await self._get_embedding_dim()
        embedding_func_kwargs = {
            "embedding_dim": embedding_dim,
            "model_name": self.embedding_model_name,
            "send_dimensions": False,
            "func": self._ark_multimodal_embed,
        }
        if self.embedding_max_token_size is not None:
            embedding_func_kwargs["max_token_size"] = self.embedding_max_token_size
        embedding_func = self.EmbeddingFunc(**embedding_func_kwargs)

        rag_kwargs = {
            "working_dir": self.store_path,
            "workspace": self.workspace,
            "llm_model_name": self.llm_model or "custom-llm",
            "llm_model_func": self._llm_model_func,
            "embedding_func": embedding_func,
        }
        if self.chunk_token_size is not None:
            rag_kwargs["chunk_token_size"] = self.chunk_token_size
        if self.chunk_overlap_token_size is not None:
            rag_kwargs["chunk_overlap_token_size"] = self.chunk_overlap_token_size
        if self.embedding_batch_num is not None:
            rag_kwargs["embedding_batch_num"] = self.embedding_batch_num
        if self.embedding_func_max_async is not None:
            rag_kwargs["embedding_func_max_async"] = self.embedding_func_max_async
        if self.enable_llm_cache is not None:
            rag_kwargs["enable_llm_cache"] = self.enable_llm_cache
        if self.max_parallel_insert is not None:
            rag_kwargs["max_parallel_insert"] = self.max_parallel_insert
        if self._should_enable_rerank():
            rag_kwargs["rerank_model_func"] = self._rerank_model_func
            if self.rerank_threshold is not None:
                rag_kwargs["min_rerank_score"] = self.rerank_threshold

        rag = self.LightRAG(**rag_kwargs)
        await rag.initialize_storages()
        self._rag = rag
        return rag

    def _ensure_rag(self):
        with self._rag_lock:
            if self._rag is not None:
                return self._rag
            rag = self._run_async(self._ensure_rag_async())
            self._rag = rag
            return rag

    def _build_query_param(
        self, topk: Optional[int], *, only_need_context: bool = False
    ) -> Any:
        query_param_kwargs = {
            "mode": self.query_mode,
            "stream": False,
            "enable_rerank": self._should_enable_rerank(),
        }
        if only_need_context:
            query_param_kwargs["only_need_context"] = True
            query_param_kwargs["only_need_prompt"] = False
        if topk is not None:
            query_param_kwargs["top_k"] = topk
            query_param_kwargs["chunk_top_k"] = topk
        return self.QueryParam(**query_param_kwargs)

    def _build_result_from_raw_result(
        self,
        result: Any,
        *,
        native_generation_used: bool,
        usage: Dict[str, int],
    ) -> LightRAGResult:
        result_dict = result if isinstance(result, dict) else {}
        resources = self._extract_resources_from_raw_result(result_dict)
        lightrag_context = ""
        if not native_generation_used:
            lightrag_context = self._extract_context_from_raw_result(result_dict)
        return LightRAGResult(
            resources=resources,
            lightrag_context=lightrag_context,
            retrieve_input_tokens=usage["prompt_tokens"],
            retrieve_output_tokens=usage["completion_tokens"],
            native_generation_used=native_generation_used,
            native_final_answer=self._extract_native_answer_from_raw_result(result_dict)
            if native_generation_used
            else "",
            native_input_tokens=usage["prompt_tokens"] if native_generation_used else 0,
            native_output_tokens=usage["completion_tokens"]
            if native_generation_used
            else 0,
            raw_result=result_dict,
        )

    @staticmethod
    def _extract_resources_from_raw_result(
        result: Dict[str, Any],
    ) -> List[LightRAGResource]:
        data_section = result.get("data", {}) if isinstance(result, dict) else {}
        resources: List[LightRAGResource] = []

        for chunk in data_section.get("chunks", []):
            chunk_content = chunk.get("content", "") or ""
            resources.append(
                LightRAGResource(
                    uri=chunk.get("reference_id")
                    or chunk.get("chunk_id")
                    or chunk.get("file_path")
                    or "",
                    content=chunk_content,
                    score=float(chunk.get("score", 0.0) or 0.0),
                    file_path=chunk.get("file_path", "") or "",
                    chunk_id=chunk.get("chunk_id", "") or "",
                    metadata=dict(chunk),
                )
            )

        if not resources:
            for entity in data_section.get("entities", []):
                content = "\n".join(
                    [
                        entity.get("entity_name", ""),
                        entity.get("entity_type", ""),
                        entity.get("description", ""),
                    ]
                ).strip()
                if not content:
                    continue
                resources.append(
                    LightRAGResource(
                        uri=entity.get("reference_id")
                        or entity.get("file_path")
                        or entity.get("entity_name")
                        or "",
                        content=content,
                        score=float(entity.get("score", 0.0) or 0.0),
                        file_path=entity.get("file_path", "") or "",
                        metadata=dict(entity),
                    )
                )

            for relation in data_section.get("relationships", []):
                content = "\n".join(
                    [
                        f"{relation.get('src_id', '')} -> {relation.get('tgt_id', '')}",
                        relation.get("keywords", ""),
                        relation.get("description", ""),
                    ]
                ).strip()
                if not content:
                    continue
                resources.append(
                    LightRAGResource(
                        uri=relation.get("reference_id")
                        or relation.get("file_path")
                        or "",
                        content=content,
                        score=float(relation.get("weight", 0.0) or 0.0),
                        file_path=relation.get("file_path", "") or "",
                        metadata=dict(relation),
                    )
                )

        return resources

    @staticmethod
    def _extract_native_answer_from_raw_result(result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        llm_response = result.get("llm_response", {})
        content = llm_response.get("content", "")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _extract_context_from_raw_result(result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return ""
        llm_response = result.get("llm_response", {})
        content = llm_response.get("content", "")
        return content.strip() if isinstance(content, str) else ""

    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: Optional[int] = None,
        monitor=None,
    ) -> dict:
        start_time = time.time()
        rag = self._ensure_rag()
        previous_max_parallel_insert = None
        if max_workers is not None:
            previous_max_parallel_insert = getattr(rag, "max_parallel_insert", None)
            rag.max_parallel_insert = int(max_workers)

        texts = []
        file_paths = []
        ids = []
        for sample in samples:
            for doc_path in sample.doc_paths:
                try:
                    content = self._read_document(doc_path)
                except Exception as e:
                    self.logger.error(f"Failed to read {doc_path}: {e}")
                    if monitor:
                        monitor.worker_end(success=False)
                    continue

                if not content:
                    continue

                texts.append(content)
                file_paths.append(doc_path)
                doc_name = os.path.splitext(os.path.basename(doc_path))[0]
                doc_id = f"{sample.sample_id}:{doc_name}"
                ids.append(doc_id)
                if monitor:
                    monitor.worker_start()
                    monitor.worker_end(success=True)

        scope = self._make_scope("ingest")
        scope_token = self._token_tracker.set_scope(scope)
        before = self._token_tracker.get_usage(scope)
        try:
            if texts:
                with self._operation_lock:
                    rag.insert(texts, ids=ids, file_paths=file_paths)
            after = self._token_tracker.get_usage(scope)
        finally:
            if max_workers is not None and previous_max_parallel_insert is not None:
                rag.max_parallel_insert = previous_max_parallel_insert
            self._token_tracker.reset_scope(scope_token)

        usage = self._get_token_delta(before, after)
        return {
            "time": time.time() - start_time,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }

    def retrieve(
        self, query: str, topk: Optional[int] = None, target_uri: str = None
    ) -> LightRAGResult:
        rag = self._ensure_rag()
        native_generation_used = self.use_native_answer_generation
        param = self._build_query_param(
            topk, only_need_context=not native_generation_used
        )
        scope = self._make_scope(
            "retrieve_native" if native_generation_used else "retrieve"
        )
        scope_token = self._token_tracker.set_scope(scope)
        before = self._token_tracker.get_usage(scope)
        try:
            with self._operation_lock:
                result = rag.query_llm(query, param=param)
            after = self._token_tracker.get_usage(scope)
        finally:
            self._token_tracker.reset_scope(scope_token)

        usage = self._get_token_delta(before, after)
        return self._build_result_from_raw_result(
            result,
            native_generation_used=native_generation_used,
            usage=usage,
        )

    async def aretrieve(
        self, query: str, topk: Optional[int] = None, target_uri: str = None
    ) -> LightRAGResult:
        rag = await self._ensure_rag_async()
        native_generation_used = self.use_native_answer_generation
        param = self._build_query_param(
            topk, only_need_context=not native_generation_used
        )
        scope = self._make_scope(
            "retrieve_native" if native_generation_used else "retrieve"
        )
        scope_token = self._token_tracker.set_scope(scope)
        before = self._token_tracker.get_usage(scope)
        try:
            result = await rag.aquery_llm(query, param=param)
            after = self._token_tracker.get_usage(scope)
        finally:
            self._token_tracker.reset_scope(scope_token)

        usage = self._get_token_delta(before, after)
        return self._build_result_from_raw_result(
            result,
            native_generation_used=native_generation_used,
            usage=usage,
        )

    async def aensure_ready(self) -> None:
        await self._ensure_rag_async()

    def process_retrieval_results(self, search_res: LightRAGResult):
        retrieved_texts = []
        retrieved_uris = []
        for resource in search_res.resources:
            if not resource.content:
                continue
            retrieved_uris.append(resource.uri)
            retrieved_texts.append(resource.content)

        context_blocks = []
        if search_res.lightrag_context:
            context_blocks.append(search_res.lightrag_context)
        else:
            for resource in search_res.resources:
                if not resource.content:
                    continue
                context_blocks.append(resource.content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    async def _clear_async(self) -> None:
        rag = self._rag
        if rag is not None:
            storages = [
                rag.full_docs,
                rag.text_chunks,
                rag.full_entities,
                rag.full_relations,
                rag.entity_chunks,
                rag.relation_chunks,
                rag.entities_vdb,
                rag.relationships_vdb,
                rag.chunks_vdb,
                rag.chunk_entity_relation_graph,
                rag.llm_response_cache,
                rag.doc_status,
            ]
            for storage in storages:
                if storage is None:
                    continue
                try:
                    await storage.drop()
                except Exception as e:
                    self.logger.warning(
                        f"LightRAG storage drop failed for {type(storage).__name__}: {e}"
                    )
            try:
                await rag.finalize_storages()
            except Exception as e:
                self.logger.warning(f"LightRAG finalize failed: {e}")

        self._rag = None
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
        os.makedirs(self.store_path, exist_ok=True)

    def clear(self) -> None:
        with self._operation_lock:
            self._run_async(self._clear_async())

    async def afinalize(self) -> None:
        if self._closed:
            return
        rag = self._rag
        if rag is None:
            self._shutdown_loop()
            return
        try:
            await rag.finalize_storages()
        except Exception as e:
            self.logger.warning(f"Failed to finalize LightRAG during async close: {e}")
        self._rag = None
        self._shutdown_loop()

    def close(self):
        if self._closed:
            return
        rag = self._rag
        if rag is None:
            self._shutdown_loop()
            return
        try:
            with self._operation_lock:
                self._run_async(rag.finalize_storages())
        except Exception as e:
            self.logger.warning(f"Failed to finalize LightRAG during close: {e}")
        self._rag = None
        self._shutdown_loop()
