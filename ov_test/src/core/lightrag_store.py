import asyncio
import contextvars
import hashlib
import logging
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


def _quiet_lightrag_info_logs() -> None:
    """Hide verbose LightRAG retrieval INFO logs while keeping warnings/errors."""
    lightrag_logger = logging.getLogger("lightrag")
    lightrag_logger.setLevel(logging.WARNING)
    for handler in lightrag_logger.handlers:
        handler.setLevel(logging.WARNING)


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

    def __init__(self, logger=None):
        self._usage_by_scope: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()
        self._scope_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
            "lightrag_scope", default=None
        )
        # 跨线程可见的兜底 scope：_run_async 会把协程调度到独立 event-loop 线程，
        # contextvars 不会自动传播过去，导致 add_usage 读不到 scope 而丢弃 token。
        # 由于所有统计入口都被 _operation_lock 串行化，这里用普通属性即可安全兜底。
        self._fallback_scope: Optional[str] = None
        self._logger = logger

    def set_scope(self, scope: str):
        self._fallback_scope = scope
        return self._scope_var.set(scope)

    def reset_scope(self, token) -> None:
        self._fallback_scope = None
        self._scope_var.reset(token)

    def reset(self) -> None:
        with self._lock:
            self._usage_by_scope.clear()

    def add_usage(self, token_counts: Dict[str, int]) -> None:
        # 优先用 contextvar 中的 scope；跨线程时 contextvar 不传播，回退到兜底 scope。
        scope = self._scope_var.get() or self._fallback_scope
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

        # 真实 API 调用才会带着 usage 进来，这里直接输出本次调用返回的 token 信息
        if self._logger is not None and (prompt_tokens or completion_tokens):
            self._logger.info(
                f"[TokenTracker:{scope}] API usage: "
                f"prompt={prompt_tokens}, completion={completion_tokens}, "
                f"total={total_tokens}"
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
    _quiet_lightrag_info_logs()

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
        _quiet_lightrag_info_logs()
        os.makedirs(store_path, exist_ok=True)

        self.config = dict(lightrag_config or {})
        self.delete_mode = str(self.config.get("delete_mode", "semantic")).strip().lower()
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
        self.llm_model_max_async = self._coerce_optional_int(
            self.config.get("llm_model_max_async")
        )
        self.llm_model_kwargs = self.config.get("llm_model_kwargs") or {}
        self.default_llm_timeout = self._coerce_optional_int(
            self.config.get("default_llm_timeout")
            or self.config.get("llm_timeout")
            or self.llm_model_kwargs.get("timeout")
        )
        self.chunk_token_size = self._coerce_optional_int(
            self.config.get("chunk_token_size")
        )
        self.chunk_overlap_token_size = self._coerce_optional_int(
            self.config.get("chunk_overlap_token_size")
        )
        self.entity_extract_max_gleaning = self._coerce_optional_int(
            self.config.get("entity_extract_max_gleaning")
        )
        self.max_extract_input_tokens = self._coerce_optional_int(
            self.config.get("max_extract_input_tokens")
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
        self._token_tracker = ScopedTokenTracker(logger=self.logger)
        self._embedding_dim = None
        self._closed = False
        self._rerank_warning_emitted = False
        self._loop = None
        self._loop_thread = None
        self._loop_ready = threading.Event()
        self._loop_lock = threading.Lock()

        self.LightRAG, self.QueryParam, self.EmbeddingFunc = _ensure_vendored_lightrag()
        from lightrag.base import DocStatus  # type: ignore

        self.DocStatus = DocStatus
        self._RerankClient = None  # 按需加载，只在 rerank 实际使用时才导入

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

    def _get_rerank_client(self):
        """按需加载 RerankClient，避免 LightRAG 无条件依赖 OpenViking CLI。"""
        if self._RerankClient is not None:
            return self._RerankClient
        try:
            self._RerankClient = _ensure_vendored_openviking_cli()
        except ImportError as e:
            self.logger.warning(
                f"OpenViking CLI not available, rerank disabled: {e}"
            )
            self._RerankClient = False
        return self._RerankClient

    def _should_enable_rerank(self) -> bool:
        if self.enable_rerank is False:
            return False

        rerank_available = self._has_rerank_backend()
        if not rerank_available:
            if self.enable_rerank is True and not self._rerank_warning_emitted:
                self.logger.warning(
                    "LightRAG rerank is enabled in config but rerank AK/SK are missing; rerank will be disabled."
                )
                self._rerank_warning_emitted = True
            return False

        # 只有显式启用或未配置时，才尝试加载 RerankClient。
        rerank_client = self._get_rerank_client()
        if rerank_client is False:
            return False

        if self.enable_rerank is True:
            return True

        return bool(rerank_available)

    def _read_pipeline_progress(self, workspace: str) -> tuple[int, str]:
        """读取 LightRAG 入库状态；失败时静默降级为无进度。"""
        async def _read():
            from lightrag.kg.shared_storage import get_namespace_data  # type: ignore

            status = await get_namespace_data("pipeline_status", workspace=workspace)
            cur = int(status.get("cur_batch") or 0)
            msg = str(status.get("latest_message") or "")
            return cur, msg

        try:
            return asyncio.run(_read())
        except Exception:
            return 0, ""

    def _track_insert_progress(
        self,
        total_docs: int,
        workspace: str,
        stop_event: threading.Event,
    ) -> None:
        """在 rag.insert() 运行时显示文档级进度条。"""
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None

        pbar = None
        last_cur = 0
        last_msg = ""
        if tqdm is not None:
            pbar = tqdm(
                total=total_docs,
                desc="LightRAG ingest",
                unit="doc",
                dynamic_ncols=True,
                leave=True,
            )
        else:
            self.logger.info(f"[LightRAG ingest] 0/{total_docs} docs")

        try:
            while not stop_event.wait(5):
                cur, msg = self._read_pipeline_progress(workspace)
                cur = max(0, min(cur, total_docs))
                if pbar is not None:
                    if cur > last_cur:
                        pbar.update(cur - last_cur)
                    if msg and msg != last_msg:
                        pbar.set_postfix_str(msg[:80])
                elif cur != last_cur or (msg and msg != last_msg):
                    self.logger.info(
                        f"[LightRAG ingest] {cur}/{total_docs} docs | {msg[:120]}"
                    )
                last_cur = max(last_cur, cur)
                last_msg = msg or last_msg
        finally:
            cur, msg = self._read_pipeline_progress(workspace)
            cur = max(last_cur, min(cur or total_docs, total_docs))
            if pbar is not None:
                if cur > last_cur:
                    pbar.update(cur - last_cur)
                pbar.close()
            else:
                self.logger.info(f"[LightRAG ingest] finished {cur}/{total_docs} docs")

    async def _ark_multimodal_embed(self, texts: List[str]) -> np.ndarray:
        from volcenginesdkarkruntime import Ark

        api_key = self._get_embedding_api_key()
        client = Ark(api_key=api_key, base_url=self.embedding_base_url)
        embeddings = []
        for text in texts:
            normalized = text if text and text.strip() else " "
            # 指数退避重试：最多 8 次，应对 429 TPM 限流
            last_exc = None
            for attempt in range(8):
                try:
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
                    break
                except Exception as e:
                    last_exc = e
                    exc_str = str(e).lower()
                    is_retryable = any(
                        kw in exc_str
                        for kw in ("429", "ratelimit", "rate limit", "toomanyrequests", "tpm")
                    ) or getattr(e, "status_code", None) in (429, 502, 503, 504)
                    if not is_retryable:
                        raise
                    if attempt < 7:
                        delay = min(2.0 * (2 ** attempt), 120.0)
                        import random
                        jitter = random.uniform(0, delay * 0.3)
                        total_delay = delay + jitter
                        self.logger.warning(
                            f"[LightRAG Embed Retry {attempt + 1}/8] "
                            f"Rate-limited, waiting {total_delay:.1f}s... "
                            f"Error: {str(e)[:200]}"
                        )
                        await asyncio.sleep(total_delay)
            else:
                raise last_exc
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
            client = self._get_rerank_client()(
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

    def _ensure_loop(self):
        if self._closed:
            raise RuntimeError("LightRAGStoreWrapper is already closed")
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop

            self._loop_ready.clear()

            def _runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._loop_ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

            self._loop_thread = threading.Thread(
                target=_runner, name="LightRAGEventLoop", daemon=True
            )
            self._loop_thread.start()
            self._loop_ready.wait()
            return self._loop

    def _run_async(self, coro):
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def _shutdown_loop(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)

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
        if self.llm_model_max_async is not None:
            rag_kwargs["llm_model_max_async"] = self.llm_model_max_async
        if self.llm_model_kwargs:
            rag_kwargs["llm_model_kwargs"] = self.llm_model_kwargs
        if self.default_llm_timeout is not None:
            rag_kwargs["default_llm_timeout"] = self.default_llm_timeout
        if self.entity_extract_max_gleaning is not None:
            rag_kwargs["entity_extract_max_gleaning"] = self.entity_extract_max_gleaning
        if self.max_extract_input_tokens is not None:
            rag_kwargs["max_extract_input_tokens"] = self.max_extract_input_tokens
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
                doc_id_raw = f"{sample.sample_id}:{os.path.abspath(doc_path)}"
                doc_id = f"doc_{hashlib.sha1(doc_id_raw.encode('utf-8')).hexdigest()[:24]}"
                ids.append(doc_id)
                if monitor:
                    monitor.worker_start()
                    monitor.worker_end(success=True)

        scope = self._make_scope("ingest")
        scope_token = self._token_tracker.set_scope(scope)
        before = self._token_tracker.get_usage(scope)
        progress_stop = threading.Event()
        progress_thread = None
        try:
            if texts:
                with self._operation_lock:
                    progress_thread = threading.Thread(
                        target=self._track_insert_progress,
                        args=(len(texts), getattr(rag, "workspace", ""), progress_stop),
                        daemon=True,
                    )
                    progress_thread.start()
                    self._run_async(rag.ainsert(texts, ids=ids, file_paths=file_paths))
            after = self._token_tracker.get_usage(scope)
        finally:
            progress_stop.set()
            if progress_thread is not None:
                progress_thread.join(timeout=2)
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
                result = self._run_async(rag.aquery_llm(query, param=param))
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
        return await asyncio.to_thread(self.retrieve, query, topk, target_uri)

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

    async def _list_all_doc_ids_async(self) -> List[str]:
        rag = await self._ensure_rag_async()
        statuses = [
            self.DocStatus.PENDING,
            self.DocStatus.PROCESSING,
            self.DocStatus.PREPROCESSED,
            self.DocStatus.PROCESSED,
            self.DocStatus.FAILED,
        ]
        doc_ids: List[str] = []
        seen: set[str] = set()
        for status in statuses:
            docs = await rag.get_docs_by_status(status)
            for doc_id in docs.keys():
                if doc_id not in seen:
                    seen.add(doc_id)
                    doc_ids.append(doc_id)
        return doc_ids

    async def _clear_async(self) -> None:
        rag = await self._ensure_rag_async()
        doc_ids = await self._list_all_doc_ids_async()
        for doc_id in doc_ids:
            result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
            if getattr(result, "status", None) not in {"success", "not_found"}:
                raise RuntimeError(
                    f"LightRAG delete_by_doc_id failed for {doc_id}: "
                    f"{getattr(result, 'message', 'unknown error')}"
                )
        try:
            await rag.aclear_cache()
        except Exception as e:
            self.logger.warning(f"LightRAG clear_cache failed: {e}")

    def clear(self) -> dict:
        scope = self._make_scope("deletion")
        scope_token = self._token_tracker.set_scope(scope)
        before = self._token_tracker.get_usage(scope)
        try:
            with self._operation_lock:
                self._run_async(self._clear_async())
            after = self._token_tracker.get_usage(scope)
        finally:
            self._token_tracker.reset_scope(scope_token)

        usage = self._get_token_delta(before, after)
        return {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }

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
