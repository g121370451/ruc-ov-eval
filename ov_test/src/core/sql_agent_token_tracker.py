# src/core/sql_agent_token_tracker.py
"""Token tracking with tiktoken + LangChain callback handler.

Thread-safe and async-compatible.  Uses ``contextvars`` so that each
concurrent operation (insert / retrieve / delete) collects only its own
LLM-call records even when many run in parallel.

从 LangChain-SQL-Agent 项目提取，用于 SQLAgentStoreWrapper 的 token 统计。
"""

import contextvars
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

_op_collector: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar("_op_collector", default=None)
)


class TokenTracker(BaseCallbackHandler):
    """LangChain callback that counts tokens with tiktoken (thread-safe)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._inflight: Dict[Any, Dict[str, Any]] = {}

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(str(text)))

    def _messages_to_tokens(self, messages: Sequence[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            total += 4  # per-message overhead
        total += 2  # reply priming
        return total

    def on_llm_start(
        self, serialized: Dict, prompts: List[str], *, run_id: Any = None, **kw: Any
    ) -> None:
        tokens = sum(self.count_tokens(p) for p in prompts)
        key = run_id or id(prompts)
        with self._lock:
            self._inflight[key] = {"prompt_tokens": tokens, "start": time.time()}

    def on_chat_model_start(
        self,
        serialized: Dict,
        messages: List[List[BaseMessage]],
        *,
        run_id: Any = None,
        **kw: Any,
    ) -> None:
        tokens = 0
        for msg_list in messages:
            tokens += self._messages_to_tokens(msg_list)
        key = run_id or id(messages)
        with self._lock:
            self._inflight[key] = {"prompt_tokens": tokens, "start": time.time()}

    def on_llm_end(self, response: LLMResult, *, run_id: Any = None, **kw: Any) -> None:
        with self._lock:
            state = self._inflight.pop(run_id, None) if run_id else None
            if state is None and self._inflight:
                _, state = self._inflight.popitem()
        if state is None:
            return

        elapsed = time.time() - state["start"]
        completion_tokens = 0
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    t = gen.text
                    if not t and hasattr(gen, "message"):
                        t = getattr(gen.message, "content", "")
                    completion_tokens += self.count_tokens(t or "")

        record = {
            "prompt_tokens": state["prompt_tokens"],
            "completion_tokens": completion_tokens,
            "total_tokens": state["prompt_tokens"] + completion_tokens,
            "elapsed_seconds": elapsed,
        }
        with self._lock:
            self.records.append(record)

        collector = _op_collector.get(None)
        if collector is not None:
            collector.append(record)

    def on_llm_error(
        self, error: BaseException, *, run_id: Any = None, **kw: Any
    ) -> None:
        if run_id:
            with self._lock:
                self._inflight.pop(run_id, None)

    def total(self) -> Dict[str, Any]:
        with self._lock:
            recs = list(self.records)
        return {
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "total_completion_tokens": sum(r["completion_tokens"] for r in recs),
            "total_tokens": sum(r["total_tokens"] for r in recs),
            "total_time_seconds": sum(r["elapsed_seconds"] for r in recs),
            "num_llm_calls": len(recs),
        }

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self._inflight.clear()


class OperationTracker:
    """Context-manager that collects TokenTracker records per operation.

    Safe for concurrent use: each ``with track(...)`` block uses a
    ``contextvars.ContextVar`` so parallel operations don't interfere.
    """

    def __init__(self, token_tracker: TokenTracker):
        self.tracker = token_tracker
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    class _Context:
        def __init__(self, parent: "OperationTracker", name: str):
            self._parent = parent
            self._name = name
            self._collector: List[Dict[str, Any]] = []
            self._start: float = 0.0
            self._token: Optional[contextvars.Token] = None

        def __enter__(self):
            self._collector = []
            self._start = time.time()
            self._token = _op_collector.set(self._collector)
            return self

        def __exit__(self, *args: Any):
            if self._token is not None:
                _op_collector.reset(self._token)
            elapsed = time.time() - self._start
            prompt_tok = sum(r["prompt_tokens"] for r in self._collector)
            comp_tok = sum(r["completion_tokens"] for r in self._collector)
            rec = {
                "elapsed_seconds": elapsed,
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": prompt_tok + comp_tok,
                "num_llm_calls": len(self._collector),
            }
            with self._parent._lock:
                self._parent.results.setdefault(self._name, []).append(rec)

    def track(self, operation_name: str) -> _Context:
        return self._Context(self, operation_name)

    def summary(self, operation_name: str) -> Dict[str, Any]:
        with self._lock:
            recs = list(self.results.get(operation_name, []))
        if not recs:
            return {}
        n = len(recs)
        return {
            "count": n,
            "total_time": sum(r["elapsed_seconds"] for r in recs),
            "avg_time": sum(r["elapsed_seconds"] for r in recs) / n,
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "total_completion_tokens": sum(r["completion_tokens"] for r in recs),
            "total_tokens": sum(r["total_tokens"] for r in recs),
            "avg_tokens": sum(r["total_tokens"] for r in recs) / n,
        }
