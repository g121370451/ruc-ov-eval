# src/core/sql_agent_token_tracker.py
"""
Token tracking with tiktoken + LangChain callback handler.
从 LangChain-SQL-Agent 项目提取，用于 SQLAgentStoreWrapper 的 token 统计。
"""

import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


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
