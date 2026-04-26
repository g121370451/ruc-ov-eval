# src/core/sql_agent_token_tracker.py
"""Token tracking with tiktoken + LangChain callback handler.

按 sample_id 物理隔离 token 记录，无需 lock 管理多线程竞争。
每个线程通过 contextvars 设置当前 sample_id，callback 自动归档到对应桶。
"""

import contextvars
import time
from typing import Any, Dict, List, Optional, Sequence
from collections import defaultdict

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# 当前线程/协程的 sample_id
_current_sample_id: contextvars.ContextVar[str] = (
    contextvars.ContextVar("_current_sample_id", default="")
)


class TokenTracker(BaseCallbackHandler):
    """LangChain callback，按 sample_id 分桶记录 token 消耗。"""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        # sample_id -> list of records
        self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # run_id -> inflight state
        self._inflight: Dict[Any, Dict[str, Any]] = {}

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(str(text)))

    def set_sample_id(self, sample_id: str) -> contextvars.Token:
        """设置当前上下文的 sample_id，返回 token 用于恢复。"""
        return _current_sample_id.set(sample_id)

    def restore_sample_id(self, token: contextvars.Token) -> None:
        """恢复之前的 sample_id。"""
        _current_sample_id.reset(token)

    def _messages_to_tokens(self, messages: Sequence[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            total += 4
        total += 2
        return total

    def on_llm_start(
        self, serialized: Dict, prompts: List[str], *, run_id: Any = None, **kw: Any
    ) -> None:
        tokens = sum(self.count_tokens(p) for p in prompts)
        key = run_id or id(prompts)
        sid = _current_sample_id.get("")
        self._inflight[key] = {"prompt_tokens": tokens, "start": time.time(), "sample_id": sid}

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
        sid = _current_sample_id.get("")
        self._inflight[key] = {"prompt_tokens": tokens, "start": time.time(), "sample_id": sid}

    def on_llm_end(self, response: LLMResult, *, run_id: Any = None, **kw: Any) -> None:
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
        sid = state.get("sample_id", "")
        self._records[sid].append(record)

    def on_llm_error(
        self, error: BaseException, *, run_id: Any = None, **kw: Any
    ) -> None:
        if run_id:
            self._inflight.pop(run_id, None)

    def get_usage(self, sample_id: str) -> Dict[str, Any]:
        """获取指定 sample_id 的 token 统计。"""
        recs = self._records.get(sample_id, [])
        return {
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "total_completion_tokens": sum(r["completion_tokens"] for r in recs),
            "total_tokens": sum(r["total_tokens"] for r in recs),
            "total_time_seconds": sum(r["elapsed_seconds"] for r in recs),
            "num_llm_calls": len(recs),
        }

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 sample_id 的 token 统计。"""
        return {sid: self.get_usage(sid) for sid in self._records}
