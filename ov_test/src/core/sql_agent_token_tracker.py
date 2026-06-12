# src/core/sql_agent_token_tracker.py
"""Token tracking for SQL Agent LLM calls.

只记录模型服务返回的官方 usage，不再用本地 tiktoken 估算 SQL Agent 检索 token。
每个线程通过 contextvars 设置当前 scope，模型包装器自动归档到对应桶。
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
    """按当前 scope 分桶记录 SQL Agent LLM 官方 token usage。"""

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

    def current_sample_id(self) -> str:
        return _current_sample_id.get("")

    def _messages_to_tokens(self, messages: Sequence[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            total += self.count_tokens(
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            total += 4
        total += 2
        return total

    @staticmethod
    def _usage_int(usage: Dict[str, Any], *keys: str) -> Optional[int]:
        for key in keys:
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_usage_from_mapping(self, mapping: Any) -> Optional[Dict[str, int]]:
        if not isinstance(mapping, dict) and mapping is None:
            return None

        usage = (
            (mapping.get("token_usage") if isinstance(mapping, dict) else getattr(mapping, "token_usage", None))
            or (mapping.get("usage") if isinstance(mapping, dict) else getattr(mapping, "usage", None))
            or (mapping.get("usage_metadata") if isinstance(mapping, dict) else getattr(mapping, "usage_metadata", None))
            or mapping
        )
        if not isinstance(usage, dict) and not any(
            hasattr(usage, key)
            for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
        ):
            return None

        prompt_tokens = self._usage_int(
            usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
        )
        completion_tokens = self._usage_int(
            usage,
            "completion_tokens",
            "output_tokens",
            "completion_token_count",
        )
        total_tokens = self._usage_int(usage, "total_tokens", "total_token_count")

        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None

        return {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
        }

    def _extract_official_usage(self, response: LLMResult) -> Optional[Dict[str, int]]:
        """Extract provider-reported token usage from common LangChain locations."""
        usage = self._extract_usage_from_mapping(getattr(response, "llm_output", None))
        if usage is not None:
            return usage

        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        found = False
        generations = getattr(response, "generations", None) or []
        if generations and not isinstance(generations[0], (list, tuple)):
            generations = [generations]

        for gen_list in generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                candidates = [
                    getattr(message, "usage_metadata", None),
                    getattr(message, "response_metadata", None),
                    getattr(gen, "generation_info", None),
                ]
                for candidate in candidates:
                    usage = self._extract_usage_from_mapping(candidate)
                    if usage is None:
                        continue
                    found = True
                    totals["prompt_tokens"] += usage["prompt_tokens"]
                    totals["completion_tokens"] += usage["completion_tokens"]
                    totals["total_tokens"] += usage["total_tokens"]
                    break
        return totals if found else None

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
        local_completion_tokens = 0
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    t = gen.text
                    if not t and hasattr(gen, "message"):
                        t = getattr(gen.message, "content", "")
                    local_completion_tokens += self.count_tokens(t or "")

        official_usage = self._extract_official_usage(response)
        prompt_tokens = state["prompt_tokens"]
        completion_tokens = local_completion_tokens
        usage_source = "estimated"
        if official_usage is not None:
            prompt_tokens = official_usage["prompt_tokens"] or prompt_tokens
            completion_tokens = official_usage["completion_tokens"] or completion_tokens
            usage_source = "official"

        record = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_seconds": elapsed,
            "usage_source": usage_source,
        }
        if official_usage is not None and official_usage["total_tokens"]:
            record["total_tokens"] = official_usage["total_tokens"]
        sid = state.get("sample_id", "")
        self._records[sid].append(record)

    def record_official_usage(self, response: Any, elapsed: float = 0.0) -> None:
        """Record provider-reported usage from a ChatResult/LLMResult-like object."""
        sid = self.current_sample_id()
        if not sid:
            return

        official_usage = self._extract_official_usage(response)
        if official_usage is None:
            self._records[sid].append({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds": elapsed,
                "usage_source": "missing_official",
            })
            return

        self._records[sid].append({
            "prompt_tokens": official_usage["prompt_tokens"],
            "completion_tokens": official_usage["completion_tokens"],
            "total_tokens": official_usage["total_tokens"],
            "elapsed_seconds": elapsed,
            "usage_source": "official",
        })

    def record_openai_response_usage(self, response: Any, elapsed: float = 0.0) -> None:
        """Record official usage directly from an OpenAI-compatible API response."""
        sid = self.current_sample_id()
        if not sid:
            return

        official_usage = self._extract_usage_from_mapping(response)
        if official_usage is None:
            self._records[sid].append({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds": elapsed,
                "usage_source": "missing_official",
            })
            return

        self._records[sid].append({
            "prompt_tokens": official_usage["prompt_tokens"],
            "completion_tokens": official_usage["completion_tokens"],
            "total_tokens": official_usage["total_tokens"],
            "elapsed_seconds": elapsed,
            "usage_source": "official",
        })

    def on_llm_error(
        self, error: BaseException, *, run_id: Any = None, **kw: Any
    ) -> None:
        if run_id:
            self._inflight.pop(run_id, None)

    def get_usage(self, sample_id: str) -> Dict[str, Any]:
        """获取指定 sample_id 的 token 统计。"""
        recs = self._records.get(sample_id, [])
        official_calls = sum(1 for r in recs if r.get("usage_source") == "official")
        estimated_calls = sum(1 for r in recs if r.get("usage_source") == "estimated")
        missing_official_calls = sum(1 for r in recs if r.get("usage_source") == "missing_official")
        return {
            "total_prompt_tokens": sum(r["prompt_tokens"] for r in recs),
            "total_completion_tokens": sum(r["completion_tokens"] for r in recs),
            "total_tokens": sum(r["total_tokens"] for r in recs),
            "total_time_seconds": sum(r["elapsed_seconds"] for r in recs),
            "num_llm_calls": len(recs),
            "official_usage_calls": official_calls,
            "estimated_usage_calls": estimated_calls,
            "missing_official_usage_calls": missing_official_calls,
        }

    def get_all_usage(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 sample_id 的 token 统计。"""
        return {sid: self.get_usage(sid) for sid in self._records}
