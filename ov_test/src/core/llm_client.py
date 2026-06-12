import asyncio
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError, APIConnectionError, AuthenticationError, BadRequestError

from src.core.retry_utils import _is_retryable, calculate_retry_delay

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError)
_FATAL_ERRORS = (AuthenticationError, BadRequestError)

# 关闭 SDK 短重试，统一由外层退避控制。
_RETRY_COUNT = 8
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=api_key,
            base_url=config['base_url'],
            max_retries=0,
            timeout=config.get('timeout', 180),
        )
        self.retry_count = _RETRY_COUNT

    @staticmethod
    def _usage_int(usage, *keys: str):
        for key in keys:
            value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _extract_usage_from_mapping(cls, mapping):
        if mapping is None:
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
        input_tokens = cls._usage_int(usage, "prompt_tokens", "input_tokens", "prompt_token_count")
        output_tokens = cls._usage_int(usage, "completion_tokens", "output_tokens", "completion_token_count")
        total_tokens = cls._usage_int(usage, "total_tokens", "total_token_count")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens or (input_tokens or 0) + (output_tokens or 0),
            "usage_source": "official",
        }

    @classmethod
    def _extract_usage(cls, response):
        candidates = [
            getattr(response, "usage_metadata", None),
            getattr(response, "response_metadata", None),
            getattr(response, "llm_output", None),
        ]
        for candidate in candidates:
            usage = cls._extract_usage_from_mapping(candidate)
            if usage is not None:
                return usage
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "usage_source": "missing_official",
        }

    def _backoff(self, exc: Exception, attempt: int, sync: bool = True):
        total_delay = calculate_retry_delay(exc, attempt)
        logger.warning(
            f"[LLM Retry {attempt + 1}/{self.retry_count}] "
            f"Rate-limited or transient error, waiting {total_delay:.1f}s... "
            f"Error: {str(exc)[:200]}"
        )
        if sync:
            time.sleep(total_delay)
        else:
            return asyncio.sleep(total_delay)

    def generate(self, prompt: str) -> str:
        content, _ = self.generate_with_usage(prompt)
        return content

    def generate_with_usage(self, prompt: str):
        last_err = None
        for attempt in range(self.retry_count):
            try:
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                return resp.content, self._extract_usage(resp)
            except Exception as e:
                last_err = e
                if not _is_retryable(e):
                    raise
                if attempt < self.retry_count - 1:
                    self._backoff(e, attempt)
        raise last_err

    async def agenerate(self, prompt: str) -> str:
        content, _ = await self.agenerate_with_usage(prompt)
        return content

    async def agenerate_with_usage(self, prompt: str):
        last_err = None
        messages = [HumanMessage(content=prompt)]
        for attempt in range(self.retry_count):
            try:
                if hasattr(self.llm, "ainvoke"):
                    resp = await self.llm.ainvoke(messages)
                else:
                    resp = await asyncio.to_thread(self.llm.invoke, messages)
                return resp.content, self._extract_usage(resp)
            except Exception as e:
                last_err = e
                if not _is_retryable(e):
                    raise
                if attempt < self.retry_count - 1:
                    await self._backoff(e, attempt, sync=False)
        raise last_err
    
    
    def explain_not_mentioned(
        self,
        question: str,
        context_texts: list,
    ) -> str:
        """
        当生成答案为 'Not mentioned' 时，让 LLM 解释为什么提供的上下文无法回答该问题。
        """
        context_str = "\n\n".join(context_texts[:10])
        prompt = f"""The following context was retrieved to answer a question, but the system concluded "Not mentioned".
    Explain briefly why the context is insufficient to answer the question.

    Context:
    {context_str}

    Question: {question}

    Respond with a short explanation (2-3 sentences).
    """
        try:
            resp = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant that analyzes retrieval quality."),
                HumanMessage(content=prompt),
            ])
            return resp.content.strip() if resp and hasattr(resp, "content") else ""
        except Exception:
            return ""

    async def aexplain_not_mentioned(
        self,
        question: str,
        context_texts: list,
    ) -> str:
        """
        异步版本的 Not mentioned 原因解释。
        """
        context_str = "\n\n".join(context_texts[:10])
        prompt = f"""The following context was retrieved to answer a question, but the system concluded "Not mentioned".
    Explain briefly why the context is insufficient to answer the question.

    Context:
    {context_str}

    Question: {question}

    Respond with a short explanation (2-3 sentences).
    """
        messages = [
            SystemMessage(content="You are a helpful assistant that analyzes retrieval quality."),
            HumanMessage(content=prompt),
        ]
        try:
            if hasattr(self.llm, "ainvoke"):
                resp = await self.llm.ainvoke(messages)
            else:
                resp = await asyncio.to_thread(self.llm.invoke, messages)
            return resp.content.strip() if resp and hasattr(resp, "content") else ""
        except Exception:
            return ""
