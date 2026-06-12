import functools
import hashlib
import json
import os
import random
import sqlite3
import time
from copy import deepcopy
from typing import List, Tuple

from filelock import FileLock
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

# 退避重试参数
_RETRY_MAX = 8
_RETRY_BASE_DELAY = 2.0
_RETRY_MAX_DELAY = 120.0


def _is_retryable_error(exc: Exception) -> bool:
    """判断异常是否可重试（429 / 502 / 503 / 504 / 连接错误）"""
    exc_str = str(exc).lower()
    if any(kw in exc_str for kw in ("429", "ratelimit", "rate limit", "toomanyrequests", "tpm")):
        return True
    status = getattr(exc, "status_code", None)
    if status in (429, 502, 503, 504):
        return True
    cls_name = type(exc).__name__.lower()
    if any(kw in cls_name for kw in ("connection", "timeout", "ratelimit")):
        return True
    return False


def _convert_messages(messages: List[TextChatMessage]):
    """将 HippoRAG 的 dict 格式消息转为 LangChain Message 对象"""
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if hasattr(content, 'safe_substitute'):
            content = str(content)
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = self.llm_config.generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata, True

        result = func(self, *args, **kwargs)
        message, metadata = result

        # 累加 token 消耗（仅非缓存命中时）
        self.total_prompt_tokens += metadata.get("prompt_tokens", 0)
        self.total_completion_tokens += metadata.get("completion_tokens", 0)

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


class LangChainLLM(BaseLLM):
    """基于 LangChain ChatOpenAI 的 LLM 实现，接口与 CacheOpenAI 完全一致。"""

    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "LangChainLLM":
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None, **kwargs):
        super().__init__(global_config=global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config
        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        # token 累加器
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()

        max_tokens = global_config.max_new_tokens or 2048

        client_kwargs = {
            "model": self.llm_name,
            "temperature": global_config.temperature,
            "max_tokens": max_tokens,
            # Disable SDK short retries; infer() performs controlled exponential backoff.
            "max_retries": 0,
            "request_timeout": 120,
        }
        if self.llm_base_url:
            client_kwargs["base_url"] = self.llm_base_url
        api_key = getattr(self.global_config, 'llm_api_key', None) or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key

        self.client = ChatOpenAI(**client_kwargs)
        logger.info(f"LangChainLLM initialized: model={self.llm_name}, base_url={self.llm_base_url}")

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__
        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
            "model": self.global_config.llm_name,
            "max_completion_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)

    @cache_response
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict]:
        lc_messages = _convert_messages(messages)

        last_exc = None
        for attempt in range(_RETRY_MAX):
            try:
                response = self.client.invoke(lc_messages)
                response_message = response.content
                usage = response.response_metadata.get("token_usage", {})
                metadata = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish_reason": response.response_metadata.get("finish_reason", "stop"),
                }
                return response_message, metadata
            except Exception as e:
                last_exc = e
                if not _is_retryable_error(e):
                    raise
                if attempt < _RETRY_MAX - 1:
                    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
                    jitter = random.uniform(0, delay * 0.3)
                    total_delay = delay + jitter
                    logger.warning(
                        f"[LangChainLLM Retry {attempt + 1}/{_RETRY_MAX}] "
                        f"Rate-limited or transient error, waiting {total_delay:.1f}s... "
                        f"Error: {str(e)[:200]}"
                    )
                    time.sleep(total_delay)
        raise last_exc

    def get_token_usage(self) -> dict:
        return {
            "input_tokens": self.total_prompt_tokens,
            "output_tokens": self.total_completion_tokens,
        }

    def reset_token_usage(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
