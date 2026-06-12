"""
共享退避重试工具模块。

为所有 LLM / Embedding 调用点提供统一的指数退避 + 随机抖动重试策略，
专门应对火山引擎 ARK API 的 429 TPM (Tokens Per Minute) 限流。
"""

import asyncio
import random
import time
import functools
import logging
from typing import Callable, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# 默认重试参数
DEFAULT_MAX_RETRIES = 8
DEFAULT_BASE_DELAY = 2.0      # 基础等待秒数
DEFAULT_MAX_DELAY = 120.0      # 最大等待秒数（TPM 限流窗口可能长达 60s）
DEFAULT_TPM_BASE_DELAY = 8.0
DEFAULT_TPM_MAX_DELAY = 180.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """判断异常是否为速率限制错误 (429 / RateLimit / TooManyRequests)"""
    exc_str = str(exc).lower()
    if "429" in exc_str:
        return True
    if "ratelimit" in exc_str or "rate limit" in exc_str:
        return True
    if "toomanyrequests" in exc_str:
        return True
    if "tpm" in exc_str or "tokens per minute" in exc_str:
        return True
    # 检查 status_code 属性
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    # LangChain / OpenAI SDK 的 RateLimitError
    cls_name = type(exc).__name__.lower()
    if "ratelimit" in cls_name:
        return True
    return False


def _is_tpm_rate_limit_error(exc: Exception) -> bool:
    """判断异常是否为分钟级 token 限流。"""
    exc_str = str(exc).lower()
    return (
        "tpm" in exc_str
        or "tokens per minute" in exc_str
        or "modelaccounttpmratelimitexceeded" in exc_str
    )


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否可重试（429 / 502 / 503 / 504 / 连接错误）"""
    if _is_rate_limit_error(exc):
        return True
    status = getattr(exc, "status_code", None)
    if status in (502, 503, 504):
        return True
    cls_name = type(exc).__name__.lower()
    if "connection" in cls_name or "timeout" in cls_name:
        return True
    return False


def calculate_retry_delay(
    exc: Exception,
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    tpm_base_delay: float = DEFAULT_TPM_BASE_DELAY,
    tpm_max_delay: float = DEFAULT_TPM_MAX_DELAY,
) -> float:
    """计算重试等待时间；TPM 使用分钟级退避，普通瞬时错误使用短退避。"""
    if _is_tpm_rate_limit_error(exc):
        delay = min(tpm_base_delay * (2 ** attempt), tpm_max_delay)
        return delay + random.uniform(0, 10.0)

    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay + random.uniform(0, delay * 0.3)


def exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
):
    """同步指数退避装饰器。

    用法:
        @exponential_backoff(max_retries=8, base_delay=2, max_delay=120)
        def my_llm_call(...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not _is_retryable(e):
                        raise
                    if attempt < max_retries - 1:
                        total_delay = calculate_retry_delay(e, attempt, base_delay, max_delay)
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] "
                            f"Rate-limited or transient error, "
                            f"waiting {total_delay:.1f}s... "
                            f"Error: {str(e)[:200]}"
                        )
                        time.sleep(total_delay)
            raise last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


def async_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
):
    """异步指数退避装饰器。

    用法:
        @async_exponential_backoff(max_retries=8, base_delay=2, max_delay=120)
        async def my_async_llm_call(...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if not _is_retryable(e):
                        raise
                    if attempt < max_retries - 1:
                        total_delay = calculate_retry_delay(e, attempt, base_delay, max_delay)
                        logger.warning(
                            f"[Retry {attempt + 1}/{max_retries}] "
                            f"Rate-limited or transient error, "
                            f"waiting {total_delay:.1f}s... "
                            f"Error: {str(e)[:200]}"
                        )
                        await asyncio.sleep(total_delay)
            raise last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs,
):
    """同步函数调用包装器（非装饰器形式）。

    用法:
        result = retry_with_backoff(my_func, arg1, arg2, max_retries=8)
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not _is_retryable(e):
                raise
            if attempt < max_retries - 1:
                total_delay = calculate_retry_delay(e, attempt, base_delay, max_delay)
                logger.warning(
                    f"[Retry {attempt + 1}/{max_retries}] "
                    f"Rate-limited or transient error, "
                    f"waiting {total_delay:.1f}s... "
                    f"Error: {str(e)[:200]}"
                )
                time.sleep(total_delay)
    raise last_exc


async def async_retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    **kwargs,
):
    """异步函数调用包装器（非装饰器形式）。

    用法:
        result = await async_retry_with_backoff(my_async_func, arg1, max_retries=8)
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not _is_retryable(e):
                raise
            if attempt < max_retries - 1:
                total_delay = calculate_retry_delay(e, attempt, base_delay, max_delay)
                logger.warning(
                    f"[Retry {attempt + 1}/{max_retries}] "
                    f"Rate-limited or transient error, "
                    f"waiting {total_delay:.1f}s... "
                    f"Error: {str(e)[:200]}"
                )
                await asyncio.sleep(total_delay)
    raise last_exc
