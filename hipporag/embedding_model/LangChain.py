import os
import random
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


def _is_retryable_embedding_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if any(
        marker in text
        for marker in (
            "429",
            "ratelimit",
            "rate limit",
            "toomanyrequests",
            "tpm",
            "tokens per minute",
            "modelaccounttpmratelimitexceeded",
        )
    ):
        return True
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    cls_name = type(exc).__name__.lower()
    return any(marker in cls_name for marker in ("ratelimit", "connection", "timeout"))


def _embedding_retry_delay(exc: Exception, attempt: int, base_delay: float = 2.0, max_delay: float = 120.0) -> float:
    text = str(exc).lower()
    if "tpm" in text or "tokens per minute" in text or "modelaccounttpmratelimitexceeded" in text:
        return min(8.0 * (2 ** attempt), 180.0) + random.uniform(0, 10.0)
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay + random.uniform(0, delay * 0.3)


class LangChainEmbeddingModel(BaseEmbeddingModel):
    """Embedding 适配器，支持标准 OpenAI 接口和火山引擎多模态接口。"""

    def __init__(self, global_config: Optional[BaseConfig] = None,
                 embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name

        self._init_embedding_config()

        api_key = getattr(self.global_config, 'embedding_api_key', None) or os.environ.get("OPENAI_API_KEY")
        base_url = self.global_config.embedding_base_url

        # 判断是否为火山引擎多模态 embedding 模型
        self._use_volcengine = "doubao-embedding" in self.embedding_model_name and "vision" in self.embedding_model_name

        if self._use_volcengine:
            from volcenginesdkarkruntime import Ark
            self.client = Ark(api_key=api_key, base_url=base_url)
            logger.info(f"VolcEngine multimodal embedding: model={self.embedding_model_name}")
        else:
            from langchain_openai import OpenAIEmbeddings
            embed_kwargs = {
                "model": self.embedding_model_name,
                "check_embedding_ctx_length": False,
                "max_retries": 0,
                "request_timeout": 120,
            }
            if base_url:
                embed_kwargs["base_url"] = base_url
            if api_key:
                embed_kwargs["api_key"] = api_key
            self.client = OpenAIEmbeddings(**embed_kwargs)
            logger.info(f"OpenAI embedding: model={self.embedding_model_name}")

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)

    def _run_with_exponential_retry(self, func, *, max_retries: int = 10):
        last_exc = None
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exc = e
                if _is_retryable_embedding_error(e):
                    if attempt >= max_retries - 1:
                        break
                    total_delay = _embedding_retry_delay(e, attempt)
                    logger.warning(f"Embedding retry {attempt + 1}/{max_retries} after {total_delay:.1f}s")
                    time.sleep(total_delay)
                else:
                    raise
        raise RuntimeError(f"Embedding failed after {max_retries} retries: {last_exc}")

    def _volcengine_embed_with_retry(self, text: str, max_retries: int = 10):
        return self._run_with_exponential_retry(
            lambda: self.client.multimodal_embeddings.create(
                model=self.embedding_model_name,
                input=[{"type": "text", "text": text}]
            ).data.embedding,
            max_retries=max_retries,
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]

        if self._use_volcengine:
            all_embeddings = []
            for t in texts:
                emb = self._volcengine_embed_with_retry(t)
                all_embeddings.append(emb)
            return np.array(all_embeddings)
        else:
            embeddings = self._run_with_exponential_retry(lambda: self.client.embed_documents(texts))
            return np.array(embeddings)

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.append(self.encode(batch))
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            results = results / norms

        return results
