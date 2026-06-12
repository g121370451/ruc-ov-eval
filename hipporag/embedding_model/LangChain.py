import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


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

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]

        if self._use_volcengine:
            # 火山引擎多模态接口：逐条调用，每条包装成 {"type":"text","text":"..."}
            all_embeddings = []
            for t in texts:
                resp = self.client.multimodal_embeddings.create(
                    model=self.embedding_model_name,
                    input=[{"type": "text", "text": t}]
                )
                all_embeddings.append(resp.data.embedding)
            return np.array(all_embeddings)
        else:
            embeddings = self.client.embed_documents(texts)
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
