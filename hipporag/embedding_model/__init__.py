from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .LangChain import LangChainEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2", use_langchain: bool = False):
    # LangChain 适配器优先（当 use_langchain=True 时）
    if use_langchain:
        return LangChainEmbeddingModel
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"