import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .langchain_llm import LangChainLLM
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)

    # 优先使用 LangChain 适配器（通过配置 use_langchain=True 或 llm_backend="langchain"）
    use_langchain = getattr(config, 'use_langchain', False)
    if use_langchain:
        return LangChainLLM.from_experiment_config(config)

    return CacheOpenAI.from_experiment_config(config)
    