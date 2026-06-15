from typing import List, Optional

import os
import time
import volcenginesdkarkruntime
from src.core.token_tracer_util import ThreadLocalTokenTracker

embedding_token_tracker = ThreadLocalTokenTracker()

_EMBED_MAX_RETRIES = 5
_EMBED_BASE_DELAY = 2.0
_EMBED_MAX_DELAY = 60.0

def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "rate" in msg or "throttl" in msg or "tpm" in msg or "rpm" in msg or "too many requests" in msg

def _backoff_delay(attempt: int, is_rate_limit: bool) -> float:
    delay = _EMBED_BASE_DELAY * (2 ** attempt)
    if is_rate_limit:
        delay *= 2
    return min(delay, _EMBED_MAX_DELAY)

def truncate_and_normalize(embedding: List[float], dimension: Optional[int]) -> List[float]:
    """Truncate and L2 normalize embedding vector

    Args:
        embedding: The embedding vector to process
        dimension: Target dimension for truncation, None to skip truncation

    Returns:
        Processed embedding vector
    """
    if not dimension or len(embedding) <= dimension:
        return embedding

    import math

    embedding = embedding[:dimension]
    norm = math.sqrt(sum(x**2 for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    return embedding

class VolcengineEmbedder():
    """Volcengine Embedder Implementation

    Supports Volcengine embedding models such as doubao-embedding.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimension: Optional[int] = None,
        input_type: str = "text",
        tracker=None,
    ):
        """Initialize Volcengine Embedder

        Args:
            model_name: Volcengine model name (e.g., doubao-embedding)
            api_key: API key for authentication
            api_base: API base URL
            dimension: Target dimension for truncation (optional)
            input_type: Input type - "text" or "multimodal" (default: "multimodal")
            config: Additional configuration dict

        Raises:
            ValueError: If api_key is not provided
        """

        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base or "https://ark.cn-beijing.volces.com/api/v3"
        self.dimension = dimension
        self.input_type = input_type
        self.tracker = tracker if tracker is not None else embedding_token_tracker

        if not self.api_key:
            raise ValueError("api_key is required")

        # Initialize Volcengine client
        ark_kwargs = {"api_key": self.api_key}
        if self.api_base:
            ark_kwargs["base_url"] = self.api_base
        self.client = volcenginesdkarkruntime.Ark(**ark_kwargs)

        # Auto-detect dimension
        self._dimension = dimension
        if self._dimension is None:
            self._dimension = self._detect_dimension()

    def _detect_dimension(self) -> int:
        """Detect dimension by making an actual API call"""
        try:
            result = self.embed("test")
            return len(result) if result else 2048
        except Exception:
            return 2048  # Default dimension

    def _update_telemetry_token_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return

        def _usage_value(key: str, default: int = 0) -> int:
            if isinstance(usage, dict):
                return int(usage.get(key, default) or default)
            return int(getattr(usage, key, default) or default)

        prompt_tokens = _usage_value("prompt_tokens", 0)
        total_tokens = _usage_value("total_tokens", prompt_tokens)
        completion_tokens = max(total_tokens - prompt_tokens, 0)

        self.tracker.add(prompt_tokens, completion_tokens)
        # print("prompt_tokens", prompt_tokens)
        # print("total_tokens", total_tokens)
        # print("completion_tokens", completion_tokens)

    def embed(self, text: str) -> List[float]:
        """Perform dense embedding on text

        Args:
            text: Input text
            is_query: Flag to indicate if this is a query embedding

        Returns:
            List[float]: Result containing dense_vector

        Raises:
            RuntimeError: When API call fails
        """
        # Handle empty or whitespace-only text to avoid API errors
        if not text or not text.strip():
            return [0.0] * self.dimension

        def _embed_call():
            if self.input_type == "multimodal":
                # Use multimodal embeddings API
                response = self.client.multimodal_embeddings.create(
                    input=[{"type": "text", "text": text}], model=self.model_name
                )
                self._update_telemetry_token_usage(response)
                vector = response.data.embedding
            else:
                # Use text embeddings API
                response = self.client.embeddings.create(input=text, model=self.model_name)
                self._update_telemetry_token_usage(response)
                vector = response.data[0].embedding

            vector = truncate_and_normalize(vector, self.dimension)
            return vector

        try:
            return _embed_call()
        except Exception as e:
            last_err = e
            for attempt in range(_EMBED_MAX_RETRIES):
                is_rl = _is_rate_limit_error(last_err)
                if not is_rl:
                    break
                delay = _backoff_delay(attempt, is_rl)
                time.sleep(delay)
                try:
                    return _embed_call()
                except Exception as retry_err:
                    last_err = retry_err
            raise RuntimeError(f"Volcengine embedding failed after retries: {str(last_err)}") from last_err

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding

        Args:
            texts: List of texts
            is_query: Flag to indicate if these are query embeddings

        Returns:
            List[List[float]]: List of embedding results

        Raises:
            RuntimeError: When API call fails
        """
        if not texts:
            return []

        def _call() -> List[List[float]]:
            if self.input_type == "multimodal":
                results = []
                for t in texts:
                    # Skip empty or whitespace-only texts to avoid API errors
                    if t and t.strip():
                        results.append(self.embed(text=t))
                    else:
                        # Return zero vector for empty texts
                        results.append([0.0] * self.dimension)
                return results
            else:
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                self._update_telemetry_token_usage(response)

            return [
                truncate_and_normalize(item.embedding, self.dimension)
                for item in response.data
            ]

        try:
            return _call()
        except Exception as e:
            last_err = e
            for attempt in range(_EMBED_MAX_RETRIES):
                is_rl = _is_rate_limit_error(last_err)
                if not is_rl:
                    break
                delay = _backoff_delay(attempt, is_rl)
                time.sleep(delay)
                try:
                    return _call()
                except Exception as retry_err:
                    last_err = retry_err
            print(
                f"Volcengine batch embedding failed, texts length: {len(texts)}, input_type: {self.input_type}, model_name: {self.model_name}"
            )
            raise RuntimeError(f"Volcengine batch embedding failed after retries: {str(last_err)}") from last_err

    def get_dimension(self) -> int:
        return self._dimension
    
def main():
    test_text = "let's test VolcengineEmbedder!"
    test_text2 = "let's test VolcengineEmbedder!"
    api_key = os.environ.get("EMBEDDING_API_KEY", "").strip()
    api_base = os.environ.get("EMBEDDING_BASE_URL", "").strip()
    model_name = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not api_key or not api_base or not model_name:
        raise ValueError(
            "EMBEDDING_API_KEY/EMBEDDING_BASE_URL/EMBEDDING_MODEL "
            "must be configured in ov_test/.env"
        )
    embedder = VolcengineEmbedder(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            input_type="multimodal",
            dimension=2048,
        )
    result1 = []
    result1.append(embedder.embed(test_text))
    result1.append(embedder.embed(test_text2))
    import numpy as np
    
    arr1 = np.asarray(result1, dtype=np.float16)
    print(arr1)
    print(len(arr1[0]))

    texts = [test_text, test_text2, "doc 3", "doc 4", "doc 5"]
    result2 = embedder.embed_batch(texts=texts)
    arr2 = np.asarray(result2, dtype=np.float16)
    print(arr2)
    print(len(arr2[0]))

if __name__ == "__main__":
    main()
