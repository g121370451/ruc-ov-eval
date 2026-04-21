from typing import Any, Dict, List, Optional

import volcenginesdkarkruntime

class TokenTracker:
    """追踪ChatAPI调用过程中的输入/输出token消耗（线程安全）"""
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int):
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def reset(self):
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0

    def get(self):
        with self._lock:
            return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

# embedding token追踪器实例
embedding_token_tracker = TokenTracker()
# 全局token追踪器实例
token_tracker = TokenTracker()

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

        embedding_token_tracker.add(prompt_tokens, completion_tokens)
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
            raise RuntimeError(f"Volcengine embedding failed: {str(e)}") from e

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
            print(
                f"Volcengine batch embedding failed, texts length: {len(texts)}, input_type: {self.input_type}, model_name: {self.model_name}"
            )
            raise RuntimeError(f"Volcengine batch embedding failed: {str(e)}") from e

    def get_dimension(self) -> int:
        return self._dimension
    
def main():
    test_text = "let's test VolcengineEmbedder!"
    test_text2 = "let's test VolcengineEmbedder!"
    embedder = VolcengineEmbedder(
            model_name="doubao-embedding-vision-250615",
            api_key="68e15b71-7673-4734-bf7a-01bb80a127ea",
            api_base="https://ark.cn-beijing.volces.com/api/v3",
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

    return
    bs = 1
    emb_list: List[List[float]] = []

    embed_api_key = "68e15b71-7673-4734-bf7a-01bb80a127ea"
    embed_base_url = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"

    def _http_embed(model: str, inputs: List[str]) -> List[List[float]]:
        import requests

        url = embed_base_url
        headers = {"Content-Type": "application/json"}
        if embed_api_key:
            headers["Authorization"] = f"Bearer {embed_api_key}"
        payload = {"model": model, "input": [{"type":"text", "text": t} for t in inputs]}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        return [data.get("data").get("embedding")]

    for j in range(0, len(texts), bs):
        batch = texts[j : j + bs]
        outs = _http_embed("doubao-embedding-vision-250615", batch)  # raw text
        emb_list.extend(outs)

    arr = np.asarray(emb_list, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / (norm + 1e-12)

    print(arr.astype(np.float16))
    print(len(arr[0]))
    
test_config = """{
        "model": "doubao-embedding-vision-250615",
        "api_key": "your_api_key_here",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "dimension": 1024,
        "provider": "volcengine",
        "input": "multimodal",
        "batch_size": 8
    }"""

if __name__ == "__main__":
    main()
