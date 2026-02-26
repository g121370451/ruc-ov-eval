import os
import time
import tiktoken
import openviking as ov
from openviking.storage.queuefs.queue_manager import get_queue_manager

class VikingStoreWrapper:
    def __init__(self, store_path: str):
        self.store_path = store_path
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        
        # 初始化 OpenViking 客户端
        self.client = ov.SyncOpenViking(path=store_path)
        
        # 初始化 Tokenizer (cl100k_base)
        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples, base_dir) -> dict:
        start_time = time.time()
        
        # 1. 提交资源
        for sample in samples:         
            doc_path = os.path.join(base_dir, f"{sample.sample_id}_doc.md")
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(sample.doc_content)
            self.client.add_resource(doc_path, wait=False)

        # 2. 等待处理完成
        self.client.wait_processed()

        # 3. 获取并解析 Token 消耗
        semantic_queue = get_queue_manager().get_queue("Semantic")
        tokens_cost = semantic_queue.get_tokens_cost()

        # 初始化统计值
        input_tokens = 0
        output_tokens = 0

        if isinstance(tokens_cost, dict):
            input_tokens = tokens_cost.get("summary_tokens_cost", 0) + \
                        tokens_cost.get("overview_tokens_cost", 0)
            output_tokens = tokens_cost.get("summary_output_tokens_cost", 0) + \
                            tokens_cost.get("overview_output_tokens_cost", 0)
            # input_tokens += tokens_cost.get("embedding_tokens_cost", 0)

        return {
            "time": time.time() - start_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        """执行检索"""
        return self.client.find(query=query, limit=topk, target_uri=target_uri)

    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        return str(self.client.read(uri))

    def clear(self):
        """清空库"""
        self.client.rm("viking://resources", recursive=True)