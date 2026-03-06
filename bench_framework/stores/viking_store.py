import os
import time
from typing import List, Optional
import tiktoken
import openviking as ov
from openviking.storage.queuefs.queue_manager import get_queue_manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from bench_framework.adapters.base import StandardDoc
from bench_framework.stores.base import VectorStoreBase
from bench_framework.core.monitor import BenchmarkMonitor
from bench_framework.types import IngestStats, SearchResult

class VikingStoreWrapper(VectorStoreBase):
    def __init__(self, store_path: str, doc_output_dir:str):
        self.store_path = store_path
        self.doc_output_dir = doc_output_dir
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

    def ingest(self, samples: List[StandardDoc], max_workers: int = 10, monitor: Optional[BenchmarkMonitor] = None) -> IngestStats:
        start_time = time.time()

        def _submit_sample(sample: StandardDoc):
            if monitor:
                monitor.worker_start()
            try:
                self.client.add_resource(sample.doc_path, wait=False)
                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                if monitor:
                    monitor.worker_end(success=False)
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(samples), desc="Ingesting Docs", unit="file")
            futures = [executor.submit(_submit_sample, s) for s in samples]
            for _ in as_completed(futures):
                if monitor:
                    pbar.set_postfix(monitor.get_status_dict())
                pbar.update(1)
            pbar.close()

        self.client.wait_processed()

        semantic_queue = get_queue_manager().get_queue("Semantic")
        tokens_cost = semantic_queue.get_tokens_cost()

        input_tokens = 0
        output_tokens = 0
        if isinstance(tokens_cost, dict):
            input_tokens = tokens_cost.get("summary_tokens_cost", 0) + \
                        tokens_cost.get("overview_tokens_cost", 0)
            output_tokens = tokens_cost.get("summary_output_tokens_cost", 0) + \
                            tokens_cost.get("overview_output_tokens_cost", 0)

        return IngestStats(
            time=time.time() - start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def retrieve(self, query: str, topk: int = 10, target_uri: Optional[str] = None) -> SearchResult:
        """执行检索"""
        ov_result = self.client.find(query=query, limit=topk, target_uri=target_uri or "viking://resources")
        return ov_result

    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        return str(self.client.read(uri))

    def clear(self) -> None:
        """清空库"""
        self.client.rm("viking://resources", recursive=True)