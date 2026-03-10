import os
import time
import logging
from typing import List, Dict
from src.adapters.base import StandardDoc, StandardSample
import tiktoken
import openviking as ov
from openviking.storage.queuefs.queue_manager import get_queue_manager
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def ingest(self, samples: List[StandardDoc], max_workers=10, monitor=None) -> dict:
        start_time = time.time()

        # 展开 doc_paths 并去重（保持顺序）
        seen = set()
        all_paths = []
        for sample in samples:
            for p in sample.doc_paths:
                if p not in seen:
                    seen.add(p)
                    all_paths.append(p)

        def _submit_path(path: str):
            if monitor:
                monitor.worker_start()
            try:
                self.client.add_resource(path, wait=False)
                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                if monitor:
                    monitor.worker_end(success=False)
                raise e

        # 1. 使用线程池并发提交资源
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(total=len(all_paths), desc="Ingesting Docs", unit="file")

            futures = [executor.submit(_submit_path, p) for p in all_paths]

            for _ in as_completed(futures):
                if monitor:
                    pbar.set_postfix(monitor.get_status_dict())
                pbar.update(1)
            pbar.close()

        # 2. 等待服务端处理完成
        self.client.wait_processed()

        # 3. 获取 Token 统计
        semantic_queue = get_queue_manager().get_queue("Semantic")
        tokens_cost = semantic_queue.get_tokens_cost()
        
        input_tokens = 0
        output_tokens = 0
        if isinstance(tokens_cost, dict):
            input_tokens = tokens_cost.get("summary_tokens_cost", 0) + \
                        tokens_cost.get("overview_tokens_cost", 0)
            output_tokens = tokens_cost.get("summary_output_tokens_cost", 0) + \
                            tokens_cost.get("overview_output_tokens_cost", 0)

        return {
            "time": time.time() - start_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        """执行检索"""
        return self.client.find(query=query, limit=topk, target_uri=target_uri)

    def process_retrieval_results(self, search_res):
        """
        从检索结果中提取 retrieved_texts / context_blocks / retrieved_uris。
        Viking 根据 resource.level 决定读全文还是拼接 abstract+overview。
        """
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []
        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            if getattr(r, 'level', 2) == 2:
                content = self.read_resource(r.uri)
            else:
                content = f"{getattr(r, 'abstract', '')}\n{getattr(r, 'overview', '')}"
            retrieved_texts.append(content)
            context_blocks.append(content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    def build_uri_map(self, doc_info: List[StandardDoc]) -> Dict[str, list]:
        """构建 sample_id -> [viking:// URI] 映射，通过 client.ls 验证 URI 存在性"""
        logger = logging.getLogger(__name__)
        uri_map = {}
        for doc in doc_info:
            for path in doc.doc_paths:
                basename = os.path.splitext(os.path.basename(path))[0]
                basename = basename.replace('.', '')
                basename = basename.replace(' ', '_')
                basename = basename.replace('(', '')
                basename = basename.replace(')', '')
                candidate_uri = f"viking://resources/{basename}"
                try:
                    self.client.ls(candidate_uri)
                    uri_map.setdefault(doc.sample_id, []).append(candidate_uri)
                except Exception:
                    logger.warning(f"URI not found for doc: {basename} (sample_id={doc.sample_id})")
        return uri_map

    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        return str(self.client.read(uri))

    def clear(self):
        """清空库：删除所有资源 + 删除 store 目录"""
        import shutil
        self.client.rm("viking://resources", recursive=True)
        self.client.close()
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)