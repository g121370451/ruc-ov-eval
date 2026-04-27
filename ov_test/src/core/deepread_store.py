import os
import json
import time
import numpy as np
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

from src.adapters.base import StandardDoc
from src.core.monitor import BenchmarkMonitor
from src.core.logger import get_logger

import sys
DEEPREAD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "DeepRead")
sys.path.insert(0, DEEPREAD_PATH)

from DeepRead.DeepRead import (
    DocIndex,
    load_corpus,
    run_agent,
    JsonlLogger,
    _normalize_neighbor_window,
)
from DeepRead.parser_pdf import parse_markdown_to_corpus
from DeepRead.utils import VolcengineEmbedder, embedding_token_tracker
import utils as _deepread_utils


@dataclass
class DeepReadResource:
    uri: str = ""
    content: str = ""
    score: float = 0.0


@dataclass
class DeepReadResult:
    resources: List[DeepReadResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    retrieved_texts: List[str] = field(default_factory=list)


class DeepReadWrapper:
    """
    DeepRead 向量存储包装器，接口与 VikingStoreWrapper 对齐。

    每个 sample_id 对应 doc_output_dir/{sample_id}/ 下的独立目录。
    ingest 阶段：PDF -> PaddleOCR -> Markdown -> corpus JSON + embedding .npy
    retrieve 阶段：按 target_uri（即 sample_id）加载对应目录的 corpus，
    调用 run_agent 返回最终答案字符串。
    """

    def __init__(
        self,
        store_path: str,
        doc_output_dir: str,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        enable_vector: bool = True,
        enable_hybrid: bool = False,
        enable_semantic: bool = False,
        neighbor_window: str = "1,-1",
        max_rounds: int = 50,
        use_pymupdf: bool = False,
    ):
        self.store_path = store_path
        self.doc_output_dir = doc_output_dir
        self.logger = get_logger()

        os.makedirs(self.store_path, exist_ok=True)

        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.enable_vector = enable_vector
        self.enable_hybrid = enable_hybrid
        self.enable_semantic = enable_semantic
        self.max_rounds = max_rounds
        self.use_pymupdf = use_pymupdf

        # Neighbor window
        try:
            parts = [p.strip() for p in str(neighbor_window).split(",")]
            if len(parts) == 2:
                self.neighbor_window = _normalize_neighbor_window((int(parts[0]), int(parts[1])))
            else:
                self.neighbor_window = None
        except Exception:
            self.neighbor_window = None

        self.log_path = os.path.join(self.store_path, "deepread_run.log")

        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed: {e}")
            self.enc = None

    @classmethod
    def from_config(cls, store_path: str, doc_output_dir: str, llm_cfg: dict, store_cfg: dict) -> "DeepReadWrapper":
        """从 config.yaml 的三个子块构造实例，供 run.py 调用。"""
        neighbor_window = store_cfg.get("neighbor_window", "1,-1")
        return cls(
            store_path=store_path,
            doc_output_dir=doc_output_dir,
            model=llm_cfg.get("model", ""),
            base_url=llm_cfg.get("base_url", ""),
            api_key=llm_cfg.get("api_key", ""),
            temperature=llm_cfg.get("temperature", 0.0),
            enable_vector=store_cfg.get("enable_vector", True),
            enable_hybrid=store_cfg.get("enable_hybrid", False),
            enable_semantic=store_cfg.get("enable_semantic", False),
            neighbor_window=str(neighbor_window),
            max_rounds=store_cfg.get("max_rounds", 12),
            use_pymupdf=store_cfg.get("use_pymupdf", False),
        )
    
    def _pdf_to_markdown_pymupdf(self, pdf_path: str, md_path: str, sample_id: str):
        """
        用 pymupdf 从数字原生 PDF 提取文本，写成 Markdown 文件。
        每页以 `## Page N` 作为标题，保留段落换行。
        仅适用于数字原生 PDF（非扫描版）。
        """
        import fitz
        # pymupdf

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        with open(md_path, "w", encoding="utf-8") as f:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                f.write(f"## Page {page_num}\n\n")
                f.write(text)
                f.write("\n\n")
        doc.close()
        self.logger.info(f"[{sample_id}] pymupdf extracted {page_count} pages -> {md_path}")

    def _sample_dir(self, sample_id: str) -> str:
        """返回 sample 的独立工作目录路径。"""
        return os.path.join(self.doc_output_dir, sample_id)
    
    def _corpus_path(self, sample_id: str) -> str:
        """返回 sample 的 corpus JSON 路径。"""
        return os.path.join(self._sample_dir(sample_id), f"{sample_id}_corpus.json")

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))
    
    def build_uri_map(self, doc_info: list[StandardDoc]) -> Dict[str, list]:
        """"""
        return {doc.sample_id: [doc.sample_id] for doc in doc_info}

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor: Optional[BenchmarkMonitor] = None) -> dict:
        """
        将文档转换为 DeepRead 的 corpus JSON 格式并写入 store_path，
        然后构建 DocIndex。
        """

        # TODO 完善并发处理逻辑（monitor在并发处理时起到展示作用）
        start_time = time.time()
        embedding_token_tracker.reset()

        if self.use_pymupdf:
            ocr_pipeline = None
        else:
            from paddleocr import PaddleOCRVL
            ocr_pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url="http://127.0.0.1:8956/v1",
            )

        embedder = VolcengineEmbedder(
            model_name="doubao-embedding-vision-250615",
            api_key=self.api_key,
            api_base=self.base_url,
            input_type="multimodal",
            dimension=2048,
        )

        for sample in tqdm(samples, desc="Ingesting Docs to DeepRead"):
            if monitor:
                monitor.worker_start()

            try:
                self._ingest_one(sample, ocr_pipeline, embedder)
                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                self.logger.error(f"Failed to ingest sample {sample.sample_id}: {e}")
                if monitor:
                    monitor.worker_end(success=False)
                raise e

        token_usage = embedding_token_tracker.get()
        return {
            "time": time.time() - start_time,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        }
    
    def _ingest_one(self, sample: StandardDoc, ocr_pipeline, embedder: VolcengineEmbedder):
        doc_paths = sample.doc_paths

        for path in doc_paths:
            if not os.path.exists(path):
                self.logger.warning(f"Document path does not exist: {path}")
                return
            name = os.path.splitext(os.path.basename(path))[0]
            name = re.sub(r'[\\/*?:"<>|]', '_', name)[:120]
            name = name if name else hashlib.sha1(path.encode('utf-8')).hexdigest()[:16]

            merged_md_path = os.path.join(self.store_path, f"{name}.md")
            corpus_path = os.path.join(self.store_path, f"{name}_corpus.json")
            emb_path = os.path.join(self.store_path, f"{name}_emb.npy")
            idmap_path = os.path.join(self.store_path, f"{name}_idmap.json")

            #  ---Step 1: PDF -> Markdown (or Markdown directly) ---
            ext = os.path.splitext(path)[1].lower()
            if ext in [".md", ".markdown"]:
                import shutil
                shutil.copy2(path, merged_md_path)
                self.logger.info(f"[{path}] Copied existing Markdown {merged_md_path}")
            elif self.use_pymupdf:
                self._pdf_to_markdown_pymupdf(path, merged_md_path, name)
            else:
                # TODO 没有经过验证，暂时忽略
                merged_json_path = os.path.join(self.store_path, f"{name}.json")
                temp_json_path = os.path.join(self.store_path, f"{name}_temp_page.json")
                temp_md_path = os.path.join(self.store_path, f"{name}_temp_page.md")
                
                output = ocr_pipeline.predict(path)
                all_json_data = []

                with open(merged_md_path, "w", encoding="utf-8"):
                    pass

                for i, res in enumerate(output):
                    try:
                        res.save_to_json(save_path=temp_json_path)
                        with open(temp_json_path, "r", encoding="utf-8") as f:
                            all_json_data.append(json.load(f))
                    except Exception as e:
                        self.logger.warning(f"[{name}] JSON page {i+1} error: {e}")

                    try:
                        res.save_to_markdown(save_path=temp_md_path)
                        with open(temp_md_path, "r", encoding="utf-8") as f:
                            page_content = f.read()
                        with open(merged_md_path, "a", encoding="utf-8") as f:
                            f.write(page_content)
                            if i < len(output) - 1:
                                f.write("\n\n")
                    except Exception as e:
                        self.logger.warning(f"[{name}] Markdown page {i+1} error: {e}")

                try:
                    with open(merged_json_path, "w", encoding="utf-8") as f:
                        json.dump(all_json_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.warning(f"[{name}] Save merged JSON error: {e}")

                for p in (temp_json_path, temp_md_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

            # --- Step 2: Markdown -> corpus ---
            corpus = parse_markdown_to_corpus(merged_md_path)

            # --- Step 3: Embedding ---
            texts: List[str] = []
            id_map: List[Dict[str, Any]] = []

            for n in corpus.get("nodes", []):
                nid = n.get("id")
                for pi, p in enumerate(n.get("paragraphs", [])):
                    if isinstance(p, str):
                        t = p.strip()
                    elif isinstance(p, dict):
                        t = str(p.get("content", "")).strip()
                    else:
                        t = str(p).strip()

                    if not t:
                        continue
                    texts.append(t)
                    id_map.append({"node_id": nid, "paragraph_index": pi})

            if texts:
                emb_list: List[List[float]] = [
                    embedder.embed(text=t) 
                    for t in tqdm(texts, desc=f"[{name}] Embedding", unit="chunk", leave=False)]
                arr = np.asarray(emb_list, dtype=np.float16)

                np.save(emb_path, arr)
                with open(idmap_path, "w", encoding="utf-8") as f:
                    json.dump(id_map, f, ensure_ascii=False)

                corpus["vector_store"] = {
                    "matrix_path": emb_path,
                    "id_map_path": idmap_path,
                    "model_name": "doubao-embedding-vision-250615",
                    "normalized": True,
                    "dtype": "float16",
                    "embed_base_url": self.base_url,
                }

            # --- Step 4: save corpus JSON ---
            with open(corpus_path, "w", encoding="utf-8") as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[{name}] Corpus saved to {corpus_path}")

                
    def retrieve(self, query: str, topk: int = 5, target_uri: str = None) -> DeepReadResult:
        """
        使用 run_agent 对指定 sample 执行多轮检索并返回最终答案。

        target_uri: sample_id，用于定位该 sample 的 corpus 目录。
        返回值中 resources[0].content 即为 agent 的最终答案字符串。
        """
        # 找到store_path下所有_corpus.json文件
        corpus_paths = [os.path.join(self.store_path, f) 
                        for f in os.listdir(self.store_path) 
                        if f.endswith('_corpus.json')]
        
        try:
            doc_index = load_corpus(corpus_paths, neighbor_window=self.neighbor_window)
        except Exception as e:
            self.logger.error(f"Failed to load corpus for '{self.store_path}': {e}")
            return DeepReadResult()

        jsonl_logger = JsonlLogger(self.log_path)

        # reset 追踪器，确保只统计本次 run_agent 的 token 消耗
        # 必须用 utils（非 DeepRead.utils），与 DeepRead.py 内部的 _token_tracker 是同一实例
        token_tracker = _deepread_utils.token_tracker
        token_tracker.reset()

        collected_texts: List[str] = []

        try:
            answer = run_agent(
                model=self.model,
                base_url=self.base_url,
                doc_index=doc_index,
                user_question=query,
                logger=jsonl_logger,
                max_rounds=self.max_rounds,
                temperature=self.temperature,
                api_key=self.api_key,
                enable_vector=self.enable_vector,
                enable_hybrid=self.enable_hybrid,
                enable_semantic=self.enable_semantic,
                disable_bm25=False,
                disable_regex=False,
                disable_read=False,
                neighbor_window=self.neighbor_window,
                vector_topk=1,
                hybrid_topk=1,
                semantic_topk1=30,
                semantic_topk2=1,
                collected_texts=collected_texts,
            )
        except Exception as e:
            self.logger.error(f"run_agent failed for '{self.store_path}': {e}")
            answer = f"[Error] {str(e)}"

        usage = token_tracker.get()
        return DeepReadResult(
            resources=[DeepReadResource(
                uri=f"deepread://{self.store_path}",
                content=answer,
                score=1.0,
            )],
            retrieve_input_tokens=usage["input_tokens"],
            retrieve_output_tokens=usage["output_tokens"],
            retrieved_texts=collected_texts,
        )

    def process_retrieval_results(self, search_res: DeepReadResult):
        """
        从检索结果中提取 retrieved_texts / context_blocks / retrieved_uris。
        """
        retrieved_uris = [r.uri for r in search_res.resources]
        context_blocks = [r.content for r in search_res.resources]
        retrieved_texts = search_res.retrieved_texts if search_res.retrieved_texts else context_blocks

        return retrieved_texts, context_blocks, retrieved_uris

    def read_resource(self, uri: str) -> str:
        """读取 sample 对应的 Markdown 文件内容。uri 格式为 deepread://{sample_id}。"""
        sample_id = uri.replace("deepread://", "")
        md_path = os.path.join(self._sample_dir(sample_id), f"{sample_id}.md")
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def clear(self):
        """清空所有 sample 的 corpus 目录。"""
        if os.path.exists(self.store_path):
            import shutil
            for entry in os.listdir(self.store_path):
                entry_path = os.path.join(self.store_path, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                    self.logger.info(f"Removed sample dir: {entry_path}")

def main():
    # 简单测试
    from paddleocr import PaddleOCRVL
    ocr_pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://10.77.110.187:8956/v1",
    )
    output = ocr_pipeline.predict("/Users/zhangqianyi/Desktop/ruc-ov/Data/FinanceBench/pdfs/3M_2015_10K.pdf")
    a = 1

if __name__ == "__main__":
    main()