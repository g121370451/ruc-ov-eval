import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from paddleocr import PaddleOCRVL 

from src.adapters.base import StandardDoc
from src.core.monitor import BenchmarkMonitor
from src.core.logger import get_logger

# Import DeepRead modules
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


@dataclass
class DeepReadResource:
    """DeepRead 检索返回的单个资源"""
    uri: str = ""
    content: str = ""
    level: int = 2
    score: float = 0.0
    abstract: str = ""
    overview: str = ""


@dataclass
class DeepReadResult:
    """DeepRead 检索返回结果，与 OpenViking 的 find 返回格式对齐"""
    resources: List[DeepReadResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0


class DeepReadWrapper:
    """
    DeepRead 向量存储包装器，接口与 VikingStoreWrapper 对齐。

    DeepRead 是一个 Agent-based RAG 系统，使用 DocIndex 管理文档索引，
    通过 run_agent 驱动多轮工具调用（BM25/Vector/Regex Search + Read Section）
    来回答用户问题。
    """

    def __init__(
        self,
        store_path: str,
        doc_output_dir: str = "",
        embedding_batch_size: int = 1,
        config_path: str = None,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        enable_vector: bool = False,
        enable_hybrid: bool = False,
        enable_semantic: bool = False,
        neighbor_window: str = "1,-1",
        max_rounds: int = 12,
        temperature: float = 0.0,
    ):
        self.store_path = store_path
        self.doc_output_dir = doc_output_dir
        self.embedding_batch_size = embedding_batch_size
        self.config_path = config_path
        self.logger = get_logger()

        os.makedirs(self.store_path, exist_ok=True)

        # LLM 配置（用于 run_agent）
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.enable_vector = enable_vector
        self.enable_hybrid = enable_hybrid
        self.enable_semantic = enable_semantic
        self.max_rounds = max_rounds
        self.temperature = temperature

        # Neighbor window
        try:
            parts = [p.strip() for p in str(neighbor_window).split(",")]
            if len(parts) == 2:
                self.neighbor_window = _normalize_neighbor_window((int(parts[0]), int(parts[1])))
            else:
                self.neighbor_window = None
        except Exception:
            self.neighbor_window = None

        # DocIndex 实例
        self.doc_index: Optional[DocIndex] = None

        # JSONL logger for run_agent
        self.log_path = os.path.join(self.store_path, "deepread_run.log")
        self.jsonl_logger: Optional[JsonlLogger] = None

        # 加载已有的索引
        self._load_existing_index()

        # Tokenizer
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def _load_existing_index(self):
        """加载硬盘上已有的 corpus JSON 文件构建 DocIndex"""
        corpus_files = []
        if os.path.exists(self.store_path):
            for fname in os.listdir(self.store_path):
                if fname.endswith(".json") and not fname.startswith("_"):
                    corpus_files.append(os.path.join(self.store_path, fname))

        if corpus_files:
            try:
                self.doc_index = load_corpus(corpus_files, neighbor_window=self.neighbor_window)
                self.logger.info(f"Loaded existing index with {len(corpus_files)} corpus files")
            except Exception as e:
                self.logger.warning(f"Failed to load existing index: {e}")
                self.doc_index = None

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor: Optional[BenchmarkMonitor] = None) -> dict:
        """
        将文档转换为 DeepRead 的 corpus JSON 格式并写入 store_path，
        然后构建 DocIndex。
        """

        # TODO 完善并发处理逻辑（monitor在并发处理时起到展示作用）
        start_time = time.time()

        output_base_path = self.doc_output_dir
        # Create output directory
        os.makedirs(f"{output_base_path}/", exist_ok=True)

        pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url="http://127.0.0.1:8956/v1",
        )

        # TODO 参数化embedder
        embedder = VolcengineEmbedder(
            model_name="doubao-embedding-vision-250615",
            api_key="68e15b71-7673-4734-bf7a-01bb80a127ea",
            api_base="https://ark.cn-beijing.volces.com/api/v3",
            input_type="multimodal",
            dimension=2048,
        )

        for sample in tqdm(samples, desc="Ingesting Docs to DeepRead"):
            if monitor:
                monitor.worker_start()

            try:
                output = pipeline.predict(sample.doc_path)

                basename = output_base_path.split("/")[-1]
                merged_json_path = f"{output_base_path}/{basename}.json"
                merged_md_path = f"{output_base_path}/{basename}.md"
                temp_json_path = f"{output_base_path}/{basename}_temp_page.json"
                temp_md_path = f"{output_base_path}/{basename}_temp_page.md"

                all_json_data = []

                # ensure merged md exists/cleared
                with open(merged_md_path, "w", encoding="utf-8"):
                    pass

                for i, res in enumerate(output):

                    # JSON
                    try:
                        res.save_to_json(save_path=temp_json_path)
                        with open(temp_json_path, "r", encoding="utf-8") as f_temp_json:
                            page_data = json.load(f_temp_json)
                            all_json_data.append(page_data)
                    except Exception as e:
                        print(f"    ! Error processing JSON for page {i + 1}: {e}")

                    # Markdown
                    try:
                        res.save_to_markdown(save_path=temp_md_path)
                        with open(temp_md_path, "r", encoding="utf-8") as f_temp_md:
                            page_content = f_temp_md.read()
                        with open(merged_md_path, "a", encoding="utf-8") as f_final_md:
                            f_final_md.write(page_content)
                            if i < len(output) - 1:
                                f_final_md.write("\n\n")
                    except Exception as e:
                        print(f"    ! Error processing Markdown for page {i + 1}: {e}")

                try:
                    with open(merged_json_path, "w", encoding="utf-8") as f_final_json:
                        json.dump(all_json_data, f_final_json, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"  ! Error saving merged JSON: {e}")

                try:
                    if os.path.exists(temp_json_path):
                        os.remove(temp_json_path)
                    if os.path.exists(temp_md_path):
                        os.remove(temp_md_path)
                except Exception as e:
                    print(f"  ! Warning: could not remove temporary files: {e}")

                corpus = parse_markdown_to_corpus(merged_md_path)

                texts: List[str] = []
                id_map: List[Dict[str, Any]] = []

                for n in corpus.get("nodes", []):
                    nid = n.get("id")
                    pars = n.get("paragraphs", [])
                    for pi, p in enumerate(pars):
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
                    emb_list: List[List[float]] = []

                    for text in texts:
                        outs = embedder.embed(text=text)
                        emb_list.append(outs)
                    
                    arr = np.asarray(emb_list, dtype=np.float16)

                    emb_path = f"{output_base_path}/{basename}_emb.npy"
                    idmap_path = f"{output_base_path}/{basename}_idmap.json"

                    np.save(emb_path, arr.astype(np.float16))
                    with open(idmap_path, "w", encoding="utf-8") as f_id:
                        json.dump(id_map, f_id, ensure_ascii=False)

                    # TODO 参数化embedder相关信息
                    corpus["vector_store"] = {
                        "matrix_path": emb_path,
                        "id_map_path": idmap_path,
                        "model_name": "doubao-embedding-vision-250615",
                        "normalized": True,
                        "dtype": "float16",
                        "embed_base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    }

                corpus_path = f"{output_base_path}/{basename}_corpus.json"
                try:
                    with open(corpus_path, "w", encoding="utf-8") as f_c:
                        json.dump(corpus, f_c, indent=2, ensure_ascii=False)
                    print(f"--- Corpus JSON saved to {corpus_path} ---")
                except Exception as e:
                    print(f"  ! Error saving corpus JSON: {e}")
            except Exception as e:
                if monitor:
                    monitor.worker_end(success=False) # 线程异常结束
                raise e
        
        token_usage = embedding_token_tracker.get()
        return {
            "time": time.time() - start_time,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        }

    def _build_corpus_nodes(self, sample: StandardDoc) -> List[Dict[str, Any]]:
        """
        将文档转换为 DeepRead 的 node 格式。
        每个 node 包含: id, title, paragraphs, children
        """
        nodes = []
        doc_id = os.path.basename(sample.doc_path)

        try:
            if sample.doc_path.endswith('.md'):
                with open(sample.doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif sample.doc_path.endswith('.txt'):
                with open(sample.doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                self.logger.warning(f"Unsupported file type for DeepRead: {sample.doc_path}")
                return nodes

            # 简单的段落分割
            lines = content.split('\n')
            paragraphs = []
            current_para = []
            for line in lines:
                if line.strip():
                    current_para.append(line)
                else:
                    if current_para:
                        paragraphs.append('\n'.join(current_para))
                        current_para = []
            if current_para:
                paragraphs.append('\n'.join(current_para))

            if paragraphs:
                node_id = doc_id
                nodes.append({
                    "id": node_id,
                    "title": doc_id,
                    "paragraphs": paragraphs,
                    "children": []
                })

        except Exception as e:
            self.logger.error(f"Error reading {sample.doc_path}: {e}")

        return nodes

    def retrieve(self, query: str, topk: int = 5, target_uri: str = None) -> DeepReadResult:
        """
        使用 DeepRead 的 run_agent 执行多轮检索。

        注意：DeepRead 是一个 Agent 系统，它会自行决定调用哪些搜索工具
        和读取哪些章节，最终返回一个答案字符串。我们将其包装为
        与 VikingStoreWrapper.retrieve() 一致的返回格式。
        """
        if self.doc_index is None:
            self.logger.error("DocIndex not initialized. Please run ingest first.")
            return DeepReadResult()

        # 初始化 logger
        self.jsonl_logger = JsonlLogger(self.log_path)

        # 调用 run_agent
        try:
            answer = run_agent(
                model=self.model,
                base_url=self.base_url,
                doc_index=self.doc_index,
                user_question=query,
                logger=self.jsonl_logger,
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
            )
        except Exception as e:
            self.logger.error(f"run_agent failed: {e}")
            answer = f"[Error] {str(e)}"

        # 将答案包装为 DeepReadResult
        resource = DeepReadResource(
            uri="deepread://result",
            content=answer,
            score=1.0,
        )
        return DeepReadResult(resources=[resource])

    def process_retrieval_results(self, search_res: DeepReadResult):
        """
        从检索结果中提取 retrieved_texts / context_blocks / retrieved_uris。
        """
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []

        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            retrieved_texts.append(r.content)
            context_blocks.append(r.content[:2000] if r.content else "")

        return retrieved_texts, context_blocks, retrieved_uris

    def build_uri_map(self, doc_info: List[StandardDoc]) -> Dict[str, list]:
        """构建 sample_id -> [doc_id] 映射"""
        uri_map = {}
        for doc in doc_info:
            doc_id = os.path.basename(doc.doc_path)
            uri_map.setdefault(doc.sample_id, []).append(doc_id)
        return uri_map

    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        # DeepRead 的 run_agent 已经完成了文档读取
        # 这里返回空字符串，因为答案在 retrieve 阶段已经获取
        return ""

    def clear(self):
        """清空索引"""
        self.doc_index = None
        if os.path.exists(self.store_path):
            for fname in os.listdir(self.store_path):
                if fname.endswith(".json"):
                    os.remove(os.path.join(self.store_path, fname))
