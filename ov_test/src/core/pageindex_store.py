
import os
import json
import time
import asyncio
import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm
from langchain_core.messages import HumanMessage

from src.adapters.base import StandardDoc
from src.core.monitor import BenchmarkMonitor
from src.core.logger import get_logger

from pageindex.page_index import page_index
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader, token_tracker
from pageindex.config_utils import get_api_client, get_pageindex_config


@dataclass
class PageIndexResource:
    """PageIndex 检索返回的单个资源"""
    uri: str
    content: str = ""
    level: int = 2
    score: float = 0.0
    abstract: str = ""
    overview: str = ""


@dataclass
class PageIndexResult:
    """PageIndex 检索返回结果，与 OpenViking 的 find 返回格式对齐"""
    resources: List[PageIndexResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0


class PageIndexStoreWrapper:
    """PageIndex 向量存储包装器，接口与 VikingStoreWrapper 对齐"""

    def __init__(self, store_path: str = "./local_index", doc_output_dir: str = "./processed_docs", config_path: str = None):
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)
        self.config_path = config_path
        self.doc_output_dir = doc_output_dir
        self.logger = get_logger()

        # 加载配置并创建API客户端
        self.config = get_pageindex_config(config_path)
        self.llm_client = get_api_client(config_path)
        self.model_name = self.config.get_model_name()
        self.pageindex_config = self.config.get_pageindex_config()

        # 内存缓存
        self.doc_trees: Dict[str, dict] = {}
        self.doc_paths: Dict[str, str] = {}

        self._load_local_trees()

        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def _load_local_trees(self):
        """启动时加载硬盘上已有的树结构文件和原文路径映射"""
        if not os.path.exists(self.store_path):
            return
        if not os.path.exists(self.doc_output_dir):
            return
        paths_file = os.path.join(self.store_path, "_doc_paths.json")
        if os.path.exists(paths_file):
            with open(paths_file, 'r', encoding='utf-8') as f:
                self.doc_paths = json.load(f)
        for filename in os.listdir(self.store_path):
            if filename.endswith("_structure.json"):
                doc_id = filename.replace("_structure.json", "")
                with open(os.path.join(self.store_path, filename), 'r', encoding='utf-8') as f:
                    self.doc_trees[doc_id] = json.load(f)
                source_path = os.path.join(self.doc_output_dir, doc_id)
                if os.path.exists(source_path):
                    self.doc_paths[doc_id] = source_path

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor: Optional[BenchmarkMonitor] = None) -> dict:
        """入库：根据文件类型调用不同的本地解析方法生成树，返回 dict 与 VikingStoreWrapper 对齐"""
        start_time = time.time()
        token_tracker.reset()

        # 展开 doc_paths 并去重（保持顺序）
        seen = set()
        all_paths = []
        for sample in samples:
            for p in sample.doc_paths:
                if p not in seen:
                    seen.add(p)
                    all_paths.append(p)

        for doc_path in tqdm(all_paths, desc="Generating Local Trees"):
            if monitor:
                monitor.worker_start()
            try:
                ext = os.path.splitext(doc_path)[1].lower()
                doc_id = os.path.basename(doc_path)
                tree_data = None

                if ext == '.pdf':
                    opt = self._get_pageindex_opt()
                    tree_data = page_index(doc_path, **vars(opt))
                elif ext in ['.md', '.markdown']:
                    opt = self._get_pageindex_opt()
                    tree_data = asyncio.run(md_to_tree(
                        md_path=doc_path,
                        if_thinning=False,
                        min_token_threshold=5000,
                        if_add_node_summary=opt.if_add_node_summary,
                        summary_token_threshold=200,
                        model=opt.model,
                        if_add_doc_description=opt.if_add_doc_description,
                        if_add_node_text=opt.if_add_node_text,
                        if_add_node_id=opt.if_add_node_id
                    ))
                else:
                    print(f"[Warning] Unsupported file type: {ext}")
                    if monitor:
                        monitor.worker_end(success=False)
                    continue

                if tree_data:
                    self.doc_trees[doc_id] = tree_data
                    self.doc_paths[doc_id] = doc_path
                    output_file = os.path.join(self.store_path, f"{doc_id}_structure.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(tree_data, f, indent=2, ensure_ascii=False)

                if monitor:
                    monitor.worker_end(success=True)
            except Exception as e:
                print(f"[Error] Failed to process {doc_path}: {e}")
                if monitor:
                    monitor.worker_end(success=False)

        # 持久化 doc_paths 映射
        paths_file = os.path.join(self.store_path, "_doc_paths.json")
        with open(paths_file, 'w', encoding='utf-8') as f:
            json.dump(self.doc_paths, f, indent=2, ensure_ascii=False)

        token_usage = token_tracker.get()
        return {
            "time": time.time() - start_time,
            "input_tokens": token_usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"],
        }

    def _get_pageindex_opt(self):
        config_loader = ConfigLoader()
        user_opt = self.pageindex_config.copy()
        user_opt['model'] = self.model_name
        return config_loader.load(user_opt)

    def retrieve(self, query: str, topk: int = 10, target_uri: Optional[str] = None) -> PageIndexResult:
        """两阶段检索：先筛选相关文档并打分，再对 top 文档做节点级搜索。

        Args:
            query: 查询字符串
            topk: 最多返回的文档数量
            target_uri: 限定搜索的 doc_id，为 None 时走文档级筛选
        """
        if not self.doc_trees:
            return PageIndexResult()

        local_in = 0
        local_out = 0

        # --- 阶段 1：确定要搜索的文档 ---
        if target_uri is not None:
            scored_docs = [(target_uri, 1.0)] if target_uri in self.doc_trees else []
        elif len(self.doc_trees) == 1:
            doc_id = next(iter(self.doc_trees))
            scored_docs = [(doc_id, 1.0)]
        else:
            scored_docs, rank_in, rank_out = self._rank_documents(query, topk)
            local_in += rank_in
            local_out += rank_out

        # --- 阶段 2：对筛出的文档做节点级搜索 ---
        resources = []
        for doc_id, doc_score in scored_docs[:topk]:
            tree = self.doc_trees.get(doc_id)
            if tree is None:
                continue
            try:
                content, search_in, search_out = self._search_nodes_in_doc(query, doc_id, tree)
                local_in += search_in
                local_out += search_out
                if content.strip():
                    resources.append(PageIndexResource(
                        uri=doc_id,
                        content=content,
                        score=doc_score,
                    ))
            except Exception as e:
                print(f"[Warning] Retrieval failed for {doc_id}: {e}")

        result = PageIndexResult(resources=resources)
        result.retrieve_input_tokens = local_in
        result.retrieve_output_tokens = local_out

        return result

    def _rank_documents(self, query: str, topk: int) -> tuple:
        """阶段 1：一次 LLM 调用，对所有文档做相关性打分。
        返回 ([(doc_id, score), ...], input_tokens, output_tokens)"""
        doc_profiles = {}
        for doc_id, tree in self.doc_trees.items():
            desc = tree.get('doc_description', '')
            # 用第一层子节点的 title + summary 作为文档画像
            section_summaries = self._get_top_level_summaries(tree.get('structure', []))
            if desc:
                profile = f"{desc}\n{section_summaries}"
            else:
                profile = section_summaries if section_summaries else doc_id
            doc_profiles[doc_id] = profile

        doc_list_str = "\n\n".join(
            f'Document "{doc_id}":\n{profile}' for doc_id, profile in doc_profiles.items()
        )

        rank_prompt = f"""You are given a question and a list of documents with their descriptions.
Score each document's relevance to the question from 0 to 1 (0 = irrelevant, 1 = highly relevant).
Only include documents with score > 0.

Question: {query}

Documents:
{doc_list_str}

Reply in JSON format:
{{
    "results": [
        {{"doc_id": "<document id>", "score": <0-1>, "reason": "<brief reason>"}}
    ]
}}
Directly return the JSON. Do not output anything else."""

        try:
            response = self.llm_client.invoke([HumanMessage(content=rank_prompt)])
            in_t = self.count_tokens(rank_prompt)
            out_t = self.count_tokens(response.content)
            token_tracker.add(in_t, out_t)
            result_json = self._extract_json(response.content)
            results = result_json.get("results", [])
            scored = [(r["doc_id"], float(r.get("score", 0))) for r in results if r.get("score", 0) > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:topk], in_t, out_t
        except Exception as e:
            print(f"[Warning] Document ranking failed: {e}, falling back to all docs")
            return [(did, 0.5) for did in list(self.doc_trees.keys())[:topk]], 0, 0

    def _search_nodes_in_doc(self, query: str, doc_id: str, tree: dict) -> tuple:
        """阶段 2：在单个文档的树结构中搜索相关节点。
        返回 (拼接正文, input_tokens, output_tokens)"""
        tree_without_text = self._remove_text_field(tree)

        search_prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}
Document tree structure:
{json.dumps(tree_without_text, indent=2, ensure_ascii=False)}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process>",
    "node_list": [
        {{
            "thinking": "<Your thinking process>",
            "node_list": ["node_id_1", "node_id_2"]
        }}
    ]
}}
Directly return the final JSON structure. Do not output anything else."""

        response = self.llm_client.invoke([HumanMessage(content=search_prompt)])
        in_t = self.count_tokens(search_prompt)
        out_t = self.count_tokens(response.content)
        token_tracker.add(in_t, out_t)
        result_json = self._extract_json(response.content)

        raw_node_list = result_json.get("node_list", [])
        target_node_ids: List[str] = []
        for entry in raw_node_list:
            if isinstance(entry, dict):
                target_node_ids.extend(entry.get("node_list", []))
            elif isinstance(entry, str):
                target_node_ids.append(entry)

        node_map = self._create_node_mapping(tree['structure'])
        doc_path = self.doc_paths.get(doc_id, "")
        content = self._resolve_nodes_content(target_node_ids, node_map, doc_path)
        return content, in_t, out_t

    def process_retrieval_results(self, search_res: PageIndexResult):
        """
        从检索结果中提取 retrieved_texts / context_blocks / retrieved_uris。
        PageIndex 在 retrieve 阶段已经提取了节点正文，直接使用 resource.content。
        """
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []
        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            retrieved_texts.append(r.content)
            context_blocks.append(r.content[:2000])
        return retrieved_texts, context_blocks, retrieved_uris

    def read_resource(self, uri: str) -> str:
        """读取资源内容"""
        if uri not in self.doc_trees:
            return ""
        doc_path = self.doc_paths.get(uri, "")
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        node_map = self._create_node_mapping(self.doc_trees[uri].get('structure', []))
        parts = []
        for node in node_map.values():
            s = node.get('summary', node.get('prefix_summary', ''))
            if s:
                parts.append(s)
        return "\n\n".join(parts)

    def clear(self) -> None:
        """清空库：清除内存缓存 + 删除所有 JSON 文件"""
        self.doc_trees.clear()
        self.doc_paths.clear()
        if os.path.exists(self.store_path):
            for filename in os.listdir(self.store_path):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.store_path, filename))

    # --- 辅助方法 ---

    def _get_top_level_summaries(self, structure, max_child_summary_len: int = 100) -> str:
        """提取前两层子节点的 title + summary，第二层 summary 截断控制 token 开销"""
        lines = []
        nodes = structure if isinstance(structure, list) else [structure]
        for node in nodes:
            title = node.get('title', '')
            summary = node.get('summary', node.get('prefix_summary', ''))
            if title and summary:
                lines.append(f"- {title}: {summary}")
            elif title:
                lines.append(f"- {title}")
            # 第二层子节点，summary 截断
            for child in node.get('nodes', []):
                c_title = child.get('title', '')
                c_summary = child.get('summary', child.get('prefix_summary', ''))
                if c_summary and len(c_summary) > max_child_summary_len:
                    c_summary = c_summary[:max_child_summary_len] + "..."
                if c_title and c_summary:
                    lines.append(f"  - {c_title}: {c_summary}")
                elif c_title:
                    lines.append(f"  - {c_title}")
        return "\n".join(lines)

    def _resolve_nodes_content(self, target_node_ids: List[str], node_map: Dict[str, dict], doc_path: str) -> str:
        source_lines: List[str] = []
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

        sorted_nodes = sorted(node_map.values(), key=lambda n: n.get('line_num', 0))
        sorted_idx_by_id: Dict[str, int] = {}
        for idx, n in enumerate(sorted_nodes):
            sorted_idx_by_id[n.get('node_id', '')] = idx

        parts: List[str] = []
        for nid in target_node_ids:
            node = node_map.get(nid)
            if node is None:
                continue
            has_children = bool(node.get('nodes'))
            if has_children:
                summaries = self._collect_children_summaries(node)
                if summaries:
                    parts.append(f"[{node.get('title', '')}]\n" + "\n".join(summaries))
            else:
                if not source_lines:
                    summary = node.get('summary', node.get('prefix_summary', ''))
                    if summary:
                        parts.append(f"[{node.get('title', '')}]\n{summary}")
                    continue
                line_start = node.get('line_num', 1)
                si = sorted_idx_by_id.get(nid, -1)
                if si >= 0 and si + 1 < len(sorted_nodes):
                    line_end = sorted_nodes[si + 1].get('line_num', len(source_lines) + 1)
                else:
                    line_end = len(source_lines) + 1
                text = "".join(source_lines[line_start - 1 : line_end - 1]).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def _collect_children_summaries(self, node: dict) -> List[str]:
        results: List[str] = []
        for child in node.get('nodes', []):
            s = child.get('summary', child.get('prefix_summary', ''))
            if s:
                results.append(s)
            results.extend(self._collect_children_summaries(child))
        return results

    def _remove_text_field(self, tree_data):
        data = copy.deepcopy(tree_data)
        def _recursive_remove(node):
            if isinstance(node, list):
                for item in node:
                    _recursive_remove(item)
            elif isinstance(node, dict):
                node.pop('text', None)
                if 'nodes' in node:
                    _recursive_remove(node['nodes'])
        _recursive_remove(data)
        return data

    def _create_node_mapping(self, tree_data, mapping=None) -> dict:
        if mapping is None:
            mapping = {}
        if isinstance(tree_data, list):
            for item in tree_data:
                self._create_node_mapping(item, mapping)
        elif isinstance(tree_data, dict):
            if 'node_id' in tree_data:
                mapping[tree_data['node_id']] = tree_data
            if 'nodes' in tree_data:
                self._create_node_mapping(tree_data['nodes'], mapping)
        return mapping

    def _extract_json(self, content):
        try:
            start_idx = content.find("```json")
            if start_idx != -1:
                start_idx += 7
                end_idx = content.rfind("```")
                json_content = content[start_idx:end_idx].strip()
            else:
                json_content = content.strip()
            json_content = json_content.replace('None', 'null')
            return json.loads(json_content)
        except Exception as e:
            print(f"[Warning] Failed to extract JSON: {e}")
            return {"node_list": []}
