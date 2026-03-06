
import os
import json
import time
import asyncio
import copy
import re
from typing import Dict, List, Optional
from tqdm import tqdm
from langchain_core.messages import HumanMessage

from bench_framework.adapters.base import StandardDoc
from bench_framework.stores.base import VectorStoreBase
from bench_framework.core.monitor import BenchmarkMonitor
from bench_framework.core.logger import get_logger
from bench_framework.types import IngestStats, SearchResult, SearchResource

from bench_framework.pageindex.page_index import page_index
from bench_framework.pageindex.page_index_md import md_to_tree
from bench_framework.pageindex.utils import ConfigLoader, token_tracker
from bench_framework.pageindex.config_utils import get_api_client, get_pageindex_config

class PageIndexStoreWrapper(VectorStoreBase):
    """PageIndex向量存储包装器，继承自VectorStoreBase基类"""
    
    def __init__(self, store_path: str = "./local_index", doc_output_dir:str = "./processed_docs",config_path: str = None):
        """
        初始化PageIndex Wrapper
        
        Args:
            store_path: 存储路径
            config_path: pageindex.conf配置文件路径（可选）
        """
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
        
        # 内存缓存：用于存储每个文档的完整树结构
        self.doc_trees: Dict[str, dict] = {}
        # 原文路径映射：doc_id -> 原始文件路径
        self.doc_paths: Dict[str, str] = {}

        # 启动时加载本地已有的索引文件
        self._load_local_trees()
        
        # 初始化tokenizer
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def _load_local_trees(self):
        """启动时加载硬盘上已有的树结构文件和原文路径映射"""
        if not os.path.exists(self.store_path):
            self.logger.error("store_path not found, skip loading local trees")
            return
        if not os.path.exists(self.doc_output_dir):
            self.logger.error("doc_output_dir not found, skip loading local trees")
            return
        # 加载 doc_paths 映射
        paths_file = os.path.join(self.store_path, "_doc_paths.json")
        if os.path.exists(paths_file):
            with open(paths_file, 'r', encoding='utf-8') as f:
                self.doc_paths = json.load(f)
        # 加载树结构（从 store_path 读 JSON）+ 原文文本（从 doc_output_dir 读 md）
        for filename in os.listdir(self.store_path):
            if filename.endswith("_structure.json"):
                # 树结构 JSON
                doc_id = filename.replace("_structure.json", "")
                with open(os.path.join(self.store_path, filename), 'r', encoding='utf-8') as f:
                    self.doc_trees[doc_id] = json.load(f)
                # 原文文本：doc_id 本身就是文件名（如 xxx_doc.md）
                source_path = os.path.join(self.doc_output_dir, doc_id)
                if os.path.exists(source_path):
                    self.doc_paths[doc_id] = source_path

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor: Optional[BenchmarkMonitor] = None) -> IngestStats:
        """
        本地 Ingest：根据文件类型调用不同的本地解析方法生成树
        
        Args:
            samples: 标准文档列表
            max_workers: 最大并发数
            monitor: 监控器对象
            
        Returns:
            包含时间和token统计信息的字典
        """
        start_time = time.time()
        token_tracker.reset()

        for sample in tqdm(samples, desc="Generating Local Trees"):
            if monitor: 
                monitor.worker_start()
            
            try:
                ext = os.path.splitext(sample.doc_path)[1].lower()
                doc_id = os.path.basename(sample.doc_path)
                tree_data = None

                if ext == '.pdf':
                    # 使用pageindex.conf中的配置
                    opt = self._get_pageindex_opt()
                    tree_data = page_index(sample.doc_path, **vars(opt))

                elif ext in ['.md', '.markdown']:
                    # 配置 Markdown 解析参数
                    opt = self._get_pageindex_opt()
                    # md_to_tree 是异步的，需要用 asyncio 运行
                    tree_data = asyncio.run(md_to_tree(
                        md_path=sample.doc_path,
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
                    # 保存到内存
                    self.doc_trees[doc_id] = tree_data
                    self.doc_paths[doc_id] = sample.doc_path

                    # 保存到本地硬盘
                    output_file = os.path.join(self.store_path, f"{doc_id}_structure.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(tree_data, f, indent=2, ensure_ascii=False)

                if monitor: 
                    monitor.worker_end(success=True)

            except Exception as e:
                print(f"[Error] Failed to process {sample.doc_path}: {e}")
                if monitor: 
                    monitor.worker_end(success=False)

        # 持久化 doc_paths 映射
        paths_file = os.path.join(self.store_path, "_doc_paths.json")
        with open(paths_file, 'w', encoding='utf-8') as f:
            json.dump(self.doc_paths, f, indent=2, ensure_ascii=False)

        token_usage = token_tracker.get()
        return IngestStats(
            time=time.time() - start_time,
            input_tokens=token_usage["input_tokens"],
            output_tokens=token_usage["output_tokens"],
        )
    
    def _get_pageindex_opt(self):
        """获取pageindex配置对象"""
        config_loader = ConfigLoader()
        user_opt = self.pageindex_config.copy()
        # 确保model参数被正确设置
        user_opt['model'] = self.model_name
        return config_loader.load(user_opt)

    def retrieve(self, query: str, topk: int = 10, target_uri: Optional[str] = None) -> SearchResult:
        """
        本地检索与推理
        
        Args:
            query: 查询字符串
            topk: 返回的最相关文档数量（在PageIndex中未使用，保留接口兼容性）
            target_uri: 目标URI（在PageIndex中未使用，保留接口兼容性）
            
        Returns:
            检索结果对象，模拟OpenViking的返回格式
        """
        if not self.doc_trees:
            return self._create_empty_search_result()

        all_relevant_content = []
        retrieved_uris = []

        for doc_id, tree in self.doc_trees.items():
            # 1. 剔除原文内容，只留目录给模型看
            tree_without_text = self._remove_text_field(tree)

            search_prompt = f"""
            You are given a question and a tree structure of a document.
            Each node contains a node id, node title, and a corresponding summary.
            Your task is to find all nodes that are likely to contain the answer to the question.

            Question: {query}
            Document tree structure:
            {json.dumps(tree_without_text, indent=2, ensure_ascii=False)}

            Please reply in the following JSON format:
            {{
                "thinking": "&lt;Your thinking process&gt;",
                "doc_name": ["related"]
                "node_list": [
                    {{
                        "thinking": "&lt;Your thinking process&gt;",
                        "node_list": ["node_id_1", "node_id_2"]
                    }}
                ]
            }}
            Directly return the final JSON structure. Do not output anything else.
            """

            try:
                # 2. 调用大模型找节点
                response = self.llm_client.invoke([HumanMessage(content=search_prompt)])

                result_json = self._extract_json(response.content)
                # node_list 是 [{"thinking": "...", "node_list": ["id1", "id2"]}, ...]
                # 展开所有内层 node_list 得到扁平的 node_id 列表
                raw_node_list = result_json.get("node_list", [])
                target_node_ids: List[str] = []
                for entry in raw_node_list:
                    if isinstance(entry, dict):
                        target_node_ids.extend(entry.get("node_list", []))
                    elif isinstance(entry, str):
                        # 兼容模型直接返回扁平列表的情况
                        target_node_ids.append(entry)

                # 3. 提取找出的节点的正文文本
                node_map = self._create_node_mapping(tree['structure'])
                doc_path = self.doc_paths.get(doc_id, "")
                doc_content = self._resolve_nodes_content(
                    target_node_ids, node_map, doc_path,
                )

                if doc_content.strip():
                    all_relevant_content.append(f"--- Document: {doc_id} ---\n{doc_content}")
                    retrieved_uris.append(doc_content)

            except Exception as e:
                print(f"[Warning] Retrieval failed for {doc_id}: {e}")

        # 4. 生成最终回答
        final_context = "\n\n".join(all_relevant_content)

        # 返回模拟OpenViking格式的结果
        return self._create_search_result(final_context, retrieved_uris)

    def read_resource(self, uri: str) -> str:
        """
        读取资源内容。

        优先从原文文件读取全文，退化到拼接节点 summary。
        """
        if uri not in self.doc_trees:
            return ""
        # 优先读原文
        doc_path = self.doc_paths.get(uri, "")
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        # 退化：拼接所有节点 summary
        node_map = self._create_node_mapping(self.doc_trees[uri].get('structure', []))
        parts = []
        for node in node_map.values():
            s = node.get('summary', node.get('prefix_summary', ''))
            if s:
                parts.append(s)
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.doc_trees.clear()
        if os.path.exists(self.store_path):
            for filename in os.listdir(self.store_path):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.store_path, filename))

    # --- 以下是操作本地 JSON 树的辅助方法 ---

    def _resolve_nodes_content(
        self,
        target_node_ids: List[str],
        node_map: Dict[str, dict],
        doc_path: str,
    ) -> str:
        """
        根据目标 node_id 列表提取内容。

        - 有子节点的 node（存在 nodes 属性）：汇总所有子节点的 summary
        - 叶子节点（无 nodes 属性）：用 line_num 去原文按行范围截取

        行范围规则：
        - line_start = 当前 node 的 line_num
        - line_end   = 按 line_num 排序后下一个 node 的 line_num（不含），
                        若为最后一个 node 则取到文件末尾
        """
        # 读取原文行（懒加载，仅在需要时读取）
        source_lines: List[str] = []
        if doc_path and os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

        # 按 line_num 排序所有节点，用于确定行范围边界
        sorted_nodes = sorted(
            node_map.values(),
            key=lambda n: n.get('line_num', 0),
        )
        # 建立 node_id -> 在排序列表中的索引
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
                # 父节点：汇总所有子节点的 summary / prefix_summary
                summaries = self._collect_children_summaries(node)
                if summaries:
                    parts.append(f"[{node.get('title', '')}]\n" + "\n".join(summaries))
            else:
                # 叶子节点：按行范围从原文截取
                if not source_lines:
                    # 没有原文，退化到 summary
                    summary = node.get('summary', node.get('prefix_summary', ''))
                    if summary:
                        parts.append(f"[{node.get('title', '')}]\n{summary}")
                    continue

                line_start = node.get('line_num', 1)
                # 找下一个 node 的 line_num 作为截止
                si = sorted_idx_by_id.get(nid, -1)
                if si >= 0 and si + 1 < len(sorted_nodes):
                    line_end = sorted_nodes[si + 1].get('line_num', len(source_lines) + 1)
                else:
                    line_end = len(source_lines) + 1

                # line_num 是 1-based，转为 0-based 切片 [start-1, end-1)
                text = "".join(source_lines[line_start - 1 : line_end - 1]).strip()
                if text:
                    parts.append(text)

        return "\n\n".join(parts)

    def _collect_children_summaries(self, node: dict) -> List[str]:
        """递归收集一个父节点下所有子节点的 summary"""
        results: List[str] = []
        for child in node.get('nodes', []):
            s = child.get('summary', child.get('prefix_summary', ''))
            if s:
                results.append(s)
            # 递归子节点
            results.extend(self._collect_children_summaries(child))
        return results

    def _remove_text_field(self, tree_data):
        """递归剔除 `text` 字段以压缩 Prompt"""
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
        """递归建立 node_id -> node_data 的字典映射"""
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
        """从LLM响应中提取JSON"""
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

    def _create_empty_search_result(self) -> SearchResult:
        """创建空的搜索结果"""
        return SearchResult(resources=[])

    def _create_search_result(self, content: str, uris: List[str]) -> SearchResult:
        """创建搜索结果"""
        resources = [
            SearchResource(uri=uri, level=2, abstract="", overview="")
            for uri in uris
        ]
        return SearchResult(resources=resources)

