import os
import json
import time
import asyncio
import copy
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 引入你的基础接口定义
from src.adapters.base import StandardDoc

from src.pageindex.page_index import *
from src.pageindex.page_index_md import md_to_tree
from src.pageindex.utils import ConfigLoader

load_dotenv()

class LocalPageIndexWrapper:
    def __init__(self, store_path: str = "./local_index", model_name: str = "gpt-4o-mini"):
        """
        初始化本地 PageIndex Wrapper
        """
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)
        self.model_name = model_name
        
        # 初始化大模型客户端 (默认使用环境变量中的 OPENAI_API_KEY)
        # 如果你使用本地 Ollama，可以在 .env 中配置 OPENAI_BASE_URL
        self.llm_client = OpenAI() 
        
        # 内存缓存：用于存储每个文档的完整树结构
        self.doc_trees: Dict[str, Any] = {}
        
        # 启动时加载本地已有的索引文件
        self._load_local_trees()

    def _load_local_trees(self):
        """启动时加载硬盘上已有的树结构文件"""
        for filename in os.listdir(self.store_path):
            if filename.endswith("_structure.json"):
                doc_id = filename.replace("_structure.json", "")
                with open(os.path.join(self.store_path, filename), 'r', encoding='utf-8') as f:
                    self.doc_trees[doc_id] = json.load(f)

    def ingest(self, samples: List[StandardDoc], max_workers: int = 4, monitor=None) -> dict:
        """
        本地 Ingest：根据文件类型调用不同的本地解析方法生成树。
        """
        start_time = time.time()

        for sample in tqdm(samples, desc="Generating Local Trees"):
            if monitor: monitor.worker_start()
            
            try:
                ext = os.path.splitext(sample.doc_path)[1].lower()
                doc_id = os.path.basename(sample.doc_path)
                tree_data = None

                if ext == '.pdf':
                    # 配置 PDF 解析参数 (强制加上 text 和 summary)
                    opt = config(
                        model=self.model_name,
                        toc_check_page_num=20,
                        max_page_num_each_node=10,
                        max_token_num_each_node=20000,
                        if_add_node_id='yes',
                        if_add_node_summary='yes',
                        if_add_doc_description='no',
                        if_add_node_text='yes' # RAG 必须提取原文
                    )
                    tree_data = page_index_main(sample.doc_path, opt)

                elif ext in ['.md', '.markdown']:
                    # 配置 Markdown 解析参数
                    config_loader = ConfigLoader()
                    user_opt = {
                        'model': self.model_name,
                        'if_add_node_summary': 'yes',
                        'if_add_doc_description': 'no',
                        'if_add_node_text': 'yes', # RAG 必须提取原文
                        'if_add_node_id': 'yes'
                    }
                    opt = config_loader.load(user_opt)
                    opt.model = "doubao-seed-2-0-pro-260215"
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
                    if monitor: monitor.worker_end(success=False)
                    continue

                if tree_data:
                    # 保存到内存
                    self.doc_trees[doc_id] = tree_data
                    
                    # 保存到本地硬盘
                    output_file = os.path.join(self.store_path, f"{doc_id}_structure.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(tree_data, f, indent=2, ensure_ascii=False)

                if monitor: monitor.worker_end(success=True)

            except Exception as e:
                print(f"[Error] Failed to process {sample.doc_path}: {e}")
                if monitor: monitor.worker_end(success=False)

        return {
            "time": time.time() - start_time,
            "input_tokens": 0,
            "output_tokens": 0
        }

    def retrieve(self, query: str, topk: int = None) -> str:
        """
        本地检索与推理
        """
        if not self.doc_trees:
            return "No documents indexed. Please ingest documents first."

        all_relevant_content = []

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
                "thinking": "<Your thinking process>",
                "node_list": ["node_id_1", "node_id_2"]
            }}
            Directly return the final JSON structure. Do not output anything else.
            """

            try:
                # 2. 调用大模型找节点
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0,
                    response_format={ "type": "json_object" }
                )
                
                result_json = json.loads(response.choices[0].message.content.strip())
                target_node_ids = result_json.get("node_list", [])

                # 3. 提取找出的节点的正文文本
                node_map = self._create_node_mapping(tree)
                doc_content = "\n\n".join(
                    node_map[nid].get("text", "") 
                    for nid in target_node_ids if nid in node_map
                )
                
                if doc_content.strip():
                    all_relevant_content.append(f"--- Document: {doc_id} ---\n{doc_content}")

            except Exception as e:
                print(f"[Warning] Retrieval failed for {doc_id}: {e}")

        # 4. 生成最终回答
        final_context = "\n\n".join(all_relevant_content)
        if not final_context.strip():
            return "No relevant information found in the local index."

        answer_prompt = f"""
        Answer the question based strictly on the provided context:
        Question: {query}
        
        Context: 
        {final_context}
        """
        
        final_response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.2
        )
        
        return final_response.choices[0].message.content.strip()

    # --- 以下是操作本地 JSON 树的辅助方法 ---

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

    def clear(self):
        """清空内存并删除本地文件"""
        self.doc_trees.clear()
        for filename in os.listdir(self.store_path):
            if filename.endswith(".json"):
                os.remove(os.path.join(self.store_path, filename))