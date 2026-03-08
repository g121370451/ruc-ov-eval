
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

import unicodedata

def sanitize_filename(name: str, max_length: int = 150) -> str:
    name = unicodedata.normalize("NFKD", name)

    name = re.sub(r'[\\/*?:"<>|]', "", name)

    name = re.sub(r'[\x00-\x1f\x7f]', "", name)

    name = name.strip(" .")

    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

    if name.upper() in reserved_names:
        name = f"{name}_file"

    if len(name) > max_length:
        name = name[:max_length].rstrip()

    if not name:
        name = "untitled"

    return name

QA_PROMPT = """Based on the above context, write an answer to the following question. 
Use information from the context to answer. If no information is available, write 'Not mentioned'.

Question: {} 
Answer:
"""

def convert_to_md(raw_text: str) -> str:
    """
    将：
    navigation            -&gt;  ## navigation\n
    Contents ( hide )     -&gt;  ## Contents ( hide )\n
    """

    # 1. 压缩多余空白为单空格
    text = re.sub(r"\s+", " ", raw_text).strip()
    
    text = re.sub(r"Contents\s*\(\s*hide\s*\)", "## Contents ( hide )\n", text)

    # 3. 让标题前强制换行（避免粘在正文后）
    text = re.sub(r"\s*## ", r"\n\n## ", text)

    # 4. 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip() + "\n"

class ClapNQAdapter(BaseAdapter):
    """
    专门用于处理 ClapNQ 数据集的适配器。
    """
    
    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        self.logger.info(f"[ClapNQAdapter initialized")

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        加载original_document目录下dev和train目录下的answerable_orig.jsonl文件
        并转换为Markdown格式
        """
        doc_files = []
        
        # 从标注数据路径推导原始文档目录
        orig_dir = os.path.join(self.raw_file_path, "original_documents")
        
        if not os.path.exists(orig_dir):
            raise FileNotFoundError(f"Original documents directory not found: {orig_dir}")
        
        # 查找dev和train目录下的answerable_orig.jsonl文件
        for split in ['dev']:
            split_dir = os.path.join(orig_dir, split)
            if os.path.exists(split_dir):
                for filename in os.listdir(split_dir):
                    if filename.endswith('answerable_orig.jsonl') and not filename.endswith('unanswerable_orig.jsonl'):
                        doc_files.append(os.path.join(split_dir, filename))
        
        if not doc_files:
            raise FileNotFoundError(f"No answerable_orig.jsonl files found in {orig_dir}/dev and {orig_dir}/train")
        
        self.logger.info(f"Found {len(doc_files)} answerable_orig.jsonl files")

        res: List[StandardDoc] = []
        os.makedirs(doc_dir, exist_ok=True)

        for doc_file in doc_files:
            self.logger.info(f"Processing: {doc_file}")
            with open(doc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    example_id = item.get("example_id", "unknown")
                    
                    # 提取文档纯文本并转换为Markdown
                    document_plaintext = item.get("document_plaintext", "")
                    document_title = item.get("document_title", "")
                    doc_content = convert_to_md(document_plaintext)
                    
                    # 添加文档标题到开头
                    final_content = f"# {document_title}\n\n{doc_content}"
                    
                    try:
                        # 使用document_title作为文件名（清理非法字符）
                        # safe_title = document_title
                        # safe_title = safe_title[:100]  # 限制文件名长度
                        # doc_filename = f"{safe_title}_{example_id}.md"
                        doc_filename = f"{document_title}.md"
                        doc_filename = sanitize_filename(doc_filename)
                        doc_path = os.path.join(doc_dir, doc_filename)
                        self.logger.info(f"doc_path is {doc_path}")
                        with open(doc_path, "w", encoding="utf-8") as f_out:
                            f_out.write(final_content)
                        res.append(StandardDoc(example_id, doc_path))
                    except Exception as e:
                        self.logger.error(f"[clapnq adapter] doc:{example_id} prepare error {e}")
                        raise e

        self.logger.info(f"Total {len(res)} documents prepared")
        return res
        # return [StandardDoc("123123",doc_dir)]

    def load_and_transform(self) -> List[StandardSample]:
        """
        加载原始 JSONL 数据并转换为标准化的 StandardSample 对象列表。
        """
        data_files = []
        if os.path.isdir(self.raw_file_path):
            # 如果是目录，查找所有.jsonl文件
            for root, _, files in os.walk(self.raw_file_path):
                for filename in files:
                    if root.endswith('dev') and filename.endswith('answerable.jsonl') and not filename.endswith('unanswerable.jsonl') and not filename.endswith('_orig.jsonl'):
                        data_files.append(os.path.join(root, filename))
        else:
            data_files.append(self.raw_file_path)
        
        if not data_files:
            raise FileNotFoundError(f"No annotated data files found.")
        
        self.logger.info(f"Found {len(data_files)} annotated data files")

        standard_samples = []
        processed_ids = set()

        for data_file in data_files:
            self.logger.info(f"Processing annotated data file: {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    sample_id = item.get("id", "unknown")
                    
                    # 避免重复处理
                    if sample_id in processed_ids:
                        continue
                    processed_ids.add(sample_id)
                    
                    qa_pairs = []
                    question = item.get("input", "")
                    
                    gold_answers = []
                    evidence = []
                    
                    outputs = item.get("output", [])
                    for output in outputs:
                        answer = output.get("answer", "")
                        if answer:
                            gold_answers.append(answer)
                        selected_sentences = output.get("selected_sentences", [])
                        evidence.extend(selected_sentences)
                    
                    qa_pairs.append(StandardQA(
                        question=question,
                        gold_answers=gold_answers if gold_answers else ["Not mentioned"],
                        evidence=evidence,
                        category=None,
                        metadata={"original_id": sample_id}
                    ))

                    standard_samples.append(StandardSample(
                        sample_id=sample_id,
                        qa_pairs=qa_pairs
                    ))

        self.logger.info(f"Total {len(standard_samples)} samples loaded")
        return standard_samples

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{QA_PROMPT.format(qa.question)}"
        meta = {}
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        return raw_answer.strip()

