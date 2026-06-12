
# src/adapters/clapnq_adapter.py
"""
ClapNQ Dataset Adapter

ClapNQ is a QA dataset with Wikipedia articles as documents.
"""

import json
import os
import re
import unicodedata
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from base import BaseAdapter, StandardDoc, StandardSample, StandardQA

MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."


def sanitize_filename(name, max_length=150):
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'[\x00-\x1f\x7f]', "", name)
    name = name.strip(" .")
    
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    
    if name.upper() in reserved_names:
        name = name + "_file"
    
    if len(name) > max_length:
        name = name[:max_length].rstrip()
    
    if not name:
        name = "untitled"
    
    return name


def convert_to_md(raw_text):
    """
    将ClapNQ的原始文本转换为Markdown格式。
    """
    text = raw_text
    
    # 移除开头的Wikipedia导航信息："标题 - wikipedia 标题 Jump to : navigation , search"
    text = re.sub(r'^.*?Jump to\s*:\s*navigation\s*,\s*search', '', text)
    
    # 修复多余的空格（单词之间只有一个空格）
    text = re.sub(r'\s+', ' ', text)
    
    # 移除"( edit )"编辑标记
    text = re.sub(r'\(\s*edit\s*\)', '', text)
    
    # 处理Contents部分
    text = re.sub(r'Contents\s*\(\s*hide\s*\)', '\n\n## Contents\n\n', text)
    
    # 在句号后且后面跟着大写字母的地方创建段落
    text = re.sub(r'(\.)\s+([A-Z])', r'\1\n\n\2', text)
    
    # 处理冒号后创建列表项的情况
    text = re.sub(r'(:)\s+(\d)', r'\1\n\n\2', text)
    
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip() + '\n'


class ClapNQAdapter(BaseAdapter):
    """
    ClapNQ Dataset Adapter.
    Processes QA data with Wikipedia articles as documents.
    """

    def __init__(self, raw_file_path):
        super().__init__(raw_file_path)
        data_dir = os.path.dirname(self.raw_file_path)
        self.doc_file_path = os.path.join(data_dir, "clapnq_dev_answerable_orig.jsonl")

    def data_prepare(self, doc_dir):
        if not os.path.exists(self.doc_file_path):
            raise FileNotFoundError(f"Document file not found: {self.doc_file_path}")

        res = []
        os.makedirs(doc_dir, exist_ok=True)

        with open(self.doc_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                document_plaintext = item.get("document_plaintext", "")
                document_title = item.get("document_title", "")
                doc_content = convert_to_md(document_plaintext)
                
                final_content = "# " + document_title + "\n\n" + doc_content
                
                try:
                    doc_filename = document_title + ".md"
                    doc_filename = sanitize_filename(doc_filename)
                    doc_path = os.path.join(doc_dir, doc_filename)
                    with open(doc_path, "w", encoding="utf-8") as f_out:
                        f_out.write(final_content)
                    res.append(StandardDoc(sample_id=document_title, doc_path=doc_path))
                except Exception as e:
                    self.logger.error(f"[clapnq adapter] doc prepare error {e}")
                    raise e

        self.logger.info(f"Total {len(res)} documents prepared")
        return res

    def load_and_transform(self):
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        standard_samples = []

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                qa_id = item.get("id", "")
                question = item.get("input", "")
                
                gold_answers = []
                evidence = []
                
                # 从passages中提取evidence（如果有）
                passages = item.get("passages", [])
                for passage in passages:
                    sentences = passage.get("sentences", [])
                    evidence.extend(sentences)
                
                # 从output中提取答案（如果有）
                outputs = item.get("output", [])
                for output in outputs:
                    answer = output.get("answer", "")
                    if answer:
                        gold_answers.append(answer)
                    selected_sentences = output.get("selected_sentences", [])
                    evidence.extend(selected_sentences)
                
                # 如果没有从output中找到答案，尝试从其他字段提取
                if not gold_answers:
                    answers_text = item.get("answers", "")
                    if answers_text:
                        if isinstance(answers_text, str):
                            answers = answers_text.split('::')
                            gold_answers = [ans.strip() for ans in answers if ans.strip()]
                        elif isinstance(answers_text, list):
                            gold_answers = [ans.strip() for ans in answers_text if ans.strip()]
                
                qa_pairs = [StandardQA(
                    question=question,
                    gold_answers=gold_answers if gold_answers else ["Not mentioned"],
                    evidence=evidence,
                    category=None,
                    metadata={
                        "id": qa_id,
                        "passages": passages
                    }
                )]
                
                sample_id = qa_id
                standard_samples.append(StandardSample(
                    sample_id=sample_id,
                    qa_pairs=qa_pairs
                ))

        self.logger.info(f"Total {len(standard_samples)} samples loaded")
        return standard_samples

    def build_prompt(self, qa, context_blocks):
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"""{context_text}

{MISSING_RULE}

Interpret “who” and “what” broadly as context-dependent descriptions, not necessarily as a specific person or object.
If a question includes constraints and the context provides the relevant facts, answer within that scope even if the constraint is not repeated explicitly.

Question: {qa.question}

Answer:"""
        meta = {
            "id": qa.metadata.get("id", ""),
        }
        return full_prompt, meta
