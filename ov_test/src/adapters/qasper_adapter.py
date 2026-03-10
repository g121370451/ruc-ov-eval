# src/adapters/qasper_adapter.py
"""
Qasper 数据集适配器

Qasper 是一个学术论文问答数据集，包含 1585 篇 NLP 论文和 5049 个问题。
每个问题由多个标注者回答，答案类型包括：
- extractive_spans: 从论文中提取的文本片段
- free_form_answer: 自由形式的答案
- yes_no: 是/否答案
- unanswerable: 无法从论文中找到答案

数据集特点：
1. 每篇论文包含标题、摘要、章节内容和图表信息
2. 每个问题可能有多个标注者的答案
3. 每个答案有对应的 evidence（证据文本）

适配器功能：
- data_prepare: 将论文转换为 Markdown 格式，保留章节结构
- load_and_transform: 解析 QA 数据，保留答案与证据的对应关系
- build_prompt: 构建问答提示词
- post_process_answer: 后处理 LLM 输出
"""

import json
import os
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

# 问答提示词模板
QA_PROMPT = """You are a professional academic research assistant. Your task is to answer questions based on the provided research paper snippets.

### INSTRUCTIONS:
1. **Source Grounding**: Answer the question using ONLY the provided context. 
2. **Conciseness**: Provide the answer as a short phrase, entity, or specific value. Avoid full sentences unless absolutely necessary.
3. **Yes/No Questions**: If the question is a Yes/No question, respond with ONLY "Yes" or "No".
4. **Lists**: If the answer involves multiple items, separate them with a comma.
5. **Exact Extraction**: Use exact terminology from the text whenever possible.

### ABSENCE RULE:
{missing_rule}

---
### CONTEXT (Excerpts from the paper):
{context_text}

---
### QUESTION:
{question}

---
### ANSWER:
"""

# 无法回答时的规则说明
MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."


class QasperAdapter(BaseAdapter):
    """
    专门用于处理 Qasper 数据集的适配器。
    
    将学术论文转换为带有章节结构的 Markdown 文档，
    并将 QA 数据转换为标准化的 StandardSample 格式。
    
    Attributes:
        raw_file_path: 原始 JSON 数据文件路径
        logger: 日志记录器
    """
    
    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        加载原始数据并转换为 OpenViking 友好格式。
        
        将每篇论文转换为 Markdown 文档，保留以下结构：
        - 标题（# Title）
        - 摘要（## Abstract）
        - 章节（## Section Name）
        - 图表（## Figures and Tables）
        
        Args:
            doc_dir: 文档输出目录路径
            
        Returns:
            List[StandardDoc]: 标准化文档对象列表，每个包含 paper_id 和文档路径
            
        Raises:
            FileNotFoundError: 原始数据文件不存在
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        res: List[StandardDoc] = []

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        os.makedirs(doc_dir, exist_ok=True)
        
        for paper_id, paper_data in data.items():
            doc_content = self._convert_paper_to_markdown(paper_id, paper_data)

            try:
                doc_path = os.path.join(doc_dir, f"{paper_id}_doc.md")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                res.append(StandardDoc(paper_id, [doc_path]))
            except Exception as e:
                self.logger.error(f"[qasper adapter] doc:{paper_id} prepare error {e}")
                raise e
        return res

    def load_and_transform(self) -> List[StandardSample]:
        """
        加载原始 JSON 数据并转换为标准化的 StandardSample 对象列表。
        
        处理逻辑：
        1. 遍历每篇论文的 QA 列表
        2. 对每个问题，收集所有标注者的答案
        3. 保留答案与证据的对应关系（存储在 metadata 中）
        4. 问题格式化为 "Based on the paper "{title}", {question}"
        
        答案类型处理：
        - extractive_spans: 直接使用提取的文本片段
        - free_form_answer: 使用自由形式答案
        - yes_no: 转换为 "Yes" 或 "No"
        - unanswerable: 转换为 "Not mentioned"
        
        Returns:
            List[StandardSample]: 标准化样本对象列表
            
        Raises:
            FileNotFoundError: 原始数据文件不存在
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        standard_samples = []

        for paper_id, paper_data in data.items():
            qa_pairs = []
            paper_title = paper_data.get("title", "Unknown Title")
            
            for qa_item in paper_data.get("qas", []):
    
                # --- unanswerable过滤逻辑 ---
                # 检查是否所有答案都被标记为无法回答
                is_unanswerable = all(
                    ans.get("answer", {}).get("unanswerable", False) 
                    for ans in qa_item.get("answers", [])
                )
                if is_unanswerable:
                    continue  # 直接跳过该问题，不进入后续处理
                # ------------------
                
                raw_question = qa_item.get("question", "")
                question_id = qa_item.get("question_id", "")
                # 将论文标题附加到问题上，便于检索时定位到正确的论文
                question = f'Based on the paper "{paper_title}", {raw_question}'
                
                gold_answers = []
                evidence_list = []
                answer_types = []
                answer_evidence_pairs = []
                
                # 遍历所有标注者的答案
                for answer_wrapper in qa_item.get("answers", []):
                    answer_obj = answer_wrapper.get("answer", {})
                    
                    current_answer = None
                    answer_type = self._get_answer_type(answer_obj)
                    
                    # 处理不同类型的答案
                    if answer_obj.get("unanswerable", False):
                        current_answer = "Not mentioned"
                        gold_answers.append(current_answer)
                    else:
                        extractive_spans = answer_obj.get("extractive_spans", [])
                        free_form_answer = answer_obj.get("free_form_answer", "")
                        yes_no = answer_obj.get("yes_no")
                        
                        if extractive_spans:
                            # 提取式答案：可能有多个片段
                            for span in extractive_spans:
                                if span and span.strip():
                                    gold_answers.append(span.strip())
                            current_answer = extractive_spans[0] if extractive_spans else None
                        elif free_form_answer and free_form_answer.strip():
                            # 自由形式答案
                            current_answer = free_form_answer.strip()
                            gold_answers.append(current_answer)
                        elif yes_no is not None:
                            # 是/否答案
                            current_answer = "Yes" if yes_no else "No"
                            gold_answers.append(current_answer)
                    
                    # 收集证据文本
                    current_evidence = []
                    evidence = answer_obj.get("evidence", [])
                    for ev in evidence:
                        if ev and ev.strip():
                            current_evidence.append(ev)
                            if ev not in evidence_list:
                                evidence_list.append(ev)
                    
                    # 记录答案类型（去重）
                    if answer_type not in answer_types:
                        answer_types.append(answer_type)
                    
                    # 保存答案-证据对应关系
                    if current_answer:
                        answer_evidence_pairs.append({
                            "answer": current_answer,
                            "evidence": current_evidence,
                            "answer_type": answer_type
                        })
                
                # 如果没有答案，默认为 "Not mentioned"
                if not gold_answers:
                    gold_answers = ["Not mentioned"]
                
                # 去重（保持顺序）
                gold_answers = list(dict.fromkeys(gold_answers))
                
                qa_pairs.append(StandardQA(
                    question=question,
                    gold_answers=gold_answers,
                    evidence=evidence_list,
                    category=None,
                    metadata={
                        "question_id": question_id,
                        "answer_types": answer_types,
                        "answer_evidence_pairs": answer_evidence_pairs
                    }
                ))

            standard_samples.append(StandardSample(
                sample_id=paper_id,
                qa_pairs=qa_pairs
            ))

        return standard_samples
    
    def _get_answer_type(self, answer_obj: Dict[str, Any]) -> str:
        """
        根据答案对象判断答案类型。
        
        Args:
            answer_obj: 答案对象，包含 extractive_spans、free_form_answer、yes_no 等字段
            
        Returns:
            str: 答案类型，可能的值：
                - "unanswerable": 无法回答
                - "extractive": 提取式答案
                - "free_form": 自由形式答案
                - "yes_no": 是/否答案
                - "unknown": 未知类型
        """
        if answer_obj.get("unanswerable", False):
            return "unanswerable"
        if answer_obj.get("extractive_spans"):
            return "extractive"
        if answer_obj.get("free_form_answer", "").strip():
            return "free_form"
        if answer_obj.get("yes_no") is not None:
            return "yes_no"
        return "unknown"

    def _convert_paper_to_markdown(self, paper_id: str, paper_data: Dict[str, Any]) -> str:
        """
        将 Qasper 论文结构转换为 Markdown 字符串。
        
        转换格式：
        ```markdown
        # {title}
        Paper ID: {paper_id}
        
        ## Abstract
        {abstract}
        
        ## Section Name 1
        {paragraph 1}
        {paragraph 2}
        
        ## Section Name 2
        ...
        
        ## Figures and Tables
        ### Figure 1
        Caption: {caption}
        File: {filename}
        
        ### Table 1
        Caption: {caption}
        File: {filename}
        ```
        
        Args:
            paper_id: 论文 ID
            paper_data: 论文数据，包含 title、abstract、full_text、figures_and_tables
            
        Returns:
            str: Markdown 格式的论文内容
        """
        md_lines = []
        
        # 标题
        title = paper_data.get("title", "Unknown Title")
        md_lines.append(f"# {title}")
        md_lines.append(f"Paper ID: {paper_id}\n")
        
        # 摘要
        abstract = paper_data.get("abstract", "")
        if abstract:
            md_lines.append("## Abstract")
            md_lines.append(abstract)
            md_lines.append("")
        
        # 正文章节
        full_text = paper_data.get("full_text", [])
        for section in full_text:
            section_name = section.get("section_name", "")
            paragraphs = section.get("paragraphs", [])
            
            if section_name:
                md_lines.append(f"## {section_name}")
            
            for para in paragraphs:
                if para and para.strip():
                    md_lines.append(para.strip())
                    md_lines.append("")
        
        # 图表信息
        figures_and_tables = paper_data.get("figures_and_tables", [])
        if figures_and_tables:
            md_lines.append("## Figures and Tables")
            for idx, fig in enumerate(figures_and_tables, 1):
                caption = fig.get("caption", "")
                file_name = fig.get("file", "")
                
                # 根据文件名或标题判断是图还是表
                if "Figure" in file_name or "figure" in caption.lower():
                    md_lines.append(f"### Figure {idx}")
                else:
                    md_lines.append(f"### Table {idx}")
                
                if caption:
                    md_lines.append(f"Caption: {caption}")
                if file_name:
                    md_lines.append(f"File: {file_name}")
                md_lines.append("")

        return "\n".join(md_lines)

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        
        full_prompt = QA_PROMPT.format(
            missing_rule=MISSING_RULE,
            context_text=context_text,
            question=qa.question
        )

        meta = {
            "question_id": qa.metadata.get("question_id", ""),
            "answer_types": qa.metadata.get("answer_types", [])
        }
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        后处理 LLM 生成的原始答案。
        
        当前实现仅去除首尾空白字符。
        
        Args:
            qa: 标准化问答对象
            raw_answer: LLM 生成的原始答案
            meta: 元数据字典
            
        Returns:
            str: 处理后的答案
        """
        return raw_answer.strip()
