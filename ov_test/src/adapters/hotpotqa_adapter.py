# src/adapters/hotpotqa_adapter.py
"""
HotpotQA 数据集适配器

HotpotQA 是一个多跳问答数据集，需要推理多个文档才能回答问题。
问题类型包括：
- bridge: 桥接型问题，需要先找到一个中间实体
- comparison: 比较型问题，需要比较两个实体

数据集特点：
1. 每个问题需要多个支持文档（supporting facts）
2. 答案类型包括：实体、是/否、片段
3. 问题难度分为：easy、medium、hard

适配器功能：
- data_prepare: 将文章转换为 Markdown 格式
- load_and_transform: 解析 QA 数据，保留支持事实信息
- build_prompt: 构建问答提示词
- post_process_answer: 后处理 LLM 输出
"""

import json
import os
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

QA_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 

### INSTRUCTIONS:
1. **Source Grounding**: Answer the question using ONLY the provided context.
2. **Conciseness**: Provide the answer as a short phrase, entity, or specific value. Avoid full sentences unless absolutely necessary.
3. **Yes/No Questions**: If the question is a Yes/No question, respond with ONLY "yes" or "no".
4. **Lists**: If the answer involves multiple items, separate them with a comma.
5. **Exact Extraction**: Use exact terminology from the text whenever possible.

### ABSENCE RULE:
{missing_rule}

---
### CONTEXT:
{context_text}

---
### QUESTION:
{question}

---
### ANSWER:
"""

MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."


class HotpotQAAdapter(BaseAdapter):
    """
    专门用于处理 HotpotQA 数据集的适配器。
    
    将 Wikipedia 文章转换为 Markdown 文档，
    并将多跳 QA 数据转换为标准化的 StandardSample 格式。
    
    Attributes:
        raw_file_path: 原始 JSON 数据文件路径（QA 数据）
        articles_file_path: 文章数据文件路径
        logger: 日志记录器
    """
    
    def __init__(self, raw_file_path: str, **kwargs):
        """
        初始化 HotpotQAAdapter。
        
        Args:
            raw_file_path: 原始数据文件路径（QA 数据 JSON）
            **kwargs: 其他参数
                - articles_file_path: 文章数据文件路径（可选）
        """
        super().__init__(raw_file_path)
        
        if 'articles_file_path' in kwargs:
            self.articles_file_path = kwargs['articles_file_path']
        else:
            data_dir = os.path.dirname(raw_file_path)
            qa_filename = os.path.basename(raw_file_path)
            
            if qa_filename == 'hotpot_qa_test.json':
                self.articles_file_path = os.path.join(data_dir, 'hotpot_articles_test.json')
            else:
                self.articles_file_path = os.path.join(data_dir, 'hotpot_articles.json')
    
    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        加载原始文章数据并转换为 OpenViking 友好格式。
        
        将每篇 Wikipedia 文章转换为 Markdown 文档，保留以下结构：
        - 标题（# Title）
        - 段落（保持原始段落结构）
        
        Args:
            doc_dir: 文档输出目录路径
            
        Returns:
            List[StandardDoc]: 标准化文档对象列表
            
        Raises:
            FileNotFoundError: 文章数据文件不存在
        """
        if not os.path.exists(self.articles_file_path):
            raise FileNotFoundError(f"Articles file not found: {self.articles_file_path}")

        res: List[StandardDoc] = []

        with open(self.articles_file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        os.makedirs(doc_dir, exist_ok=True)
        
        required_titles = self._get_required_titles()
        self.logger.info(f"[HotpotQAAdapter] Required articles: {len(required_titles)}")
        
        for article in articles:
            title = article.get("title", "")
            
            if title not in required_titles:
                continue
            
            doc_content = self._convert_article_to_markdown(article)
            
            try:
                safe_title = self._safe_filename(title)
                doc_path = os.path.join(doc_dir, f"{safe_title}_doc.md")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                res.append(StandardDoc(title, doc_path))
            except Exception as e:
                self.logger.error(f"[hotpotqa adapter] doc:{title} prepare error {e}")
                raise e
        
        self.logger.info(f"[HotpotQAAdapter] Processed {len(res)} articles")
        return res
    
    def _get_required_titles(self) -> set:
        """
        获取 QA 数据中涉及的文章标题列表。
        
        Returns:
            set: 文章标题集合
        """
        required = set()
        
        if not os.path.exists(self.raw_file_path):
            return required
        
        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        for item in qa_data:
            context = item.get("context", {})
            titles = context.get("title", [])
            required.update(titles)
        
        return required
    
    def _safe_filename(self, title: str) -> str:
        """
        将标题转换为安全的文件名。
        
        Args:
            title: 文章标题
            
        Returns:
            str: 安全的文件名
        """
        safe_chars = []
        for char in title:
            if char.isalnum() or char in (' ', '-', '_'):
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        return ''.join(safe_chars).strip()
    
    def _convert_article_to_markdown(self, article: Dict[str, Any]) -> str:
        """
        将 Wikipedia 文章结构转换为 Markdown 字符串。
        
        转换格式：
        ```markdown
        # {title}
        
        {paragraph 1}
        
        {paragraph 2}
        ...
        ```
        
        Args:
            article: 文章数据，包含 title、text 等字段
            
        Returns:
            str: Markdown 格式的文章内容
        """
        md_lines = []
        
        title = article.get("title", "Unknown Title")
        md_lines.append(f"# {title}")
        md_lines.append("")
        
        text = article.get("text", [])
        for paragraph in text:
            if isinstance(paragraph, list):
                para_text = " ".join(paragraph)
            else:
                para_text = paragraph
            
            if para_text and para_text.strip():
                md_lines.append(para_text.strip())
                md_lines.append("")
        
        return "\n".join(md_lines)

    def load_and_transform(self) -> List[StandardSample]:
        """
        加载原始 JSON 数据并转换为标准化的 StandardSample 对象列表。
        
        处理逻辑：
        1. 读取 QA 数据
        2. 按 sample_id 分组（使用 id 的前缀作为 sample_id）
        3. 提取 supporting facts 作为 evidence
        4. category 存储问题类型（bridge/comparison）
        
        Returns:
            List[StandardSample]: 标准化样本对象列表
            
        Raises:
            FileNotFoundError: 原始数据文件不存在
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        standard_samples = []
        
        for item in qa_data:
            qa_id = item.get("id", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            qa_type = item.get("type", "")
            level = item.get("level", "")
            
            supporting_facts = item.get("supporting_facts", {})
            context = item.get("context", {})
            
            evidence = self._extract_evidence(context, supporting_facts)
            
            gold_answers = [answer] if answer else ["Not mentioned"]
            
            sample_id = qa_id[:8] if len(qa_id) >= 8 else qa_id
            
            qa_pairs = [StandardQA(
                question=question,
                gold_answers=gold_answers,
                evidence=evidence,
                category=qa_type,
                metadata={
                    "id": qa_id,
                    "level": level,
                    "supporting_fact_titles": supporting_facts.get("title", []),
                    "supporting_fact_sent_ids": supporting_facts.get("sent_id", [])
                }
            )]
            
            standard_samples.append(StandardSample(
                sample_id=sample_id,
                qa_pairs=qa_pairs
            ))

        return standard_samples
    
    def _extract_evidence(self, context: Dict[str, Any], supporting_facts: Dict[str, Any]) -> List[str]:
        """
        从上下文中提取支持事实作为证据。
        
        Args:
            context: 上下文数据，包含 title 和 sentences
            supporting_facts: 支持事实，包含 title 和 sent_id
            
        Returns:
            List[str]: 证据文本列表
        """
        evidence = []
        
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        fact_titles = supporting_facts.get("title", [])
        fact_sent_ids = supporting_facts.get("sent_id", [])
        
        title_to_sentences = {}
        for i, title in enumerate(titles):
            title_to_sentences[title] = sentences[i] if i < len(sentences) else []
        
        for fact_title, sent_id in zip(fact_titles, fact_sent_ids):
            if fact_title in title_to_sentences:
                sents = title_to_sentences[fact_title]
                if sent_id < len(sents):
                    evidence_text = sents[sent_id]
                    if evidence_text and evidence_text.strip():
                        evidence.append(evidence_text.strip())
        
        return evidence

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        构建发送给 LLM 的完整提示词。
        
        提示词结构：
        1. 指导说明
        2. 无法回答的规则说明
        3. 上下文内容
        4. 问题
        
        Args:
            qa: 标准化问答对象
            context_blocks: 检索到的上下文文本块列表
            
        Returns:
            tuple[str, Dict[str, Any]]: 
                - 完整的提示词字符串
                - 元数据字典，包含 id、level 等
        """
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        
        full_prompt = QA_PROMPT.format(
            missing_rule=MISSING_RULE,
            context_text=context_text,
            question=qa.question
        )

        meta = {
            "id": qa.metadata.get("id", ""),
            "level": qa.metadata.get("level", ""),
            "type": qa.category
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
