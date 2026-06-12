
# src/adapters/hotpotqa_adapter.py
"""
HotpotQA Dataset Adapter

HotpotQA is a multi-hop QA dataset that requires reasoning across multiple documents.
Question types include:
- bridge: Bridge-type questions, need to find an intermediate entity first
- comparison: Comparison-type questions, need to compare two entities
"""

import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from base import BaseAdapter, StandardDoc, StandardSample, StandardQA

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
    HotpotQA Dataset Adapter.
    Processes multi-hop QA data with Wikipedia articles as documents.
    """

    def __init__(self, raw_file_path: str, **kwargs):
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
        Prepare document list for ingestion. Convert Wikipedia articles to Markdown.
        """
        if not os.path.exists(self.articles_file_path):
            raise FileNotFoundError(f"Articles file not found: {self.articles_file_path}")

        with open(self.articles_file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        os.makedirs(doc_dir, exist_ok=True)
        
        docs: List[StandardDoc] = []
        for article in articles:
            title = article.get("title", "")
            
            doc_content = self._convert_article_to_markdown(article)
            
            try:
                safe_title = self._safe_filename(title)
                doc_path = os.path.join(doc_dir, f"{safe_title}_doc.md")
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(doc_content)
                docs.append(StandardDoc(sample_id=title, doc_path=doc_path))
            except Exception as e:
                self.logger.error(f"[hotpotqa adapter] doc:{title} prepare error {e}")
                raise e
        
        self.logger.info(f"[HotpotQAAdapter] Processed {len(docs)} articles")
        return docs
    
    def _safe_filename(self, title: str) -> str:
        """Convert title to safe filename."""
        safe_chars = []
        for char in title:
            if char.isalnum() or char in (' ', '-', '_'):
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        return ''.join(safe_chars).strip()
    
    def _strip_html_tags(self, text: str) -> str:
        """Remove HTML tags from text, including hyperlinks."""
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove HTML entities like &nbsp;, &amp;
        text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
        return text
    
    def _convert_article_to_markdown(self, article: Dict[str, Any]) -> str:
        """Convert Wikipedia article to Markdown string."""
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
            
            # Strip HTML tags from article content
            para_text = self._strip_html_tags(para_text)
            
            if para_text and para_text.strip():
                md_lines.append(para_text.strip())
                md_lines.append("")
        
        return "\n".join(md_lines)

    def load_and_transform(self) -> List[StandardSample]:
        """
        Parse JSON question file.
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
        """Extract supporting facts from context as evidence."""
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

