import json
import os
from typing import List, Dict, Any, Set

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

QA_PROMPT = """Based on the above context, answer the following question with a short phrase or single word.

Question: {} Answer:
"""

class HotpotQAAdapter(BaseAdapter):
    """
    HotpotQA 数据集适配器。
    每个 title + sentences 作为一个独立文档入库。
    """
    
    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        将 HotpotQA 的 context 转换为独立文档。
        每个 title 生成一个 markdown 文档。
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        os.makedirs(doc_dir, exist_ok=True)
        
        seen_titles: Set[str] = set()
        res: List[StandardDoc] = []

        for item in data:
            context = item.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])

            for i, title in enumerate(titles):
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                sentences = sentences_list[i] if i < len(sentences_list) else []
                doc_content = self._convert_to_markdown(title, sentences)

                safe_title = self._safe_filename(title)
                doc_path = os.path.join(doc_dir, f"{safe_title}_doc.md")

                try:
                    with open(doc_path, "w", encoding="utf-8") as f:
                        f.write(doc_content)
                    res.append(StandardDoc(sample_id=title, doc_path=doc_path))
                except Exception as e:
                    self.logger.error(f"[hotpotqa adapter] doc:{title} prepare error {e}")
                    raise e

        self.logger.info(f"[hotpotqa adapter] Created {len(res)} unique documents")
        return res

    def _convert_to_markdown(self, title: str, sentences: List[str]) -> str:
        """
        将 title 和 sentences 转换为 markdown 格式。
        """
        lines = [f"# {title}", ""]
        for sent in sentences:
            lines.append(sent)
        return "\n".join(lines)

    def _safe_filename(self, title: str) -> str:
        """
        将标题转换为安全的文件名。
        """
        safe = title.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe = safe.replace("?", "_").replace("*", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_").replace("|", "_")
        return safe[:100]

    def load_and_transform(self) -> List[StandardSample]:
        """
        加载原始 JSON 数据并转换为标准化的 StandardSample 对象列表。
        evidence 使用 supporting_facts 中的 title 列表（去重）。
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        standard_samples = []

        for item in data:
            sample_id = item.get("id", "unknown")
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            supporting_facts = item.get("supporting_facts", {})
            supporting_titles = supporting_facts.get("title", [])
            evidence = list(set(supporting_titles))

            raw_ans = answer
            if isinstance(raw_ans, list):
                golds = [str(g) for g in raw_ans]
            elif raw_ans is None or raw_ans == "":
                golds = ["Not mentioned"]
            else:
                golds = [str(raw_ans)]

            qa_pairs = [StandardQA(
                question=question,
                gold_answers=golds,
                evidence=evidence,
                category=item.get("type", "unknown"),
                metadata={
                    "level": item.get("level", "unknown"),
                    "original_id": sample_id
                }
            )]

            standard_samples.append(StandardSample(
                sample_id=sample_id,
                qa_pairs=qa_pairs
            ))

        return standard_samples

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        构建 QA prompt。
        """
        context_text = "\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{QA_PROMPT.format(qa.question)}"
        meta = {"category": qa.category}
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        后处理答案。
        """
        return raw_answer.strip()
