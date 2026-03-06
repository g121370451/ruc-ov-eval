import json
import os
from typing import Dict, List, Tuple

from bench_framework.adapters.base import BaseAdapter, StandardDoc, StandardSample, StandardQA
from bench_framework.types import Metadata

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

MISSING_RULE = "If no information is available to answer the question, write 'Not mentioned'."


class QasperAdapter(BaseAdapter):
    """专门用于处理 Qasper 数据集的适配器。"""

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
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
                res.append(StandardDoc(paper_id, doc_path))
            except Exception as e:
                self.logger.error(f"[qasper adapter] doc:{paper_id} prepare error {e}")
                raise e
        return res

    def load_and_transform(self) -> List[StandardSample]:
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        standard_samples = []

        for paper_id, paper_data in data.items():
            qa_pairs = []
            paper_title = paper_data.get("title", "Unknown Title")

            for qa_item in paper_data.get("qas", []):
                is_unanswerable = all(
                    ans.get("answer", {}).get("unanswerable", False)
                    for ans in qa_item.get("answers", [])
                )
                if is_unanswerable:
                    continue

                raw_question = qa_item.get("question", "")
                question_id = qa_item.get("question_id", "")
                question = f'Based on the paper "{paper_title}", {raw_question}'

                gold_answers = []
                evidence_list = []
                answer_types = []
                answer_evidence_pairs = []

                for answer_wrapper in qa_item.get("answers", []):
                    answer_obj = answer_wrapper.get("answer", {})

                    current_answer = None
                    answer_type = self._get_answer_type(answer_obj)

                    if answer_obj.get("unanswerable", False):
                        current_answer = "Not mentioned"
                        gold_answers.append(current_answer)
                    else:
                        extractive_spans = answer_obj.get("extractive_spans", [])
                        free_form_answer = answer_obj.get("free_form_answer", "")
                        yes_no = answer_obj.get("yes_no")

                        if extractive_spans:
                            for span in extractive_spans:
                                if span and span.strip():
                                    gold_answers.append(span.strip())
                            current_answer = extractive_spans[0] if extractive_spans else None
                        elif free_form_answer and free_form_answer.strip():
                            current_answer = free_form_answer.strip()
                            gold_answers.append(current_answer)
                        elif yes_no is not None:
                            current_answer = "Yes" if yes_no else "No"
                            gold_answers.append(current_answer)

                    current_evidence = []
                    evidence = answer_obj.get("evidence", [])
                    for ev in evidence:
                        if ev and ev.strip():
                            current_evidence.append(ev)
                            if ev not in evidence_list:
                                evidence_list.append(ev)

                    if answer_type not in answer_types:
                        answer_types.append(answer_type)

                    if current_answer:
                        answer_evidence_pairs.append({
                            "answer": current_answer,
                            "evidence": current_evidence,
                            "answer_type": answer_type
                        })

                if not gold_answers:
                    gold_answers = ["Not mentioned"]

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

    def _get_answer_type(self, answer_obj: dict) -> str:
        if answer_obj.get("unanswerable", False):
            return "unanswerable"
        if answer_obj.get("extractive_spans"):
            return "extractive"
        if answer_obj.get("free_form_answer", "").strip():
            return "free_form"
        if answer_obj.get("yes_no") is not None:
            return "yes_no"
        return "unknown"

    def _convert_paper_to_markdown(self, paper_id: str, paper_data: dict) -> str:
        md_lines = []

        title = paper_data.get("title", "Unknown Title")
        md_lines.append(f"# {title}")
        md_lines.append(f"Paper ID: {paper_id}\n")

        abstract = paper_data.get("abstract", "")
        if abstract:
            md_lines.append("## Abstract")
            md_lines.append(abstract)
            md_lines.append("")

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

        figures_and_tables = paper_data.get("figures_and_tables", [])
        if figures_and_tables:
            md_lines.append("## Figures and Tables")
            for idx, fig in enumerate(figures_and_tables, 1):
                caption = fig.get("caption", "")
                file_name = fig.get("file", "")

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

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> Tuple[str, Metadata]:
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

        full_prompt = QA_PROMPT.format(
            missing_rule=MISSING_RULE,
            context_text=context_text,
            question=qa.question
        )

        meta: Metadata = {
            "question_id": qa.metadata.get("question_id", ""),
            "answer_types": qa.metadata.get("answer_types", [])
        }
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Metadata) -> str:
        return raw_answer.strip()
