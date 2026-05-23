# src/adapters/open_ragbench_adapter.py
"""
Open RAG Benchmark Dataset Adapter

Open RAG Benchmark is a multimodal RAG evaluation dataset built from arXiv PDFs.
It contains 1000 papers (400 positive + 600 hard negatives) and 3045 QA pairs.
Each query is mapped to a specific doc and section via qrels.

Since only parsed JSONs (not raw PDFs) are provided, this adapter:
1. Converts each paper JSON to Markdown for ingestion
2. Preserves original section headings and replaces table/image placeholders
3. Uses qrels section_id to construct evidence for recall evaluation
"""

import json
import os
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

QA_PROMPT = """Based on the provided research paper excerpts, answer the following question accurately and concisely.

Question: {}
Answer:"""

MISSING_RULE = "If the provided context does not contain sufficient information to answer the question, respond with 'Not mentioned'."


class OpenRAGBenchAdapter(BaseAdapter):
    """
    Adapter for Open RAG Benchmark (arXiv PDF multimodal dataset).

    Expects raw_file_path to point to the arxiv directory containing:
        - corpus/         (paper JSON files)
        - queries.json    (query_id -> {query, type, source})
        - qrels.json      (query_id -> {doc_id, section_id})
        - answers.json    (query_id -> answer text)
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        # raw_file_path may be the arxiv directory itself or a file inside it
        self.base_dir = raw_file_path if os.path.isdir(raw_file_path) else os.path.dirname(raw_file_path)
        self.corpus_dir = os.path.join(self.base_dir, "corpus")
        self.queries_path = os.path.join(self.base_dir, "queries.json")
        self.qrels_path = os.path.join(self.base_dir, "qrels.json")
        self.answers_path = os.path.join(self.base_dir, "answers.json")

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        Convert all corpus JSONs to Markdown and return doc list for ingestion.
        Processes all 1000 papers (including hard negatives) to preserve the
        full retrieval evaluation scenario.
        """
        os.makedirs(doc_dir, exist_ok=True)
        docs: List[StandardDoc] = []

        if not os.path.exists(self.corpus_dir):
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")

        reused = 0    
        converted = 0 

        for filename in sorted(os.listdir(self.corpus_dir)):
            if not filename.endswith(".json"):
                continue
            doc_id = filename[:-5]  # strip .json

            md_path = os.path.join(doc_dir, f"{doc_id}.md")

            # 已有转化好的 md，直接复用
            if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
                docs.append(StandardDoc(sample_id=doc_id, doc_paths=[md_path]))
                reused += 1
                continue

            json_path = os.path.join(self.corpus_dir, filename)
            with open(json_path, "r", encoding="utf-8") as f:
                paper = json.load(f)

            md_content = self._convert_paper_to_markdown(paper)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            docs.append(StandardDoc(sample_id=doc_id, doc_paths=[md_path]))
            converted += 1

        self.logger.info(
            f"[OpenRAGBench] Prepared {len(docs)} documents "
            f"(reused {reused}, converted {converted})"
        )                                      
        return docs

    def _convert_paper_to_markdown(self, paper: Dict[str, Any]) -> str:
        """
        Convert a single paper JSON to Markdown string.

        Structure:
            # {title}
            **Authors**: ...
            **Categories**: ...
            **Published**: ...
            **Updated**: ...
            **Paper ID**: ...

            ---

            {section 0 text (with tables/images resolved)}

            {section 1 text (with tables/images resolved)}
            ...
        """
        md_lines: List[str] = []

        # Title
        title = paper.get("title", "Unknown Title")
        md_lines.append(f"# {title}")

        # Metadata
        authors = paper.get("authors", [])
        if authors:
            md_lines.append(f"**Authors**: {', '.join(str(a) for a in authors)}")

        categories = paper.get("categories", [])
        if categories:
            md_lines.append(f"**Categories**: {', '.join(str(c) for c in categories)}")

        if paper.get("published"):
            md_lines.append(f"**Published**: {paper['published']}")
        if paper.get("updated"):
            md_lines.append(f"**Updated**: {paper['updated']}")

        md_lines.append(f"**Paper ID**: {paper.get('id', '')}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Sections
        for sec in paper.get("sections", []):
            text = sec.get("text", "")

            # Replace table placeholders with actual markdown tables
            for table_id, table_md in sec.get("tables", {}).items():
                placeholder = f"![{table_id}]({table_id})"
                table_block = f"\n\n{table_md}\n\n"
                if placeholder in text:
                    text = text.replace(placeholder, table_block)
                else:
                    text += table_block

            # Replace image placeholders with simple markers (base64 too large for text RAG)
            for img_id in sec.get("images", {}).keys():
                placeholder = f"![{img_id}]({img_id})"
                img_block = f"\n\n[Image: {img_id}]\n\n"
                if placeholder in text:
                    text = text.replace(placeholder, img_block)
                else:
                    text += img_block

            if text.strip():
                md_lines.append(text.strip())
                md_lines.append("")

        return "\n".join(md_lines)

    def load_and_transform(self) -> List[StandardSample]:
        """
        Load queries, qrels, and answers; group by doc_id into StandardSamples.
        Evidence is constructed from the qrels-specified section text.
        """
        with open(self.queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)
        with open(self.answers_path, "r", encoding="utf-8") as f:
            answers = json.load(f)

        # Group queries by doc_id
        doc_qa_map: Dict[str, List[Dict[str, Any]]] = {}
        for query_id, rel in qrels.items():
            doc_id = rel["doc_id"]
            section_id = rel["section_id"]
            query_data = queries.get(query_id, {})
            answer = answers.get(query_id, "")

            doc_qa_map.setdefault(doc_id, []).append({
                "query_id": query_id,
                "question": query_data.get("query", ""),
                "type": query_data.get("type", ""),
                "source": query_data.get("source", ""),
                "answer": answer,
                "section_id": section_id,
            })

        samples: List[StandardSample] = []
        for doc_id, qa_list in doc_qa_map.items():
            # Load corpus sections for evidence construction
            sections = []
            doc_json_path = os.path.join(self.corpus_dir, f"{doc_id}.json")
            if os.path.exists(doc_json_path):
                with open(doc_json_path, "r", encoding="utf-8") as f:
                    paper = json.load(f)
                sections = paper.get("sections", [])

            qa_pairs: List[StandardQA] = []
            for qa in qa_list:
                section_id = qa["section_id"]
                evidence_texts: List[str] = []

                if 0 <= section_id < len(sections):
                    sec = sections[section_id]
                    sec_text = sec.get("text", "")
                    # Append tables into evidence so recall can match them too
                    for table_id, table_md in sec.get("tables", {}).items():
                        sec_text += f"\n\n{table_md}"
                    if sec_text.strip():
                        evidence_texts.append(sec_text.strip())

                qa_pairs.append(StandardQA(
                    question=qa["question"],
                    gold_answers=[str(qa["answer"])],
                    evidence=evidence_texts,
                    category=qa["type"],
                    metadata={
                        "query_id": qa["query_id"],
                        "source": qa["source"],
                        "doc_id": doc_id,
                        "section_id": section_id,
                    }
                ))

            samples.append(StandardSample(
                sample_id=doc_id,
                qa_pairs=qa_pairs
            ))

        total_qa = sum(len(s.qa_pairs) for s in samples)
        self.logger.info(
            f"[OpenRAGBench] Loaded {total_qa} questions across {len(samples)} documents"
        )
        return samples

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        Construct LLM prompt from retrieved context blocks and question.
        """
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
        full_prompt = f"{context_text}\n\n{MISSING_RULE}\n\n{QA_PROMPT.format(qa.question)}"

        meta = {
            "query_id": qa.metadata.get("query_id", ""),
            "source": qa.metadata.get("source", ""),
            "question_type": qa.category,
        }
        return full_prompt, meta

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """Default post-processing: strip whitespace."""
        return raw_answer.strip()
