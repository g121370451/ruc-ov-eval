# src/adapters/loong_adapter.py
"""
Loong Dataset Adapter (Paper-only)

Loong is a long-context benchmark with three subsets: paper, financial, and legal.
This adapter only handles the 'paper' subset because financial/legal lack Markdown structure.

Dataset characteristics:
- Mixed jsonl file: loong.jsonl contains paper/financial/legal interleaved
- Each paper sample involves multiple documents (2~N arXiv-style .md papers)
- Documents are heavily shared across samples (501/698 docs appear in multiple samples)
- Two task levels:
  - Level 3: Given a target paper title + several papers, identify Reference/Citation relationships
  - Level 4: Given several papers, construct the longest citation chain
- Answers are structured (JSON dict or list), not free-form text

The vector_store.ingest() automatically deduplicates doc_paths, so shared docs
across samples are only ingested once.
"""

import json
import os
from typing import List, Dict, Any

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

MISSING_RULE = "If the information is insufficient to answer, respond with an empty JSON object or empty list as appropriate."


class LoongAdapter(BaseAdapter):
    """
    Adapter for Loong dataset (paper subset only).
    """

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        Collect all Markdown document paths referenced by paper samples.

        Documents are already in .md format under <dataset_dir>/doc/paper/.
        No conversion needed — we return the existing paths directly.
        The ingest layer deduplicates paths, so docs shared across samples
        are submitted only once.

        Args:
            doc_dir: Document output directory (unused because docs are pre-built)

        Returns:
            List[StandardDoc]: One entry per unique document, sample_id = filename without .md
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        dataset_dir = os.path.dirname(self.raw_file_path)
        doc_root = os.path.join(dataset_dir, "doc", "paper")

        res: List[StandardDoc] = []
        seen_docs = set()
        paper_count = 0

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("type") != "paper":
                    continue

                paper_count += 1
                for doc_name in data.get("doc", []):
                    if doc_name in seen_docs:
                        continue
                    seen_docs.add(doc_name)

                    doc_path = os.path.join(doc_root, doc_name)
                    if os.path.exists(doc_path):
                        sample_id = doc_name.replace(".md", "")
                        res.append(StandardDoc(sample_id, [doc_path]))
                    else:
                        self.logger.warning(f"[loong adapter] Doc not found: {doc_path}")

        self.logger.info(
            f"[loong adapter] Loaded {paper_count} paper samples, "
            f"{len(res)} unique docs to ingest from {doc_root}"
        )
        return res

    def load_and_transform(self) -> List[StandardSample]:
        """
        Load paper samples from the mixed jsonl and convert to StandardSample.

        Question construction:
        - Level 3: instruction + "The paper you need to analyze: " + question (paper title)
        - Level 4: instruction only (question field is empty)

        Answer handling:
        - Level 3: dict {"Reference": [...], "Citation": [...]} → JSON string
        - Level 4: list ["Title 1", "Title 2", ...] → JSON string

        Evidence:
        - Loong has no explicit evidence annotations → empty list
          (Recall will naturally be 0; this dataset focuses on multi-doc reasoning)

        Returns:
            List[StandardSample]: Paper samples only
        """
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found: {self.raw_file_path}")

        standard_samples = []

        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("type") != "paper":
                    continue

                sample_id = data.get("id", "unknown")
                level = data.get("level", 0)
                instruction = data.get("instruction", "")
                target_question = data.get("question", "")

                # Compose the full question
                if target_question and target_question.strip():
                    full_question = (
                        f"{instruction}\n\n"
                        f"The paper you need to analyze: {target_question.strip()}"
                    )
                else:
                    full_question = instruction

                # Format answer as JSON string for uniform handling
                answer = data.get("answer")
                if isinstance(answer, (dict, list)):
                    gold_answers = [json.dumps(answer, ensure_ascii=False)]
                else:
                    gold_answers = [str(answer)] if answer is not None else [""]

                qa_pairs = [
                    StandardQA(
                        question=full_question,
                        gold_answers=gold_answers,
                        evidence=[],  # Loong has no explicit evidence
                        category=level,
                        metadata={
                            "level": level,
                            "set": data.get("set"),
                            "doc_list": data.get("doc", []),
                            "language": data.get("language"),
                            "original_question": target_question,
                            "original_instruction": instruction,
                        },
                    )
                ]

                standard_samples.append(
                    StandardSample(sample_id=sample_id, qa_pairs=qa_pairs)
                )

        self.logger.info(
            f"[loong adapter] Transformed {len(standard_samples)} paper samples"
        )
        return standard_samples

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build prompt from retrieved context blocks and the question.

        The original Loong prompt_template is:
            #Papers:\n{docs}\n\n{instruction}\n\n#The paper you need to analyze:\n{question}

        We map:
        - context_blocks → {docs} (retrieved paper excerpts)
        - qa.question    → {instruction} + target paper info
        """
        context_text = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No relevant papers retrieved."
        )

        full_prompt = f"""You are given several papers. Please carefully analyze them and answer the following question.

# Papers:
{context_text}

{qa.question}

Answer (provide your response in the required format, e.g. JSON with Reference/Citation fields or a list of paper titles):
"""

        meta = {
            "level": qa.metadata.get("level"),
            "doc_list": qa.metadata.get("doc_list", []),
        }
        return full_prompt, meta

    def post_process_answer(
        self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]
    ) -> str:
        """
        Post-process LLM raw output.

        Default: strip whitespace. The LLM is expected to output JSON or a list
        directly; no extra extraction needed.
        """
        return raw_answer.strip()