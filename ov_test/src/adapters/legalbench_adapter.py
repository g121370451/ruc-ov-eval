import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAdapter, StandardDoc, StandardQA, StandardSample


SUBSETS = ["contractnli", "cuad", "maud", "privacy_qa"]


def sanitize_filename(name: str, max_length: int = 200) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = name.strip(" .")
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        name = f"{name}_file"
    if len(name) > max_length:
        name = name[:max_length].rstrip()
    return name or "untitled"


def convert_legal_to_md(content: str, title: str = "") -> str:
    lines = content.splitlines()
    result: List[str] = []

    if title:
        result.append(f"# {title}")
        result.append("")

    for line in lines:
        stripped = line.rstrip()
        text = stripped.strip()
        if not text:
            result.append("")
            continue

        if (
            text == text.upper()
            and re.search(r"[A-Z]{2}", text)
            and 3 < len(text) < 120
            and not re.search(r"[.-]{4,}", text)
            and not re.fullmatch(r"[\s\d,()\-\u2013\u2014]+", text)
        ):
            result.append(f"## {text}")
            result.append("")
            continue

        if re.match(r"^(ARTICLE|Section)\s+\S", text, re.IGNORECASE):
            result.append(f"### {text}")
            result.append("")
            continue

        if re.match(r"^\d+(\.\d+)*\.\s+[A-Z]", text):
            result.append(f"### {text}")
            result.append("")
            continue

        if re.match(r"^\([a-zA-Z]\)\s+\S", text):
            result.append(f"- {text}")
            continue

        result.append(stripped)

    final = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return final.strip() + "\n"


QA_PROMPT = """Based on the following legal document excerpts, answer the question concisely and accurately.

- Quote or closely paraphrase the relevant contract language when possible.
- If the answer involves a date, party name, or specific term, state it exactly.
- If the context contains no information relevant to the question, write "Not mentioned".

Question: {question}
Answer:"""


class LegalBenchAdapter(BaseAdapter):
    """
    LegalBench adapter.

    raw_file_path may point to either:
    1. Dataset root containing benchmarks/ and corpus/.
    2. A single benchmark JSON file under benchmarks/.
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        self._corpus_root: Optional[str] = None
        self._benchmark_files: List[str] = []
        self._resolve_paths()

    def _resolve_paths(self):
        path = Path(self.raw_file_path)
        if path.is_dir():
            self._corpus_root = str(path / "corpus")
            benchmarks_dir = path / "benchmarks"
            for subset in SUBSETS:
                benchmark_path = benchmarks_dir / f"{subset}.json"
                if benchmark_path.exists():
                    self._benchmark_files.append(str(benchmark_path))
            self.logger.info(
                f"[LegalBenchAdapter] root mode: {len(self._benchmark_files)} subsets, "
                f"corpus={self._corpus_root}"
            )
        elif path.is_file() and path.suffix == ".json":
            self._corpus_root = str(path.parent.parent / "corpus")
            self._benchmark_files = [str(path)]
            self.logger.info(
                f"[LegalBenchAdapter] single-file mode: {path.name}, corpus={self._corpus_root}"
            )
        else:
            raise FileNotFoundError(
                f"raw_file_path must be a directory or a .json benchmark file, got: {self.raw_file_path}"
            )

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        os.makedirs(doc_dir, exist_ok=True)
        seen: Dict[str, str] = {}

        for benchmark_file in self._benchmark_files:
            with open(benchmark_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            for test in data["tests"]:
                for snippet in test["snippets"]:
                    rel_path = snippet["file_path"]
                    if rel_path in seen:
                        continue

                    txt_path = os.path.join(self._corpus_root or "", rel_path)
                    if not os.path.exists(txt_path):
                        self.logger.warning(f"Corpus file not found: {txt_path}")
                        continue

                    safe_name = sanitize_filename(rel_path.replace("/", "_").replace(".txt", ".md"))
                    md_path = os.path.join(doc_dir, safe_name)

                    with open(txt_path, "r", encoding="utf-8", errors="replace") as handle:
                        raw_text = handle.read()

                    stem = Path(txt_path).stem
                    md_content = convert_legal_to_md(raw_text, title=stem)

                    with open(md_path, "w", encoding="utf-8") as handle:
                        handle.write(md_content)

                    seen[rel_path] = md_path

        docs = [
            StandardDoc(sample_id=rel_path, doc_paths=[md_path])
            for rel_path, md_path in seen.items()
        ]
        self.logger.info(f"[LegalBenchAdapter] data_prepare: {len(docs)} documents written to {doc_dir}")
        return docs

    def load_and_transform(self) -> List[StandardSample]:
        samples: List[StandardSample] = []
        global_idx = 0

        for benchmark_file in self._benchmark_files:
            subset_name = Path(benchmark_file).stem
            with open(benchmark_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            for test in data["tests"]:
                query = test["query"]
                snippets = test["snippets"]
                if not snippets:
                    continue

                gold_answers = [
                    snippet["answer"].strip()
                    for snippet in snippets
                    if snippet.get("answer", "").strip()
                ]
                if not gold_answers:
                    gold_answers = ["Not mentioned"]

                evidence = self._extract_evidence(snippets)
                primary_file = snippets[0]["file_path"]

                qa = StandardQA(
                    question=query,
                    gold_answers=gold_answers,
                    evidence=evidence,
                    category=subset_name,
                    metadata={
                        "subset": subset_name,
                        "global_idx": global_idx,
                        "file_paths": [snippet["file_path"] for snippet in snippets],
                    },
                )
                samples.append(StandardSample(sample_id=primary_file, qa_pairs=[qa]))
                global_idx += 1

        self.logger.info(f"[LegalBenchAdapter] load_and_transform: {len(samples)} samples loaded")
        return samples

    def _extract_evidence(self, snippets: List[dict]) -> List[str]:
        evidence: List[str] = []
        for snippet in snippets:
            rel_path = snippet["file_path"]
            span = snippet.get("span")
            answer = snippet.get("answer", "").strip()

            if not span or len(span) != 2:
                if answer:
                    evidence.append(answer)
                continue

            txt_path = os.path.join(self._corpus_root or "", rel_path)
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
                extracted = content[span[0] : span[1]].strip()
                evidence.append(answer if answer else extracted)
            except Exception:
                if answer:
                    evidence.append(answer)

        return evidence

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n---\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{QA_PROMPT.format(question=qa.question)}"
        return full_prompt, {}

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        return raw_answer.strip()
