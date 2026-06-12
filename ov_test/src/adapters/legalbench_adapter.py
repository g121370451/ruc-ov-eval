import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

# ---------
# 支持的子集
# ---------
SUBSETS = ["contractnli", "cuad", "maud", "privacy_qa"]

# ---------
# 文件名清理 (同 ClapNQ)
# ---------

def sanitize_filename(name: str, max_length: int = 200) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'[\x00-\x1f\x7f]', "", name)
    name = name.strip(" .")
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved:
        name = f"{name}_file"
    if len(name) > max_length:
        name = name[:max_length].rstrip()
    return name or "untitled"

def convert_legal_to_md(content: str, title: str = "") -> str:
    """
    将法律合同/政策 txt 转换为结构化 Markdown。

    转换规则 (优先级从高到低)：
    1. 全大写短行[3-120字符，含至少2个大写字母] → ## 二级标题
      排除目录点线行 (含 4 个以上连续点)
    2. "Article I" / "Section 1.01." 开头行 → ### 三级标题
    3. 纯数字条款 "1." / "1.1." 开头行 (首字母大写) → ### 三级标题
    4. "(a) ..." 字母枚举行 → Markdown 列表项 (不影响段落文本)
    5. 其余行保持原样，多余空行收缩为单个空行

    与 ClapNQ 的核心区别：保留原始换行结构，不做全局空白压缩，
    确保法律条款粒度在检索时可以被独立召回。
    """
    lines = content.splitlines()
    result: list[str] = []

    if title:
        result.append(f"# {title}")
        result.append("")

    for line in lines:
        stripped = line.rstrip()
        text = stripped.strip()

        # 空行
        if not text:
            result.append("")
            continue

        # 规则 1: 全大写标题行
        # 排除：目录虚线行 (连续 4 个以上点/横线) 、纯符号行
        if (
            text == text.upper()
            and re.search(r"[A-Z]{2}", text)
            and 3 < len(text) < 120
            and not re.search(r"[.-]{4,}", text)
            and not re.fullmatch(r"[\s\d,()\-\\u2013\\u2014]+", text)
        ):
            result.append(f"## {text}")
            result.append("")
            continue

        # 规则 2: Article / Section 标题
        if re.match(r"^(ARTICLE|Section)\s+\S", text, re.IGNORECASE):
            result.append(f"### {text}")
            result.append("")
            continue

        # 规则 3: 数字条款标题 ("1." 或 "1.1." 开头，后接大写字母)
        if re.match(r"^\d+(\.\d+)*\.\s+[A-Z]", text):
            result.append(f"### {text}")
            result.append("")
            continue

        # 规则 4: 字母枚举项 "(a) "
        if re.match(r"^\([a-zA-Z]\)\s+\S", text):
            result.append(f"- {text}")
            continue

        # 默认：保留原始缩进 (用于列举子项)
        result.append(stripped)

    # 收缩连续空行为最多 2 行 (1 个空行)
    final = re.sub(r"\n{3,}", "\n\n", "\n".join(result))
    return final.strip() + "\n"


# ----------
# Prompt 模板
# ----------

QA_PROMPT = """Based on the following legal document excerpts, answer the question concisely and accurately.

- Quote or closely paraphrase the relevant contract language when possible.
- If the answer involves a date, party name, or specific term, state it exactly.
- If the context contains no information relevant to the question, write "Not mentioned".

Question: {question}
Answer:"""


class LegalBenchAdapter(BaseAdapter):
    """
    LegalBench 数据集适配器。

    raw_file_path 可以是：
    1. 数据集根目录 (含 benchmarks/ 和 corpus/ 的目录)，处理所有子集
    2. 单个 benchmark JSON 文件路径 (如 benchmarks/cuad.json)，仅处理该子集
    """
    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        self._corpus_root: Optional[str] = None
        self._benchmark_files: list[str] = []
        self._resolve_paths()

    def _resolve_paths(self):
        p = Path(self.raw_file_path)
        if p.is_dir():
            # 根目录模式
            self._corpus_root = str(p / "corpus")
            benchmarks_dir = p / "benchmarks"
            for subset in SUBSETS:
                fp = benchmarks_dir / f"{subset}.json"
                if fp.exists():
                    self._benchmark_files.append(str(fp))
            self.logger.info(
                f"[LegalBenchAdapter] root mode: {len(self._benchmark_files)} subsets, corpus={self._corpus_root}"
            )
        elif p.is_file() and p.suffix == ".json":
            # 单文件模式: corpus 目录与 benchmarks/ 同级
            self._corpus_root = str(p.parent.parent / "corpus")
            self._benchmark_files = [str(p)]
            self.logger.info(
                f"[LegalBenchAdapter] single-file mode: {p.name}, corpus={self._corpus_root}"
            )
        else:
            raise FileNotFoundError(
                f"raw_file_path must be a directory or a .json benchmark file, got: {self.raw_file_path}"
            )

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        """
        遍历所有 benchmark JSON，收集引用的 corpus 文件路径，
        将 .txt 转换为 Markdown 并写入 doc_dir。

        返回值：每个唯一原始文件对应一个 StandardDoc
            sample_id 为 corpus 相对路径 (如 "cuad/foo.txt")，
            doc_paths 为转换后的 .md 文件路径列表 (单元素) 。
        """
        os.makedirs(doc_dir, exist_ok=True)
        seen: dict[str, str] = {}  # relative_path -> md_path

        for bm_file in self._benchmark_files:
            with open(bm_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for test in data["tests"]:
                for snippet in test["snippets"]:
                    rel_path = snippet["file_path"]  # e.g. "cuad/foo.txt"
                    if rel_path in seen:
                        continue

                    txt_path = os.path.join(self._corpus_root, rel_path)
                    if not os.path.exists(txt_path):
                        self.logger.warning(f"Corpus file not found: {txt_path}")
                        continue

                    # 生成不冲突的 md 文件名
                    # 将 "/" 替换为 "_" 保留子集前缀，避免不同子集重名
                    safe_name = sanitize_filename(rel_path.replace("/", "_").replace(".txt", ".md"))
                    md_path = os.path.join(doc_dir, safe_name)

                    try:
                        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                            raw_text = f.read()

                        # 用文件名 (去扩展名) 作为文档标题
                        stem = Path(txt_path).stem
                        md_content = convert_legal_to_md(raw_text, title=stem)

                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(md_content)

                        seen[rel_path] = md_path
                        self.logger.debug(f"Converted: {rel_path} -> {safe_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to convert {txt_path}: {e}")
                        raise

        res = [
            StandardDoc(sample_id=rel_path, doc_paths=[md_path])
            for rel_path, md_path in seen.items()
        ]
        self.logger.info(f"[LegalBenchAdapter] data_prepare: {len(res)} documents written to {doc_dir}")
        return res

    # ----------
    # load_and_transform: benchmark JSON → StandardSample 列表
    # ----------

    def load_and_transform(self) -> List[StandardSample]:
        """
        将 benchmark JSON 转换为 StandardSample 列表。

        设计说明:
        - sample_id = query 对应的 corpus 文件相对路径 ("cuad/foo.txt")
        一个文档可能对应多个 query，每个 query 是独立的 StandardSample，
        但它们共享同一个 sample_id，从而检索时只查该文档的向量库。
        - StandardQA.gold_answers = 所有 snippet.answer [可能多个]
        - StandardQA.evidence = span 对应的原始文本 (用于 Recall 评测)
        - StandardQA.metadata 存储子集名称，供 judge_util 做针对性评分
        """
        samples: list[StandardSample] = []
        global_idx = 0

        for bm_file in self._benchmark_files:
            subset_name = Path(bm_file).stem  # e.g. "cuad"
            with open(bm_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for test in data["tests"]:
                query = test["query"]
                snippets = test["snippets"]

                if not snippets:
                    continue

                # 所有 snippet 的答案文本作为 gold_answers
                gold_answers = [s["answer"].strip() for s in snippets if s.get("answer", "").strip()]
                if not gold_answers:
                    gold_answers = ["Not mentioned"]

                # evidence: 从原始文档按 span 提取，与 gold_answers 对齐
                # (span 是字符级偏移，document_text[start:end] == answer)
                evidence = self._extract_evidence(snippets)

                # sample_id 使用第一个 snippet 的文件路径
                # 若多个 snippet 来自不同文件 (极少数情况)，
                # 仍以第一个为主文档，其他文档的向量库不参与检索
                primary_file = snippets[0]["file_path"]

                qa = StandardQA(
                    question=query,
                    gold_answers=gold_answers,
                    evidence=evidence,
                    category=subset_name,
                    metadata={
                        "subset": subset_name,
                        "global_idx": global_idx,
                        "file_paths": list(s["file_path"] for s in snippets),
                    },
                )

                samples.append(StandardSample(
                    sample_id=primary_file,
                    qa_pairs=[qa],
                ))
                global_idx += 1

        self.logger.info(f"[LegalBenchAdapter] load_and_transform: {len(samples)} samples loaded")
        return samples

    def _extract_evidence(self, snippets: List[dict]) -> List[str]:
        """从原始 txt 按 span 提取证据文本，验证与 answer 字段一致。"""
        evidence = []
        for snippet in snippets:
            rel_path = snippet["file_path"]
            span = snippet.get("span")
            answer = snippet.get("answer", "").strip()

            if not span or len(span) != 2:
                if answer:
                    evidence.append(answer)
                continue

            txt_path = os.path.join(self._corpus_root, rel_path)
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                extracted = content[span[0]:span[1]].strip()
                # 理论上 extracted == answer，直接用 answer 更可靠
                evidence.append(answer if answer else extracted)
            except Exception:
                if answer:
                    evidence.append(answer)

        return evidence

    # ----------
    # build_prompt
    # ----------

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n---\n\n".join(context_blocks)
        full_prompt = f"{context_text}\n\n{QA_PROMPT.format(question=qa.question)}"
        return full_prompt, {}

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        return raw_answer.strip()
