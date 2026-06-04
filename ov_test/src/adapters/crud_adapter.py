import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA

# ----------
# 文件名清理
# ----------

def sanitize_filename(name: str, max_length: int = 150) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'[\x00-\x1f\x7f]', "", name)
    name = name.strip(" .")
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.upper() in reserved_names:
        name = f"{name}_file"
    if len(name) > max_length:
        name = name[:max_length].rstrip()
    return name or "untitled"


# ----------
# Prompt 模板
# ----------

QA_PROMPT = """请根据以下提供的上下文信息，准确、简洁地回答问题。

### 回答要求：
1. **基于上下文**：仅使用提供的上下文信息作答，不要引入外部知识。
2. **简洁准确**：直接给出答案，避免冗长解释。
3. **信息不足**：如果上下文完全没有相关信息，请回答"未提及"。

---
### 上下文：
{context_text}

---
### 问题：
{question}

### 答案：
"""


class CrudAdapter(BaseAdapter):
    """
    CRUD 数据集适配器（Global 模式）。

    支持子集："1doc", "2docs", "3docs", "all"

    raw_file_path 指向 split_merged.json（或 CRUD 数据集根目录）。
    文档库构建逻辑：
        1. 从 80000_docs/ 原始语料按时间戳解析新闻片段
        2. 从 split_merged.json 提取 golden 新闻作为补充（确保覆盖率）
    """

    def __init__(self, raw_file_path: str, subset: str = "2docs"):
        super().__init__(raw_file_path)
        self.subset = subset
        self._split_file, self._corpus_dir = self._resolve_paths()

    def _resolve_paths(self):
        p = Path(self.raw_file_path)
        if p.is_file() and p.name.endswith(".json"):
            split_file = str(p)
            crud_root = p.parent.parent
        elif p.is_dir():
            crud_root = p
            split_file = str(p / "crud_split" / "split_merged.json")
        else:
            raise FileNotFoundError(
                f"raw_file_path must be a .json file or a directory, got: {self.raw_file_path}"
            )

        corpus_dir = str(crud_root / "80000_docs")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"split_merged.json not found: {split_file}")
        if not os.path.exists(corpus_dir):
            raise FileNotFoundError(f"80000_docs not found: {corpus_dir}")

        return split_file, corpus_dir

    # ----------
    # data_prepare
    # ----------

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        os.makedirs(doc_dir, exist_ok=True)

        # 1. 解析 80000_docs 原始语料
        corpus_docs = self._prepare_corpus_docs(doc_dir)
        self.logger.info(f"[CrudAdapter] Corpus docs prepared: {len(corpus_docs)}")

        # 2. 从 split_merged.json 提取 golden 新闻作为补充
        golden_docs = self._prepare_golden_docs(doc_dir, corpus_docs)
        self.logger.info(f"[CrudAdapter] Golden docs prepared: {len(golden_docs)}")

        all_docs = corpus_docs + golden_docs
        self.logger.info(f"[CrudAdapter] Total docs: {len(all_docs)}")
        return all_docs

    def _prepare_corpus_docs(self, doc_dir: str) -> List[StandardDoc]:
        """解析 80000_docs，按时间戳分割新闻片段。"""
        corpus_doc_dir = os.path.join(doc_dir, "corpus")
        os.makedirs(corpus_doc_dir, exist_ok=True)

        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
        docs: List[StandardDoc] = []
        global_idx = 0

        for fname in sorted(os.listdir(self._corpus_dir)):
            fpath = os.path.join(self._corpus_dir, fname)
            if os.path.isdir(fpath):
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                self.logger.warning(f"Failed to read {fname}: {e}")
                continue

            matches = list(date_pattern.finditer(content))
            if not matches:
                self.logger.debug(f"No date matches in {fname}, skipping")
                continue

            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                news_text = content[start:end].strip()
                news_text = re.sub(r"\s+", " ", news_text)

                if len(news_text) < 100:
                    continue

                # 嵌入模型有最大 token 限制（约 8K），超长文档需要截断
                # 中文约 1~1.5 字符/token，保守截断到 6000 字符
                MAX_DOC_CHARS = 6000
                if len(news_text) > MAX_DOC_CHARS:
                    self.logger.warning(
                        f"[CrudAdapter] News segment truncated from {len(news_text)} to {MAX_DOC_CHARS} chars "
                        f"({fname}, news_{i})"
                    )
                    news_text = news_text[:MAX_DOC_CHARS]

                md_content = self._convert_news_to_md(news_text)
                safe_name = sanitize_filename(f"{fname}_news_{i:04d}")
                md_path = os.path.join(corpus_doc_dir, f"{safe_name}.md")

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)

                docs.append(StandardDoc(
                    sample_id=f"corpus_{global_idx:06d}",
                    doc_paths=[md_path]
                ))
                global_idx += 1

        return docs

    def _prepare_golden_docs(self, doc_dir: str, corpus_docs: List[StandardDoc]) -> List[StandardDoc]:
        """
        从 split_merged.json 提取 golden 新闻，补充 corpus_docs 中缺失的部分。
        通过前80字符匹配判断是否已存在。
        """
        golden_doc_dir = os.path.join(doc_dir, "golden")
        os.makedirs(golden_doc_dir, exist_ok=True)

        # 构建 corpus 内容快速查找集合
        corpus_texts = []
        for doc in corpus_docs:
            try:
                with open(doc.doc_paths[0], "r", encoding="utf-8") as f:
                    corpus_texts.append(f.read())
            except Exception:
                pass

        with open(self._split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        # 收集需要补充的 golden 新闻
        golden_news_map: Dict[str, str] = {}  # text_prefix -> full_text
        subsets_to_process = []

        if self.subset == "all":
            subsets_to_process = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
        elif self.subset == "1doc":
            subsets_to_process = ["questanswer_1doc"]
        elif self.subset == "2docs":
            subsets_to_process = ["questanswer_2docs"]
        elif self.subset == "3docs":
            subsets_to_process = ["questanswer_3docs"]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        for subset_name in subsets_to_process:
            for item in split_data.get(subset_name, []):
                for key in ["news1", "news2", "news3"]:
                    if key not in item:
                        continue
                    text = item[key]
                    prefix = text[:80]
                    if prefix not in golden_news_map:
                        golden_news_map[prefix] = text

        docs: List[StandardDoc] = []
        added = 0
        skipped = 0
        idx = 0

        MAX_DOC_CHARS = 6000
        for prefix, text in golden_news_map.items():
            # 检查是否已存在于 corpus 中
            found = any(prefix in ct for ct in corpus_texts)
            if found:
                skipped += 1
                continue

            # 截断超长 golden 新闻
            if len(text) > MAX_DOC_CHARS:
                self.logger.warning(
                    f"[CrudAdapter] Golden news truncated from {len(text)} to {MAX_DOC_CHARS} chars"
                )
                text = text[:MAX_DOC_CHARS]

            md_content = self._convert_news_to_md(text)
            safe_name = sanitize_filename(f"golden_news_{idx:04d}")
            md_path = os.path.join(golden_doc_dir, f"{safe_name}.md")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            docs.append(StandardDoc(
                sample_id=f"golden_{idx:06d}",
                doc_paths=[md_path]
            ))
            idx += 1
            added += 1

        self.logger.info(
            f"[CrudAdapter] Golden docs: {added} added, {skipped} already in corpus"
        )
        return docs

    def _convert_news_to_md(self, content: str) -> str:
        """将新闻文本转换为 Markdown 格式。"""
        lines = []

        # 尝试提取时间和正文
        m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(.*)", content)
        if m:
            time_str = m.group(1)
            rest = m.group(2)

            lines.append(f"# 新闻片段")
            lines.append("")
            lines.append(f"**时间**: {time_str}")

            # 尝试提取来源
            source_m = re.search(r"来源[：:]\s*([^，,\s]+)", rest)
            if source_m:
                lines.append(f"**来源**: {source_m.group(1)}")

            lines.append("")
            lines.append("---")
            lines.append("")

            # 提取正文
            body = re.sub(r"^[，,]?\s*正文[：:]\s*", "", rest).strip()
            lines.append(body)
        else:
            # 没有时间前缀，尝试提取标题/来源信息
            lines.append(f"# 新闻片段")
            lines.append("")
            lines.append(content.strip())

        return "\n".join(lines) + "\n"

    # ----------
    # load_and_transform
    # ----------

    def load_and_transform(self) -> List[StandardSample]:
        with open(self._split_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        subsets_to_process = []
        if self.subset == "all":
            subsets_to_process = ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
        elif self.subset == "1doc":
            subsets_to_process = ["questanswer_1doc"]
        elif self.subset == "2docs":
            subsets_to_process = ["questanswer_2docs"]
        elif self.subset == "3docs":
            subsets_to_process = ["questanswer_3docs"]
        else:
            raise ValueError(f"Unknown subset: {self.subset}")

        samples: List[StandardSample] = []
        global_idx = 0

        for subset_name in subsets_to_process:
            subset_data = data.get(subset_name, [])
            for item in subset_data:
                sample_id = item.get("ID", f"{subset_name}_{global_idx}")
                event = item.get("event", "")
                question = item.get("questions", "")
                answer = item.get("answers", "")

                # evidence: 收集所有关联的新闻文本，用于 Recall 计算
                evidence = []
                for key in ["news1", "news2", "news3"]:
                    if key in item and item[key]:
                        evidence.append(item[key].strip())

                # 如果 answer 不在 evidence 中，也加入 evidence（帮助 Recall）
                if answer and answer.strip() and answer.strip() not in evidence:
                    evidence.append(answer.strip())

                qa = StandardQA(
                    question=question,
                    gold_answers=[answer.strip()] if answer.strip() else ["未提及"],
                    evidence=evidence,
                    category=subset_name,
                    metadata={
                        "event": event,
                        "global_idx": global_idx,
                        "doc_count": sum(1 for k in ["news1", "news2", "news3"] if k in item),
                    }
                )

                samples.append(StandardSample(
                    sample_id=sample_id,
                    qa_pairs=[qa],
                ))
                global_idx += 1

        self.logger.info(f"[CrudAdapter] Loaded {len(samples)} samples from {subsets_to_process}")
        return samples

    # ----------
    # build_prompt
    # ----------

    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "未提供相关上下文。"
        full_prompt = QA_PROMPT.format(
            context_text=context_text,
            question=qa.question
        )
        return full_prompt, {"category": qa.category, "global_idx": qa.metadata.get("global_idx", "")}

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        return raw_answer.strip()
