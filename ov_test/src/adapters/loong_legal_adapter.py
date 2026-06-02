import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import BaseAdapter, StandardDoc, StandardSample, StandardQA


# ---------
# 文件名清理
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


# ---------
# Markdown 转换
# ---------

def convert_legal_doc_to_md(title: str, doc: dict) -> str:
    """
    将单篇判决文书转换为结构化 Markdown。
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")

    meta_fields = [
        ("案号", doc.get("number", "")),
        ("法院", doc.get("court", "")),
        ("大类案由", doc.get("case", "")),
        ("子案由", doc.get("sub_case", "")),
        ("文书类型", doc.get("legal_type", "")),
    ]
    for label, value in meta_fields:
        if value:
            lines.append(f"- **{label}**: {value}")

    result = doc.get("result", "").strip()
    if result:
        lines.append(f"- **判决结果**: {result}")

    lines.append("")
    lines.append("## 正文")
    lines.append("")

    content = doc.get("content", "").strip()
    if content:
        lines.append(content)

    return "\n".join(lines) + "\n"


# ----------
# Prompt 模板
# ----------

DEFAULT_INSTRUCTION = "仅根据上述看到的判决文书回答以下问题。"


class LoongLegalAdapter(BaseAdapter):
    """
    Loong 数据集 Legal 部分适配器。

    只处理 loong.jsonl 中 type="legal" 的样本（共 500 条，Level 1~4）。
    判决文书存储在同级目录 doc/legal/legal.json 中（共 629 篇）。

    设计要点：
    1. data_prepare 将全部 629 篇文书转为 Markdown。
    2. load_and_transform 过滤 legal 样本，将答案中的《判决文书X》占位符
       映射为实际标题；Level 4 额外将 判决结果Y 映射为实际文本。
    3. build_prompt 使用样本预存的全部文档内容（all_docs），因为 Loong Legal
       的任务（分类、匹配、多文档定位）需要跨文档理解，不能仅依赖 Top-K 检索块。
    4. evidence 使用每篇引用文书的「子案由 + 判决结果」构造，用于 Recall 评测。
    """

    def __init__(self, raw_file_path: str):
        super().__init__(raw_file_path)
        self._legal_json_path: Optional[str] = None
        self._legal_docs: Optional[Dict[str, dict]] = None
        self._samples: Optional[List[dict]] = None
        self._md_paths: Dict[str, str] = {}       # title -> md_path
        self._md_contents: Dict[str, str] = {}    # title -> md_content (缓存)
        self._resolve_paths()

    def _resolve_paths(self):
        p = Path(self.raw_file_path)
        self._legal_json_path = str(p.parent / "doc" / "legal" / "legal.json")
        if not os.path.exists(self._legal_json_path):
            raise FileNotFoundError(
                f"Legal docs not found: {self._legal_json_path} "
                f"(expected alongside {self.raw_file_path})"
            )
        self.logger.info(f"[LoongLegalAdapter] legal_json={self._legal_json_path}")

    def _load_legal_docs(self) -> Dict[str, dict]:
        if self._legal_docs is None:
            with open(self._legal_json_path, "r", encoding="utf-8") as f:
                self._legal_docs = json.load(f)
        return self._legal_docs

    def _load_samples(self) -> List[dict]:
        if self._samples is None:
            samples = []
            with open(self.raw_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("type") == "legal":
                        samples.append(obj)
            self._samples = samples
            self.logger.info(
                f"[LoongLegalAdapter] loaded {len(samples)} legal samples "
                f"(levels: {sorted(set(s.get('level', 0) for s in samples))})"
            )
        return self._samples

    # ----------
    # 占位符解析工具
    # ----------

    @staticmethod
    def _resolve_placeholders(value: Any, doc_titles: List[str]) -> Any:
        """
        递归将答案中的《判决文书X》替换为实际标题。
        支持 str / dict / list。
        """
        if isinstance(value, str):
            def repl(m):
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(doc_titles):
                    return f"《{doc_titles[idx]}》"
                return m.group(0)
            return re.sub(r'《判决文书(\d+)》', repl, value)
        elif isinstance(value, dict):
            return {
                LoongLegalAdapter._resolve_placeholders(k, doc_titles):
                LoongLegalAdapter._resolve_placeholders(v, doc_titles)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                LoongLegalAdapter._resolve_placeholders(item, doc_titles)
                for item in value
            ]
        return value

    @staticmethod
    def _extract_verdict_results(question: str) -> Dict[str, str]:
        """
        从 Level 4 问题文本中提取 判决结果X -> 实际文本 的映射。
        问题中包含类似 {'判决结果1': '...', ...} 的字典字符串。
        """
        mapping = {}
        # 匹配 '判决结果N': '...内容...'（支持转义序列 \n \' \\
        pattern = r"'判决结果(\d+)':\s*'((?:[^'\\]|\\.)*)'"
        for m in re.finditer(pattern, question):
            key = f"判决结果{m.group(1)}"
            val = (
                m.group(2)
                .replace("\\n", "\n")
                .replace("\\'", "'")
                .replace("\\\\", "\\")
            )
            mapping[key] = val
        return mapping

    @staticmethod
    def _resolve_verdict_results(answer: Any, question: str) -> Any:
        """将 Level 4 答案中的 判决结果Y 替换为实际文本。"""
        if not isinstance(answer, dict):
            return answer
        mapping = LoongLegalAdapter._extract_verdict_results(question)
        new_ans = {}
        for k, v in answer.items():
            if isinstance(v, str) and v.startswith("判决结果") and v in mapping:
                new_ans[k] = mapping[v]
            else:
                new_ans[k] = v
        return new_ans

    # ----------
    # data_prepare
    # ----------

    def data_prepare(self, doc_dir: str) -> List[StandardDoc]:
        os.makedirs(doc_dir, exist_ok=True)
        legal_docs = self._load_legal_docs()
        self._md_paths = {}
        self._md_contents = {}

        for title, doc in legal_docs.items():
            safe_name = sanitize_filename(title) + ".md"
            md_path = os.path.join(doc_dir, safe_name)

            md_content = convert_legal_doc_to_md(title, doc)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            self._md_paths[title] = md_path
            self._md_contents[title] = md_content

        res = [
            StandardDoc(sample_id=title, doc_paths=[md_path])
            for title, md_path in self._md_paths.items()
        ]
        self.logger.info(
            f"[LoongLegalAdapter] data_prepare: {len(res)} documents written to {doc_dir}"
        )
        return res

    # ----------
    # load_and_transform
    # ----------

    def load_and_transform(self) -> List[StandardSample]:
        samples = self._load_samples()
        legal_docs = self._load_legal_docs()
        # 仅保留 Level 1（单一文档定位），丢弃 Level 2/3/4
        samples = [s for s in samples if s.get("level") == 1]
        result: List[StandardSample] = []

        for s in samples:
            doc_titles = s.get("doc", [])
            level = s.get("level", 0)
            question = s.get("question", "")
            instruction = s.get("instruction", DEFAULT_INSTRUCTION)
            answer_raw = s.get("answer", "")
            sample_id = s.get("id", "")

            # 收集引用的文档路径、内容、evidence
            doc_paths = []
            doc_contents = []
            evidence_items = []

            for idx, title in enumerate(doc_titles):
                md_path = self._md_paths.get(title)
                if not md_path:
                    self.logger.warning(f"Document not prepared: {title}")
                    continue
                doc_paths.append(md_path)

                # 优先使用缓存的 md 内容，避免重复转换
                md_content = self._md_contents.get(title)
                if md_content is None:
                    doc_info = legal_docs.get(title, {})
                    md_content = convert_legal_doc_to_md(title, doc_info)
                    self._md_contents[title] = md_content

                # 为 prompt 添加占位符标记，方便模型理解引用
                doc_with_ref = f"《判决文书{idx + 1}》\n\n{md_content}"
                doc_contents.append(doc_with_ref)

                # evidence 用于 Recall 评测
                doc_info = legal_docs.get(title, {})
                sub_case = doc_info.get("sub_case", "")
                result_text = doc_info.get("result", "")
                evp = f"文书《{title}》"
                if sub_case:
                    evp += f"，案由={sub_case}"
                if result_text:
                    evp += f"，判决结果={result_text}"
                evidence_items.append(evp)

            # 处理答案
            answer_resolved = self._resolve_placeholders(answer_raw, doc_titles)
            if level == 4:
                answer_resolved = self._resolve_verdict_results(answer_resolved, question)

            # 序列化 gold_answers（统一存入 List[str]）
            if isinstance(answer_resolved, (dict, list)):
                gold_answers = [json.dumps(answer_resolved, ensure_ascii=False)]
            else:
                gold_answers = [str(answer_resolved)]

            qa = StandardQA(
                question=question,
                gold_answers=gold_answers,
                evidence=evidence_items,
                category=level,
                metadata={
                    "level": level,
                    "set": s.get("set", 0),
                    "instruction": instruction,
                    "doc_titles": doc_titles,
                    "all_docs": doc_contents,
                    "id": sample_id,
                },
            )

            result.append(StandardSample(
                sample_id=sample_id,
                qa_pairs=[qa],
                metadata={"doc_paths": doc_paths},
            ))

        self.logger.info(
            f"[LoongLegalAdapter] load_and_transform: {len(result)} samples"
        )
        return result

    # ----------
    # build_prompt
    # ----------

    def build_prompt(
        self, qa: StandardQA, context_blocks: List[str]
    ) -> tuple[str, Dict[str, Any]]:
        """
        按数据集原生的 prompt_template 格式构造最终 Prompt：
            {docs}\n\n{instruction}\n\n{question}

        由于 Loong Legal 的任务（分类、匹配、多文档定位）需要跨文档理解，
        这里使用样本中预存的全部文档内容（all_docs），而非仅 Top-K 检索块。
        """
        all_docs = qa.metadata.get("all_docs", [])
        if all_docs:
            docs_text = "\n\n---\n\n".join(all_docs)
        else:
            # 兜底：使用检索结果（理论上不会发生）
            docs_text = "\n\n---\n\n".join(context_blocks)

        instruction = qa.metadata.get("instruction", DEFAULT_INSTRUCTION)
        question = qa.question

        full_prompt = f"{docs_text}\n\n{instruction}\n\n{question}"
        meta = {"level": qa.metadata.get("level"), "set": qa.metadata.get("set")}
        return full_prompt, meta

    # ----------
    # post_process_answer
    # ----------

    def post_process_answer(
        self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]
    ) -> str:
        """
        清洗模型输出：去除首尾空白，尝试提取 Markdown 代码块内容。
        """
        ans = raw_answer.strip()
        # 若被 ```json / ``` 包裹，提取内部内容
        if ans.startswith("```"):
            m = re.search(r"```(?:\w+)?\n?(.*?)```", ans, re.DOTALL)
            if m:
                ans = m.group(1).strip()
        return ans
