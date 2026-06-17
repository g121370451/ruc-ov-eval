import json
import math
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.adapters.base import StandardDoc
from src.core.logger import get_logger


VIEW_PROMPT = """You are helping reproduce MC-indexing for long-document retrieval.
For the section below, produce two views:
1. "summary": no more than 10 sentences or 200 words.
2. "keywords": an array of 8-20 concepts, entities, and important descriptions useful for retrieval.

Return JSON only with keys "summary" and "keywords".

Section Name: {title}
Section Text:
{text}
"""


@dataclass
class MCSection:
    section_id: str
    doc_id: str
    sample_id: str
    title: str
    text: str
    order: int
    source_path: str


@dataclass
class MCView:
    section_id: str
    doc_id: str
    sample_id: str
    title: str
    raw_text: str
    summary: str
    keywords: List[str]
    generation_method: str


@dataclass
class MCChunk:
    chunk_id: str
    doc_id: str
    sample_id: str
    method: str
    title: str
    text: str
    source_sections: List[str]
    return_text: str


@dataclass
class MCResource:
    uri: str
    content: str = ""
    level: int = 2
    score: float = 0.0
    abstract: str = ""
    overview: str = ""
    source_view: str = ""


@dataclass
class MCSearchResult:
    resources: List[MCResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def truncate(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def lexical_tokens(text: str) -> List[str]:
    lowered = str(text).lower()
    tokens = re.findall(r"[a-z][a-z0-9.+_-]{1,}", lowered)
    for han_run in re.findall(r"[\u4e00-\u9fff]{2,}", str(text)):
        tokens.extend(han_run[idx : idx + 2] for idx in range(len(han_run) - 1))
        if len(han_run) <= 8:
            tokens.append(han_run)
    return tokens


def sentence_split(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", str(text))
    return [normalize_space(piece) for piece in pieces if normalize_space(piece)]


def fallback_summary(text: str) -> str:
    return truncate(" ".join(sentence_split(text)[:4]) or text, 900)


def fallback_keywords(title: str, text: str, limit: int = 16) -> List[str]:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.+_-]{1,}|[\u4e00-\u9fff]{2,}", f"{title} {text}")
        if len(token) > 1
    ]
    counts = Counter(tokens)
    keywords = []
    for token, _count in counts.most_common(50):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class SimpleBM25Retriever:
    name = "BM25"

    def fit(self, chunks: List[MCChunk]) -> None:
        self.chunks = chunks
        self.tokenized = [lexical_tokens(chunk.text) for chunk in chunks]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized:
            self.doc_freq.update(set(tokens))
        self.avgdl = sum(len(tokens) for tokens in self.tokenized) / max(len(self.tokenized), 1)
        self.n_docs = len(self.tokenized)
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, top_k: int) -> List[Tuple[MCChunk, float]]:
        if not self.chunks:
            return []
        query_terms = lexical_tokens(query)
        scores = []
        for idx, tokens in enumerate(self.tokenized):
            freqs = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                tf = freqs.get(term, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
            scores.append(score)
        order = np.argsort(np.asarray(scores))[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in order]


class SimpleTfidfRetriever:
    def __init__(self, name: str = "TFIDF") -> None:
        self.name = name

    def fit(self, chunks: List[MCChunk]) -> None:
        self.chunks = chunks
        self.vocab: Dict[str, int] = {}
        doc_tokens = [lexical_tokens(chunk.text) for chunk in chunks]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        for term in sorted(df):
            self.vocab[term] = len(self.vocab)
        matrix = np.zeros((len(chunks), max(len(self.vocab), 1)), dtype=np.float32)
        n_docs = max(len(chunks), 1)
        for row, tokens in enumerate(doc_tokens):
            counts = Counter(tokens)
            for term, count in counts.items():
                col = self.vocab.get(term)
                if col is None:
                    continue
                idf = math.log((1 + n_docs) / (1 + df[term])) + 1
                matrix[row, col] = float(count) * idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    def search(self, query: str, top_k: int) -> List[Tuple[MCChunk, float]]:
        if not self.chunks:
            return []
        query_vec = np.zeros((max(len(self.vocab), 1),), dtype=np.float32)
        for term, count in Counter(lexical_tokens(query)).items():
            col = self.vocab.get(term)
            if col is not None:
                query_vec[col] = float(count)
        norm = np.linalg.norm(query_vec)
        if norm:
            query_vec = query_vec / norm
        scores = self.matrix @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in order]


class SentenceTransformerRetriever:
    def __init__(self, name: str, model_name: str) -> None:
        self.name = name
        self.model_name = model_name
        self.is_e5 = "e5" in model_name.lower()

    def fit(self, chunks: List[MCChunk]) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                f"{self.name} requires sentence-transformers. "
                "Install it or set allow_dense_fallback: true in mc_indexing_config."
            ) from exc
        self.chunks = chunks
        self.model = SentenceTransformer(self.model_name)
        passages = [f"passage: {chunk.text}" if self.is_e5 else chunk.text for chunk in chunks]
        self.matrix = np.asarray(
            self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def search(self, query: str, top_k: int) -> List[Tuple[MCChunk, float]]:
        if not self.chunks:
            return []
        encoded_query = f"query: {query}" if self.is_e5 else query
        query_vec = np.asarray(
            self.model.encode([encoded_query], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )
        scores = self.matrix @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in order]


class DoubaoEmbeddingRetriever:
    name = "DoubaoVision"

    def __init__(self, store_path: Path, method: str) -> None:
        self.store_path = store_path
        self.method = method
        self.model_name = os.environ.get("EMBEDDING_MODEL_NAME", "")
        if not self.model_name:
            raise RuntimeError("EMBEDDING_MODEL_NAME is required for Doubao retriever.")

    def _cache_path(self, kind: str) -> Path:
        safe_model = self.model_name.replace("/", "_").replace(":", "_")
        safe_method = self.method.replace("/", "_").replace(" ", "_")
        return self.store_path / "embedding_cache" / kind / f"{safe_method}__{safe_model}.jsonl"

    def _load_cache(self, path: Path) -> Dict[str, List[float]]:
        return {row["id"]: row["embedding"] for row in read_jsonl(path)}

    def _save_cache(self, path: Path, cache: Dict[str, List[float]]) -> None:
        rows = [{"id": key, "embedding": value} for key, value in sorted(cache.items())]
        write_jsonl(path, rows)

    def _embed_cached(self, cache: Dict[str, List[float]], cache_path: Path, key: str, text: str) -> List[float]:
        if key not in cache:
            cache[key] = self._embed_text(truncate(text, 12000))
            self._save_cache(cache_path, cache)
        return cache[key]

    @staticmethod
    def _embed_text(text: str) -> List[float]:
        api_key = os.environ.get("EMBEDDING_API_KEY", "")
        base_url = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
        model = os.environ.get("EMBEDDING_MODEL_NAME", "")
        if not api_key or not base_url or not model:
            raise RuntimeError("EMBEDDING_API_KEY, EMBEDDING_BASE_URL, and EMBEDDING_MODEL_NAME are required.")
        url = base_url if base_url.endswith("/embeddings/multimodal") else f"{base_url}/embeddings/multimodal"
        payload = {
            "model": model,
            "encoding_format": "float",
            "input": [{"type": "text", "text": text}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        last_error = None
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                embedding = data.get("data", {}).get("embedding")
                if embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]
                if not isinstance(embedding, list):
                    raise RuntimeError(f"Unexpected embedding response shape: {list(data)}")
                return [float(value) for value in embedding]
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                if exc.code < 500 and exc.code != 429:
                    raise last_error from None
            except Exception as exc:
                last_error = RuntimeError(str(exc))
            time.sleep(min(2**attempt, 20))
        raise last_error or RuntimeError("Embedding request failed.")

    def fit(self, chunks: List[MCChunk]) -> None:
        self.chunks = chunks
        cache_path = self._cache_path("chunks")
        self.chunk_cache = self._load_cache(cache_path)
        vectors = [
            self._embed_cached(self.chunk_cache, cache_path, chunk.chunk_id, chunk.text)
            for chunk in chunks
        ]
        matrix = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms
        self.query_cache_path = self._cache_path("queries")
        self.query_cache = self._load_cache(self.query_cache_path)

    def search(self, query: str, top_k: int) -> List[Tuple[MCChunk, float]]:
        if not self.chunks:
            return []
        query_vec = np.asarray(
            self._embed_cached(self.query_cache, self.query_cache_path, query, query),
            dtype=np.float32,
        )
        norm = np.linalg.norm(query_vec)
        if norm:
            query_vec = query_vec / norm
        scores = self.matrix @ query_vec
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(idx)], float(scores[int(idx)])) for idx in order]


class MCIndexingStoreWrapper:
    """MC-indexing retrieval backend aligned with ruc-ov-eval StoreWrapper API."""

    def __init__(self, store_path: str, doc_output_dir: str = "", mc_indexing_config: Optional[dict] = None):
        self.store_path = Path(store_path)
        self.doc_output_dir = doc_output_dir
        self.config = mc_indexing_config or {}
        self.logger = get_logger()
        self.method = self.config.get("method", "MC-indexing")
        self.retriever_name = self.config.get("retriever", "bm25")
        self.generate_views = bool(self.config.get("generate_views", True))
        self.cache_views = bool(self.config.get("cache_views", True))
        self.view_workers = int(self.config.get("view_workers", 1) or 1)
        self.max_view_input_chars = int(self.config.get("max_view_input_chars", 12000))
        self.max_context_chars = int(self.config.get("max_context_chars", 12000))
        self.context_block_chars = int(self.config.get("context_block_chars", 2000))
        self.return_source_text = bool(self.config.get("return_source_text", True))
        self.allow_dense_fallback = bool(self.config.get("allow_dense_fallback", False))
        self.llm_json_mode = bool(self.config.get("llm_json_mode", True))
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._token_lock = threading.Lock()
        self.view_input_tokens = 0
        self.view_output_tokens = 0

        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            self.logger.warning(f"tiktoken init failed, token counting will return 0: {exc}")
            self.enc = None

        self.sections: List[MCSection] = []
        self.views: List[MCView] = []
        self.chunks_by_method: Dict[str, List[MCChunk]] = {}
        self.retrievers: Dict[str, object] = {}
        self._load_index_if_present()

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 10, monitor=None) -> dict:
        start = time.time()
        self.view_input_tokens = 0
        self.view_output_tokens = 0
        self.sections = self._parse_documents(samples)
        self.views = self._generate_or_load_views(self.sections)
        self.chunks_by_method = self._build_all_chunks(self.sections, self.views)
        self._build_retrievers()
        self._save_index()
        return {
            "time": time.time() - start,
            "input_tokens": self.view_input_tokens,
            "output_tokens": self.view_output_tokens,
        }

    def retrieve(self, query: str, topk: int = 10, target_uri: Optional[str] = None) -> MCSearchResult:
        if not self.retrievers:
            self._load_index_if_present()
            self._build_retrievers()
        resources = self._search(query, topk or 10)
        return MCSearchResult(resources=resources, retrieve_input_tokens=self.count_tokens(query), retrieve_output_tokens=0)

    def process_retrieval_results(self, search_res: MCSearchResult):
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []
        remaining_context_chars = self.max_context_chars if self.max_context_chars > 0 else None
        for resource in search_res.resources:
            retrieved_uris.append(resource.uri)
            retrieved_texts.append(resource.content)
            block_limit = self.context_block_chars
            if remaining_context_chars is not None:
                if remaining_context_chars <= 0:
                    continue
                block_limit = min(block_limit, remaining_context_chars)
            if block_limit > 0:
                block = truncate(resource.content, block_limit)
                context_blocks.append(block)
                if remaining_context_chars is not None:
                    remaining_context_chars -= len(block)
        return retrieved_texts, context_blocks, retrieved_uris

    def clear(self):
        if self.store_path.exists():
            shutil.rmtree(self.store_path)
        self.sections = []
        self.views = []
        self.chunks_by_method = {}
        self.retrievers = {}

    def _parse_documents(self, samples: List[StandardDoc]) -> List[MCSection]:
        sections = []
        order = 0
        for sample in samples:
            for doc_path in sample.doc_paths:
                path = Path(doc_path)
                if not path.exists():
                    self.logger.warning(f"MC-indexing skipped missing doc: {doc_path}")
                    continue
                raw = path.read_text(encoding="utf-8", errors="replace")
                doc_id = path.stem
                parsed = self._markdown_sections(raw, sample.sample_id, doc_id, str(path))
                for section in parsed:
                    order += 1
                    section.order = order
                    sections.append(section)
        self.logger.info(f"MC-indexing parsed {len(sections)} sections")
        return sections

    @staticmethod
    def _markdown_sections(raw: str, sample_id: str, doc_id: str, source_path: str) -> List[MCSection]:
        sections = []
        current_title = doc_id
        current_lines = []
        local_order = 0

        def finish():
            nonlocal local_order, current_lines
            text = normalize_space("\n".join(current_lines))
            if text:
                local_order += 1
                section_id = f"{sample_id}::{doc_id}::s{local_order:04d}"
                sections.append(
                    MCSection(
                        section_id=section_id,
                        doc_id=doc_id,
                        sample_id=sample_id,
                        title=current_title,
                        text=f"{current_title}. {text}",
                        order=local_order,
                        source_path=source_path,
                    )
                )
            current_lines = []

        for raw_line in raw.splitlines():
            line = raw_line.rstrip()
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading:
                finish()
                current_title = heading.group(2).strip()
            else:
                current_lines.append(line)
        finish()
        if not sections and normalize_space(raw):
            sections.append(
                MCSection(
                    section_id=f"{sample_id}::{doc_id}::s0001",
                    doc_id=doc_id,
                    sample_id=sample_id,
                    title=doc_id,
                    text=normalize_space(raw),
                    order=1,
                    source_path=source_path,
                )
            )
        return sections

    def _generate_or_load_views(self, sections: List[MCSection]) -> List[MCView]:
        views_path = self.store_path / "views.jsonl"
        existing = {}
        if self.cache_views and views_path.exists():
            existing = {row["section_id"]: MCView(**row) for row in read_jsonl(views_path)}

        rows = [existing[section.section_id] for section in sections if section.section_id in existing]
        todo = [section for section in sections if section.section_id not in existing]
        if not todo:
            return rows

        use_llm = self.generate_views and bool(os.environ.get("LLM_API_KEY")) and bool(os.environ.get("LLM_BASE_URL"))
        model = self.config.get("llm_model") or os.environ.get("LLM_MODEL", "")
        if self.generate_views and not use_llm:
            self.logger.warning("LLM env not fully set; MC-indexing will generate fallback summary/keywords.")

        def build(section: MCSection) -> MCView:
            return self._generate_view(section, model=model, use_llm=use_llm)

        if self.view_workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=self.view_workers) as executor:
                futures = {executor.submit(build, section): section for section in todo}
                for future in as_completed(futures):
                    rows.append(future.result())
        else:
            for section in todo:
                rows.append(build(section))

        rows.sort(key=lambda view: (view.sample_id, view.doc_id, view.section_id))
        if self.cache_views:
            write_jsonl(views_path, [asdict(row) for row in rows])
        return rows

    def _generate_view(self, section: MCSection, model: str, use_llm: bool) -> MCView:
        text = truncate(section.text, self.max_view_input_chars)
        if use_llm:
            try:
                prompt = VIEW_PROMPT.format(title=section.title, text=text)
                obj = self._call_llm_json(prompt, model=model)
                summary = str(obj.get("summary", "")).strip() or fallback_summary(text)
                raw_keywords = obj.get("keywords", [])
                keywords = [str(item).strip() for item in raw_keywords if str(item).strip()]
                if not keywords:
                    keywords = fallback_keywords(section.title, text)
                with self._token_lock:
                    self.view_input_tokens += self.count_tokens(prompt)
                    self.view_output_tokens += self.count_tokens(
                        json.dumps({"summary": summary, "keywords": keywords}, ensure_ascii=False)
                    )
                method = f"llm:chat:{model}"
            except Exception as exc:
                self.logger.warning(f"View generation failed for {section.section_id}: {exc}; using fallback views.")
                summary = fallback_summary(text)
                keywords = fallback_keywords(section.title, text)
                method = "extractive-fallback:llm-error"
        else:
            summary = fallback_summary(text)
            keywords = fallback_keywords(section.title, text)
            method = "extractive-fallback:no-llm-key"

        return MCView(
            section_id=section.section_id,
            doc_id=section.doc_id,
            sample_id=section.sample_id,
            title=section.title,
            raw_text=f"{section.title}. {text}",
            summary=summary,
            keywords=keywords,
            generation_method=method,
        )

    def _call_llm_json(self, prompt: str, model: str) -> dict:
        from openai import OpenAI

        if not model:
            raise RuntimeError("LLM_MODEL is required for MC-indexing view generation.")
        client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only. Do not include markdown fences."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(self.config.get("view_max_tokens", 600)),
        }
        if self.llm_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or "{}"
        if text.strip().startswith("```"):
            text = text.strip().strip("`")
            text = text.split("\n", 1)[-1]
        return json.loads(text)

    def _build_all_chunks(self, sections: List[MCSection], views: List[MCView]) -> Dict[str, List[MCChunk]]:
        by_section = {section.section_id: section for section in sections}
        by_view = {view.section_id: view for view in views}
        chunks = {
            "Content raw-text": self._content_chunks(sections, "Content raw-text", "text", by_section),
            "Content summary": self._view_chunks(views, "Content summary", "summary", by_section),
            "Content keyword": self._view_chunks(views, "Content keyword", "keywords", by_section),
        }
        for length in self._flc_lengths():
            chunks[f"FLC-{length}"] = self._flc_chunks(sections, length)
        chunks["MC raw-text"] = self._content_chunks(sections, "MC raw-text", "text", by_section)
        chunks["MC summary"] = self._view_chunks(views, "MC summary", "summary", by_section)
        chunks["MC keyword"] = self._view_chunks(views, "MC keyword", "keywords", by_section)
        return chunks

    def _flc_lengths(self) -> List[int]:
        raw = self.config.get("flc_lengths", [100, 200, 300])
        if isinstance(raw, str):
            return [int(item.strip()) for item in raw.split(",") if item.strip()]
        return [int(item) for item in raw]

    @staticmethod
    def _content_chunks(sections: List[MCSection], method: str, text_key: str, by_section: Dict[str, MCSection]) -> List[MCChunk]:
        chunks = []
        for section in sections:
            return_text = section.text
            chunks.append(
                MCChunk(
                    chunk_id=f"{section.section_id}::{method}",
                    doc_id=section.doc_id,
                    sample_id=section.sample_id,
                    method=method,
                    title=section.title,
                    text=section.text,
                    source_sections=[section.section_id],
                    return_text=return_text,
                )
            )
        return chunks

    @staticmethod
    def _view_chunks(views: List[MCView], method: str, view_key: str, by_section: Dict[str, MCSection]) -> List[MCChunk]:
        chunks = []
        for view in views:
            value = getattr(view, view_key)
            text = " ".join(str(item) for item in value) if isinstance(value, list) else str(value)
            source = by_section.get(view.section_id)
            return_text = source.text if source else view.raw_text
            chunks.append(
                MCChunk(
                    chunk_id=f"{view.section_id}::{method}",
                    doc_id=view.doc_id,
                    sample_id=view.sample_id,
                    method=method,
                    title=view.title,
                    text=text,
                    source_sections=[view.section_id],
                    return_text=return_text,
                )
            )
        return chunks

    @staticmethod
    def _flc_chunks(sections: List[MCSection], length: int) -> List[MCChunk]:
        chunks = []
        by_doc: Dict[str, List[MCSection]] = {}
        for section in sections:
            by_doc.setdefault(section.doc_id, []).append(section)
        for doc_id, doc_sections in by_doc.items():
            ordered = sorted(doc_sections, key=lambda row: row.order)
            units = []
            source_sections = []
            sample_id = ordered[0].sample_id if ordered else doc_id
            for section in ordered:
                section_units = re.findall(r"[A-Za-z0-9.+_-]+|[\u4e00-\u9fff]", section.text)
                units.extend(section_units)
                source_sections.extend([section.section_id] * len(section_units))
            for start in range(0, len(units), length):
                end = min(start + length, len(units))
                if start >= end:
                    continue
                chunk_index = start // length + 1
                text = " ".join(units[start:end])
                chunks.append(
                    MCChunk(
                        chunk_id=f"{sample_id}::{doc_id}::flc{length}-{chunk_index:04d}",
                        doc_id=doc_id,
                        sample_id=sample_id,
                        method=f"FLC-{length}",
                        title=f"{doc_id} FLC-{length} chunk {chunk_index}",
                        text=text,
                        source_sections=sorted(set(source_sections[start:end])),
                        return_text=text,
                    )
                )
        return chunks

    def _make_retriever(self, method: str):
        normalized = str(self.retriever_name).lower()
        if normalized == "bm25":
            return SimpleBM25Retriever()
        if normalized in {"tfidf", "tf-idf"}:
            return SimpleTfidfRetriever("TFIDF")
        if normalized == "e5":
            if self.allow_dense_fallback:
                return SimpleTfidfRetriever("E5-TFIDF-FALLBACK")
            return SentenceTransformerRetriever("E5", self.config.get("e5_model", "intfloat/e5-small-v2"))
        if normalized == "bge":
            if self.allow_dense_fallback:
                return SimpleTfidfRetriever("BGE-TFIDF-FALLBACK")
            return SentenceTransformerRetriever("BGE", self.config.get("bge_model", "BAAI/bge-small-en-v1.5"))
        if normalized in {"doubao", "doubao-vision", "doubaovision"}:
            return DoubaoEmbeddingRetriever(self.store_path, method=method)
        raise ValueError(f"Unknown MC-indexing retriever: {self.retriever_name}")

    def _build_retrievers(self) -> None:
        self.retrievers = {}
        methods = ["MC raw-text", "MC summary", "MC keyword"] if self.method == "MC-indexing" else [self.method]
        for method in methods:
            chunks = self.chunks_by_method.get(method, [])
            retriever = self._make_retriever(method)
            retriever.fit(chunks)
            self.retrievers[method] = retriever
        self.logger.info(f"MC-indexing built retrievers for methods: {', '.join(self.retrievers)}")

    def _search(self, query: str, topk: int) -> List[MCResource]:
        if self.method == "MC-indexing":
            combined: Dict[Tuple[str, ...], Tuple[MCChunk, float]] = {}
            for method, retriever in self.retrievers.items():
                for rank, (chunk, score) in enumerate(retriever.search(query, topk), start=1):
                    key = tuple(chunk.source_sections)
                    weighted = float(score) / rank
                    if key not in combined:
                        combined[key] = (chunk, 0.0)
                    combined[key] = (combined[key][0], combined[key][1] + weighted)
            ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)[:topk]
            return [self._resource_from_chunk(chunk, score) for chunk, score in ranked]

        retriever = self.retrievers.get(self.method)
        if retriever is None:
            raise RuntimeError(f"MC-indexing method not indexed: {self.method}")
        return [self._resource_from_chunk(chunk, score) for chunk, score in retriever.search(query, topk)]

    def _resource_from_chunk(self, chunk: MCChunk, score: float) -> MCResource:
        content = chunk.return_text if self.return_source_text else chunk.text
        return MCResource(
            uri=f"mc-indexing://{chunk.chunk_id}",
            content=content,
            level=2,
            score=round(float(score), 6),
            abstract=truncate(content, 300),
            overview=truncate(content, 700),
            source_view=chunk.method,
        )

    def _save_index(self) -> None:
        write_jsonl(self.store_path / "sections.jsonl", [asdict(row) for row in self.sections])
        write_jsonl(self.store_path / "views.jsonl", [asdict(row) for row in self.views])
        all_chunks = []
        for chunks in self.chunks_by_method.values():
            all_chunks.extend(asdict(chunk) for chunk in chunks)
        write_jsonl(self.store_path / "chunks.jsonl", all_chunks)
        meta = {
            "method": self.method,
            "retriever": self.retriever_name,
            "return_source_text": self.return_source_text,
        }
        (self.store_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index_if_present(self) -> None:
        sections_path = self.store_path / "sections.jsonl"
        chunks_path = self.store_path / "chunks.jsonl"
        if not sections_path.exists() or not chunks_path.exists():
            return
        self.sections = [MCSection(**row) for row in read_jsonl(sections_path)]
        views_path = self.store_path / "views.jsonl"
        self.views = [MCView(**row) for row in read_jsonl(views_path)] if views_path.exists() else []
        chunks_by_method: Dict[str, List[MCChunk]] = {}
        for row in read_jsonl(chunks_path):
            chunk = MCChunk(**row)
            chunks_by_method.setdefault(chunk.method, []).append(chunk)
        self.chunks_by_method = chunks_by_method
