import json
import hashlib
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


PROMPT_VERSION = "mcindex-paper-figure8-9-v1"


SUMMARY_PROMPT = """You are a helpful summarization assistant. Please help me summarize the following section into no more than 10 sentences or 200 words.
**Section Name**:
{title}
**Section Text**:
{text}
"""


KEYWORD_PROMPT = """You are a helpful keyword extractor. You need to extract keywords from the following section. The keywords should consist of concepts, entities, or important descriptions that are related to the section text, which could be used to answer any questions from users.
**Section Name**:
{title}
**Section Text**:
**Beginning of text**
{text}
**End of text**
Please output format in list format: [...]. Do not output anything else aside from this list.
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
    content_hash: str = ""
    view_config_hash: str = ""


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
    raw = str(text)
    try:
        from nltk.tokenize import sent_tokenize

        pieces = sent_tokenize(raw)
    except Exception:
        pieces = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", raw)
    return [normalize_space(piece) for piece in pieces if normalize_space(piece)]


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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


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
        self.cache_views = bool(self.config.get("cache_views", True))
        self.view_workers = int(self.config.get("view_workers", 1) or 1)
        self.summary_min_tokens = int(self.config.get("summary_min_tokens", 200))
        self.max_context_chars = int(self.config.get("max_context_chars", 12000))
        self.context_block_chars = int(self.config.get("context_block_chars", 2000))
        self.return_source_text = bool(self.config.get("return_source_text", True))
        self.allow_dense_fallback = bool(self.config.get("allow_dense_fallback", False))
        self.llm_json_mode = bool(self.config.get("llm_json_mode", True))
        self.pdf_pages_as_sections = bool(self.config.get("pdf_pages_as_sections", True))
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._token_lock = threading.Lock()
        self.view_input_tokens = 0
        self.view_output_tokens = 0
        self.view_api_input_tokens = 0
        self.view_api_output_tokens = 0
        self.view_estimated_input_tokens = 0
        self.view_estimated_output_tokens = 0

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

    def _section_token_count(self, text: str) -> int:
        counted = self.count_tokens(text)
        return counted if counted else len(lexical_tokens(text))

    def ingest(self, samples: List[StandardDoc], max_workers: int = 10, monitor=None) -> dict:
        start = time.time()
        self.view_input_tokens = 0
        self.view_output_tokens = 0
        self.view_api_input_tokens = 0
        self.view_api_output_tokens = 0
        self.view_estimated_input_tokens = 0
        self.view_estimated_output_tokens = 0
        parse_start = time.time()
        doc_count = sum(len(sample.doc_paths) for sample in samples)
        self.logger.info(
            f"MC-indexing ingest started: samples={len(samples)}, doc_paths={doc_count}, "
            f"method={self.method}, retriever={self.retriever_name}, "
            f"view_workers={self.view_workers}, cache_views={self.cache_views}"
        )
        self.sections = self._parse_documents(samples)
        parse_time = time.time() - parse_start
        self.logger.info("MC-indexing stage: generate/load views")
        view_start = time.time()
        self.views = self._generate_or_load_views(self.sections)
        view_time = time.time() - view_start
        self.logger.info(f"MC-indexing views ready: {len(self.views)}")
        self.logger.info("MC-indexing stage: build chunks")
        chunk_start = time.time()
        self.chunks_by_method = self._build_all_chunks(self.sections, self.views)
        chunk_time = time.time() - chunk_start
        chunk_summary = ", ".join(
            f"{method}={len(chunks)}" for method, chunks in sorted(self.chunks_by_method.items())
        )
        self.logger.info(f"MC-indexing chunks ready: {chunk_summary}")
        self.logger.info("MC-indexing stage: build retrievers")
        retriever_start = time.time()
        self._build_retrievers()
        retriever_time = time.time() - retriever_start
        self.logger.info("MC-indexing stage: save index")
        save_start = time.time()
        self._save_index()
        save_time = time.time() - save_start
        elapsed = time.time() - start
        self.logger.info(
            f"MC-indexing ingest finished in {elapsed:.2f}s; "
            f"view_input_tokens={self.view_input_tokens}, view_output_tokens={self.view_output_tokens}, "
            f"api_input_tokens={self.view_api_input_tokens}, api_output_tokens={self.view_api_output_tokens}, "
            f"estimated_input_tokens={self.view_estimated_input_tokens}, "
            f"estimated_output_tokens={self.view_estimated_output_tokens}"
        )
        return {
            "time": elapsed,
            "input_tokens": self.view_input_tokens,
            "output_tokens": self.view_output_tokens,
            "api_input_tokens": self.view_api_input_tokens,
            "api_output_tokens": self.view_api_output_tokens,
            "estimated_input_tokens": self.view_estimated_input_tokens,
            "estimated_output_tokens": self.view_estimated_output_tokens,
            "stage_times": {
                "parse_documents": parse_time,
                "generate_views": view_time,
                "build_chunks": chunk_time,
                "build_retrievers": retriever_time,
                "save_index": save_time,
            },
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
        seen_paths = set()
        for sample in samples:
            for doc_path in sample.doc_paths:
                path = Path(doc_path)
                if not path.exists():
                    self.logger.warning(f"MC-indexing skipped missing doc: {doc_path}")
                    continue
                resolved_path = str(path.resolve())
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                doc_id = path.stem
                if path.suffix.lower() == ".pdf":
                    parsed = self._pdf_sections(path, sample.sample_id, doc_id)
                else:
                    raw = path.read_text(encoding="utf-8", errors="replace")
                    parsed = self._markdown_sections(raw, sample.sample_id, doc_id, str(path))
                self.logger.info(
                    f"MC-indexing parsed document: {path.name}, sections={len(parsed)}, "
                    f"cumulative_sections={len(sections) + len(parsed)}"
                )
                for section in parsed:
                    order += 1
                    section.order = order
                    sections.append(section)
        self.logger.info(f"MC-indexing parsed {len(sections)} sections")
        return sections

    def _pdf_sections(self, path: Path, sample_id: str, doc_id: str) -> List[MCSection]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError("MC-indexing PDF input requires pdfplumber.") from exc

        sections = []
        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = normalize_space(page.extract_text() or "")
                if not text:
                    continue
                title = f"{doc_id} page {page_index}"
                section_id = f"{sample_id}::{doc_id}::p{page_index:04d}"
                sections.append(
                    MCSection(
                        section_id=section_id,
                        doc_id=doc_id,
                        sample_id=sample_id,
                        title=title,
                        text=f"{title}. {text}" if self.pdf_pages_as_sections else text,
                        order=page_index,
                        source_path=str(path),
                    )
                )
        if sections:
            return sections

        self.logger.warning(f"MC-indexing extracted no text from PDF: {path}")
        return []

    @staticmethod
    def _markdown_sections(raw: str, sample_id: str, doc_id: str, source_path: str) -> List[MCSection]:
        sections = []
        current_title = doc_id
        current_lines = []
        title_stack: List[str] = []
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
                level = len(heading.group(1))
                title = heading.group(2).strip()
                del title_stack[level - 1 :]
                title_stack.append(title)
                current_title = " > ".join(title_stack) or doc_id
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
        model = self.config.get("llm_model") or os.environ.get("LLM_MODEL", "")
        self._require_llm(model)
        view_config_hash = self._view_config_hash(model=model)
        existing = {}
        if self.cache_views and views_path.exists():
            for row in read_jsonl(views_path):
                view = MCView(**row)
                if view.view_config_hash == view_config_hash:
                    existing[view.section_id] = view

        rows = []
        todo = []
        for section in sections:
            cached = existing.get(section.section_id)
            if cached and cached.content_hash == self._section_hash(section):
                rows.append(cached)
            else:
                todo.append(section)
        estimated_summary_calls = sum(
            1 for section in todo
            if self._section_token_count(normalize_space(section.text)) > self.summary_min_tokens
        )
        estimated_llm_calls = len(todo) + estimated_summary_calls
        self.logger.info(
            f"MC-indexing view cache: total_sections={len(sections)}, cached={len(rows)}, "
            f"todo={len(todo)}, estimated_summary_calls={estimated_summary_calls}, "
            f"estimated_keyword_calls={len(todo)}, estimated_llm_calls={estimated_llm_calls}, "
            f"cache_path={views_path}"
        )
        if not todo:
            self.logger.info("MC-indexing view generation skipped: all views loaded from cache")
            return rows

        def build(section: MCSection) -> MCView:
            return self._generate_view(section, model=model, view_config_hash=view_config_hash)

        progress_every = max(1, int(self.config.get("view_log_every", 25) or 25))
        progress_lock = threading.Lock()
        completed = 0

        def record_progress(view: MCView) -> None:
            nonlocal completed
            rows.append(view)
            completed += 1
            if self.cache_views:
                append_jsonl(views_path, asdict(view))
            if completed == 1 or completed == len(todo) or completed % progress_every == 0:
                self.logger.info(
                    f"MC-indexing view generation progress: {completed}/{len(todo)} "
                    f"latest={view.section_id}, input_tokens={self.view_input_tokens}, "
                    f"output_tokens={self.view_output_tokens}"
                )

        self.logger.info(
            f"MC-indexing view generation started: todo={len(todo)}, workers={self.view_workers}, "
            f"log_every={progress_every}"
        )
        if self.view_workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=self.view_workers) as executor:
                futures = {executor.submit(build, section): section for section in todo}
                for future in as_completed(futures):
                    with progress_lock:
                        record_progress(future.result())
        else:
            for section in todo:
                record_progress(build(section))

        rows.sort(key=lambda view: (view.sample_id, view.doc_id, view.section_id))
        if self.cache_views:
            write_jsonl(views_path, [asdict(row) for row in rows])
            self.logger.info(f"MC-indexing view cache finalized: {views_path}")
        return rows

    def _section_hash(self, section: MCSection) -> str:
        payload = {
            "section_id": section.section_id,
            "title": section.title,
            "text": section.text,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _require_llm(self, model: str) -> None:
        missing = []
        if not os.environ.get("LLM_API_KEY"):
            missing.append("LLM_API_KEY")
        if not os.environ.get("LLM_BASE_URL"):
            missing.append("LLM_BASE_URL")
        if not model:
            missing.append("LLM_MODEL")
        if missing:
            raise RuntimeError(f"MC-indexing view generation requires {', '.join(missing)}.")

    def _view_config_hash(self, model: str) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "summary_min_tokens": self.summary_min_tokens,
            "llm_model": model,
            "view_max_tokens": int(self.config.get("view_max_tokens", 600)),
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _generate_view(self, section: MCSection, model: str, view_config_hash: str) -> MCView:
        text = normalize_space(section.text)
        content_hash = self._section_hash(section)
        needs_summary = self._section_token_count(text) > self.summary_min_tokens

        input_tokens = 0
        output_tokens = 0
        api_input_tokens = 0
        api_output_tokens = 0
        estimated_input_tokens = 0
        estimated_output_tokens = 0
        if needs_summary:
            summary_prompt = SUMMARY_PROMPT.format(title=section.title, text=text)
            summary, usage = self._call_llm_text_with_usage(summary_prompt, model=model)
            summary = summary.strip()
            if not summary:
                raise RuntimeError(f"LLM returned empty summary for {section.section_id}.")
            est_in = self.count_tokens(summary_prompt)
            est_out = self.count_tokens(summary)
            estimated_input_tokens += est_in
            estimated_output_tokens += est_out
            api_in = int(usage.get("input_tokens", 0) or 0)
            api_out = int(usage.get("output_tokens", 0) or 0)
            api_input_tokens += api_in
            api_output_tokens += api_out
            input_tokens += api_in or est_in
            output_tokens += api_out or est_out
        else:
            summary = text

        keyword_prompt = KEYWORD_PROMPT.format(title=section.title, text=text)
        raw_keywords, usage = self._call_llm_jsonish_with_usage(keyword_prompt, model=model)
        keywords = self._parse_keywords(raw_keywords)
        if not keywords:
            raise RuntimeError(f"LLM returned no keywords for {section.section_id}.")
        est_in = self.count_tokens(keyword_prompt)
        est_out = self.count_tokens(json.dumps(keywords, ensure_ascii=False))
        estimated_input_tokens += est_in
        estimated_output_tokens += est_out
        api_in = int(usage.get("input_tokens", 0) or 0)
        api_out = int(usage.get("output_tokens", 0) or 0)
        api_input_tokens += api_in
        api_output_tokens += api_out
        input_tokens += api_in or est_in
        output_tokens += api_out or est_out
        with self._token_lock:
            self.view_input_tokens += input_tokens
            self.view_output_tokens += output_tokens
            self.view_api_input_tokens += api_input_tokens
            self.view_api_output_tokens += api_output_tokens
            self.view_estimated_input_tokens += estimated_input_tokens
            self.view_estimated_output_tokens += estimated_output_tokens
        method = f"llm:chat:{model}"

        return MCView(
            section_id=section.section_id,
            doc_id=section.doc_id,
            sample_id=section.sample_id,
            title=section.title,
            raw_text=f"{section.title}. {text}",
            summary=summary,
            keywords=keywords,
            generation_method=method,
            content_hash=content_hash,
            view_config_hash=view_config_hash,
        )

    def _call_llm_text(self, prompt: str, model: str, json_mode: bool = False) -> str:
        text, _usage = self._call_llm_text_with_usage(prompt, model=model, json_mode=json_mode)
        return text

    @staticmethod
    def _openai_usage(response) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        return {
            "input_tokens": int(prompt_tokens),
            "output_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

    def _call_llm_text_with_usage(self, prompt: str, model: str, json_mode: bool = False) -> tuple[str, Dict[str, int]]:
        from openai import OpenAI

        if not model:
            raise RuntimeError("LLM_MODEL is required for MC-indexing view generation.")
        client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"])
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(self.config.get("view_max_tokens", 600)),
        }
        if json_mode and self.llm_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        if text.strip().startswith("```"):
            text = text.strip().strip("`")
            text = text.split("\n", 1)[-1]
        return text.strip(), self._openai_usage(response)

    def _call_llm_jsonish(self, prompt: str, model: str):
        text, _usage = self._call_llm_jsonish_with_usage(prompt, model=model)
        return text

    def _call_llm_jsonish_with_usage(self, prompt: str, model: str):
        return self._call_llm_text_with_usage(prompt, model=model, json_mode=False)

    @staticmethod
    def _parse_keywords(raw) -> List[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        text = str(raw).strip()
        if not text:
            return []
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, dict):
                for key in ("keywords", "keyword", "items"):
                    items = value.get(key)
                    if isinstance(items, list):
                        return [str(item).strip() for item in items if str(item).strip()]
        except json.JSONDecodeError:
            pass
        text = text.strip("[]")
        parts = re.split(r",|\n|;", text)
        return [part.strip().strip("\"' -") for part in parts if part.strip().strip("\"' -")]

    def _build_all_chunks(self, sections: List[MCSection], views: List[MCView]) -> Dict[str, List[MCChunk]]:
        by_section = {section.section_id: section for section in sections}
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
            sentence_rows = []
            sample_id = ordered[0].sample_id if ordered else doc_id
            for section in ordered:
                sentences = sentence_split(section.text) or [section.text]
                for sentence in sentences:
                    sentence_rows.append((sentence, section.section_id))

            current_sentences = []
            current_sources = []
            current_tokens = 0
            chunk_index = 0

            def flush():
                nonlocal chunk_index, current_sentences, current_sources, current_tokens
                if not current_sentences:
                    return
                chunk_index += 1
                text = normalize_space(" ".join(current_sentences))
                chunks.append(
                    MCChunk(
                        chunk_id=f"{sample_id}::{doc_id}::flc{length}-{chunk_index:04d}",
                        doc_id=doc_id,
                        sample_id=sample_id,
                        method=f"FLC-{length}",
                        title=f"{doc_id} FLC-{length} chunk {chunk_index}",
                        text=text,
                        source_sections=unique_preserve_order(current_sources),
                        return_text=text,
                    )
                )
                current_sentences = []
                current_sources = []
                current_tokens = 0

            for sentence, section_id in sentence_rows:
                token_count = max(1, len(lexical_tokens(sentence)))
                if current_sentences and current_tokens + token_count > length:
                    flush()
                current_sentences.append(sentence)
                current_sources.append(section_id)
                current_tokens += token_count
            flush()
        return chunks

    @staticmethod
    def _mc_view_topk(topk: int) -> int:
        if topk <= 1:
            return 1
        return max(1, int(round(2 * topk / 3)))

    @staticmethod
    def _merge_view_results(results_by_view: List[Tuple[str, List[Tuple[MCChunk, float]]]], topk: int) -> List[Tuple[MCChunk, float]]:
        merged: Dict[Tuple[str, ...], Tuple[MCChunk, float]] = {}
        for _method, results in results_by_view:
            for chunk, score in results:
                key = tuple(chunk.source_sections)
                if key in merged:
                    existing_chunk, existing_score = merged[key]
                    merged[key] = (existing_chunk, max(existing_score, float(score)))
                    continue
                merged[key] = (chunk, float(score))
        return list(merged.values())[:topk]

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
            self.logger.info(f"MC-indexing fitting retriever: method={method}, chunks={len(chunks)}")
            retriever = self._make_retriever(method)
            retriever.fit(chunks)
            self.retrievers[method] = retriever
        self.logger.info(f"MC-indexing built retrievers for methods: {', '.join(self.retrievers)}")

    def _search(self, query: str, topk: int) -> List[MCResource]:
        if self.method == "MC-indexing":
            per_view_k = self._mc_view_topk(topk)
            results_by_view = []
            for method, retriever in self.retrievers.items():
                results_by_view.append((method, retriever.search(query, per_view_k)))
            ranked = self._merge_view_results(results_by_view, topk)
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
