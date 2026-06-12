import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import math
import threading
from concurrent.futures import ThreadPoolExecutor

# Align with existing benchmark modules (see vector_store.py)
sys.path.append(str(Path(__file__).parent.parent))

from adapters.base import StandardDoc  # noqa: E402

try:
    import tiktoken  # noqa: E402
except Exception:
    tiktoken = None

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore

    # Use Volcengine embedder so multimodal embedding models from ov.conf work correctly.
    from openviking.models.embedder.volcengine_embedders import (  # noqa: E402
        VolcengineDenseEmbedder,
    )

    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


class _NaiveResource:
    """Minimal resource object that matches BenchmarkPipeline's expectations."""

    def __init__(self, uri: str, level: int = 2, abstract: str = "", overview: str = "", score: float = 0.0):
        self.uri = uri
        self.level = level
        self.abstract = abstract
        self.overview = overview
        self.score = score


class _NaiveSearchResult:
    """Minimal search result object with `.resources`."""

    def __init__(self, resources: List[_NaiveResource], telemetry: Optional[dict] = None):
        self.resources = resources
        self.telemetry = telemetry or {}


class NaiveStoreWrapper:
    """
    Naive RAG baseline vector store:
    - chunk docs -> embed chunks -> FAISS index
    - embed query -> top-k search -> return chunk uris
    - read_resource(uri) returns chunk text
    """

    def __init__(
        self,
        store_path: str,
        doc_output_dir: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "ep-20240910084318-g9vqn",
        api_key: str = "",
        api_base: str = "https://ark.cn-beijing.volces.com/api/v3",
        dimension: int = 1024,
        input_type: str = "multimodal",
        batch_size: int = 8,
        max_concurrent: int = 10,
    ):
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "Naive RAG requires faiss + numpy + openviking (VolcengineDenseEmbedder). "
                "Please install dependencies first."
            )

        self.retriever_type = "naive"
        self.logger = logging.getLogger(__name__)

        self.store_path = store_path
        self.doc_output_dir = str(doc_output_dir) if doc_output_dir else None
        if self.doc_output_dir:
            os.makedirs(self.doc_output_dir, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.api_base = api_base
        self.dimension = int(dimension)
        self.input_type = str(input_type or "multimodal")
        self.batch_size = max(int(batch_size or 8), 1)
        # Align with ov.conf embedding.max_concurrent by default.
        self.max_concurrent = max(int(max_concurrent or 10), 1)

        # Keep embedder params so we can create per-thread embedders (safer than sharing one client).
        self._embedder_params = {
            "model_name": self.embedding_model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "dimension": self.dimension,
            "input_type": self.input_type,
        }
        self._thread_local = threading.local()

        os.makedirs(store_path, exist_ok=True)
        self.index_path = os.path.join(store_path, "faiss_index.bin")
        self.chunks_path = os.path.join(store_path, "chunks.json")
        self._target_indices_cache = {}

        self.enc = None
        if tiktoken is not None:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = None

        # A main-thread embedder instance for query-time embed(). For ingestion we create per-thread instances.
        self.embedder = VolcengineDenseEmbedder(**self._embedder_params)

        self.index = None
        self.chunks = []  # [{id:int, uri:str, text:str}]
        self._load_or_init()

    def _safe_filename(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return "document"
        # Replace path separators and other odd characters.
        s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
        return s[:180] or "document"

    def _persist_processed_doc(self, sample_id: str, text: str) -> Optional[str]:
        """Persist extracted document text to doc_output_dir (if configured)."""
        if not self.doc_output_dir:
            return None
        if not str(text or "").strip():
            return None

        fname = f"{self._safe_filename(sample_id)}.md"
        out_path = os.path.join(self.doc_output_dir, fname)
        try:
            # Avoid rewriting large files if they already exist.
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        except Exception:
            pass

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            return out_path
        except Exception as exc:
            self.logger.warning("Failed to persist processed doc to %s: %s", out_path, exc)
            return None

    def _load_processed_doc(self, sample_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Load cached extracted document text from doc_output_dir, if available."""
        if not self.doc_output_dir:
            return None, None

        cached_path = os.path.join(self.doc_output_dir, f"{self._safe_filename(sample_id)}.md")
        try:
            if not os.path.exists(cached_path) or os.path.getsize(cached_path) <= 0:
                return None, None
            with open(cached_path, "r", encoding="utf-8") as f:
                cached_text = f.read()
            if not str(cached_text or "").strip():
                return None, None
            return cached_path, cached_text
        except Exception as exc:
            self.logger.warning("Failed to load processed doc from %s: %s", cached_path, exc)
            return None, None

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF: pdfplumber -> docling -> pypdf fallback chain.

        All dependencies are optional; if none are installed or extraction fails, return empty string.
        """
        if not pdf_path:
            return ""

        # Priority 1: pdfplumber
        try:
            import pdfplumber  # type: ignore

            self.logger.info("Attempting to extract text using pdfplumber: %s", pdf_path)
            pages_text: List[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
            content = "\n\n".join(pages_text)
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)

        # Priority 2: docling
        try:
            from docling.document_converter import DocumentConverter  # type: ignore

            self.logger.info("Attempting to extract text using docling: %s", pdf_path)
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            content = result.document.export_to_markdown()
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("docling failed for %s: %s, falling back", pdf_path, exc)

        # Priority 3: pypdf
        try:
            from pypdf import PdfReader  # type: ignore

            self.logger.info("Attempting to extract text using pypdf: %s", pdf_path)
            reader = PdfReader(pdf_path)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    # If we can't decrypt, extraction will likely return empty.
                    pass
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            content = "\n".join(pages_text)
            if content.strip():
                return content
        except ImportError:
            pass
        except Exception as exc:
            self.logger.warning("pypdf failed for %s: %s, falling back", pdf_path, exc)

        self.logger.error(
            "Cannot extract text from %s. Install one of: "
            "pip install 'docling>=2' / pip install pypdf / pip install pdfplumber",
            pdf_path,
        )
        return ""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.enc is None:
            return len(str(text))
        return len(self.enc.encode(str(text)))

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []

        if self.enc is None:
            # Fallback: char-based chunking (only used if tiktoken missing)
            size = max(self.chunk_size, 1)
            overlap = max(min(self.chunk_overlap, size - 1), 0)
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start = max(end - overlap, 0)
            return chunks

        tokens = self.enc.encode(text)
        n = len(tokens)
        size = max(self.chunk_size, 1)
        overlap = max(min(self.chunk_overlap, size - 1), 0)

        chunks: List[str] = []
        start = 0
        while start < n:
            end = min(start + size, n)
            chunks.append(self.enc.decode(tokens[start:end]))
            if end == n:
                break
            start = max(end - overlap, 0)
        return chunks

    def _init_new_index(self, dim: int) -> None:
        self.index = faiss.IndexFlatL2(int(dim))
        self.dimension = int(dim)
        self.chunks = []
        self._target_indices_cache.clear()

    def _load_or_init(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                # Keep dimension consistent with index
                self.dimension = int(self.index.d)
                return
            except Exception:
                pass
        self._init_new_index(self.dimension)

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def _chunk_uri(self, doc_path: str, chunk_id: int) -> str:
        abs_path = os.path.abspath(doc_path)
        return f"naive://{abs_path}#chunk={int(chunk_id)}"

    def _chunk_doc_path(self, uri: str) -> str:
        raw = str(uri or "")
        if raw.startswith("naive://"):
            raw = raw[len("naive://") :]
        marker = "#chunk="
        pos = raw.rfind(marker)
        if pos != -1:
            raw = raw[:pos]
        return os.path.abspath(raw) if raw else ""

    def _target_uri_to_paths(self, target_uri: str) -> List[str]:
        raw = str(target_uri or "").strip()
        if not raw or raw == "viking://resources":
            return []
        if raw.startswith("naive://"):
            raw = raw[len("naive://") :]
        elif raw.startswith("file://"):
            raw = raw[len("file://") :]
        elif raw.startswith("viking://"):
            # Naive chunks are keyed by local file path, so Viking resource URIs cannot filter them.
            return []
        marker = "#chunk="
        pos = raw.rfind(marker)
        if pos != -1:
            raw = raw[:pos]
        if not raw:
            return []

        target_path = os.path.abspath(raw)
        paths = [target_path]
        if target_path.lower().endswith(".pdf") and self.doc_output_dir:
            stem = os.path.splitext(os.path.basename(target_path))[0]
            cached_path = os.path.abspath(os.path.join(self.doc_output_dir, f"{self._safe_filename(stem)}.md"))
            paths.append(cached_path)
        return paths

    def _matches_target_uri(self, chunk_uri: str, target_uri: str) -> bool:
        target_paths = self._target_uri_to_paths(target_uri)
        if not target_paths:
            return True

        doc_path = self._chunk_doc_path(chunk_uri)
        if not doc_path:
            return False
        for target_path in target_paths:
            if doc_path == target_path:
                return True
            if os.path.isdir(target_path):
                try:
                    if os.path.commonpath([doc_path, target_path]) == target_path:
                        return True
                except ValueError:
                    continue
            if doc_path.startswith(target_path.rstrip(os.sep) + os.sep):
                return True
        return False

    def _target_chunk_indices(self, target_uri: str) -> List[int]:
        target_paths = self._target_uri_to_paths(target_uri)
        if not target_paths:
            return []

        cache_key = tuple(target_paths)
        cached = self._target_indices_cache.get(cache_key)
        if cached is not None:
            return cached

        indices: List[int] = []
        for pos, chunk in enumerate(self.chunks):
            uri = str(chunk.get("uri", ""))
            if self._matches_target_uri(uri, target_uri):
                indices.append(pos)
        self._target_indices_cache[cache_key] = indices
        return indices

    def _parse_chunk_id(self, uri: str) -> Optional[int]:
        if not uri:
            return None
        marker = "#chunk="
        pos = uri.rfind(marker)
        if pos == -1:
            return None
        raw = uri[pos + len(marker) :]
        try:
            return int(raw)
        except Exception:
            return None

    def ingest(self, samples: List[StandardDoc], max_workers=10, monitor=None, ingest_mode="per_file") -> dict:
        start_time = time.time()
        total_input_tokens = 0

        if not samples:
            return {"time": 0.0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}

        texts: List[str] = []
        metas: List[dict] = []
        base_id = len(self.chunks)

        # Optional progress bar (kept optional to avoid hard dependency issues).
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None

        def _load_doc_content(s: StandardDoc) -> Tuple[Optional[str], Optional[str]]:
            doc_path = getattr(s, "doc_path", None)
            if not doc_path:
                return None, None
            sample_id = getattr(s, "sample_id", "") or os.path.basename(str(doc_path))

            content = getattr(s, "content", None)
            if content is not None:
                return str(doc_path), str(content)

            # FinanceBench provides PDFs; naiveRAG needs to extract text before chunking.
            if str(doc_path).lower().endswith(".pdf"):
                cached_path, cached_text = self._load_processed_doc(str(sample_id))
                if cached_text is not None:
                    return cached_path, cached_text

                pdf_text = self._extract_pdf_text(str(doc_path))
                if not str(pdf_text or "").strip():
                    return str(doc_path), None
                persisted = self._persist_processed_doc(str(sample_id), pdf_text)
                # For traceability, point uri at the processed doc if we persisted one.
                return (persisted or str(doc_path)), pdf_text

            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    return str(doc_path), f.read()
            except Exception:
                return str(doc_path), None

        # PDF parsing can be slow. Parallelize doc reading/extraction (bounded) and show progress.
        parse_desc = "Reading documents"
        try:
            if any(str(getattr(s, "doc_path", "")).lower().endswith(".pdf") for s in samples):
                parse_desc = "Parsing PDFs"
        except Exception:
            pass

        # Use the `max_workers` argument if provided by benchmark config.
        mw = int(max_workers) if max_workers is not None else 1
        mw = max(mw, 1)

        if mw > 1:
            with ThreadPoolExecutor(max_workers=mw) as executor:
                it = executor.map(_load_doc_content, samples)
                if tqdm is not None:
                    it = tqdm(it, total=len(samples), desc=parse_desc, unit="doc")
                for doc_path, content in it:
                    if not doc_path:
                        continue
                    if not str(content or "").strip():
                        self.logger.warning("Empty/failed content for %s, skipping", doc_path)
                        continue
                    for chunk in self._split_text(str(content)):
                        chunk_id = base_id + len(metas)
                        uri = self._chunk_uri(doc_path, chunk_id)
                        texts.append(chunk)
                        metas.append({"id": chunk_id, "uri": uri, "text": chunk})
                        total_input_tokens += self.count_tokens(chunk)
        else:
            seq_iter = samples
            if tqdm is not None:
                seq_iter = tqdm(seq_iter, total=len(samples), desc=parse_desc, unit="doc")
            for s in seq_iter:
                doc_path, content = _load_doc_content(s)
                if not doc_path:
                    continue
                if not str(content or "").strip():
                    self.logger.warning("Empty/failed content for %s, skipping", doc_path)
                    continue
                for chunk in self._split_text(str(content)):
                    chunk_id = base_id + len(metas)
                    uri = self._chunk_uri(doc_path, chunk_id)
                    texts.append(chunk)
                    metas.append({"id": chunk_id, "uri": uri, "text": chunk})
                    total_input_tokens += self.count_tokens(chunk)

        if not texts:
            return {"time": time.time() - start_time, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}

        def _get_thread_embedder() -> VolcengineDenseEmbedder:
            e = getattr(self._thread_local, "embedder", None)
            if e is None:
                e = VolcengineDenseEmbedder(**self._embedder_params)
                self._thread_local.embedder = e
            return e

        # Multimodal embedding batch responses can be unstable across SDK versions.
        # Use per-text embed() but execute with bounded parallelism, aligned with ov.conf max_concurrent.
        if self.input_type == "multimodal":
            def _embed_one(text: str):
                return _get_thread_embedder().embed(text, is_query=False)

            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                it = executor.map(_embed_one, texts)
                if tqdm is not None:
                    it = tqdm(it, total=len(texts), desc="Embedding chunks", unit="chunk")
                embed_results = list(it)

            vectors = np.asarray([r.dense_vector for r in embed_results], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(metas):
                raise RuntimeError("Unexpected embedding output shape for naive RAG ingestion.")
        else:
            # Text embeddings API supports true batch calls. Keep small batches to avoid request-size limits.
            vectors_list = []
            total_batches = int(math.ceil(len(texts) / float(self.batch_size)))
            batch_iter = range(0, len(texts), self.batch_size)
            if tqdm is not None:
                batch_iter = tqdm(
                    batch_iter,
                    total=total_batches,
                    desc="Embedding chunks",
                    unit="batch",
                )

            for i in batch_iter:
                batch_texts = texts[i : i + self.batch_size]
                batch_results = self.embedder.embed_batch(batch_texts, is_query=False)
                batch_vectors = np.asarray([r.dense_vector for r in batch_results], dtype=np.float32)
                if batch_vectors.ndim != 2 or batch_vectors.shape[0] != len(batch_texts):
                    raise RuntimeError("Unexpected embedding output shape for naive RAG ingestion batch.")
                vectors_list.append(batch_vectors)

            vectors = (
                np.concatenate(vectors_list, axis=0)
                if vectors_list
                else np.zeros((0, self.dimension), dtype=np.float32)
            )
        if vectors.shape[0] != len(metas):
            raise RuntimeError("Embedding batch results count mismatch for naive RAG ingestion.")

        dim = int(vectors.shape[1])
        if self.index is None or int(getattr(self.index, "d", dim)) != dim or self.index.ntotal == 0:
            # (Re)create index if empty or dim mismatched.
            if self.index is not None and self.index.ntotal > 0 and int(getattr(self.index, "d", dim)) != dim:
                raise RuntimeError("Embedding dimension changed while index already has data.")
            self._init_new_index(dim)

        self.index.add(vectors)
        self.chunks.extend(metas)
        self._target_indices_cache.clear()
        self._save()

        # We don't have true embedding token usage from Ark here, approximate with input tokens.
        return {
            "time": time.time() - start_time,
            "input_tokens": total_input_tokens,
            "output_tokens": 0,
            "embedding_tokens": total_input_tokens,
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources") -> Tuple[_NaiveSearchResult, int]:
        embedding_t0 = time.perf_counter()
        query_tokens = self.count_tokens(query)
        if self.index is None or self.index.ntotal == 0:
            return _NaiveSearchResult(resources=[]), query_tokens

        q = self.embedder.embed(str(query), is_query=True)
        embedding_time = time.perf_counter() - embedding_t0
        search_t0 = time.perf_counter()
        vec = np.asarray([q.dense_vector], dtype=np.float32)
        if vec.ndim != 2 or int(vec.shape[1]) != int(self.dimension):
            raise RuntimeError("Query embedding dimension mismatch for naive RAG.")

        resources: List[_NaiveResource] = []
        target_paths = self._target_uri_to_paths(target_uri)
        if target_paths:
            target_indices = self._target_chunk_indices(target_uri)
            if not target_indices:
                search_time = time.perf_counter() - search_t0
                telemetry = {
                    "search_scope": "target",
                    "target_chunk_count": 0,
                    "searched_chunk_count": 0,
                    "retrieval_embedding_time_sec": embedding_time,
                    "retrieval_search_time_sec": search_time,
                }
                return _NaiveSearchResult(resources=[], telemetry=telemetry), query_tokens

            # Keep the same FAISS search path as non per-query retrieval, then
            # filter to the per-query target document. Searching the full flat
            # index preserves the exact target-scoped top-k without Python-side
            # vector reconstruction or numpy distance scans.
            k = min(int(self.index.ntotal), max(int(topk), 1))
            target_set = set(int(i) for i in target_indices)
            faiss_search_time = 0.0
            first_faiss_search_time = None
            while len(resources) < int(topk) and k <= int(self.index.ntotal):
                faiss_t0 = time.perf_counter()
                _dist, idx = self.index.search(vec, k)
                this_faiss_search_time = time.perf_counter() - faiss_t0
                faiss_search_time += this_faiss_search_time
                if first_faiss_search_time is None:
                    first_faiss_search_time = this_faiss_search_time
                resources = []
                seen = set()
                for dist, i in zip(_dist[0], idx[0]):
                    chunk_idx = int(i)
                    if chunk_idx < 0 or chunk_idx in seen or chunk_idx not in target_set:
                        continue
                    seen.add(chunk_idx)
                    uri = self.chunks[chunk_idx]["uri"]
                    resources.append(_NaiveResource(uri=uri, level=2, score=-float(dist)))
                    if len(resources) >= int(topk):
                        break
                if len(resources) >= int(topk) or k == int(self.index.ntotal):
                    break
                k = min(int(self.index.ntotal), max(k * 2, k + 1))
            telemetry = {
                "search_scope": "target",
                "target_chunk_count": len(target_indices),
                "searched_chunk_count": k,
                "retrieval_faiss_search_time_sec": faiss_search_time,
                "retrieval_filter_time_sec": max((time.perf_counter() - search_t0) - faiss_search_time, 0.0),
                "retrieval_core_time_sec": embedding_time + float(first_faiss_search_time or 0.0),
            }
        else:
            k = min(int(topk), int(self.index.ntotal))
            faiss_t0 = time.perf_counter()
            _dist, idx = self.index.search(vec, k)
            faiss_search_time = time.perf_counter() - faiss_t0
            for dist, i in zip(_dist[0], idx[0]):
                if 0 <= int(i) < len(self.chunks):
                    uri = self.chunks[int(i)]["uri"]
                    resources.append(_NaiveResource(uri=uri, level=2, score=-float(dist)))
            telemetry = {
                "search_scope": "global",
                "target_chunk_count": 0,
                "searched_chunk_count": k,
                "retrieval_faiss_search_time_sec": faiss_search_time,
                "retrieval_filter_time_sec": 0.0,
                "retrieval_core_time_sec": embedding_time + faiss_search_time,
            }

        search_time = time.perf_counter() - search_t0
        telemetry.update({
            "retrieval_embedding_time_sec": embedding_time,
            "retrieval_search_time_sec": search_time,
        })
        return _NaiveSearchResult(resources=resources, telemetry=telemetry), query_tokens

    def read_resource(self, uri: str) -> str:
        chunk_id = self._parse_chunk_id(uri)
        if chunk_id is None:
            return ""
        for c in self.chunks:
            if int(c.get("id", -1)) == int(chunk_id):
                return str(c.get("text", ""))
        return ""

    def clear(self):
        # Reset in-memory
        self._init_new_index(self.dimension)
        self._target_indices_cache.clear()
        # Remove persisted files
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        except Exception:
            pass
        try:
            if os.path.exists(self.chunks_path):
                os.remove(self.chunks_path)
        except Exception:
            pass
