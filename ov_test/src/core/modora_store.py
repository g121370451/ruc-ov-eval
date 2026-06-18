from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from src.adapters.base import StandardDoc
from src.core.logger import get_logger

logger = get_logger()


@dataclass
class ModoraResource:
    """Single evidence item returned by MoDora."""

    uri: str
    content: str = ""
    score: float = 0.0
    file_name: str = ""
    page: int = 0
    bboxes: List[list] = field(default_factory=list)
    retrievers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModoraResult:
    """Result object aligned with the benchmark store interface."""

    resources: List[ModoraResource] = field(default_factory=list)
    retrieve_input_tokens: int = 0
    retrieve_output_tokens: int = 0
    native_generation_used: bool = True
    native_final_answer: str = ""
    native_input_tokens: int = 0
    native_output_tokens: int = 0
    raw_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ModoraLibraryState:
    tree: Any
    source_paths: Dict[str, str]
    loaded_at: float
    query_lock: threading.Lock = field(default_factory=threading.Lock)


class ModoraStoreWrapper:
    """MoDora QA backend wrapper for ruc-ov-eval.

    Ingestion is expected to be done by MoDora CLI beforehand. This wrapper loads
    all cached CCTrees, merges them into a library-level tree, and lets MoDora
    perform retrieval + native answer generation for each benchmark question.
    """

    _state_cache: Dict[tuple[str, str, str, str], _ModoraLibraryState] = {}
    _state_cache_lock = threading.Lock()

    def __init__(
        self,
        store_path: str,
        modora_config: Optional[dict] = None,
    ):
        self.store_path = store_path
        self.logger = logger
        os.makedirs(store_path, exist_ok=True)

        cfg = dict(modora_config or {})
        self.repo_root = Path(__file__).resolve().parents[3]
        self.workspace_root = self.repo_root.parent

        self.modora_backend_path = self._resolve_optional_path(
            cfg.get("modora_backend_path"),
            prefer_repo=True,
        )
        self.modora_config_path = self._resolve_optional_path(
            cfg.get("modora_config"),
            prefer_repo=True,
        )
        self.docs_dir = self._resolve_optional_path(cfg.get("docs_dir"))
        self.cache_dir = self._resolve_optional_path(cfg.get("cache_dir"))
        self.inline_modora_config = self._build_inline_modora_config(cfg)

        self.ingest_mode = str(cfg.get("ingest_mode", "python") or "python").lower()
        self.preload_library = bool(cfg.get("preload_library", True)) and (
            self.ingest_mode in {"validate", "cache", "external", "none"}
        )
        self.share_library_cache = bool(cfg.get("share_library_cache", True))
        self.ensure_local_llm = bool(cfg.get("ensure_local_llm", False))
        self.force_full_library_retrieval = bool(
            cfg.get("force_full_library_retrieval", True)
        )
        self.serial_queries = bool(cfg.get("serial_queries", True))
        self.max_context_chars = int(cfg.get("max_context_chars", 2000) or 2000)
        self.run_ocr = bool(cfg.get("run_ocr", True))
        self.run_build_tree = bool(cfg.get("run_build_tree", True))
        self.resume_ingest = bool(cfg.get("resume", True))
        self.overwrite_ingest = bool(cfg.get("overwrite", False))
        self.materialize_clean_unreferenced_pdfs = self._bool_value(
            cfg.get("materialize_clean_unreferenced_pdfs", True),
            default=True,
        )
        self.component_workers = cfg.get("component_workers")
        self.text_extract_workers = cfg.get("text_extract_workers")
        self.pdf_text_empty_policy = cfg.get("pdf_text_empty_policy")
        self.ocr_batch_size = int(cfg.get("ocr_batch_size", 1) or 1)
        self.build_tree_concurrency = cfg.get("build_tree_concurrency")
        self.remote_llm_per_query_concurrency = self._optional_positive_int(
            cfg.get("remote_llm_per_query_concurrency", 1)
        )
        self.query_trace_logging = self._bool_value(
            cfg.get("query_trace_logging", True),
            default=True,
        )
        delete_mode = str(cfg.get("delete_mode", "cache_only") or "cache_only").lower()
        delete_mode_aliases = {
            "cache": "cache_only",
            "cache_only": "cache_only",
            "docs_and_cache": "docs_and_cache",
            "all": "docs_and_cache",
            "none": "none",
            "skip": "none",
        }
        if delete_mode not in delete_mode_aliases:
            raise ValueError(
                "Unsupported MoDora delete_mode: "
                f"{delete_mode}. Use cache_only, docs_and_cache, or none."
            )
        self.delete_mode = delete_mode_aliases[delete_mode]

        self._imports_ready = False
        self._state: _ModoraLibraryState | None = None
        self._load_lock = threading.Lock()

        try:
            import tiktoken

            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            self.logger.warning(f"tiktoken init failed, token counting disabled: {e}")
            self.enc = None

        if self.preload_library:
            self._ensure_library_loaded()

    def _build_inline_modora_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for key in [
            "env",
            "service_name",
            "api_port",
            "log_level",
            "log_format",
            "log_to_file",
            "log_dir",
            "chroma_persist_path",
            "embedding_api_base",
            "embedding_api_key",
            "embedding_model_name",
            "rerank_api_base",
            "rerank_api_key",
            "rerank_model_name",
            "model_instances",
            "llm_local_startup_timeout_s",
            "remote_llm_max_concurrency",
            "remote_llm_per_query_concurrency",
            "remote_llm_max_attempts",
            "remote_llm_base_delay_s",
            "remote_llm_max_delay_s",
            "query_trace_logging",
            "ocr_model",
            "ocr_device",
            "ocr_lang",
            "ocr_layout_unclip_ratio",
            "ocr_text_recognition_batch_size",
            "ocr_use_table_recognition",
            "ocr_use_doc_unwarping",
            "allow_pdf_text_fallback",
            "text_extract_workers",
            "pdf_text_empty_policy",
            "enable_vector_search",
            "ui_settings",
        ]:
            if key in cfg:
                config[key] = cfg[key]
        return config

    def _ensure_modora_config_file(self) -> None:
        if self.modora_config_path is not None or not self.inline_modora_config:
            return

        config = dict(self.inline_modora_config)
        if self.docs_dir is not None:
            config["docs_dir"] = str(self.docs_dir)
        if self.cache_dir is not None:
            config["cache_dir"] = str(self.cache_dir)

        config_dir = Path(self.store_path) / "_modora_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "local.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.modora_config_path = config_path.resolve()

    def _resolve_optional_path(
        self,
        raw: Any,
        *,
        prefer_repo: bool = False,
    ) -> Path | None:
        if raw is None or str(raw).strip() == "":
            return None

        path = Path(str(raw)).expanduser()
        if path.is_absolute():
            return path.resolve()

        base_dirs = (
            [self.repo_root, self.workspace_root, Path.cwd()]
            if prefer_repo
            else [self.workspace_root, self.repo_root, Path.cwd()]
        )
        candidates = []
        seen = set()
        for base_dir in base_dirs:
            candidate = base_dir / path
            candidate_key = str(candidate)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            candidates.append(candidate)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def _optional_positive_int(raw: Any) -> int | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            text = raw.strip().lower()
            if text in {"", "none", "null", "unlimited", "off"}:
                return None
        return max(1, int(raw))

    @staticmethod
    def _bool_value(raw: Any, *, default: bool = False) -> bool:
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        return default

    def _ensure_modora_importable(self) -> None:
        if self._imports_ready:
            return

        backend_path = self.modora_backend_path
        default_backends = [
            self.repo_root / "modora" / "MoDora-backend",
            self.workspace_root / "modora-main" / "MoDora-backend",
        ]
        if backend_path is not None and not backend_path.exists():
            self.logger.warning(
                f"Configured MoDora backend path does not exist: {backend_path}; "
                "trying default locations."
            )
            backend_path = None

        if backend_path is None:
            for default_backend in default_backends:
                if default_backend.exists():
                    backend_path = default_backend.resolve()
                    break
        self.modora_backend_path = backend_path

        if backend_path is not None:
            src_path = backend_path / "src"
            for item in (src_path, backend_path):
                if item.exists():
                    item_str = str(item)
                    if item_str not in sys.path:
                        sys.path.insert(0, item_str)

        try:
            import modora  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MoDora backend is not importable. Configure "
                "store.modora_backend_path to the MoDora-backend directory, "
                "install MoDora into this environment, or run "
                "`git submodule update --init --recursive` from the ruc-ov-eval root."
            ) from e

        self._imports_ready = True

    def _load_settings_and_service(self):
        self._ensure_modora_importable()
        self._ensure_modora_config_file()

        config_path = str(self.modora_config_path) if self.modora_config_path else None
        if config_path:
            os.environ["MODORA_CONFIG"] = config_path

        from modora.core.infra.llm.process import ensure_llm_local_loaded
        from modora.core.services.qa_service import QAService
        from modora.core.settings import Settings
        from modora.core.utils.config import (
            load_ui_settings_from_config,
            settings_from_ui_payload,
        )

        settings = Settings.load(config_path)
        if self.docs_dir is None and settings.docs_dir:
            self.docs_dir = Path(settings.docs_dir).expanduser().resolve()
        if self.cache_dir is None and settings.cache_dir:
            self.cache_dir = Path(settings.cache_dir).expanduser().resolve()

        if self.ensure_local_llm:
            ensure_llm_local_loaded(settings, self.logger, config_path=config_path)

        ui_settings = load_ui_settings_from_config(config_path)
        qa_settings, _, qa_instance, cfg = settings_from_ui_payload(
            settings, ui_settings, module_key="qaService"
        )
        retriever_settings, _, retriever_instance, _ = settings_from_ui_payload(
            settings, cfg, module_key="retriever"
        )
        qa_service = QAService(
            qa_settings,
            qa_instance=qa_instance,
            retriever_settings=retriever_settings,
            retriever_instance=retriever_instance,
        )

        if self.force_full_library_retrieval:

            async def _library_scope_location(_query: str):
                return [-1], [-1.0, -1.0]

            qa_service.extract_location = _library_scope_location

        return qa_service

    def _pdf_index(self) -> Dict[str, Path]:
        if self.docs_dir is None:
            raise ValueError(
                "MoDora docs_dir is not configured and could not be inferred from settings."
            )
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"MoDora docs_dir not found: {self.docs_dir}")

        index: Dict[str, Path] = {}
        for path in sorted(self.docs_dir.rglob("*.pdf")):
            index[path.name] = path
            index[path.stem] = path
        return index

    def _tree_paths(self) -> List[Path]:
        if self.cache_dir is None:
            raise ValueError(
                "MoDora cache_dir is not configured and could not be inferred from settings."
            )
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"MoDora cache_dir not found: {self.cache_dir}")

        seen: set[Path] = set()
        paths: List[Path] = []
        for tree_path in sorted(self.cache_dir.rglob("tree.json")):
            resolved = tree_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(resolved)
        return paths

    def _doc_paths_from_samples(self, samples: List[StandardDoc]) -> List[Path]:
        paths: List[Path] = []
        for sample in samples or []:
            raw_paths = getattr(sample, "doc_paths", None)
            if raw_paths is None:
                raw_paths = getattr(sample, "doc_path", None)
            if raw_paths is None:
                continue
            if isinstance(raw_paths, (str, os.PathLike)):
                raw_paths = [raw_paths]
            for raw_path in raw_paths:
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    path = (Path.cwd() / path).resolve()
                paths.append(path)
        return paths

    def _raw_doc_paths_for_sample(self, sample: StandardDoc) -> List[Any]:
        raw_paths = getattr(sample, "doc_paths", None)
        if raw_paths is None:
            raw_paths = getattr(sample, "doc_path", None)
        if raw_paths is None:
            return []
        if isinstance(raw_paths, (str, os.PathLike)):
            return [raw_paths]
        return list(raw_paths)

    def _resolve_doc_path(self, raw_path: Any) -> Path:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path.resolve()

    @staticmethod
    def _safe_pdf_name_part(value: Any, *, fallback: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
        text = text.strip("._-")
        return text[:80] or fallback

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _materialized_pdf_path(self, source_path: Path, sample_id: str) -> Path:
        if self.docs_dir is None:
            raise ValueError("MoDora docs_dir is not configured.")
        sample_part = self._safe_pdf_name_part(sample_id, fallback="sample")
        stem_part = self._safe_pdf_name_part(source_path.stem, fallback="document")
        path_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        return self.docs_dir / f"{sample_part}__{stem_part}__{path_hash}.pdf"

    def _load_materialized_manifest(self) -> dict[str, dict[str, Any]]:
        if self.docs_dir is None:
            return {}
        manifest_path = self.docs_dir / "_materialized_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            self.logger.warning(
                f"Could not read MoDora materialization manifest: {manifest_path}"
            )
            return {}
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {}
        return {
            str(item.get("source_path")): item
            for item in items
            if isinstance(item, dict) and item.get("source_path")
        }

    def _write_text_pdf(self, source_path: Path, target_path: Path) -> None:
        import fitz  # pymupdf

        text = source_path.read_text(encoding="utf-8", errors="replace")
        doc = fitz.open()
        page = None
        y = 54.0
        page_width = 595.0
        page_height = 842.0
        margin_x = 54.0
        line_height = 13.0

        def new_page() -> None:
            nonlocal page, y
            page = doc.new_page(width=page_width, height=page_height)
            y = 54.0

        new_page()
        try:
            for raw_line in text.splitlines() or [""]:
                line = raw_line.rstrip() or " "
                wrap_width = 74 if line.lstrip().startswith("#") else 88
                wrapped_lines = textwrap.wrap(
                    line,
                    width=wrap_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [" "]
                for wrapped in wrapped_lines:
                    if y > page_height - 54:
                        new_page()
                    page.insert_text((margin_x, y), wrapped, fontsize=10, fontname="helv")
                    y += line_height
                if not raw_line.strip():
                    y += 4.0
            target_path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(target_path)
        finally:
            doc.close()

    def _materialize_pdf_documents(
        self,
        samples: List[StandardDoc],
    ) -> tuple[List[StandardDoc], dict[str, Any]]:
        source_records: dict[Path, dict[str, Any]] = {}
        materialized_samples: List[StandardDoc] = []

        for sample in samples or []:
            sample_id = str(getattr(sample, "sample_id", "") or "sample")
            pdf_paths: List[str] = []
            for raw_path in self._raw_doc_paths_for_sample(sample):
                source_path = self._resolve_doc_path(raw_path)
                if source_path not in source_records:
                    source_records[source_path] = {
                        "sample_ids": set(),
                        "first_sample_id": sample_id,
                    }
                source_records[source_path]["sample_ids"].add(sample_id)
            materialized_samples.append(StandardDoc(sample_id=sample_id, doc_paths=pdf_paths))

        if self.cache_dir is None:
            self.cache_dir = (Path(self.store_path) / "modora_cache").resolve()
        if self.docs_dir is None:
            if source_records:
                common_parent = Path(
                    os.path.commonpath(
                        [str(source_path.parent) for source_path in source_records]
                    )
                )
                self.docs_dir = common_parent.resolve()
            else:
                self.docs_dir = (Path(self.store_path) / "modora_docs").resolve()
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        previous_manifest = self._load_materialized_manifest()
        source_to_pdf: dict[Path, Path] = {}
        manifest_items: List[dict[str, Any]] = []
        stats = {
            "input_refs": sum(
                len(self._raw_doc_paths_for_sample(sample)) for sample in samples or []
            ),
            "unique_sources": len(source_records),
            "reused_pdf": 0,
            "copied_pdf": 0,
            "converted_text": 0,
            "skipped_existing": 0,
            "moved_unreferenced_pdf": 0,
            "unsupported": 0,
            "missing": 0,
        }

        supported_text_exts = {".md", ".markdown", ".txt", ".text"}
        docs_root = self.docs_dir.resolve()
        if not source_records:
            stats["manifest_path"] = ""
            stats["pdf_count"] = len(list(self.docs_dir.glob("*.pdf")))
            return materialized_samples, stats

        for source_path, record in sorted(
            source_records.items(), key=lambda item: str(item[0])
        ):
            if not source_path.exists():
                stats["missing"] += 1
                raise FileNotFoundError(f"MoDora source document not found: {source_path}")
            if not source_path.is_file():
                stats["unsupported"] += 1
                raise ValueError(f"MoDora source document is not a file: {source_path}")

            ext = source_path.suffix.lower()
            source_hash = self._file_sha256(source_path)
            prev = previous_manifest.get(str(source_path))
            first_sample_id = str(record["first_sample_id"])
            method = ""

            if ext == ".pdf" and source_path.parent.resolve() == docs_root:
                target_path = source_path
                method = "pdf_reused"
                stats["reused_pdf"] += 1
            elif ext == ".pdf":
                target_path = self._materialized_pdf_path(source_path, first_sample_id)
                should_copy = (
                    self.overwrite_ingest
                    or not target_path.exists()
                    or not prev
                    or prev.get("source_sha256") != source_hash
                    or prev.get("output_pdf_path") != str(target_path)
                )
                if should_copy:
                    shutil.copy2(source_path, target_path)
                    stats["copied_pdf"] += 1
                    method = "pdf_copied"
                else:
                    stats["skipped_existing"] += 1
                    method = "pdf_existing"
            elif ext in supported_text_exts:
                target_path = self._materialized_pdf_path(source_path, first_sample_id)
                should_convert = (
                    self.overwrite_ingest
                    or not target_path.exists()
                    or not prev
                    or prev.get("source_sha256") != source_hash
                    or prev.get("output_pdf_path") != str(target_path)
                )
                if should_convert:
                    self._write_text_pdf(source_path, target_path)
                    stats["converted_text"] += 1
                    method = "text_pdf_generated"
                else:
                    stats["skipped_existing"] += 1
                    method = "text_pdf_existing"
            else:
                stats["unsupported"] += 1
                raise ValueError(
                    "MoDora only accepts PDFs. Unsupported source document "
                    f"extension for materialization: {source_path} "
                    f"(supported: .pdf, .md, .markdown, .txt, .text)"
                )

            target_path = target_path.resolve()
            source_to_pdf[source_path] = target_path
            manifest_items.append(
                {
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "source_size_bytes": source_path.stat().st_size,
                    "output_pdf_path": str(target_path),
                    "output_pdf_name": target_path.name,
                    "method": method,
                    "sample_ids": sorted(record["sample_ids"]),
                }
            )

        current_pdf_targets = {path.resolve() for path in source_to_pdf.values()}
        if self.materialize_clean_unreferenced_pdfs:
            stale_dir: Path | None = None
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for pdf_path in sorted(self.docs_dir.glob("*.pdf")):
                if pdf_path.resolve() in current_pdf_targets:
                    continue
                if stale_dir is None:
                    stale_dir = self.docs_dir / "_unreferenced_pdfs" / timestamp
                    stale_dir.mkdir(parents=True, exist_ok=True)
                target = stale_dir / pdf_path.name
                shutil.move(str(pdf_path), str(target))
                stats["moved_unreferenced_pdf"] += 1

        for original_sample, sample in zip(samples or [], materialized_samples):
            seen: set[str] = set()
            pdf_paths = []
            for raw_path in self._raw_doc_paths_for_sample(original_sample):
                pdf_path = source_to_pdf[self._resolve_doc_path(raw_path)]
                pdf_path_str = str(pdf_path)
                if pdf_path_str not in seen:
                    seen.add(pdf_path_str)
                    pdf_paths.append(pdf_path_str)
            sample.doc_paths = pdf_paths

        manifest_path = self.docs_dir / "_materialized_manifest.json"
        manifest = {
            "version": 1,
            "docs_dir": str(self.docs_dir),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "items": manifest_items,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.modora_config_path and Path(self.store_path) in self.modora_config_path.parents:
            self.modora_config_path = None
        stats["manifest_path"] = str(manifest_path)
        stats["pdf_count"] = len(list(self.docs_dir.glob("*.pdf")))
        return materialized_samples, stats

    def _infer_ingest_paths(self, samples: List[StandardDoc]) -> None:
        if self.cache_dir is None:
            self.cache_dir = (Path(self.store_path) / "modora_cache").resolve()

        if self.docs_dir is not None:
            return

        doc_paths = [path for path in self._doc_paths_from_samples(samples) if path.exists()]
        pdf_paths = [path for path in doc_paths if path.suffix.lower() == ".pdf"]
        if not pdf_paths:
            return

        common_parent = Path(os.path.commonpath([str(path.parent) for path in pdf_paths]))
        self.docs_dir = common_parent.resolve()

    def _invalidate_library_cache(self) -> None:
        old_key = self._cache_key()
        self._state = None
        with self._state_cache_lock:
            self._state_cache.pop(old_key, None)

    def _preprocess_args(self, max_workers: int | None) -> argparse.Namespace:
        self._ensure_modora_config_file()
        component_workers = int(self.component_workers or max_workers or 4)
        return argparse.Namespace(
            config=str(self.modora_config_path) if self.modora_config_path else None,
            dataset=str(self.docs_dir) if self.docs_dir else None,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            component_workers=component_workers,
            text_extract_workers=self.text_extract_workers,
            ocr_batch_size=self.ocr_batch_size,
            resume=self.resume_ingest,
            overwrite=self.overwrite_ingest,
        )

    def _build_tree_args(self, max_workers: int | None) -> argparse.Namespace:
        self._ensure_modora_config_file()
        concurrency = int(self.build_tree_concurrency or max_workers or 1)
        return argparse.Namespace(
            config=str(self.modora_config_path) if self.modora_config_path else None,
            dataset=str(self.docs_dir) if self.docs_dir else None,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            concurrency=concurrency,
            filter_list=None,
        )

    @staticmethod
    def _sum_usage(*stats_items: dict[str, Any]) -> tuple[int, int]:
        input_tokens = 0
        output_tokens = 0
        for stats in stats_items:
            input_tokens += int(stats.get("input_tokens", 0) or 0)
            output_tokens += int(stats.get("output_tokens", 0) or 0)
        return input_tokens, output_tokens

    def _load_library_state(self) -> _ModoraLibraryState:
        self._ensure_modora_importable()
        self._ensure_modora_config_file()

        from modora.core.domain.cctree import CCTree

        pdf_index = self._pdf_index()
        trees: Dict[str, Any] = {}
        source_paths: Dict[str, str] = {}
        skipped = 0

        for tree_path in self._tree_paths():
            doc_key = tree_path.parent.name
            pdf_path = pdf_index.get(doc_key) or pdf_index.get(f"{doc_key}.pdf")
            if pdf_path is None:
                skipped += 1
                self.logger.warning(
                    f"Skipping MoDora tree without matching PDF: {tree_path}"
                )
                continue

            try:
                tree = CCTree.load_json(str(tree_path))
            except Exception as e:
                skipped += 1
                self.logger.warning(f"Failed to load MoDora tree {tree_path}: {e}")
                continue

            file_name = pdf_path.name
            trees[file_name] = tree
            source_paths[file_name] = str(pdf_path)

        if not trees:
            raise RuntimeError(
                f"No usable MoDora trees found under {self.cache_dir}. "
                "Run `modora ocr` and `modora build-tree` first."
            )

        merged_tree = CCTree.merge_multi_trees(trees)
        self.logger.info(
            f"Loaded MoDora library: {len(trees)} documents"
            + (f", skipped {skipped}" if skipped else "")
        )
        return _ModoraLibraryState(
            tree=merged_tree,
            source_paths=source_paths,
            loaded_at=time.time(),
        )

    def _cache_key(self) -> tuple[str, str, str, str]:
        return (
            str(self.modora_backend_path or ""),
            str(self.modora_config_path or ""),
            str(self.docs_dir or ""),
            str(self.cache_dir or ""),
        )

    def _has_library_cache(self) -> bool:
        if self._state is not None:
            return True
        if not self.share_library_cache:
            return False
        key = self._cache_key()
        with self._state_cache_lock:
            return key in self._state_cache

    def _ensure_library_loaded(self) -> _ModoraLibraryState:
        if self._state is not None:
            return self._state

        with self._load_lock:
            if self._state is not None:
                return self._state

            if not self.share_library_cache:
                self._state = self._load_library_state()
                return self._state

            key = self._cache_key()
            with self._state_cache_lock:
                cached = self._state_cache.get(key)
                if cached is not None:
                    self._state = cached
                    return cached

            loaded = self._load_library_state()
            key = self._cache_key()
            with self._state_cache_lock:
                self._state_cache[key] = loaded
            self._state = loaded
            return loaded

    @staticmethod
    def _run_coro_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_box: Dict[str, Any] = {}
        error_box: Dict[str, BaseException] = {}

        def _runner():
            try:
                result_box["result"] = asyncio.run(coro)
            except BaseException as e:
                error_box["error"] = e

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in error_box:
            raise error_box["error"]
        return result_box.get("result")

    @staticmethod
    def _make_uri(file_name: str, page: int) -> str:
        safe_file = quote(file_name or "unknown", safe="")
        return f"modora://{safe_file}#page={int(page or 0)}"

    def _resource_from_doc(self, doc: dict, index: int) -> ModoraResource:
        file_name = str(doc.get("file_name") or "unknown")
        page = int(doc.get("page") or 0)
        content = str(doc.get("content") or "")
        metadata = dict(doc)
        metadata["index"] = index
        return ModoraResource(
            uri=self._make_uri(file_name, page),
            content=content,
            score=float(doc.get("score") or 0.0),
            file_name=file_name,
            page=page,
            bboxes=list(doc.get("bboxes") or []),
            retrievers=list(doc.get("retrievers") or []),
            metadata=metadata,
        )

    def count_tokens(self, text: str) -> int:
        if not text or self.enc is None:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(
        self,
        samples: List[StandardDoc],
        max_workers: int = 4,
        monitor=None,
    ) -> dict:
        """Run MoDora ingestion from Python and return token usage."""
        start_time = time.time()
        stage_times: dict[str, float] = {}
        input_tokens = 0
        output_tokens = 0

        def _finish_stage(stage_name: str, stage_t0: float) -> float:
            elapsed = time.monotonic() - stage_t0
            stage_times[stage_name] = elapsed
            return elapsed

        def _stats_summary(stats: dict[str, Any]) -> str:
            keys = [
                "return_code",
                "total",
                "ok",
                "failed",
                "skipped",
                "input_tokens",
                "output_tokens",
            ]
            parts = [f"{key}={stats.get(key)}" for key in keys if key in stats]
            return ", ".join(parts) if parts else str(stats)

        if monitor:
            monitor.worker_start()
        try:
            self.logger.info(
                "MoDora ingest started "
                f"(mode={self.ingest_mode}, max_workers={max_workers}, "
                f"run_ocr={self.run_ocr}, run_build_tree={self.run_build_tree}, "
                f"resume={self.resume_ingest}, overwrite={self.overwrite_ingest}, "
                f"configured_docs_dir={self.docs_dir}, configured_cache_dir={self.cache_dir})"
            )

            stage_t0 = time.monotonic()
            self.logger.info("MoDora ingest phase started (phase=ensure_importable)")
            self._ensure_modora_importable()
            elapsed = _finish_stage("ensure_importable", stage_t0)
            self.logger.info(
                "MoDora ingest phase finished "
                f"(phase=ensure_importable, elapsed_s={elapsed:.2f})"
            )

            stage_t0 = time.monotonic()
            self.logger.info("MoDora ingest phase started (phase=infer_paths)")
            self._infer_ingest_paths(samples)
            elapsed = _finish_stage("infer_paths", stage_t0)
            self.logger.info(
                "MoDora ingest phase finished "
                f"(phase=infer_paths, elapsed_s={elapsed:.2f}, "
                f"docs_dir={self.docs_dir}, cache_dir={self.cache_dir})"
            )

            stage_t0 = time.monotonic()
            self.logger.info("MoDora ingest phase started (phase=materialize_pdfs)")
            samples, materialize_stats = self._materialize_pdf_documents(samples)
            elapsed = _finish_stage("materialize_pdfs", stage_t0)
            self.logger.info(
                "MoDora ingest phase finished "
                f"(phase=materialize_pdfs, elapsed_s={elapsed:.2f}, "
                f"input_refs={materialize_stats.get('input_refs')}, "
                f"unique_sources={materialize_stats.get('unique_sources')}, "
                f"reused_pdf={materialize_stats.get('reused_pdf')}, "
                f"copied_pdf={materialize_stats.get('copied_pdf')}, "
                f"converted_text={materialize_stats.get('converted_text')}, "
                f"skipped_existing={materialize_stats.get('skipped_existing')}, "
                f"moved_unreferenced_pdf={materialize_stats.get('moved_unreferenced_pdf')}, "
                f"pdf_count={materialize_stats.get('pdf_count')}, "
                f"manifest={materialize_stats.get('manifest_path')})"
            )

            if self.ingest_mode in {"validate", "cache", "external", "none"}:
                stage_t0 = time.monotonic()
                self.logger.info(
                    "MoDora ingest phase started (phase=load_library, mode=validate)"
                )
                self._ensure_library_loaded()
                elapsed = _finish_stage("load_library", stage_t0)
                total_elapsed = time.time() - start_time
                self.logger.info(
                    "MoDora ingest finished "
                    f"(mode={self.ingest_mode}, elapsed_s={total_elapsed:.2f}, "
                    f"stage_times={{{', '.join(f'{k}: {v:.2f}' for k, v in stage_times.items())}}})"
                )
                if monitor:
                    monitor.worker_end(success=True)
                return {
                    "time": total_elapsed,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "stage_times": stage_times,
                }

            if self.ingest_mode not in {"python", "rebuild"}:
                raise ValueError(f"Unsupported MoDora ingest_mode: {self.ingest_mode}")

            preprocess_stats: dict[str, Any] = {
                "return_code": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            build_tree_stats: dict[str, Any] = {
                "return_code": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

            if self.run_ocr:
                from modora.lab.commands.preprocess import run_preprocess_pipeline

                preprocess_args = self._preprocess_args(max_workers)
                self.logger.info(
                    "MoDora ingest phase started "
                    f"(phase=preprocess_ocr, dataset={preprocess_args.dataset}, "
                    f"cache_dir={preprocess_args.cache_dir}, "
                    f"component_workers={preprocess_args.component_workers}, "
                    f"text_extract_workers={preprocess_args.text_extract_workers}, "
                    f"ocr_batch_size={preprocess_args.ocr_batch_size}, "
                    f"resume={preprocess_args.resume}, overwrite={preprocess_args.overwrite})"
                )
                stage_t0 = time.monotonic()
                preprocess_stats = run_preprocess_pipeline(
                    preprocess_args,
                    self.logger,
                )
                elapsed = _finish_stage("preprocess_ocr", stage_t0)
                self.logger.info(
                    "MoDora ingest phase finished "
                    f"(phase=preprocess_ocr, elapsed_s={elapsed:.2f}, "
                    f"{_stats_summary(preprocess_stats)})"
                )
                if int(preprocess_stats.get("return_code", 0) or 0):
                    raise RuntimeError(
                        f"MoDora preprocess failed: {preprocess_stats}"
                    )
            else:
                self.logger.info("MoDora ingest phase skipped (phase=preprocess_ocr)")

            if self.run_build_tree:
                from modora.lab.commands.build_tree import run_build_tree_pipeline

                build_tree_args = self._build_tree_args(max_workers)
                self.logger.info(
                    "MoDora ingest phase started "
                    f"(phase=build_tree, dataset={build_tree_args.dataset}, "
                    f"cache_dir={build_tree_args.cache_dir}, "
                    f"concurrency={build_tree_args.concurrency})"
                )
                stage_t0 = time.monotonic()
                build_tree_stats = run_build_tree_pipeline(
                    build_tree_args,
                    self.logger,
                )
                elapsed = _finish_stage("build_tree", stage_t0)
                self.logger.info(
                    "MoDora ingest phase finished "
                    f"(phase=build_tree, elapsed_s={elapsed:.2f}, "
                    f"{_stats_summary(build_tree_stats)})"
                )
                if int(build_tree_stats.get("return_code", 0) or 0):
                    raise RuntimeError(
                        f"MoDora build-tree failed: {build_tree_stats}"
                    )
            else:
                self.logger.info("MoDora ingest phase skipped (phase=build_tree)")

            input_tokens, output_tokens = self._sum_usage(
                preprocess_stats,
                build_tree_stats,
            )
            self._invalidate_library_cache()
            stage_t0 = time.monotonic()
            self.logger.info("MoDora ingest phase started (phase=load_library)")
            self._ensure_library_loaded()
            elapsed = _finish_stage("load_library", stage_t0)
            total_elapsed = time.time() - start_time
            self.logger.info(
                "MoDora ingest phase finished "
                f"(phase=load_library, elapsed_s={elapsed:.2f})"
            )
            self.logger.info(
                "MoDora ingest finished "
                f"(mode={self.ingest_mode}, elapsed_s={total_elapsed:.2f}, "
                f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                f"stage_times={{{', '.join(f'{k}: {v:.2f}' for k, v in stage_times.items())}}})"
            )
            if monitor:
                monitor.worker_end(tokens=input_tokens + output_tokens, success=True)
        except Exception:
            self.logger.exception(
                "MoDora ingest failed "
                f"(elapsed_s={time.time() - start_time:.2f}, "
                f"stage_times={{{', '.join(f'{k}: {v:.2f}' for k, v in stage_times.items())}}})"
            )
            if monitor:
                monitor.worker_end(success=False)
            raise
        return {
            "time": time.time() - start_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stage_times": stage_times,
        }

    def retrieve(
        self,
        query: str,
        topk: int = 10,
        target_uri: str | None = None,
        **_: Any,
    ) -> ModoraResult:
        retrieve_t0 = time.monotonic()
        query_id = f"{threading.get_ident()}-{int(retrieve_t0 * 1000)}"
        if self.query_trace_logging:
            self.logger.info(
                "modora retrieve started "
                f"(query_id={query_id}, serial_queries={self.serial_queries}, "
                f"per_query_llm_concurrency={self.remote_llm_per_query_concurrency}, "
                f"query={query[:80]!r})",
                extra={
                    "query_id": query_id,
                    "query_preview": query[:80],
                    "serial_queries": self.serial_queries,
                    "per_query_llm_concurrency": self.remote_llm_per_query_concurrency,
                },
            )
        timing: dict[str, Any] = {
            "query_id": query_id,
            "query_preview": query[:80],
            "serial_queries": self.serial_queries,
            "per_query_llm_concurrency": self.remote_llm_per_query_concurrency,
        }
        stage_t0 = time.monotonic()
        state_cache_hit = self._has_library_cache()
        state = self._ensure_library_loaded()
        timing["library_load_s"] = time.monotonic() - stage_t0
        timing["library_cache_hit"] = state_cache_hit

        async def _qa():
            from modora.core.infra.llm.remote import remote_llm_query_scope
            from modora.core.infra.llm.usage import track_token_usage

            stage_t0 = time.monotonic()
            qa_service = self._load_settings_and_service()
            qa_service_init_s = time.monotonic() - stage_t0
            try:
                with remote_llm_query_scope(
                    self.remote_llm_per_query_concurrency, query_id=query_id
                ) as llm_timing:
                    with track_token_usage() as usage:
                        qa_t0 = time.monotonic()
                        qa_result = await qa_service.qa(
                            state.tree,
                            query,
                            state.source_paths,
                        )
                        qa_total_s = time.monotonic() - qa_t0
            finally:
                await qa_service.aclose()
            return qa_result, usage.as_dict(), dict(llm_timing), {
                "qa_service_init_s": qa_service_init_s,
                "qa_total_s": qa_total_s,
            }

        if self.serial_queries:
            lock_t0 = time.monotonic()
            with state.query_lock:
                timing["query_lock_wait_s"] = time.monotonic() - lock_t0
                qa_result, usage, llm_timing, qa_timing = self._run_coro_sync(_qa())
        else:
            timing["query_lock_wait_s"] = 0.0
            qa_result, usage, llm_timing, qa_timing = self._run_coro_sync(_qa())
        timing.update(qa_timing)
        timing["remote_llm"] = llm_timing

        retrieved_docs = qa_result.get("retrieved_documents", []) or []
        resources = [
            self._resource_from_doc(doc, idx)
            for idx, doc in enumerate(retrieved_docs[: topk or len(retrieved_docs)])
            if isinstance(doc, dict)
        ]
        answer = str(qa_result.get("answer") or "")
        elapsed_s = round(time.monotonic() - retrieve_t0, 2)
        timing["retrieve_total_s"] = time.monotonic() - retrieve_t0
        qa_stage_timing = qa_result.get("timing", {}) if isinstance(qa_result, dict) else {}
        if self.query_trace_logging:
            self.logger.info(
                "modora retrieve finished "
                f"(query_id={query_id}, elapsed_s={elapsed_s}, "
                f"resources={len(resources)}, retrieved_docs={len(retrieved_docs)}, "
                f"answer_chars={len(answer)}, "
                f"input_tokens={int(usage.get('prompt_tokens', 0) or 0)}, "
                f"output_tokens={int(usage.get('completion_tokens', 0) or 0)})",
                extra={
                    "query_id": query_id,
                    "elapsed_s": elapsed_s,
                    "resources": len(resources),
                    "retrieved_docs": len(retrieved_docs),
                    "answer_chars": len(answer),
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                },
            )
            self.logger.info(
                "modora retrieve timing summary "
                f"(query_id={query_id}, total_s={timing['retrieve_total_s']:.2f}, "
                f"library_load_s={timing['library_load_s']:.2f}, "
                f"library_cache_hit={timing['library_cache_hit']}, "
                f"qa_service_init_s={timing['qa_service_init_s']:.2f}, "
                f"query_lock_wait_s={timing['query_lock_wait_s']:.2f}, "
                f"qa_total_s={timing['qa_total_s']:.2f}, "
                f"extract_location_s={float(qa_stage_timing.get('extract_location_s', 0.0) or 0.0):.2f}, "
                f"retriever_s={float(qa_stage_timing.get('retrieve_s', 0.0) or 0.0):.2f}, "
                f"semantic_retrieve_s={float(qa_stage_timing.get('semantic_retrieve_s', 0.0) or 0.0):.2f}, "
                f"semantic_select_children_s={float(qa_stage_timing.get('semantic_select_children_s', 0.0) or 0.0):.2f}, "
                f"semantic_relevance_s={float(qa_stage_timing.get('semantic_relevance_s', 0.0) or 0.0):.2f}, "
                f"schema_s={float(qa_stage_timing.get('schema_s', 0.0) or 0.0):.2f}, "
                f"crop_images_s={float(qa_stage_timing.get('crop_images_s', 0.0) or 0.0):.2f}, "
                f"reason_retrieved_s={float(qa_stage_timing.get('reason_retrieved_s', 0.0) or 0.0):.2f}, "
                f"check_answer_s={float(qa_stage_timing.get('check_answer_s', 0.0) or 0.0):.2f}, "
                f"reason_whole_s={float(qa_stage_timing.get('reason_whole_s', 0.0) or 0.0):.2f}, "
                f"remote_llm_calls={int(llm_timing.get('calls', 0) or 0)}, "
                f"remote_llm_attempts={int(llm_timing.get('attempts', 0) or 0)}, "
                f"remote_llm_retries={int(llm_timing.get('retries', 0) or 0)}, "
                f"remote_llm_query_wait_s={float(llm_timing.get('query_wait_s', 0.0) or 0.0):.2f}, "
                f"remote_llm_global_wait_s={float(llm_timing.get('global_wait_s', 0.0) or 0.0):.2f}, "
                f"remote_llm_api_s={float(llm_timing.get('api_s', 0.0) or 0.0):.2f}, "
                f"remote_llm_retry_sleep_s={float(llm_timing.get('retry_sleep_s', 0.0) or 0.0):.2f}, "
                f"input_tokens={int(usage.get('prompt_tokens', 0) or 0)}, "
                f"output_tokens={int(usage.get('completion_tokens', 0) or 0)}, "
                f"resources={len(resources)}, retrieved_docs={len(retrieved_docs)})"
            )

        return ModoraResult(
            resources=resources,
            retrieve_input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            retrieve_output_tokens=int(usage.get("completion_tokens", 0) or 0),
            native_final_answer=answer,
            native_input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            native_output_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw_result=qa_result,
        )

    def process_retrieval_results(self, search_res: ModoraResult):
        retrieved_texts = []
        context_blocks = []
        retrieved_uris = []

        for resource in search_res.resources:
            retrieved_uris.append(resource.uri)
            retrieved_texts.append(resource.content)
            context_blocks.append(resource.content[: self.max_context_chars])

        return retrieved_texts, context_blocks, retrieved_uris

    @staticmethod
    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _safe_delete_path(self, path: Path, root: Path, label: str) -> bool:
        resolved = path.resolve()
        resolved_root = root.resolve()
        if not self._is_under(resolved, resolved_root):
            self.logger.warning(
                "MoDora delete skipped unsafe path "
                f"(label={label}, path={resolved}, root={resolved_root})"
            )
            return False
        if not resolved.exists():
            return False
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        self.logger.info(
            f"MoDora delete removed {label} (path={resolved})"
        )
        return True

    def _delete_knowledge_base_entries(self, pdf_paths: list[Path]) -> int:
        if self.cache_dir is None:
            return 0

        kb_path = self.cache_dir / "knowledge_base.json"
        if not kb_path.exists():
            return 0
        if not self._is_under(kb_path, self.cache_dir):
            self.logger.warning(
                f"MoDora delete skipped unsafe knowledge_base path: {kb_path}"
            )
            return 0

        try:
            payload = json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning(f"MoDora delete could not read knowledge_base.json: {e}")
            return 0

        docs = payload.get("docs")
        if not isinstance(docs, dict):
            return 0

        removed = 0
        for pdf_path in pdf_paths:
            for key in (pdf_path.name, pdf_path.stem):
                if key in docs:
                    del docs[key]
                    removed += 1

        if removed:
            kb_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.logger.info(
                f"MoDora delete updated knowledge_base.json (removed={removed})"
            )
        return removed

    def clear(self):
        """Delete MoDora cache entries one PDF at a time."""
        start_time = time.time()
        if self.delete_mode == "none":
            self.logger.info("MoDora delete skipped (delete_mode=none).")
            return

        self._ensure_modora_config_file()

        if self.docs_dir is None or self.cache_dir is None:
            self.logger.warning(
                "MoDora delete skipped because docs_dir or cache_dir is not configured "
                f"(docs_dir={self.docs_dir}, cache_dir={self.cache_dir})"
            )
            return

        docs_root = self.docs_dir.resolve()
        cache_root = self.cache_dir.resolve()
        if not docs_root.exists():
            self.logger.warning(f"MoDora delete skipped; docs_dir not found: {docs_root}")
            return

        pdf_paths = sorted(path.resolve() for path in docs_root.rglob("*.pdf"))
        removed_cache_dirs = 0
        removed_docs = 0
        self.logger.info(
            "MoDora delete started "
            f"(mode={self.delete_mode}, docs_dir={docs_root}, "
            f"cache_dir={cache_root}, pdf_count={len(pdf_paths)})"
        )

        for pdf_path in pdf_paths:
            if not self._is_under(pdf_path, docs_root):
                self.logger.warning(
                    f"MoDora delete skipped unsafe PDF path: {pdf_path}"
                )
                continue

            cache_candidates = [
                cache_root / pdf_path.stem,
                cache_root / "trees" / pdf_path.stem,
            ]
            if pdf_path.stem.isdigit():
                cache_candidates.append(cache_root / str(int(pdf_path.stem)))

            seen: set[Path] = set()
            for cache_path in cache_candidates:
                resolved_cache_path = cache_path.resolve()
                if resolved_cache_path in seen:
                    continue
                seen.add(resolved_cache_path)
                if self._safe_delete_path(
                    resolved_cache_path,
                    cache_root,
                    f"cache for {pdf_path.name}",
                ):
                    removed_cache_dirs += 1

            if self.delete_mode == "docs_and_cache":
                if self._safe_delete_path(pdf_path, docs_root, f"document {pdf_path.name}"):
                    removed_docs += 1

        kb_removed = self._delete_knowledge_base_entries(pdf_paths)
        self._invalidate_library_cache()
        elapsed = time.time() - start_time
        self.logger.info(
            "MoDora delete finished "
            f"(elapsed_s={elapsed:.2f}, pdf_count={len(pdf_paths)}, "
            f"removed_cache_dirs={removed_cache_dirs}, removed_docs={removed_docs}, "
            f"removed_kb_entries={kb_removed}, mode={self.delete_mode})"
        )

    def close(self):
        pass
