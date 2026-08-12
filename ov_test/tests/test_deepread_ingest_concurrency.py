import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.adapters.base import StandardDoc
from src.core.deepread_store import DeepReadWrapper


class _FakeEmbedder:
    instances = []
    instances_lock = threading.Lock()

    def __init__(self, *, tracker, **kwargs):
        self.tracker = tracker
        with self.instances_lock:
            self.instances.append(self)

    def embed(self, text):
        self.tracker.add(len(text), 0)
        return [float(len(text)), 1.0]


class DeepReadIngestConcurrencyTest(unittest.TestCase):
    def _make_wrapper(self):
        wrapper = DeepReadWrapper.__new__(DeepReadWrapper)
        wrapper.use_pymupdf = True
        wrapper.source_header_enabled = True
        wrapper.embedding_model = "test-embedding"
        wrapper.embedding_api_key = "test-key"
        wrapper.embedding_base_url = "https://example.invalid"
        wrapper.logger = type(
            "Logger",
            (),
            {
                "info": staticmethod(lambda *args, **kwargs: None),
                "error": staticmethod(lambda *args, **kwargs: None),
            },
        )()
        wrapper.invalidate_doc_index_cache = lambda: None
        return wrapper

    def test_markdown_ingestion_uses_requested_document_concurrency(self):
        wrapper = self._make_wrapper()
        samples = [StandardDoc(f"doc-{i}", [f"doc-{i}.md"]) for i in range(8)]
        state_lock = threading.Lock()
        active = 0
        max_active = 0
        processed = []
        client_ids = []

        def fake_ingest_one(sample, ocr_pipeline, embedder):
            nonlocal active, max_active
            self.assertIsNone(ocr_pipeline)
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                processed.append(sample.sample_id)
                client_ids.append(id(embedder))
            time.sleep(0.03)
            embedder.tracker.add(3, 1)
            with state_lock:
                active -= 1

        wrapper._ingest_one = fake_ingest_one
        _FakeEmbedder.instances = []

        with patch("src.core.deepread_store.VolcengineEmbedder", _FakeEmbedder):
            stats = wrapper.ingest(samples, max_workers=4)

        self.assertGreaterEqual(max_active, 2)
        self.assertLessEqual(max_active, 4)
        self.assertCountEqual(processed, [sample.sample_id for sample in samples])
        self.assertEqual(len(client_ids), len(samples))
        self.assertEqual(len(set(client_ids)), len(samples))
        self.assertEqual(stats["input_tokens"], 24)
        self.assertEqual(stats["output_tokens"], 8)

    def test_parallel_real_ingest_writes_isolated_complete_artifacts(self):
        wrapper = self._make_wrapper()
        _FakeEmbedder.instances = []

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            store_dir = os.path.join(temp_dir, "store")
            os.makedirs(source_dir)
            os.makedirs(store_dir)
            wrapper.store_path = store_dir

            samples = []
            for i in range(6):
                path = os.path.join(source_dir, f"document_{i}.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# Section {i}\n\nUnique paragraph for document {i}.\n")
                samples.append(StandardDoc(f"doc-{i}", [path]))

            with patch("src.core.deepread_store.VolcengineEmbedder", _FakeEmbedder):
                stats = wrapper.ingest(samples, max_workers=3)

            self.assertGreater(stats["input_tokens"], 0)
            for i in range(6):
                stem = f"document_{i}"
                corpus_path = os.path.join(store_dir, f"{stem}_corpus.json")
                embedding_path = os.path.join(store_dir, f"{stem}_emb.npy")
                idmap_path = os.path.join(store_dir, f"{stem}_idmap.json")
                markdown_path = os.path.join(store_dir, f"{stem}.md")

                for artifact in (
                    corpus_path,
                    embedding_path,
                    idmap_path,
                    markdown_path,
                ):
                    self.assertTrue(os.path.isfile(artifact), artifact)

                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                with open(idmap_path, "r", encoding="utf-8") as f:
                    id_map = json.load(f)
                embeddings = np.load(embedding_path)
                with open(markdown_path, "r", encoding="utf-8") as f:
                    markdown = f.read()

                self.assertEqual(corpus["source_name"], stem)
                self.assertIn(f"Source Document: {stem}", markdown)
                self.assertEqual(embeddings.shape[0], len(id_map))
                self.assertGreater(embeddings.shape[0], 0)

    def test_worker_exception_is_propagated(self):
        wrapper = self._make_wrapper()
        samples = [StandardDoc("good", ["good.md"]), StandardDoc("bad", ["bad.md"])]

        def fake_ingest_one(sample, ocr_pipeline, embedder):
            if sample.sample_id == "bad":
                raise RuntimeError("expected ingestion failure")

        wrapper._ingest_one = fake_ingest_one
        with patch("src.core.deepread_store.VolcengineEmbedder", _FakeEmbedder):
            with self.assertRaisesRegex(RuntimeError, "expected ingestion failure"):
                wrapper.ingest(samples, max_workers=2)

    def test_paddleocr_pdf_ingestion_is_forced_serial(self):
        self.assertEqual(
            DeepReadWrapper._safe_ingest_worker_count(
                8, contains_pdf=True, ocr_enabled=True
            ),
            1,
        )
        self.assertEqual(
            DeepReadWrapper._safe_ingest_worker_count(
                8, contains_pdf=True, ocr_enabled=False
            ),
            8,
        )
        self.assertEqual(
            DeepReadWrapper._safe_ingest_worker_count(
                8, contains_pdf=False, ocr_enabled=True
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
