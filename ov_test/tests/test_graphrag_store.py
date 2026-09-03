from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import yaml


OV_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(OV_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(OV_TEST_ROOT))

from src.adapters.base import StandardDoc, StandardQA
from src.core.graphrag_store import (
    GraphRAGResult,
    GraphRAGStoreWrapper,
    _context_source_uris,
    _flatten_context_text,
)
from src.core.store_contract import store_provides_final_answer
from src.pipeline import BenchmarkPipeline


TEST_ENV = {
    "EMBEDDING_MODEL": "test-embedding",
    "EMBEDDING_API_KEY": "embedding-secret",
    "EMBEDDING_BASE_URL": "http://127.0.0.1:1/v1",
    "EMBEDDING_DIMENSION": "8",
}
TEST_LLM = {
    "model": "test-chat",
    "api_key": "chat-secret",
    "base_url": "http://127.0.0.1:1/v1",
    "temperature": 0,
}

OV_WIKI_DRIFT_CONFIGS = {
    "enterprise_rag_bench_selected_80_drift.yaml",
    "mdaqa_first_100_drift.yaml",
    "mudabench_complex_drift.yaml",
    "mudabench_simple_drift.yaml",
    "paperscope_summary_57_gap_drift.yaml",
    "paperscope_summary_57_results_comparison_drift.yaml",
    "paperscope_summary_57_trend_drift.yaml",
    "paperscope_summary_93_gap_drift.yaml",
    "paperscope_summary_93_results_comparison_drift.yaml",
    "paperscope_summary_93_trend_drift.yaml",
    "scholarqa_multi_valid_101_drift.yaml",
    "wildgraphbench_summary_all_drift.yaml",
    "wildgraphbench_summary_health_drift.yaml",
}


class _FakeEngine:
    async def search(self, query: str):
        assert query
        return SimpleNamespace(
            response="final answer",
            context_text={"Sources": ["first context", "second context"]},
            context_data={
                "Sources": pd.DataFrame(
                    [{"id": "source-1", "text": "first context"}]
                )
            },
            prompt_tokens=123,
            output_tokens=17,
            llm_calls=2,
            llm_calls_categories={"build_context": 1, "response": 1},
            prompt_tokens_categories={"build_context": 3, "response": 120},
            output_tokens_categories={"build_context": 0, "response": 17},
        )


class GraphRAGStoreTests(unittest.TestCase):
    def _make_store(self, root: str, **options) -> GraphRAGStoreWrapper:
        merged = {"query_mode": "local", **options}
        with patch.dict(os.environ, TEST_ENV, clear=False):
            return GraphRAGStoreWrapper(root, TEST_LLM, merged)

    def test_config_is_in_memory_and_public_settings_do_not_leak_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(
                tmp,
                completion={
                    "call_args": {"authorization": "nested-secret"}
                },
            )
            settings = json.dumps(store._public_index_settings())

            self.assertEqual(store._config.vector_store.vector_size, 8)
            self.assertNotIn("chat-secret", settings)
            self.assertNotIn("embedding-secret", settings)
            self.assertNotIn("nested-secret", settings)
            self.assertTrue(Path(store._config.output_storage.base_dir).is_absolute())

    def test_drift_configs_apply_server_embedding_batch_limit(self):
        config_dir = OV_TEST_ROOT / "config_graphrag"
        paths = sorted(config_dir.glob("*_drift.yaml"))
        self.assertEqual(
            {path.name for path in paths},
            OV_WIKI_DRIFT_CONFIGS | {"versionrag_drift.yaml"},
        )

        for path in paths:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            options = config["store"]["graphrag"]
            self.assertEqual(config["store"]["type"], "graphrag", msg=path.name)
            self.assertEqual(options["query_mode"], "drift", msg=path.name)
            self.assertEqual(options["indexing_method"], "standard", msg=path.name)
            self.assertEqual(options["concurrent_requests"], 8, msg=path.name)
            self.assertTrue(options["show_index_progress"], msg=path.name)
            with tempfile.TemporaryDirectory() as tmp:
                store = self._make_store(tmp, **options)
            self.assertEqual(
                store._config.embed_text.batch_size,
                10,
                msg=path.name,
            )
            self.assertEqual(
                store._config.embed_text.batch_max_tokens,
                8191,
                msg=path.name,
            )
            self.assertEqual(
                store._config.embed_text.names,
                ["entity_description", "community_full_content"],
                msg=path.name,
            )
            self.assertEqual(store._config.drift_search.n_depth, 3, msg=path.name)
            self.assertEqual(
                store._config.drift_search.drift_k_followups, 20, msg=path.name
            )

    def test_ov_wiki_drift_configs_match_existing_dataset_adapters_and_inputs(self):
        graph_dir = OV_TEST_ROOT / "config_graphrag"
        baseline_dir = OV_TEST_ROOT / "config_ov_wiki"
        for graph_name in sorted(OV_WIKI_DRIFT_CONFIGS):
            base_name = graph_name.removesuffix("_drift.yaml") + ".yaml"
            graph_config = yaml.safe_load(
                (graph_dir / graph_name).read_text(encoding="utf-8")
            )
            baseline_config = yaml.safe_load(
                (baseline_dir / base_name).read_text(encoding="utf-8")
            )
            self.assertEqual(
                graph_config["dataset_name"],
                baseline_config["dataset_name"],
                msg=graph_name,
            )
            self.assertEqual(
                graph_config["adapter"], baseline_config["adapter"], msg=graph_name
            )
            self.assertEqual(
                graph_config["paths"]["raw_data"],
                baseline_config["paths"]["raw_data"],
                msg=graph_name,
            )

    def test_shared_corpora_use_one_graphrag_index(self):
        graph_dir = OV_TEST_ROOT / "config_graphrag"

        def index_paths(names):
            configs = [
                yaml.safe_load((graph_dir / name).read_text(encoding="utf-8"))
                for name in names
            ]
            return {
                (
                    config["paths"]["doc_output_dir"],
                    config["paths"]["vector_store"],
                )
                for config in configs
            }

        self.assertEqual(
            len(
                index_paths(
                    [
                        "mudabench_simple_drift.yaml",
                        "mudabench_complex_drift.yaml",
                    ]
                )
            ),
            1,
        )
        for size in (57, 93):
            self.assertEqual(
                len(
                    index_paths(
                        [
                            f"paperscope_summary_{size}_gap_drift.yaml",
                            f"paperscope_summary_{size}_results_comparison_drift.yaml",
                            f"paperscope_summary_{size}_trend_drift.yaml",
                        ]
                    )
                ),
                1,
            )

    def test_prepare_documents_is_stable_and_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "document.md"
            document.write_text("Alpha and Beta are related.", encoding="utf-8")
            store = self._make_store(str(root / "store"))

            first, first_fingerprint, sources = store._prepare_input_documents(
                [StandardDoc(sample_id="sample", doc_paths=[str(document)])]
            )
            second, second_fingerprint, _ = store._prepare_input_documents(
                [StandardDoc(sample_id="sample", doc_paths=[str(document)])]
            )

            self.assertEqual(first_fingerprint, second_fingerprint)
            self.assertEqual(first.loc[0, "id"], second.loc[0, "id"])
            self.assertEqual(first.loc[0, "raw_data"]["sample_id"], "sample")
            self.assertEqual(sources[0]["source_path"], str(document.resolve()))

    def test_prepare_pdf_records_extractor_and_page_count(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "document.pdf"
            with pymupdf.open() as pdf:
                page = pdf.new_page()
                page.insert_text((72, 72), "PDF source for GraphRAG")
                pdf.save(document)
            store = self._make_store(str(root / "store"))

            frame, _, sources = store._prepare_input_documents(
                [StandardDoc(sample_id="sample", doc_paths=[str(document)])]
            )

        self.assertIn("PDF source for GraphRAG", frame.loc[0, "text"])
        self.assertEqual(frame.loc[0, "raw_data"]["text_extractor"], "pymupdf")
        self.assertEqual(frame.loc[0, "raw_data"]["page_count"], 1)
        self.assertEqual(sources[0]["text_extractor"], "pymupdf")
        self.assertEqual(sources[0]["page_count"], 1)

    def test_retrieve_returns_direct_answer_context_and_internal_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp, query_mode="drift")
            store.manifest_path.write_text(
                json.dumps({"complete": True}), encoding="utf-8"
            )
            for path in store._required_output_paths():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test-placeholder")

            with patch.object(store, "_create_query_engine", return_value=_FakeEngine()):
                result = store.retrieve("What is related?", topk=5)

            self.assertEqual(result.answer, "final answer")
            self.assertGreater(result.retrieve_input_tokens, 123)
            self.assertEqual(result.retrieve_output_tokens, 17)
            self.assertEqual(result.llm_calls, 2)
            self.assertGreater(result.input_tokens_categories["query_embedding"], 0)
            self.assertGreater(
                result.input_tokens_categories["user_query_messages"], 0
            )
            recalled, prompts, uris = store.process_retrieval_results(result)
            self.assertEqual(recalled, ["first context", "second context"])
            self.assertEqual(prompts, recalled)
            self.assertEqual(uris, ["graphrag://drift/Sources/source-1"])
            self.assertTrue(store_provides_final_answer(store))
            self.assertEqual(store.get_final_answer(result), "final answer")

    def test_ingest_writes_complete_manifest_and_reuses_matching_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            document = root / "document.txt"
            document.write_text("A small offline corpus.", encoding="utf-8")
            store = self._make_store(str(root / "store"), query_mode="basic")
            samples = [StandardDoc(sample_id="sample", doc_paths=[str(document)])]

            async def fake_build_index(**_kwargs):
                for output in store._required_output_paths():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"parquet-placeholder")
                return [SimpleNamespace(workflow="done", error=None)]

            metrics = [
                {"prompt_tokens": 10, "completion_tokens": 2},
                {"prompt_tokens": 35, "completion_tokens": 9},
            ]
            with (
                patch("graphrag.api.build_index", side_effect=fake_build_index),
                patch.object(store, "_collect_model_metrics", side_effect=metrics),
            ):
                stats = store.ingest(samples)

            manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["document_count"], 1)
            self.assertEqual(stats["input_tokens"], 25)
            self.assertEqual(stats["output_tokens"], 7)
            self.assertNotIn("chat-secret", json.dumps(manifest))

            reused = store.ingest(samples)
            self.assertEqual(reused, stats)

    def test_context_helpers_support_dataframe_and_nested_context(self):
        frame = pd.DataFrame([{"id": 7, "text": "evidence"}])
        blocks = _flatten_context_text({"Sources": frame, "Extra": ["summary"]})
        uris = _context_source_uris({"Sources": frame}, "global")

        self.assertEqual(blocks, ["evidence", "summary"])
        self.assertEqual(uris, ["graphrag://global/Sources/7"])


class StoreContractTests(unittest.TestCase):
    def test_contract_distinguishes_retriever_and_end_to_end_store(self):
        class PlainStore:
            pass

        class FinalStore:
            def get_final_answer(self, _result):
                return "answer"

        self.assertFalse(store_provides_final_answer(PlainStore()))
        self.assertTrue(store_provides_final_answer(FinalStore()))

    def test_pipeline_uses_final_answer_strategy_without_external_generation(self):
        class Adapter:
            def build_prompt(self, *_args):
                raise AssertionError("external prompt generation must not run")

        class LLM:
            def generate(self, *_args):
                raise AssertionError("external LLM generation must not run")

        class FinalStore:
            def retrieve(self, **_kwargs):
                return SimpleNamespace(
                    retrieve_input_tokens=11,
                    retrieve_output_tokens=4,
                    answer="internal answer",
                )

            def process_retrieval_results(self, _result):
                return ["supporting evidence"], ["supporting evidence"], ["uri"]

            def get_final_answer(self, result):
                return result.answer

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "dataset_name": "test",
                "store": {"type": "future_agentic_store"},
                "paths": {"output_dir": tmp},
                "execution": {"retrieval_topk": 5, "max_workers": 1},
                "llm": {"model": "judge"},
            }
            pipeline = BenchmarkPipeline(config, Adapter(), FinalStore(), LLM())
            qa = StandardQA(
                question="question",
                gold_answers=["internal answer"],
                evidence=["supporting evidence"],
            )
            result = pipeline._process_generation_task(
                {"id": 0, "sample_id": "sample", "qa": qa}
            )

        self.assertEqual(result["llm"]["final_answer"], "internal answer")
        self.assertEqual(result["token_usage"]["total_input_tokens"], 11)
        self.assertEqual(result["token_usage"]["llm_output_tokens"], 4)
        self.assertEqual(result["retrieval"]["latency_scope"], "end_to_end")


if __name__ == "__main__":
    unittest.main()
