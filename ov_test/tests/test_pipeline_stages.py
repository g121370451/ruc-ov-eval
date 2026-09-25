import os
import tempfile
import threading
import unittest

from src.pipeline import BenchmarkPipeline


class PipelineStageTest(unittest.TestCase):
    def test_ingest_only_stops_before_generation(self):
        class FakeAdapter:
            generation_loaded = False

            def data_prepare(self, doc_dir):
                os.makedirs(doc_dir, exist_ok=True)
                return ["prepared-document"]

            def load_and_transform(self):
                self.generation_loaded = True
                raise AssertionError("generation should not start during ingest-only")

        class FakeDatabase:
            def __init__(self):
                self.calls = []

            def ingest(self, docs, max_workers, monitor):
                self.calls.append((docs, max_workers, monitor))
                return {"time": 1.0, "input_tokens": 2, "output_tokens": 3}

        logger = type(
            "Logger",
            (),
            {
                "info": staticmethod(lambda *args, **kwargs: None),
                "warning": staticmethod(lambda *args, **kwargs: None),
                "error": staticmethod(lambda *args, **kwargs: None),
            },
        )()

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
            pipeline.config = {
                "paths": {
                    "doc_output_dir": os.path.join(temp_dir, "docs"),
                    "vector_store": os.path.join(temp_dir, "store"),
                },
                "execution": {
                    "skip_ingestion": False,
                    "ingest_workers": 2,
                },
            }
            pipeline.output_dir = temp_dir
            pipeline.report_file = os.path.join(temp_dir, "report.json")
            pipeline.records_file = os.path.join(temp_dir, "records.json")
            pipeline.records = {"ingested": False, "tasks": {}}
            pipeline._records_lock = threading.Lock()
            pipeline.metrics_summary = {
                "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
                "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
            }
            pipeline.store_type = "DeepRead"
            pipeline.monitor = object()
            pipeline.logger = logger
            pipeline.adapter = FakeAdapter()
            pipeline.db = FakeDatabase()
            pipeline._update_report = lambda *args, **kwargs: None

            pipeline.run_generation(ingest_only=True)

            self.assertEqual(len(pipeline.db.calls), 1)
            self.assertEqual(pipeline.db.calls[0][0], ["prepared-document"])
            self.assertFalse(pipeline.adapter.generation_loaded)
            self.assertTrue(pipeline.records["ingested"])


if __name__ == "__main__":
    unittest.main()
