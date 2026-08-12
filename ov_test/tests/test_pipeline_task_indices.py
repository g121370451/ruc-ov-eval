import json
import os
import tempfile
import unittest

from src.adapters.base import StandardQA, StandardSample
from src.pipeline import BenchmarkPipeline


class PipelineTaskIndicesTest(unittest.TestCase):
    def _pipeline(self, manifest_path=None, case_manifest_path=None):
        pipeline = BenchmarkPipeline.__new__(BenchmarkPipeline)
        pipeline.config = {
            "execution": {
                "max_queries": None,
                "worker_id": None,
                "num_workers": None,
            },
            "paths": {
                "task_indices": manifest_path,
                "task_cases": case_manifest_path,
            },
        }
        pipeline.logger = type(
            "Logger", (), {"info": staticmethod(lambda *args, **kwargs: None)}
        )()
        return pipeline

    @staticmethod
    def _samples():
        return [
            StandardSample(
                sample_id="sample-a",
                qa_pairs=[
                    StandardQA(question="q0", gold_answers=["a0"]),
                    StandardQA(question="q1", gold_answers=["a1"]),
                ],
            ),
            StandardSample(
                sample_id="sample-b",
                qa_pairs=[
                    StandardQA(question="q2", gold_answers=["a2"]),
                    StandardQA(question="q3", gold_answers=["a3"]),
                ],
            ),
        ]

    def test_selects_exact_global_indices_and_preserves_source_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "indices.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump([3, 1], f)

            tasks = self._pipeline(manifest)._prepare_tasks(self._samples())

        self.assertEqual([task["id"] for task in tasks], [1, 3])
        self.assertEqual([task["qa"].question for task in tasks], ["q1", "q3"])

    def test_rejects_indices_not_present_in_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "indices.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump([99], f)

            with self.assertRaisesRegex(ValueError, "unavailable global indices"):
                self._pipeline(manifest)._prepare_tasks(self._samples())

    def test_stable_cases_do_not_depend_on_global_index_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "cases.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"sample_id": "sample-b", "question": "q3", "occurrence": 0},
                        {"sample_id": "sample-a", "question": "q0", "occurrence": 0},
                    ],
                    f,
                )

            tasks = self._pipeline(case_manifest_path=manifest)._prepare_tasks(
                self._samples()
            )

        self.assertEqual(
            [(task["sample_id"], task["qa"].question) for task in tasks],
            [("sample-a", "q0"), ("sample-b", "q3")],
        )

    def test_stable_cases_fail_fast_on_dataset_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "cases.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(
                    [{
                        "sample_id": "sample-a",
                        "question": "changed question",
                        "occurrence": 0,
                    }],
                    f,
                )

            with self.assertRaisesRegex(ValueError, "unavailable in the current dataset"):
                self._pipeline(case_manifest_path=manifest)._prepare_tasks(
                    self._samples()
                )

    def test_stable_cases_allow_all_identical_duplicate_questions(self):
        duplicate_samples = [
            StandardSample(
                sample_id="sample-a",
                qa_pairs=[
                    StandardQA(question="same", gold_answers=["answer"]),
                    StandardQA(question="same", gold_answers=["answer"]),
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = os.path.join(temp_dir, "cases.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {"sample_id": "sample-a", "question": "same", "occurrence": 0},
                        {"sample_id": "sample-a", "question": "same", "occurrence": 1},
                    ],
                    f,
                )

            tasks = self._pipeline(case_manifest_path=manifest)._prepare_tasks(
                duplicate_samples
            )

        self.assertEqual(len(tasks), 2)


if __name__ == "__main__":
    unittest.main()
