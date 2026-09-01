from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


OV_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(OV_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(OV_TEST_ROOT))

from src.core.graphrag_progress import GraphRAGProgressCallbacks


class _FakeBar:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.total = kwargs.get("total")
        self.n = int(kwargs.get("initial", 0))
        self.closed = False
        self.postfixes = []

    def update(self, amount):
        self.n += amount

    def set_postfix_str(self, value, refresh=True):
        self.postfixes.append((value, refresh))

    def close(self):
        self.closed = True


class GraphRAGProgressCallbacksTests(unittest.TestCase):
    def test_renders_workflow_and_item_progress_and_closes_cleanly(self):
        bars = []

        def factory(**kwargs):
            bar = _FakeBar(**kwargs)
            bars.append(bar)
            return bar

        callbacks = GraphRAGProgressCallbacks(
            enabled=True,
            mininterval=0.25,
            tqdm_factory=factory,
        )
        callbacks.pipeline_start(["extract_graph", "generate_text_embeddings"])
        workflow_bar = bars[0]
        callbacks.workflow_start("extract_graph", None)
        callbacks.progress(
            SimpleNamespace(
                description="extract graph progress: ",
                total_items=10,
                completed_items=2,
            )
        )
        item_bar = bars[1]
        callbacks.progress(
            SimpleNamespace(
                description="extract graph progress: ",
                total_items=10,
                completed_items=7,
            )
        )

        self.assertEqual(workflow_bar.total, 2)
        self.assertEqual(item_bar.total, 10)
        self.assertEqual(item_bar.n, 7)
        self.assertEqual(item_bar.kwargs["mininterval"], 0.25)

        callbacks.progress(
            SimpleNamespace(
                description="extract graph progress: ",
                total_items=10,
                completed_items=10,
            )
        )
        self.assertTrue(item_bar.closed)
        bar_count = len(bars)

        # GraphRAG emits a final done event after the last tick. It must not
        # reopen an already-completed progress bar.
        callbacks.progress(
            SimpleNamespace(
                description="extract graph progress: ",
                total_items=10,
                completed_items=10,
            )
        )
        self.assertEqual(len(bars), bar_count)

        callbacks.workflow_end("extract_graph", None)
        callbacks.workflow_start("generate_text_embeddings", None)
        callbacks.workflow_end("generate_text_embeddings", None)
        self.assertEqual(workflow_bar.n, 2)

        callbacks.pipeline_end([])
        self.assertTrue(workflow_bar.closed)

    def test_disabled_callbacks_create_no_bars(self):
        bars = []
        callbacks = GraphRAGProgressCallbacks(
            enabled=False,
            tqdm_factory=lambda **kwargs: bars.append(kwargs),
        )
        callbacks.pipeline_start(["extract_graph"])
        callbacks.workflow_start("extract_graph", None)
        callbacks.progress(
            SimpleNamespace(
                description="progress",
                total_items=1,
                completed_items=1,
            )
        )
        callbacks.workflow_end("extract_graph", None)
        callbacks.pipeline_end([])
        self.assertEqual(bars, [])


if __name__ == "__main__":
    unittest.main()
