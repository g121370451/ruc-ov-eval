"""Terminal progress reporting for Microsoft GraphRAG indexing."""

from __future__ import annotations

import threading
from typing import Any, Callable

from tqdm import tqdm


class GraphRAGProgressCallbacks:
    """Render GraphRAG workflow and item progress with elapsed time and ETA."""

    def __init__(
        self,
        enabled: bool = True,
        mininterval: float = 0.5,
        tqdm_factory: Callable[..., Any] = tqdm,
    ) -> None:
        self.enabled = enabled
        self.mininterval = mininterval
        self._tqdm_factory = tqdm_factory
        self._lock = threading.RLock()
        self._workflow_bar = None
        self._item_bars: dict[tuple[str, str, int | None], Any] = {}
        self._finished_items: set[tuple[str, str, int | None]] = set()
        self._current_workflow = ""
        self._next_position = 1
        self._closed = False

    def pipeline_start(self, names: list[str]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._closed = False
            self._workflow_bar = self._tqdm_factory(
                total=len(names),
                desc="GraphRAG indexing",
                unit="workflow",
                position=0,
                leave=True,
                dynamic_ncols=True,
                mininterval=self.mininterval,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} workflows "
                    "[{elapsed}] {postfix}"
                ),
            )

    def pipeline_end(self, _results: list[Any]) -> None:
        self.close()

    def workflow_start(self, name: str, _instance: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._close_item_bars()
            self._current_workflow = name
            self._finished_items.clear()
            self._next_position = 1
            if self._workflow_bar is not None:
                self._workflow_bar.set_postfix_str(f"current={name}", refresh=True)

    def workflow_end(self, name: str, _instance: object) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._close_item_bars()
            if self._workflow_bar is not None:
                self._workflow_bar.update(1)
                self._workflow_bar.set_postfix_str(
                    f"completed={name}", refresh=True
                )

    def progress(self, progress: Any) -> None:
        if not self.enabled:
            return

        description = str(
            getattr(progress, "description", None) or self._current_workflow or "progress"
        ).strip().rstrip(":")
        total = getattr(progress, "total_items", None)
        completed = getattr(progress, "completed_items", None)
        total = int(total) if total is not None else None
        completed = int(completed) if completed is not None else 0
        key = (self._current_workflow, description, total)

        with self._lock:
            if total is not None and completed >= total and key in self._finished_items:
                return

            bar = self._item_bars.get(key)
            if bar is None or completed < getattr(bar, "n", 0):
                if bar is not None:
                    bar.close()
                bar = self._tqdm_factory(
                    total=total,
                    desc=f"  {description}",
                    unit="item",
                    position=self._next_position,
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=self.mininterval,
                )
                self._next_position += 1
                self._item_bars[key] = bar

            delta = completed - int(getattr(bar, "n", 0))
            if delta > 0:
                bar.update(delta)

            if total is not None and completed >= total:
                bar.close()
                self._item_bars.pop(key, None)
                self._finished_items.add(key)

    def pipeline_error(self, _error: BaseException) -> None:
        self.close()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._close_item_bars()
            if self._workflow_bar is not None:
                self._workflow_bar.close()
                self._workflow_bar = None
            self._closed = True

    def _close_item_bars(self) -> None:
        for bar in self._item_bars.values():
            bar.close()
        self._item_bars.clear()
