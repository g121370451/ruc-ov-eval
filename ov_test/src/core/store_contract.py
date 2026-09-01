"""Small behavior contracts shared by benchmark store implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class FinalAnswerStore(Protocol):
    """A store whose query strategy already produced the final answer."""

    def get_final_answer(self, search_result: Any) -> str:
        """Return the final answer carried by one retrieval result."""


def store_provides_final_answer(store: Any) -> bool:
    """Return whether ``store`` implements the end-to-end answer strategy."""

    return isinstance(store, FinalAnswerStore)
