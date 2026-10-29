"""Common interface for classifier backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from ..models import Candidate, Prediction


class Classifier(ABC):
    """A backend that assigns an adoption state to candidate sentences."""

    name: str = "base"

    @abstractmethod
    def classify(self, candidates: Sequence[Candidate]) -> list[Prediction]:
        """Return one :class:`Prediction` per candidate, in order."""

    def config(self) -> dict[str, Any]:
        """Configuration recorded in result provenance."""
        return {"backend": self.name}


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
