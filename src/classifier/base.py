"""
Abstract base for intent classifiers.

Classifiers implement a single method: `classify(query, history) -> ClassifierResult`.
The factory module selects the concrete implementation at startup based on env config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.schemas import ClassifierResult


class BaseClassifier(ABC):
    """Contract that all classifiers must satisfy."""

    @abstractmethod
    async def classify(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ClassifierResult:
        """
        Classify a user query into an intent, agent, and extracted entities.

        Args:
            query: The raw user query string.
            history: Optional list of prior conversation turns
                     [{"role": "user", "content": "..."}, ...]

        Returns:
            ClassifierResult with intent, agent, entities, confidence, and source.
        """
        ...
