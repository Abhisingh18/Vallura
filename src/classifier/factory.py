"""
Classifier factory — selects the concrete classifier implementation at startup.

Strategy:
  - If OPENAI_API_KEY is set and non-empty → OpenAIClassifier (LLM-powered)
  - Otherwise → RuleClassifier (zero-cost fallback)

This ensures the system always works, even without API credentials.
"""
from __future__ import annotations

import os
import logging

from src.classifier.base import BaseClassifier

logger = logging.getLogger(__name__)


def create_classifier() -> BaseClassifier:
    """
    Factory function. Returns the best available classifier.

    Priority:
      1. Groq (Fastest LLM)
      2. OpenAI (High-quality fallback)
      3. Rules (Zero-cost fallback)
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if groq_key and not groq_key.startswith("your_"):
        logger.info("GROQ_API_KEY found -> using GroqClassifier")
        from src.classifier.groq_classifier import GroqClassifier
        return GroqClassifier()

    if openai_key and not openai_key.startswith("sk-..."):
        logger.info("OPENAI_API_KEY found -> using OpenAIClassifier")
        from src.classifier.openai_classifier import OpenAIClassifier
        return OpenAIClassifier()

    logger.info("No LLM keys found → using RuleClassifier (fallback)")
    from src.classifier.rule_classifier import RuleClassifier
    return RuleClassifier()
