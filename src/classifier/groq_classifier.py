
"""
Groq-powered intent classifier.

Uses a single structured-output call to classify user queries using Groq's high-speed inference.
Falls back to RuleClassifier on any failure (network, rate-limit, bad parse).

Model selection:
  - Default: llama3-70b-8192 (high accuracy, fast)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.classifier.base import BaseClassifier
from src.classifier.rule_classifier import RuleClassifier
from src.models.schemas import ClassifierResult, ExtractedEntities

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for a financial assistant platform called Valura.

Your job: Given a user query (and optional conversation history), return a JSON object with:
1. **intent** — the user's primary intent
2. **agent** — which agent should handle this (exactly one of the taxonomy below)
3. **entities** — extracted entities from the query
4. **safety_verdict** — "safe" or "flagged" (informational only; the safety guard is the authority)
5. **confidence** — your confidence 0.0–1.0

## Agent Taxonomy (use EXACTLY these strings)
- portfolio_health: structured assessment of the USER'S OWN portfolio (concentration, performance, benchmarking). Use this when the user asks about "my portfolio" or "how am I doing".
- market_research: factual info about instruments, symbols, tickers, sectors, markets, prices, news. Use this for ANY question about a specific stock (e.g., "tell me about NVDA", "AAPL price").
- investment_strategy: buy/sell/rebalance/allocation advice
- financial_planning: long-term planning (retirement, education, house, FIRE, savings)
- financial_calculator: deterministic computation (DCA, mortgage, tax, FV, FX conversion)
- risk_assessment: risk metrics, exposure analysis, stress testing, what-if
- product_recommendation: recommend specific funds/ETFs matching user profile
- predictive_analysis: forecasts, trend extrapolation, forward-looking
- customer_support: platform issues, account questions, how-to-use-app
- general_query: greetings, educational questions, definitions, conversational (ONLY if no specific ticker or portfolio context is present).

## Entity Fields (include only those present in the query)
- tickers: array of uppercase ticker symbols (AAPL, NVDA, ASML.AS, 7203.T)
  - Map company names to tickers: Apple→AAPL, Microsoft→MSFT, Nvidia→NVDA, Google→GOOGL, Tesla→TSLA, Amazon→AMZN, Meta→META, AMD→AMD
- amount: number
- currency: ISO 4217 (USD, EUR, GBP, JPY)
- rate: decimal (0.08 for 8%)
- period_years: integer
- frequency: daily | weekly | monthly | yearly
- horizon: 6_months | 1_year | 5_years | etc.
- time_period: today | this_week | this_month | this_year
- topics: array of strings
- sectors: array of strings
- index: S&P 500 | FTSE 100 | NIKKEI 225 | MSCI World
- action: buy | sell | hold | hedge | rebalance
- goal: retirement | education | house | FIRE | emergency_fund

## Conversation Context Rules
- If any ticker symbol (e.g., NVDA, AAPL) is present, NEVER use general_query. Use market_research or portfolio_health.
- If the user refers to something from prior turns (pronouns, "it", "them", "that stock"), resolve the reference.
- If the user names a new topic explicitly, do NOT carry context from prior turns.
- Follow-ups like "what about X?" should carry the INTENT from prior turns but switch the entity.

Return ONLY valid JSON. No markdown. No explanation."""

_USER_TEMPLATE = """Query: {query}

{history_block}

Return your classification as JSON:
{{
  "intent": "...",
  "agent": "...",
  "entities": {{}},
  "safety_verdict": "safe",
  "confidence": 0.0
}}"""


class GroqClassifier(BaseClassifier):
    """
    LLM-based classifier using Groq's API.

    On any failure, degrades gracefully to RuleClassifier.
    """

    def __init__(self) -> None:
        self._model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        self._fallback = RuleClassifier()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    max_retries=0,
                )
            except Exception:
                logger.warning("Failed to initialize Groq client")
                return None
        return self._client

    async def classify(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ClassifierResult:
        client = self._get_client()
        if client is None:
            return await self._fallback.classify(query, history)

        try:
            return await self._llm_classify(client, query, history)
        except Exception as exc:
            logger.warning("Groq classifier failed, falling back to rules: %s", exc)
            return await self._fallback.classify(query, history)

    async def _llm_classify(
        self,
        client: Any,
        query: str,
        history: list[dict[str, str]] | None,
    ) -> ClassifierResult:
        """Make the actual LLM call to Groq."""
        # Build history block
        history_block = ""
        if history:
            turns = []
            for msg in history[-6:]:
                turns.append(f"{msg['role'].upper()}: {msg['content']}")
            history_block = "Conversation history:\n" + "\n".join(turns)

        user_msg = _USER_TEMPLATE.format(query=query, history_block=history_block)

        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
            timeout=5.0,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Parse entities
        raw_entities = data.get("entities", {})
        entities = ExtractedEntities(
            tickers=raw_entities.get("tickers", []),
            amount=raw_entities.get("amount"),
            currency=raw_entities.get("currency"),
            rate=raw_entities.get("rate"),
            period_years=raw_entities.get("period_years"),
            frequency=raw_entities.get("frequency"),
            horizon=raw_entities.get("horizon"),
            time_period=raw_entities.get("time_period"),
            topics=raw_entities.get("topics", []),
            sectors=raw_entities.get("sectors", []),
            index=raw_entities.get("index"),
            action=raw_entities.get("action"),
            goal=raw_entities.get("goal"),
        )

        return ClassifierResult(
            intent=data.get("intent", "unknown"),
            agent=data.get("agent", "general_query"),
            entities=entities,
            safety_verdict=data.get("safety_verdict", "safe"),
            confidence=float(data.get("confidence", 0.8)),
            source="groq",
        )
