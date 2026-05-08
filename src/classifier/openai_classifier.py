"""
OpenAI-powered intent classifier.

Uses a single structured-output LLM call to classify user queries.
Falls back to RuleClassifier on any failure (network, rate-limit, bad parse).

Model selection:
  - Development: gpt-4o-mini (fast, cheap)
  - Evaluation: gpt-4.1 (higher accuracy)
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
- portfolio_health: structured assessment of portfolio (concentration, performance, benchmarking)
- market_research: factual info about instruments, sectors, markets, prices, news
- investment_strategy: buy/sell/rebalance/allocation advice
- financial_planning: long-term planning (retirement, education, house, FIRE, savings)
- financial_calculator: deterministic computation (DCA, mortgage, tax, FV, FX conversion)
- risk_assessment: risk metrics, exposure analysis, stress testing, what-if
- product_recommendation: recommend specific funds/ETFs matching user profile
- predictive_analysis: forecasts, trend extrapolation, forward-looking
- customer_support: platform issues, account questions, how-to-use-app
- general_query: greetings, educational questions, definitions, conversational

## Entity Fields (include only those present in the query)
- tickers: array of uppercase ticker symbols (AAPL, NVDA, ASML.AS, 7203.T)
  - Map company names to tickers: Apple→AAPL, Microsoft→MSFT, Nvidia→NVDA, Google→GOOGL, Tesla→TSLA, Amazon→AMZN, Meta→META, AMD→AMD, HSBC→HSBA.L, Barclays→BARC.L, ASML→ASML.AS, Toyota→7203.T
  - Handle typos (e.g., "microsfot" → MSFT)
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
- If the user refers to something from prior turns (pronouns, "it", "them", "that stock"), resolve the reference.
- If the user names a new topic explicitly, do NOT carry context from prior turns.
- Follow-ups like "what about X?" should carry the INTENT from prior turns but switch the entity.
- Closers like "thanks", "thx", "ok" → general_query, no entities.

## Multi-intent
- If a query has multiple intents (e.g. an educational question AND a portfolio question), ALWAYS choose the specialist intent (e.g. portfolio_health) over general_query.
- "how is my portfolio doing and what should I sell?" → portfolio_health (primary)
- "What is an ETF and how is my portfolio performing?" → portfolio_health (primary)

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


class OpenAIClassifier(BaseClassifier):
    """
    LLM-based classifier using OpenAI's API with structured output.

    On any failure, degrades gracefully to RuleClassifier.
    """

    def __init__(self) -> None:
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._fallback = RuleClassifier()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-init the OpenAI client so import doesn't fail without the key."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(max_retries=0)
            except Exception:
                logger.warning("Failed to initialize OpenAI client")
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
            logger.warning("OpenAI classifier failed, falling back to rules: %s", exc)
            return await self._fallback.classify(query, history)

    async def _llm_classify(
        self,
        client: Any,
        query: str,
        history: list[dict[str, str]] | None,
    ) -> ClassifierResult:
        """Make the actual LLM call."""
        # Build history block
        history_block = ""
        if history:
            turns = []
            for msg in history[-6:]:  # last 6 turns for context
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
            source="openai",
        )
