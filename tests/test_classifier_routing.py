"""
Classifier routing accuracy tests against the labeled gold set.

Uses the RuleClassifier (fallback) for all tests since CI runs without
OPENAI_API_KEY. The OpenAI classifier is tested separately when credentials
are available.

Threshold: ≥ 85% routing accuracy (from ASSIGNMENT.md).
"""
from typing import Any

import pytest

from src.classifier.rule_classifier import RuleClassifier


# ---------------------------------------------------------------------------
# Entity matcher — implements the rules in fixtures/README.md
# ---------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            # Catch-all for vocabulary tokens
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    return RuleClassifier()


@pytest.mark.asyncio
async def test_classifier_routing_accuracy(gold_classifier_queries, classifier):
    """
    Threshold: ≥ 85% routing accuracy.
    """
    correct = 0
    misses = []

    for case in gold_classifier_queries:
        result = await classifier.classify(case["query"])
        if result.agent == case["expected_agent"]:
            correct += 1
        else:
            misses.append(
                f"  Query: '{case['query']}' | Expected: {case['expected_agent']} | Got: {result.agent}"
            )

    accuracy = correct / len(gold_classifier_queries)

    if misses:
        print("\n--- Classifier Misses ---")
        for m in misses:
            print(m)

    print(f"\nRouting accuracy: {accuracy:.2%} ({correct}/{len(gold_classifier_queries)})")

    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


@pytest.mark.asyncio
async def test_classifier_entity_extraction(gold_classifier_queries, classifier):
    """
    Soft signal — not a hard threshold. Reported, not failed on.
    """
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = await classifier.classify(case["query"])
        entities_dict = result.entities.model_dump(exclude_none=True)
        # Remove empty lists
        entities_dict = {k: v for k, v in entities_dict.items() if v != [] and v is not None}
        if matches_entities(entities_dict, case["expected_entities"]):
            matched += 1
        else:
            print(
                f"  Entity miss: '{case['query']}' | "
                f"Expected: {case['expected_entities']} | "
                f"Got: {entities_dict}"
            )

    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")


@pytest.mark.asyncio
async def test_classifier_greetings(classifier):
    """Greetings should route to general_query."""
    for greeting in ["hi", "hello", "thanks", "thx"]:
        result = await classifier.classify(greeting)
        assert result.agent == "general_query", f"'{greeting}' should → general_query, got {result.agent}"


@pytest.mark.asyncio
async def test_classifier_portfolio_queries(classifier):
    """Portfolio queries should route to portfolio_health."""
    queries = [
        "how is my portfolio doing",
        "give me a health check on my investments",
        "portfolio summary",
        "is my portfolio well diversified?",
    ]
    for q in queries:
        result = await classifier.classify(q)
        assert result.agent == "portfolio_health", f"'{q}' should → portfolio_health, got {result.agent}"


@pytest.mark.asyncio
async def test_classifier_single_ticker(classifier):
    """Single ticker should route to market_research."""
    result = await classifier.classify("AAPL")
    assert result.agent == "market_research"
    assert "AAPL" in [t.upper().split(".")[0] for t in result.entities.tickers]


@pytest.mark.asyncio
async def test_classifier_gibberish(classifier):
    """Gibberish should not crash, route to general_query."""
    result = await classifier.classify("abcdefg")
    assert result.agent == "general_query"
