"""
Router tests — verifies agent dispatch, stub responses, and error handling.
"""
import pytest

from src.models.schemas import ClassifierResult, ExtractedEntities
from src.router.router import route


@pytest.mark.asyncio
async def test_route_portfolio_health():
    """Portfolio health agent should return structured health data."""
    result = ClassifierResult(
        intent="portfolio_health",
        agent="portfolio_health",
    )
    response = await route(result, user_id="usr_001")
    assert response is not None
    assert "concentration_risk" in response or "message" in response


@pytest.mark.asyncio
async def test_route_stub_agent():
    """Unimplemented agents should return a structured stub response."""
    result = ClassifierResult(
        intent="market_research",
        agent="market_research",
        entities=ExtractedEntities(tickers=["AAPL"]),
    )
    response = await route(result, user_id="usr_001")
    assert response is not None
    # Check for implemented research response
    assert "research" in response["message"].lower() or "price" in response["message"].lower()
    assert response["agent"] == "market_research"


@pytest.mark.asyncio
async def test_route_unknown_agent():
    """Completely unknown agents should still get a stub, not crash."""
    result = ClassifierResult(
        intent="unknown_intent",
        agent="nonexistent_agent",
    )
    response = await route(result)
    assert response is not None
    assert "message" in response


@pytest.mark.asyncio
async def test_route_missing_user():
    """Missing user_id should not crash portfolio_health."""
    result = ClassifierResult(
        intent="portfolio_health",
        agent="portfolio_health",
    )
    response = await route(result, user_id="usr_999")
    assert response is not None
    # Should get empty portfolio guidance since user doesn't exist
    assert "observations" in response


@pytest.mark.asyncio
async def test_route_all_agent_types():
    """Every agent type from the taxonomy should return without crashing."""
    agents = [
        "portfolio_health", "market_research", "investment_strategy",
        "financial_planning", "financial_calculator", "risk_assessment",
        "product_recommendation", "predictive_analysis",
        "customer_support", "general_query",
    ]
    for agent_name in agents:
        result = ClassifierResult(intent="test", agent=agent_name)
        response = await route(result)
        assert response is not None, f"Agent '{agent_name}' returned None"
