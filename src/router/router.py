"""
Agent router — dispatches classifier results to the correct specialist agent.

The router is the central switch that maps agent names to implementations.
Adding a new agent requires only:
  1. Creating a new module in src/agents/
  2. Adding one entry to _AGENT_REGISTRY below

Design: the router never crashes. Unknown agents get a stub response.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agents import portfolio_health, stub_agent
from src.models.schemas import ClassifierResult

logger = logging.getLogger(__name__)

# Fixtures directory for loading user data
_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"

# Registry of implemented agents (agent_name → module with run())
_IMPLEMENTED_AGENTS: set[str] = {"portfolio_health", "market_research"}


def _load_user(user_id: str) -> dict[str, Any] | None:
    """Load a user profile from fixtures by user_id."""
    users_dir = _FIXTURES_DIR / "users"
    if not users_dir.exists():
        return None

    for path in users_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user.get("user_id") == user_id:
                return user
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Failed to load user fixture %s: %s", path, exc)
    return None


async def route(
    classifier_result: ClassifierResult,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Route a classified query to the appropriate agent.

    Args:
        classifier_result: Output from the intent classifier.
        user_id: Optional user ID for agents that need user data.

    Returns:
        Agent response as a dict. Always succeeds — never crashes.
    """
    agent_name = classifier_result.agent

    try:
        if agent_name == "portfolio_health":
            return await _run_portfolio_health(classifier_result, user_id)
        elif agent_name == "market_research":
            from src.agents import market_research
            return await market_research.run(classifier_result)
        else:
            # Unimplemented agent → structured stub
            return stub_agent.run(classifier_result)

    except Exception as exc:
        logger.error("Agent %s failed: %s", agent_name, exc, exc_info=True)
        return {
            "agent": agent_name,
            "error": True,
            "message": f"Agent '{agent_name}' encountered an error: {str(exc)}",
            "intent": classifier_result.intent,
            "entities": classifier_result.entities.model_dump(exclude_none=True),
        }


async def _run_portfolio_health(
    classifier_result: ClassifierResult,
    user_id: str | None,
) -> dict[str, Any]:
    """Load user data and run the portfolio health agent."""
    if user_id:
        user = _load_user(user_id)
    else:
        user = None

    if user is None:
        # No user data — provide generic guidance
        user = {
            "user_id": user_id or "unknown",
            "name": "Investor",
            "risk_profile": "moderate",
            "positions": [],
            "preferences": {"preferred_benchmark": "S&P 500"},
        }

    from src.utils.llm import generate_insight
    
    # Generate an LLM insight if we have positions
    llm_insight = None
    if user.get("positions"):
        # Summarize portfolio for the LLM
        summary = f"User has {len(user['positions'])} positions. "
        summary += f"Risk profile: {user['risk_profile']}. "
        
        prompt = (
            f"Explain why diversification is important for a {user['risk_profile']} investor "
            f"in one or two short sentences. Focus on the user's situation: {summary}"
        )
        llm_insight = await generate_insight(prompt)

    result = portfolio_health.run(user, llm_insight=llm_insight)

    # Enrich with classifier metadata
    result["_meta"] = {
        "agent": "portfolio_health",
        "intent": classifier_result.intent,
        "entities": classifier_result.entities.model_dump(exclude_none=True),
        "source": classifier_result.source,
    }

    return result
