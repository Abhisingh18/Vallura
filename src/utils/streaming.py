"""
SSE streaming utilities.

Provides helpers for building Server-Sent Events responses with
progressive status updates and structured final results.

The streaming protocol:
  1. Status events ("Running safety checks...", "Classifying intent...")
  2. Final result event (the agent's structured response)
  3. Error events (if something goes wrong mid-pipeline)

All events are JSON-encoded with an `event` type field for client parsing.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


async def stream_status(message: str) -> str:
    """Format a status update as an SSE event."""
    data = json.dumps({"event": "status", "data": {"message": message}})
    return f"event: status\ndata: {data}\n\n"


async def stream_result(result: dict[str, Any]) -> str:
    """Format the final result as an SSE event."""
    data = json.dumps({"event": "result", "data": result})
    return f"event: result\ndata: {data}\n\n"


async def stream_chunk(text: str) -> str:
    """Format a partial text chunk as an SSE event."""
    data = json.dumps({"event": "chunk", "data": {"text": text}})
    return f"event: chunk\ndata: {data}\n\n"


async def stream_error(error: str, code: str = "internal_error") -> str:
    """Format an error as an SSE event."""
    data = json.dumps({"event": "error", "data": {"error": error, "code": code}})
    return f"event: error\ndata: {data}\n\n"


async def stream_blocked(safety_result: dict[str, Any]) -> str:
    """Format a safety block as an SSE event."""
    data = json.dumps({"event": "blocked", "data": safety_result})
    return f"event: blocked\ndata: {data}\n\n"


async def stream_classification(classifier_result: dict[str, Any]) -> str:
    """Format classifier metadata as an SSE event."""
    data = json.dumps({"event": "classification", "data": classifier_result})
    return f"event: classification\ndata: {data}\n\n"


async def pipeline_stream(
    query: str,
    session_id: str,
    user_id: str,
    safety_check_fn,
    classify_fn,
    route_fn,
    memory,
) -> AsyncGenerator[str, None]:
    """
    Generator that runs the full pipeline and yields SSE events progressively.
    """
    try:
        # Step 1: Safety check
        yield await stream_status("Running safety checks...")
        await asyncio.sleep(0.05)

        safety_result = safety_check_fn(query)

        if safety_result.blocked:
            yield await stream_blocked(safety_result.model_dump())
            return

        # Step 2: Intent classification
        yield await stream_status("Classifying your intent...")
        await asyncio.sleep(0.05)

        history = memory.get_history(session_id) if memory else []
        classifier_result = await classify_fn(query, history)

        yield await stream_classification(classifier_result.model_dump())

        # Step 3: Route to agent
        agent_name = classifier_result.agent
        yield await stream_status(f"Routing to {agent_name} agent...")
        await asyncio.sleep(0.05)

        # Step 4: Execute agent with 5s timeout gate
        status_messages = {
            "portfolio_health": "Fetching live market data & calculating metrics...",
            "market_research": "Researching real-time market data...",
            "investment_strategy": "Evaluating strategy options...",
            "financial_planning": "Building your financial plan...",
            "financial_calculator": "Running calculations...",
            "risk_assessment": "Assessing risk metrics...",
            "product_recommendation": "Finding recommendations...",
            "predictive_analysis": "Running predictive analysis...",
            "customer_support": "Connecting to support...",
            "general_query": "Preparing response...",
        }
        yield await stream_status(status_messages.get(agent_name, "Processing..."))

        try:
            # Wrap the agent execution in a 5s timeout
            result = await asyncio.wait_for(route_fn(classifier_result, user_id), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Agent %s timed out after 5s", agent_name)
            yield await stream_status("Fetching data is taking longer than expected. Showing partial insights...")
            # Attempt to get a partial/cached result if the agent supports it
            # For now, we'll return a generic partial response
            result = {
                "agent": agent_name,
                "message": "I'm experiencing some latency fetching live market data. Here is what I can tell you based on available information.",
                "partial": True,
                "intent": classifier_result.intent,
                "entities": classifier_result.entities.model_dump(exclude_none=True),
                # Add default metrics for portfolio_health to prevent frontend crashes
                "performance": {"total_return_pct": 0.0, "annualized_return_pct": 0.0},
                "concentration_risk": {"top_position_pct": 0.0, "top_3_positions_pct": 0.0, "flag": "low"},
                "benchmark_comparison": {"benchmark": "S&P 500", "portfolio_return_pct": 0.0, "benchmark_return_pct": 0.0, "alpha_pct": 0.0},
                "observations": [{"severity": "warning", "text": "Market data fetch timed out. Metrics shown are placeholders."}],
                "disclaimer": "This is a partial result due to service latency."
            }

        # Step 5: Store in memory
        if memory:
            memory.add_message(session_id, "user", query)
            memory.add_message(session_id, "assistant", json.dumps(result))

        # Step 6: Stream message chunks for a "typing" effect
        message = result.get("message", "")
        if message:
            yield await stream_status("Generating insights...")
            words = message.split(' ')
            for i in range(0, len(words), 4):
                chunk = ' '.join(words[i:i+4]) + ' '
                yield await stream_chunk(chunk)
                await asyncio.sleep(0.02) # Faster typing effect

        # Step 7: Stream final structured result
        yield await stream_result(result)

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        yield await stream_error(f"An unexpected error occurred: {str(exc)}", "internal_error")

