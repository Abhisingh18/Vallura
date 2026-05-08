"""
FastAPI routes for the Valura AI financial assistant.

Single endpoint: POST /chat
  - Accepts ChatRequest
  - Runs the full pipeline (safety → classifier → router)
  - Returns SSE stream

Design decisions:
  - SSE is the ONLY response mode (no JSON fallback)
  - 30-second timeout per request
  - Errors are streamed as structured SSE error events, not stack traces
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.models.schemas import ChatRequest
from src.safety.guard import check as safety_check
from src.utils.streaming import pipeline_stream

logger = logging.getLogger(__name__)

router = APIRouter()

# Request timeout in seconds
_REQUEST_TIMEOUT = 30.0


@router.post("/chat")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint. Runs the full AI assistant pipeline via SSE.

    Pipeline: Safety Guard → Intent Classifier → Agent Router → SSE Stream

    The classifier and memory are injected via app.state (set in main.py).
    This keeps routes thin and testable.
    """
    classifier = request.app.state.classifier
    memory = request.app.state.memory

    async def _bounded_stream():
        """Wrap the pipeline stream with a timeout."""
        try:
            async for event in _timeout_stream(
                query=body.query,
                session_id=body.session_id,
                user_id=body.user_id,
                classifier=classifier,
                memory=memory,
                timeout=_REQUEST_TIMEOUT,
            ):
                yield event
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            error_data = json.dumps({
                "event": "error",
                "data": {"error": str(exc), "code": "internal_error"},
            })
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        _bounded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )


async def _timeout_stream(
    query: str,
    session_id: str,
    user_id: str,
    classifier,
    memory,
    timeout: float,
):
    """
    Async generator that wraps the pipeline stream with per-event timeout.

    We cannot use asyncio.wait_for on an async generator directly, so we
    track elapsed time and bail if it exceeds the budget.
    """
    import time

    from src.router.router import route

    start = time.monotonic()

    async for event in pipeline_stream(
        query=query,
        session_id=session_id,
        user_id=user_id,
        safety_check_fn=safety_check,
        classify_fn=classifier.classify,
        route_fn=route,
        memory=memory,
    ):
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            error_data = json.dumps({
                "event": "error",
                "data": {"error": "Request timed out.", "code": "timeout"},
            })
            yield f"event: error\ndata: {error_data}\n\n"
            return
        yield event
@router.get("/portfolio/summary")
async def get_portfolio_summary(user_id: str = "usr_001"):
    """
    Fetch real-time summary for the user's portfolio.
    Used to populate the dashboard sidebar.
    """
    from src.router.router import _load_user
    from src.agents.portfolio_health import _calculate_concentration, _calculate_performance
    
    user = _load_user(user_id)
    if not user:
        return {"error": "User not found"}
        
    from src.models.schemas import UserProfile
    profile = UserProfile(**user)
    
    concentration = _calculate_concentration(profile)
    performance = _calculate_performance(profile)
    
    # Calculate top holdings
    from src.services.market_data import MarketDataService
    market = MarketDataService()
    
    holdings = []
    total_val = 0.0
    for pos in profile.positions:
        price = market.get_current_price(pos.ticker) or pos.avg_cost
        val = pos.quantity * price
        total_val += val
        holdings.append({"ticker": pos.ticker, "value": val})
        
    holdings.sort(key=lambda x: x["value"], reverse=True)
    top_holdings = []
    sectors = set()
    if total_val > 0:
        for h in holdings[:3]:
            top_holdings.append({
                "ticker": h["ticker"],
                "pct": round((h["value"] / total_val) * 100, 1)
            })
        for pos in profile.positions:
            sectors.add(market.get_sector_information(pos.ticker))
            
    # Intelligent Diversification Score
    # 0-100 based on number of holdings and sector spread
    base_score = min(len(profile.positions) * 10, 50)
    sector_score = min(len(sectors) * 10, 50)
    diversification_score = base_score + sector_score

    return {
        "total_value": round(total_val, 2),
        "total_return_pct": performance.total_return_pct,
        "risk_profile": profile.risk_profile,
        "top_holdings": top_holdings,
        "diversification_score": diversification_score,
        "last_updated": datetime.now().strftime("%H:%M:%S")
    }
