"""
Market Research Agent — provides real-time information about stocks and markets.

Uses yfinance to fetch:
- Current price and currency
- Price change (absolute and percentage)
- Company summary/description
- Key metrics (market cap, P/E ratio)
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

from src.models.schemas import ClassifierResult
from src.services.market_data import MarketDataService

logger = logging.getLogger(__name__)
_market = MarketDataService()

async def run(classifier_result: ClassifierResult) -> dict[str, Any]:
    """
    Run market research for the tickers extracted by the classifier.
    """
    tickers = classifier_result.entities.tickers
    
    if not tickers:
        return {
            "agent": "market_research",
            "message": "I couldn't identify which stock you're asking about. Could you please provide a ticker symbol (e.g., AAPL) or a company name?",
            "data": {}
        }

    # For the prototype, we research the first ticker mentioned
    ticker = tickers[0].upper()
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract meaningful data
        name = info.get("longName", ticker)
        summary = info.get("longBusinessSummary", "No description available.")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        currency = info.get("currency", "USD")
        market_cap = info.get("marketCap")
        pe_ratio = info.get("forwardPE") or info.get("trailingPE")

        change = None
        change_pct = None
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100

        return {
            "agent": "market_research",
            "ticker": ticker,
            "name": name,
            "price": price,
            "currency": currency,
            "change": round(change, 2) if change is not None else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "summary": summary,
            "metrics": {
                "market_cap": market_cap,
                "pe_ratio": round(pe_ratio, 2) if pe_ratio else None
            },
            "message": f"Here is the latest research on {name} ({ticker})."
        }

    except Exception as exc:
        logger.error("Market research agent failed for %s: %s", ticker, exc)
        # Fallback to a simpler response if yfinance fails
        price = _market.get_current_price(ticker)
        return {
            "agent": "market_research",
            "ticker": ticker,
            "price": price,
            "message": f"I was able to fetch the current price for {ticker} (${price}), but I'm having trouble getting a full market report right now.",
            "error": str(exc)
        }
