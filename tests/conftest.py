"""
Shared pytest fixtures for the Valura AI assignment.

The most important fixture here is `mock_llm` — every test that touches the
classifier or any LLM-using code must use it. CI runs without OPENAI_API_KEY
and unmocked LLM calls will fail.
"""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_user():
    """Load a user fixture by id, e.g. load_user('usr_001')."""
    def _load(user_id: str) -> dict:
        for path in (FIXTURES_DIR / "users").glob("*.json"):
            with open(path, encoding="utf-8") as f:
                user = json.load(f)
            if user["user_id"] == user_id:
                return user
        raise FileNotFoundError(f"No fixture for user {user_id}")
    return _load


@pytest.fixture
def gold_classifier_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "intent_classification.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def gold_safety_queries() -> list[dict]:
    with open(FIXTURES_DIR / "test_queries" / "safety_pairs.json", encoding="utf-8") as f:
        return json.load(f)["queries"]


@pytest.fixture
def conversation_test_cases():
    """Returns a callable: conversation_test_cases('follow_up_session')."""
    def _load(name: str) -> list[dict]:
        path = FIXTURES_DIR / "conversations" / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["test_cases"]
    return _load


# ---------------------------------------------------------------------------
# LLM mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """
    Returns a MagicMock that you should configure per-test to return whatever
    structured output your classifier expects.
    """
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_yfinance(mocker):
    """
    Auto-mock yfinance to prevent network calls during tests.
    Sets default returns for common tickers.
    """
    mock_ticker = mocker.patch("yfinance.Ticker")
    
    # Configure mock behavior
    def get_mock_info(ticker):
        instance = MagicMock()
        # Mock varied sectors to avoid 100% concentration in tests
        sectors = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JNJ": "Healthcare",
            "PG": "Consumer Defensive",
            "KO": "Consumer Defensive",
            "VYM": "Financial",
            "SCHD": "Financial",
            "BND": "Fixed Income",
            "TLT": "Fixed Income"
        }
        sector = sectors.get(ticker.upper(), "Financial")
        
        instance.fast_info = {"last_price": 150.0}
        instance.info = {
            "sector": sector, 
            "longName": f"Mock {ticker}", 
            "currentPrice": 150.0,
            "previousClose": 145.0,
            "currency": "USD",
            "longBusinessSummary": f"Mock summary for {ticker}",
            "marketCap": 2000000000000,
            "trailingPE": 25.0
        }
        
        # Mock history for benchmark returns
        import pandas as pd
        hist_data = pd.DataFrame({
            "Close": [100.0, 110.0]
        }, index=[datetime(2023, 1, 1), datetime(2024, 1, 1)])
        instance.history.return_value = hist_data
        
        return instance

    mock_ticker.side_effect = get_mock_info
    return mock_ticker
