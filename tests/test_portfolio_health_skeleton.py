"""
Portfolio Health agent tests.

Tests cover:
  - Empty portfolio handling (user_004)
  - Concentration risk detection (user_003)
  - Disclaimer presence
  - All user profiles process without errors
  - Performance calculation sanity
"""
import pytest

from src.agents.portfolio_health import run


def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user):
    """
    user_004 has no positions. Agent must not crash.
    Must produce BUILD-oriented guidance.
    """
    user = load_user("usr_004")
    response = run(user)

    assert response is not None
    assert "disclaimer" in response
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()

    # Should have helpful observations for a new investor
    assert len(response["observations"]) > 0
    obs_text = " ".join(o["text"] for o in response["observations"])
    assert "invest" in obs_text.lower() or "start" in obs_text.lower()


def test_portfolio_health_flags_concentration(load_user):
    """
    user_003 has ~60% in NVDA. Agent must surface this.
    """
    user = load_user("usr_003")
    response = run(user)

    assert response["concentration_risk"]["flag"] in {"high", "warning"}
    assert response["concentration_risk"]["top_position_pct"] > 40

    # Check that NVDA is mentioned in observations
    obs_text = " ".join(o["text"] for o in response["observations"])
    assert "NVDA" in obs_text or "nvda" in obs_text.lower() or "concentrated" in obs_text.lower()


def test_portfolio_health_includes_disclaimer(load_user):
    user = load_user("usr_001")
    response = run(user)
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()


def test_portfolio_health_active_trader(load_user):
    """user_001 — aggressive US trader with 9 holdings."""
    user = load_user("usr_001")
    response = run(user)

    assert response is not None
    assert "concentration_risk" in response
    assert "performance" in response
    assert "benchmark_comparison" in response
    assert "observations" in response
    assert len(response["observations"]) > 0

    # Active trader is diversified — should not be "high" concentration
    assert response["concentration_risk"]["flag"] in {"low", "medium"}


def test_portfolio_health_retiree(load_user):
    """user_008 — dividend-focused retiree, conservative."""
    user = load_user("usr_008")
    response = run(user)

    assert response is not None
    assert len(response["observations"]) > 0

    # Should mention income/dividend since income_focus is true
    obs_text = " ".join(o["text"] for o in response["observations"])
    assert "dividend" in obs_text.lower() or "income" in obs_text.lower()


def test_portfolio_health_multi_currency(load_user):
    """user_006 — multi-currency holdings. Should not crash."""
    user = load_user("usr_006")
    response = run(user)

    assert response is not None
    assert "concentration_risk" in response
    assert response["benchmark_comparison"]["benchmark"]  # should have a benchmark


def test_portfolio_health_performance_sanity(load_user):
    """Performance numbers should be reasonable (not NaN, not astronomical)."""
    user = load_user("usr_001")
    response = run(user)

    perf = response["performance"]
    assert -100 < perf["total_return_pct"] < 1000
    assert -100 < perf["annualized_return_pct"] < 500


def test_portfolio_health_all_users_no_crash(load_user):
    """Every user fixture should produce a valid response."""
    for uid in ["usr_001", "usr_003", "usr_004", "usr_006", "usr_008"]:
        response = run(load_user(uid))
        assert response is not None
        assert "disclaimer" in response
        assert "observations" in response
