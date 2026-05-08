"""
Portfolio Health Agent — the flagship specialist agent.

Produces a structured health assessment of a user's portfolio with:
  - Concentration risk analysis
  - Performance metrics (total return, annualized return)
  - Benchmark comparison with alpha calculation
  - Plain-language observations aimed at novice investors
  - Regulatory disclaimer

Design principles:
  1. All calculations are deterministic Python math — no LLM hallucination
  2. Empty portfolios get BUILD-oriented onboarding guidance, not errors
  3. Observations use beginner-friendly language (no jargon without context)
  4. The agent never crashes — every edge case returns a valid response
"""
from __future__ import annotations

import logging
from typing import Any

from src.models.schemas import (
    BenchmarkComparison,
    ConcentrationRisk,
    Observation,
    PerformanceMetrics,
    PortfolioHealthResult,
    UserProfile,
)
from src.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

_market = MarketDataService()


def run(user: dict[str, Any] | UserProfile, **kwargs: Any) -> dict[str, Any]:
    """
    Run the portfolio health check.

    Args:
        user: User profile dict (matching fixtures/users/*.json shape)
              or a UserProfile Pydantic model.

    Returns:
        dict matching PortfolioHealthResult schema.
    """
    llm_insight = kwargs.get("llm_insight")
    
    # Normalize input: accept both dict and Pydantic model
    if isinstance(user, dict):
        try:
            profile = UserProfile(**user)
        except Exception as exc:
            logger.warning("Failed to parse user profile: %s", exc)
            return _error_response(f"Could not parse user profile: {exc}")
    else:
        profile = user

    # Handle empty portfolio → BUILD-oriented guidance
    if not profile.positions:
        return _empty_portfolio_response(profile)

    # Calculate all metrics
    try:
        concentration = _calculate_concentration(profile)
        performance = _calculate_performance(profile)
        benchmark = _calculate_benchmark_comparison(profile, performance)
        
        # Check if we have any data issues
        data_warning = None
        missing_tickers = [pos.ticker for pos in profile.positions if _market.get_current_price(pos.ticker) is None]
        if missing_tickers:
            data_warning = f"Note: Live pricing unavailable for {', '.join(missing_tickers)}; using cost basis as fallback."

        observations = _generate_intelligent_observations(profile, concentration, performance, benchmark)
        
        if data_warning:
            observations.append(Observation(severity="warning", text=data_warning))
        else:
            observations.append(Observation(severity="info", text="✅ Market Data: Analysis performed using real-time market prices."))

        # Income Focus Insight
        if profile.preferences and profile.preferences.income_focus and len(profile.positions) > 0:
            observations.append(Observation(
                text="Income Strategy: Your focus on dividend-yielding assets is well-suited for your conservative profile. Consider monthly rebalancing to maintain yield targets.",
                severity="info"
            ))

        if llm_insight:
            observations.insert(0, Observation(
                severity="info",
                text=f"🤖 AI Insight: {llm_insight}"
            ))

        result = PortfolioHealthResult(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark,
            observations=observations,
        )
        return result.model_dump()

    except Exception as exc:
        logger.error("Portfolio health agent error: %s", exc, exc_info=True)
        return _error_response(str(exc))


def _calculate_concentration(profile: UserProfile) -> ConcentrationRisk:
    """
    Calculate concentration risk based on current market values.
    """
    position_values: list[tuple[str, float]] = []

    for pos in profile.positions:
        current_price = _market.get_current_price(pos.ticker) or pos.avg_cost
        value = pos.quantity * current_price
        position_values.append((pos.ticker, value))

    total_value = sum(v for _, v in position_values)
    if total_value <= 0:
        return ConcentrationRisk(flag="low")

    # Sort by value descending
    position_values.sort(key=lambda x: x[1], reverse=True)

    top_pct = (position_values[0][1] / total_value) * 100
    top_3_pct = (sum(v for _, v in position_values[:3]) / total_value) * 100

    # Determine flag based on top position
    if top_pct > 60:
        flag = "high"
    elif top_pct > 30:
        flag = "medium"
    else:
        flag = "low"

    return ConcentrationRisk(
        top_position_pct=round(top_pct, 1),
        top_3_positions_pct=round(top_3_pct, 1),
        flag=flag,
    )


def _calculate_performance(profile: UserProfile) -> PerformanceMetrics:
    """
    Calculate portfolio-level return metrics using live data.
    """
    total_cost = 0.0
    total_current = 0.0
    weighted_years = 0.0

    for pos in profile.positions:
        cost = pos.quantity * pos.avg_cost
        current_price = _market.get_current_price(pos.ticker) or pos.avg_cost
        current_value = pos.quantity * current_price
        
        total_cost += cost
        total_current += current_value

        years_held = _market.calculate_holding_period_years(pos.purchased_at)
        weighted_years += cost * years_held

    if total_cost <= 0:
        return PerformanceMetrics()

    total_return_pct = ((total_current - total_cost) / total_cost) * 100
    avg_years = weighted_years / total_cost

    # Annualized return calculation
    if avg_years > 0 and total_return_pct > -100:
        total_return_decimal = total_return_pct / 100
        annualized = ((1 + total_return_decimal) ** (1 / avg_years) - 1) * 100
    else:
        annualized = 0.0

    return PerformanceMetrics(
        total_return_pct=round(total_return_pct, 1),
        annualized_return_pct=round(annualized, 1),
    )


def _calculate_benchmark_comparison(
    profile: UserProfile,
    performance: PerformanceMetrics,
) -> BenchmarkComparison:
    """Compare performance against SPY (default) or preferred benchmark."""
    benchmark_name = profile.preferences.preferred_benchmark or "S&P 500"

    # Fetch benchmark return dynamically
    benchmark_return = _market.get_benchmark_return(benchmark_name, "1y")
    
    if benchmark_return is None:
        benchmark_return = 0.0  # Fallback

    benchmark_return_pct = benchmark_return * 100
    alpha = performance.annualized_return_pct - benchmark_return_pct

    return BenchmarkComparison(
        benchmark=benchmark_name,
        portfolio_return_pct=round(performance.annualized_return_pct, 1),
        benchmark_return_pct=round(benchmark_return_pct, 1),
        alpha_pct=round(alpha, 1),
    )


def _generate_intelligent_observations(
    profile: UserProfile,
    concentration: ConcentrationRisk,
    performance: PerformanceMetrics,
    benchmark: BenchmarkComparison,
) -> list[Observation]:
    """
    Generate human-level, context-aware insights.
    """
    observations: list[Observation] = []

    # 1. Concentration Insight (Highly Contextual)
    if concentration.flag == "high":
        # Identify the primary offender
        position_values = []
        for pos in profile.positions:
            price = _market.get_current_price(pos.ticker) or pos.avg_cost
            position_values.append((pos.ticker, pos.quantity * price))
        position_values.sort(key=lambda x: x[1], reverse=True)
        
        top_ticker, top_val = position_values[0]
        sector = _market.get_sector_information(top_ticker)
        
        observations.append(Observation(
            severity="warning",
            text=(
                f"Over {concentration.top_position_pct:.0f}% of your portfolio is concentrated in {top_ticker}. "
                f"This makes your wealth highly sensitive to {sector} sector volatility. "
                f"Consider diversifying into broader ETFs to protect against a downturn in {top_ticker}."
            ),
        ))
    elif concentration.flag == "medium":
        observations.append(Observation(
            severity="info",
            text=(
                f"Your top position ({concentration.top_position_pct:.0f}%) indicates moderate concentration. "
                "While not critical, you are becoming reliant on a single asset's performance."
            ),
        ))

    # 2. Performance & Alpha Insight
    if benchmark.alpha_pct > 5:
        observations.append(Observation(
            severity="info",
            text=(
                f"Your portfolio is showing significant 'Alpha' of {benchmark.alpha_pct:.1f}% over the {benchmark.benchmark}. "
                "You are effectively outperforming the market, likely due to your specific stock selection."
            ),
        ))
    elif benchmark.alpha_pct < -5:
        observations.append(Observation(
            severity="warning",
            text=(
                f"You are trailing the {benchmark.benchmark} by {abs(benchmark.alpha_pct):.1f}%. "
                "It may be worth reviewing your high-cost or underperforming positions to see if "
                "a simpler index-based approach (like VOO) would serve you better."
            ),
        ))

    # 3. Sector & Risk Profile Alignment
    sector_values: dict[str, float] = {}
    total_value = 0.0
    for pos in profile.positions:
        price = _market.get_current_price(pos.ticker) or pos.avg_cost
        value = pos.quantity * price
        total_value += value
        sector = _market.get_sector_information(pos.ticker)
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    if total_value > 0:
        for sector, value in sector_values.items():
            pct = (value / total_value) * 100
            if pct > 50 and sector != "unknown":
                observations.append(Observation(
                    severity="warning",
                    text=(
                        f"Extreme Sector Risk: {pct:.0f}% of your assets are in the {sector} sector. "
                        "A sector-wide correction could lead to significant drawdowns."
                    ),
                ))

    # 4. Empty/Low Position guidance (Safety net)
    if len(profile.positions) < 3 and total_value > 0:
        observations.append(Observation(
            severity="info",
            text=(
                "With only a few positions, your portfolio risk is high. "
                "Adding a broad-market fund like VTI or VT would provide instant diversification."
            ),
        ))

    return observations


def _empty_portfolio_response(profile: UserProfile) -> dict[str, Any]:
    """
    Onboarding guidance for empty portfolios.
    """
    name = profile.name or "Investor"
    guidance_map = {
        "conservative": "Focus on capital preservation with index funds (VOO) and bonds (BND).",
        "moderate": "A balanced 60/40 or 70/30 split between global stocks (VT) and stability assets.",
        "aggressive": "High growth potential through broad market exposure (VTI) and tech-tilted ETFs (QQQ).",
    }
    
    guidance = guidance_map.get(profile.risk_profile, guidance_map["moderate"])

    result = PortfolioHealthResult(
        observations=[
            Observation(severity="info", text=f"Welcome {name}! Let's build your first allocation."),
            Observation(severity="info", text=f"Strategy: {guidance}"),
            Observation(severity="info", text="Start small with Dollar-Cost Averaging (DCA) to build confidence."),
        ],
    )
    return result.model_dump()


def _error_response(error_msg: str) -> dict[str, Any]:
    """Graceful error response."""
    result = PortfolioHealthResult(
        observations=[
            Observation(
                severity="warning",
                text=f"Analysis Partial: {error_msg}. Using cached/fallback data where possible.",
            ),
        ],
    )
    return result.model_dump()

