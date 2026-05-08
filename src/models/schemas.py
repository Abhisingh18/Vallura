"""
Pydantic v2 schemas for the Valura AI financial assistant pipeline.

All inter-module data contracts live here to enforce a single source of truth
and prevent implicit coupling between layers.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Inbound user message."""
    session_id: str = Field(..., min_length=1, description="Client-generated session identifier")
    user_id: str = Field(..., min_length=1, description="User ID matching fixtures/users/*.json")
    query: str = Field(..., min_length=1, max_length=4000, description="User's natural-language query")


class SSEEvent(BaseModel):
    """Typed wrapper for a single SSE event pushed to the client."""
    event: str = Field(..., description="SSE event name: status | result | error")
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyResult(BaseModel):
    """Output of the synchronous safety guard."""
    blocked: bool = False
    category: Optional[str] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ExtractedEntities(BaseModel):
    """Entities extracted from the user query."""
    tickers: list[str] = Field(default_factory=list)
    amount: Optional[float] = None
    currency: Optional[str] = None
    rate: Optional[float] = None
    period_years: Optional[int] = None
    frequency: Optional[str] = None
    horizon: Optional[str] = None
    time_period: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    index: Optional[str] = None
    action: Optional[str] = None
    goal: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # forward-compat: new entity fields won't break old code


class ClassifierResult(BaseModel):
    """Structured output from the intent classifier."""
    intent: str = "unknown"
    agent: str = "general_query"
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    safety_verdict: str = "safe"
    confidence: float = 0.0
    source: str = "fallback_rules"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class StubAgentResponse(BaseModel):
    """Placeholder response for agents not yet implemented."""
    agent: str
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    message: str = "This agent is not implemented in this prototype build."


# ---------------------------------------------------------------------------
# Portfolio Health
# ---------------------------------------------------------------------------

class ConcentrationRisk(BaseModel):
    top_position_pct: float = 0.0
    top_3_positions_pct: float = 0.0
    flag: str = "low"  # low | medium | high


class PerformanceMetrics(BaseModel):
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0


class BenchmarkComparison(BaseModel):
    benchmark: str = ""
    portfolio_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0


class SeverityEnum(str, Enum):
    info = "info"
    warning = "warning"


class Observation(BaseModel):
    severity: str = "info"
    text: str = ""


class PortfolioHealthResult(BaseModel):
    """Full output of the portfolio health agent."""
    concentration_risk: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    benchmark_comparison: BenchmarkComparison = Field(default_factory=BenchmarkComparison)
    observations: list[Observation] = Field(default_factory=list)
    disclaimer: str = (
        "This is not investment advice. The information provided is for educational "
        "and informational purposes only. Past performance does not guarantee future "
        "results. Please consult a qualified financial advisor before making any "
        "investment decisions."
    )


# ---------------------------------------------------------------------------
# User data (mirrors fixtures/users/*.json)
# ---------------------------------------------------------------------------

class Position(BaseModel):
    ticker: str
    exchange: str = ""
    quantity: float
    avg_cost: float
    currency: str = "USD"
    purchased_at: str = ""


class KYC(BaseModel):
    status: str = "unverified"


class UserPreferences(BaseModel):
    preferred_benchmark: str = "S&P 500"
    reporting_currency: str = "USD"
    income_focus: bool = False

    model_config = ConfigDict(extra="allow")


class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    age: int = 0
    country: str = ""
    base_currency: str = "USD"
    kyc: KYC = Field(default_factory=KYC)
    risk_profile: str = "moderate"
    positions: list[Position] = Field(default_factory=list)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
