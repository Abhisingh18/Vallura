"""
Rule-based fallback classifier.

Used when OPENAI_API_KEY is not set, or as a fallback when the OpenAI classifier
fails. Uses keyword matching and simple heuristics.

Design trade-off: This will have lower accuracy on edge cases than the LLM
classifier, but it guarantees zero cost, zero latency, and zero external
dependencies. In a production system this serves as the circuit-breaker
fallback so the service never goes fully dark.
"""
from __future__ import annotations

import re

from src.classifier.base import BaseClassifier
from src.models.schemas import ClassifierResult, ExtractedEntities


# ---------------------------------------------------------------------------
# Known ticker aliases (company name → ticker)
# ---------------------------------------------------------------------------
_COMPANY_TICKERS: dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "microsfot": "MSFT",  # common typo
    "nvidia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "amd": "AMD",
    "hsbc": "HSBA.L",
    "barclays": "BARC.L",
    "toyota": "7203.T",
    "asml": "ASML.AS",
}

# Known tickers (upper-case, for direct match)
_KNOWN_TICKERS: set[str] = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "QQQ", "VTI", "VXUS", "VOO", "BND", "VYM", "SCHD", "TLT",
    "JNJ", "PG", "KO", "HSBA.L", "BARC.L", "ASML.AS", "7203.T",
    "GOLD", "GLD", "SPY",
}

# ---------------------------------------------------------------------------
# Agent routing patterns (order matters: more specific first)
# ---------------------------------------------------------------------------
_AGENT_RULES: list[tuple[str, str, list[str]]] = [
    # (agent, intent, keyword patterns)
    ("customer_support", "support",
     [r"\blog\s*in\b", r"\baccount\b", r"\btransaction\s+history\b",
      r"\brecurring\s+investment\b", r"\blinked\s+bank\b", r"\bcan'?t\s+(?:log|sign)\b",
      r"\bhow\s+do\s+i\s+(?:change|update|reset)\b.*(?:account|bank|password)"]),

    ("portfolio_health", "portfolio_health",
     [r"\bportfolio\s+(?:health|check|summary|doing|review|performance|analysis)\b",
      r"\banalyze\s+(?:my\s+)?(?:portfolio|investments?)\b",
      r"\bhow\s+(?:is|are)\s+my\s+(?:portfolio|investments?|holdings?)\s+(?:doing|performing)\b",
      r"\bhealth\s+check\b", r"\breview\s+my\s+holdings?\b",
      r"\bdiversif(?:ied|ication)\b", r"\bconcentration\s+risk\b",
      r"\bam\s+i\s+beating\b", r"\bportfolio\s+summary\b",
      r"\bhow\s+is\s+my\s+portfolio\b",
      r"\bhow\s+much\s+(?:do\s+i\s+)?own\b",
      r"\bwhat\s+do\s+i\s+hold\b",
      r"\bgive\s+me\s+a\s+health\s+check\b"]),

    ("financial_calculator", "calculation",
     [r"calculat", r"future\s+value", r"mortgage\s+payment",
      r"capital\s+gains?\s+tax", r"convert\s+\d+",
      r"if\s+i\s+invest\s+\d+", r"ltcg",
      r"monthly\s+for\s+\d+\s+years?",
      r"\b\d+\s+monthly\s+for\s+\d+\s+years?"]),

    ("financial_planning", "planning",
     [r"\bretir(?:e|ement)\b", r"\bcollege\s+fund\b", r"\beducation\b.*\bsav\b",
      r"\bhouse\s+(?:down\s+)?payment\b", r"\bfire\s+plan\b",
      r"\bsav(?:e|ing)\s+for\b.*(?:house|college|retirement|education)",
      r"\bon\s+track\b.*\bretir\b", r"\bfire\b.*\b(?:earning|income)\b",
      r"\bchild(?:'s)?\s+(?:college|education)\b"]),

    ("risk_assessment", "risk_analysis",
     [r"\bdownside\s+risk\b", r"\bbeta\b", r"\bmax\s+drawdown\b",
      r"\bstress\s+test\b", r"\bexpos(?:ed|ure)\b",
      r"\brisk\b.*\bmarkets?\s+drop\b", r"\brecession\b",
      r"\bcurrency\s+(?:risk|exposure)\b"]),

    ("product_recommendation", "recommendation",
     [r"\brecommend\b", r"\bbest\b.*\b(?:etf|fund|index)\b",
      r"\bwhich\s+(?:etf|fund)\b", r"\blow.?cost\s+(?:world|index|global)\b",
      r"\bdividend\s+etf\b", r"\bemerging\s+market\b.*\bfund\b"]),

    ("predictive_analysis", "prediction",
     [r"\bpredict\b", r"\bwhere\s+will\b", r"\bforecast\b",
      r"\bin\s+(?:6\s+months?|a\s+year|5\s+years?)\b.*(?:value|be|worth)"]),

    ("investment_strategy", "strategy",
     [r"\bshould\s+i\s+(?:buy|sell|hold|hedge)\b",
      r"\brebalanc\b", r"\bequity.?bond\s+split\b",
      r"\bgood\s+time\s+to\s+invest\b", r"\bhedge\b.*\bexposure\b",
      r"\bsell\s+(?:my|some)\b", r"\bbuy\s+more\b"]),

    ("market_research", "market_research",
     [r"\bprice\s+of\b", r"\btell\s+me\s+about\b", r"\bnews\s+on\b",
      r"\bhow\s+(?:is|are)\s+.+\s+doing\b", r"\bcompare\b",
      r"\btop\s+(?:gainers?|losers?)\b", r"\bmarkets?\s+today\b",
      r"\bwhat(?:'s|\s+is)\s+happening\b", r"\beur/usd\b",
      r"\bgold\s+price\b", r"\bhow\s+is\s+the\b.*(?:ftse|nikkei|s&p)",
      r"\bwhat(?:'s)?\s+happening\s+with\b"]),
]

# Greeting / general patterns
_GENERAL_PATTERNS: list[str] = [
    r"^(?:hi|hello|hey|thanks?|thank\s+you|bye|thx|ok)[\s!?.]*$",
    r"\bwhat\s+(?:is|are|does)\b.*\b(?:mutual\s+fund|compound\s+interest|etf|index\s+fund|p/?e\s+ratio)\b",
    r"\bexplain\b.*\b(?:compound|interest|etf|mutual|index)\b",
    r"\bdifference\s+between\b",
]

# ---------------------------------------------------------------------------
# Entity extraction helpers
# ---------------------------------------------------------------------------

def _extract_tickers(query: str) -> list[str]:
    """Extract stock tickers from the query."""
    tickers: list[str] = []
    q_lower = query.lower()

    # 1. Check company names (e.g. "Apple" -> "AAPL")
    for company, ticker in _COMPANY_TICKERS.items():
        if company in q_lower:
            tickers.append(ticker)

    # 2. Check for direct ticker mentions (2-5 letters, optionally with exchange suffix)
    # We look for word boundaries to avoid matching partial words
    # This handles both "AAPL" and "aapl"
    for token in re.findall(r'\b([a-zA-Z]{2,5}(?:\.[a-zA-Z]{1,2})?)\b', query):
        normalized = token.upper()
        if normalized in _KNOWN_TICKERS:
            tickers.append(normalized)

    # 3. Handle specific suffixes explicitly (e.g. "asml.as")
    for token in re.findall(r'\b([a-z0-9]+\.[a-z]{1,2})\b', q_lower):
        normalized = token.upper()
        if normalized in _KNOWN_TICKERS:
            tickers.append(normalized)

    # 4. Single-word query: check if it's a ticker or company even without boundaries
    stripped = query.strip()
    if " " not in stripped:
        upper = stripped.upper()
        if upper in _KNOWN_TICKERS:
            tickers.append(upper)
        if stripped.lower() in _COMPANY_TICKERS:
            tickers.append(_COMPANY_TICKERS[stripped.lower()])

    # Deduplicate while preserving order and normalizing to base ticker for uniqueness check
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        base = t.upper().split(".")[0]
        if base not in seen:
            seen.add(base)
            result.append(t.upper())
    return result


def _extract_amount(query: str) -> float | None:
    """Extract a dollar/monetary amount."""
    # Patterns: "$500k", "500000", "200k", "2500 monthly", "150k a year"
    m = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*[kK]', query)
    if m:
        return float(m.group(1).replace(",", "")) * 1000

    m = re.search(r'\b([\d,]+(?:\.\d+)?)\b(?:\s+(?:USD|EUR|GBP|JPY|usd|eur|gbp|jpy|dollars?|a\s+year|monthly|a\s+month))', query)
    if m:
        return float(m.group(1).replace(",", ""))

    # "invest AMOUNT monthly for..."
    m = re.search(r'invest\s+([\d,]+)', query, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    # "value of AMOUNT at"
    m = re.search(r'(?:value\s+of|loan|profit|convert)\s+\$?([\d,]+)', query, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    # "plan for ... of AMOUNT"
    m = re.search(r'of\s+\$?([\d,]+)\s*[kK]?\b', query, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(",", ""))
        if "k" in query[m.start():m.end()].lower():
            val *= 1000
        if val >= 100:
            return val

    # Bare number at start: "1500 monthly"
    m = re.search(r'^(\d{3,})\s+(?:monthly|weekly|daily|yearly)', query, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def _extract_rate(query: str) -> float | None:
    """Extract an interest/return rate."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', query)
    if m:
        return float(m.group(1)) / 100
    return None


def _extract_period_years(query: str) -> int | None:
    """Extract a time period in years."""
    m = re.search(r'(\d+)\s*(?:year|yr)', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _extract_frequency(query: str) -> str | None:
    q = query.lower()
    for freq in ["monthly", "weekly", "daily", "yearly"]:
        if freq in q:
            return freq
    if "a month" in q:
        return "monthly"
    if "a year" in q or "per year" in q:
        return "yearly"
    return None


def _extract_currency(query: str) -> str | None:
    for cur in ["USD", "EUR", "GBP", "JPY"]:
        if cur in query.upper():
            return cur
    return None


def _extract_action(query: str) -> str | None:
    q = query.lower()
    for action in ["rebalance", "buy", "sell", "hold", "hedge"]:
        if action in q:
            return action
    return None


def _extract_goal(query: str) -> str | None:
    q = query.lower()
    if "fire" in q:
        return "FIRE"
    for goal in ["retirement", "education", "house", "emergency_fund"]:
        if goal.replace("_", " ") in q or goal in q:
            return goal
    if "college" in q:
        return "education"
    if "retire" in q:
        return "retirement"
    return None


def _extract_time_period(query: str) -> str | None:
    q = query.lower()
    if "today" in q:
        return "today"
    if "this week" in q:
        return "this_week"
    if "this month" in q:
        return "this_month"
    if "this year" in q:
        return "this_year"
    return None


def _extract_horizon(query: str) -> str | None:
    q = query.lower()
    m = re.search(r'(?:in|next|within)\s+(\d+)\s+(?:month|year)', q)
    if m:
        num = int(m.group(1))
        if "month" in q[m.start():m.end()]:
            return f"{num}_months"
        return f"{num}_years" if num > 1 else "1_year"
    if "6 months" in q:
        return "6_months"
    return None


def _extract_index(query: str) -> str | None:
    q = query.lower()
    if "s&p 500" in q or "s&p500" in q or "sp500" in q:
        return "S&P 500"
    if "ftse" in q:
        return "FTSE 100"
    if "nikkei" in q:
        return "NIKKEI 225"
    if "msci world" in q:
        return "MSCI World"
    return None


def _extract_topics(query: str) -> list[str]:
    """Extract discussion topics."""
    topics: list[str] = []
    q_lower = query.lower()

    topic_patterns = {
        "mutual fund": r"\bmutual\s+fund\b",
        "compound interest": r"\bcompound\s+interest\b",
        "ETF": r"\betf\b",
        "index fund": r"\bindex\s+fund\b",
        "P/E ratio": r"\bp/?e\s+ratio\b",
        "DCA": r"\b(?:dollar\s+cost\s+averag|dca)\b",
        "lump-sum": r"\blump.?sum\b",
        "beta": r"\bbeta\b",
        "max drawdown": r"\bmax\s+drawdown\b",
        "recession": r"\brecession\b",
        "login": r"\blog\s*in\b",
        "bank account": r"\bbank\s+account\b",
        "transaction history": r"\btransaction\s+history\b",
        "recurring investment": r"\brecurring\s+investment\b",
        "FX": r"\b(?:eur/usd|fx|foreign\s+exchange)\b",
        "LTCG": r"\b(?:ltcg|long.?term\s+capital\s+gains?)\b",
        "large cap": r"\blarge\s+cap\b",
        "emerging markets": r"\bemerging\s+market\b",
        "world": r"\bworld\s+(?:index|fund)\b",
        "dividend": r"\bdividend\b",
    }

    for topic, pattern in topic_patterns.items():
        if re.search(pattern, q_lower):
            topics.append(topic)

    return topics


def _extract_sectors(query: str) -> list[str]:
    q_lower = query.lower()
    sectors = []
    if re.search(r"\btech\b|\btechnology\b", q_lower):
        sectors.append("technology")
    if "healthcare" in q_lower or "health care" in q_lower:
        sectors.append("healthcare")
    if "energy" in q_lower:
        sectors.append("energy")
    if "financ" in q_lower:
        sectors.append("financials")
    return sectors


def _build_entities(query: str) -> ExtractedEntities:
    """Extract all entities from the query."""
    return ExtractedEntities(
        tickers=_extract_tickers(query),
        amount=_extract_amount(query),
        currency=_extract_currency(query),
        rate=_extract_rate(query),
        period_years=_extract_period_years(query),
        frequency=_extract_frequency(query),
        horizon=_extract_horizon(query),
        time_period=_extract_time_period(query),
        topics=_extract_topics(query),
        sectors=_extract_sectors(query),
        index=_extract_index(query),
        action=_extract_action(query),
        goal=_extract_goal(query),
    )


class RuleClassifier(BaseClassifier):
    """
    Keyword-based fallback classifier.

    Zero-cost, zero-latency. Always available. Handles ~85%+ of common
    financial assistant queries via pattern matching.
    """

    async def classify(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ClassifierResult:
        q_lower = query.strip().lower()

        # Check agent-specific patterns first (to prioritize specialist tasks over general education)
        for agent, intent, patterns in _AGENT_RULES:
            for pattern in patterns:
                if re.search(pattern, q_lower):
                    entities = _build_entities(query)
                    if history:
                        entities = self._apply_context_entities(entities, history)
                    
                    return ClassifierResult(
                        intent=intent,
                        agent=agent,
                        entities=entities,
                        confidence=0.75,
                        source="fallback_rules",
                    )

        # Check general / greeting patterns
        for pattern in _GENERAL_PATTERNS:
            if re.search(pattern, q_lower):
                return ClassifierResult(
                    intent="general",
                    agent="general_query",
                    entities=_build_entities(query),
                    confidence=0.7,
                    source="fallback_rules",
                )

        # Single ticker / company name → market_research
        entities = _build_entities(query)
        if entities.tickers and len(query.split()) <= 3:
            if history:
                entities = self._apply_context_entities(entities, history)
            return ClassifierResult(
                intent="market_research",
                agent="market_research",
                entities=entities,
                confidence=0.65,
                source="fallback_rules",
            )

        # Fallback with context from history
        if history:
            return self._classify_with_context(query, history, entities)

        # True fallback: general_query
        return ClassifierResult(
            intent="general",
            agent="general_query",
            entities=entities,
            confidence=0.3,
            source="fallback_rules",
        )

    def _apply_context_entities(self, entities: ExtractedEntities, history: list[dict[str, str]]) -> ExtractedEntities:
        """Carry forward tickers from history if missing in current turn."""
        if entities.tickers:
            return entities
            
        prior_tickers: list[str] = []
        for msg in reversed(history[-5:]):
            if msg.get("role") == "user":
                prior_tickers.extend(_extract_tickers(msg["content"]))
        
        if prior_tickers:
            entities.tickers = list(dict.fromkeys(prior_tickers[:3]))
        return entities

    def _classify_with_context(
        self,
        query: str,
        history: list[dict[str, str]],
        entities: ExtractedEntities,
    ) -> ClassifierResult:
        """Use session history to resolve ambiguous queries."""
        q_lower = query.lower()
        
        # Look at the last few user messages for context
        prior_tickers: list[str] = []
        prior_agents: list[str] = []
        
        # Helper to classify a string without history (to avoid recursion)
        def _quick_classify(text: str) -> tuple[str, str, ExtractedEntities]:
            t_lower = text.lower()
            for agent, intent, patterns in _AGENT_RULES:
                for pattern in patterns:
                    if re.search(pattern, t_lower):
                        return agent, intent, _build_entities(text)
            return "general_query", "general", _build_entities(text)

        for msg in reversed(history[-5:]):
            if msg.get("role") == "user":
                agent, intent, p_entities = _quick_classify(msg["content"])
                prior_agents.append(agent)
                prior_tickers.extend(p_entities.tickers)

        # Carry forward tickers if current query has none
        if not entities.tickers and prior_tickers:
            entities.tickers = list(dict.fromkeys(prior_tickers[:3]))

        # Handle "what about X?" style switches
        if re.search(r"\bwhat\s+about\s+", q_lower) or re.search(r"\band\s+", q_lower):
            # Inherit the last non-general agent if we have new entities
            for agent in prior_agents:
                if agent != "general_query":
                    return ClassifierResult(
                        intent="follow_up",
                        agent=agent,
                        entities=entities,
                        confidence=0.6,
                        source="fallback_rules",
                    )

        # Handle "Should I X?" or "Compare them" follow-ups
        if (re.search(r"\bshould\s+i\b", q_lower) or "compare" in q_lower) and prior_agents:
             for agent in prior_agents:
                if agent != "general_query":
                    return ClassifierResult(
                        intent="follow_up",
                        agent=agent,
                        entities=entities,
                        confidence=0.6,
                        source="fallback_rules",
                    )

        return ClassifierResult(
            intent="general",
            agent="general_query",
            entities=entities,
            confidence=0.4,
            source="fallback_rules",
        )
