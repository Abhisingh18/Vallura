"""
Safety Guard — rule-based, deterministic, zero-latency content filter.

Runs *before* the LLM classifier. No network calls, no ML models.
Design philosophy: block obvious harmful financial intent while allowing
educational queries about the same topics. The heuristic is:
  - harmful queries contain *actionable* phrases ("help me", "how do I", "tell me to")
    combined with dangerous topic keywords
  - educational queries use *descriptive* language ("what is", "explain", "how does ... work")

Trade-off documented in README: a small number of edge cases may be over-blocked.
This is intentional — safety recall > pass-through rate for a financial platform.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from src.models.schemas import SafetyResult


# ---------------------------------------------------------------------------
# Category-specific refusal messages
# ---------------------------------------------------------------------------

_REFUSAL_MESSAGES: dict[str, str] = {
    "insider_trading": (
        "I can't help with trading based on material non-public information. "
        "Insider trading is illegal under securities law in all major jurisdictions. "
        "If you'd like to learn about insider trading regulations, feel free to ask."
    ),
    "market_manipulation": (
        "I can't assist with schemes designed to artificially move prices or "
        "create misleading market activity. These practices are illegal and carry "
        "severe penalties. I'm happy to explain how regulators detect manipulation."
    ),
    "money_laundering": (
        "I can't help with structuring transactions to evade reporting requirements "
        "or obscure the origin of funds. Anti-money laundering laws require full "
        "transparency. I can explain AML regulations if you're interested."
    ),
    "guaranteed_returns": (
        "I can't promise or guarantee specific investment returns — no one can. "
        "Claims of guaranteed returns are a hallmark of investment fraud. "
        "I can share historical market data and explain realistic expectations."
    ),
    "reckless_advice": (
        "I can't encourage putting all your savings into a single high-risk position "
        "or recommend actions that could jeopardize your financial security. "
        "I'm here to help you build a balanced approach to investing."
    ),
    "sanctions_evasion": (
        "I can't assist with circumventing international sanctions or OFAC restrictions. "
        "Sanctions evasion is a serious federal offense. "
        "I can explain how sanctions screening works if that's helpful."
    ),
    "fraud": (
        "I can't help create fraudulent documents or assist with any form of financial fraud. "
        "These activities carry severe criminal penalties. "
        "If you need help understanding compliance requirements, I'm happy to assist."
    ),
}

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------
# Each category has:
#   topic_patterns  — words/phrases that identify the *subject area*
#   intent_patterns — phrases that signal *actionable harmful intent*
#   block_patterns  — high-confidence phrases that alone are sufficient to block

class _CategoryRule(NamedTuple):
    topic_patterns: list[re.Pattern[str]]
    intent_patterns: list[re.Pattern[str]]
    block_patterns: list[re.Pattern[str]]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_RULES: dict[str, _CategoryRule] = {
    "insider_trading": _CategoryRule(
        topic_patterns=_compile([
            "insider", "non-public", "nonpublic", "unannounced",
            "confidential", "tip about earnings", "before the announcement",
            "before the call", "tip from", "gave me a tip",
            "material non-public", "mnpi",
        ]),
        intent_patterns=_compile([
            "should i buy", "should i trade", "when should i buy",
            "load up", "help me trade", "help me buy",
            "i need to know the earnings before",
            "trade on this", "before tomorrow",
        ]),
        block_patterns=_compile([
            r"i\s+work\s+at\s+.+\b(?:unannounced|confidential|secret)\b.*\bbuy\b",
            r"(?:tip|inside)\s+(?:about|on)\s+earnings.*(?:buy|trade|load)",
            r"help\s+me\s+trade\s+on\s+(?:this\s+)?confidential",
            r"gave\s+me\s+a\s+tip.*(?:load\s+up|buy|trade)",
            r"i\s+need\s+to\s+know\s+the\s+earnings\s+before",
            r"trade\s+on\s+this\s+confidential",
        ]),
    ),
    "market_manipulation": _CategoryRule(
        topic_patterns=_compile([
            "pump", "dump", "wash trad", "spoof", "layering",
            "coordinated buying", "manipulat", "move this stock",
            "move the price", "create volume", "artificial",
        ]),
        intent_patterns=_compile([
            "how can i", "help me", "design a", "scheme to",
            "pump up", "sell after",
        ]),
        block_patterns=_compile([
            r"(?:pump\s+up|pump\s+and\s+dump).*(?:sell|before\s+selling)",
            r"(?:coordinated|organized)\s+buying\s+scheme",
            r"help\s+me\s+wash\s+trad",
            r"how\s+can\s+i\s+pump",
            r"design\s+a\s+.*(?:buying|trading)\s+scheme",
            r"create\s+(?:fake\s+)?volume",
        ]),
    ),
    "money_laundering": _CategoryRule(
        topic_patterns=_compile([
            "launder", "structur", "reporting threshold", "10k reporting",
            "obscure the source", "hide.*profits", "without reporting",
            "avoid.*report", "cash through",
        ]),
        intent_patterns=_compile([
            "how do i", "help me", "move.*cash", "layer my trades",
            "without it being traced", "hide",
        ]),
        block_patterns=_compile([
            r"move\s+\d+.*cash.*without\s+report",
            r"structure\s+deposits?\s+to\s+avoid",
            r"avoid\s+the\s+10k\s+reporting",
            r"layer\s+my\s+trades\s+to\s+obscure",
            r"obscure\s+the\s+source\s+of\s+(?:these\s+)?funds",
            r"hide\s+(?:trading\s+)?profits?\s+from\s+(?:the\s+)?tax",
        ]),
    ),
    "guaranteed_returns": _CategoryRule(
        topic_patterns=_compile([
            "guarantee", "guaranteed", "100% certain", "foolproof",
            "double in a year", "certain to go up", "promise me",
            "promise my money", r"risk.?free\s+return",
        ]),
        intent_patterns=_compile([
            "guarantee me", "promise me", "tell me a stock that",
            "foolproof way", "certain to go up",
        ]),
        block_patterns=_compile([
            r"guarantee\s+me\s+\d+%?\s+return",
            r"promise\s+me\s+my\s+money\s+will",
            r"100%\s+certain\s+to\s+go\s+up",
            r"(?:stock|investment)\s+that(?:'s|s)?\s+(?:100%\s+)?certain",
            r"foolproof\s+way\s+to\s+make\s+\d+",
            r"double\s+(?:my\s+money\s+)?in\s+a\s+(?:year|month)",
        ]),
    ),
    "reckless_advice": _CategoryRule(
        topic_patterns=_compile([
            "all my retirement", "all my savings", r"entire.*savings",
            "emergency fund into", "margin loan", "mortgage my house",
            "put everything", "entire emergency",
        ]),
        intent_patterns=_compile([
            "do it", "tell me to", "should i put",
            "tell me which stock to mortgage",
        ]),
        block_patterns=_compile([
            r"(?:all|entire)\s+(?:my\s+)?(?:retirement|savings).*(?:crypto|option|margin|single)",
            r"margin\s+loan\s+to\s+buy\s+more",
            r"tell\s+me\s+to\s+take\s+a\s+margin",
            r"(?:entire\s+)?emergency\s+fund\s+into\s+options?",
            r"should\s+i\s+put\s+my\s+(?:entire\s+)?emergency\s+fund\s+into",
            r"mortgage\s+my\s+house\s+(?:for|to\s+buy)",
            r"tell\s+me\s+which\s+stock\s+to\s+mortgage",
            r"i'm\s+\d{2,}\s+and\s+want\s+to\s+put\s+all.*(?:crypto|options?|margin)",
        ]),
    ),
    "sanctions_evasion": _CategoryRule(
        topic_patterns=_compile([
            "sanction", "ofac", "sanctioned", r"bypass.*sanction",
            "shell company", r"without.*traced",
        ]),
        intent_patterns=_compile([
            "help me", "how do i", "route trades", r"invest in.*sanctioned",
        ]),
        block_patterns=_compile([
            r"(?:bypass|circumvent|evade|avoid)\s+(?:ofac\s+)?sanctions?",
            r"route\s+trades?\s+through\s+.*(?:shell|front)\s+company",
            r"invest\s+in\s+(?:a\s+)?sanctioned.*without",
            r"sanctioned\s+(?:russian|iranian|north\s+korean).*without.*traced",
        ]),
    ),
    "fraud": _CategoryRule(
        topic_patterns=_compile([
            "fake contract", "fraudulent", "forge", "falsify",
            "draft a fake", r"fake.*claim",
        ]),
        intent_patterns=_compile([
            "draft", "create", "make", "forge",
        ]),
        block_patterns=_compile([
            r"(?:draft|create|forge|fabricate)\s+(?:a\s+)?fake\s+(?:contract|document|note)",
            r"fake\s+contract\s+note\s+to\s+claim",
        ]),
    ),
}

# Educational pass-through signals — if these appear without harmful intent,
# the query is likely educational and should be allowed through.
_EDUCATIONAL_SIGNALS: list[re.Pattern[str]] = _compile([
    r"\bwhat\s+(?:is|are|does|was)\b",
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\bhow\s+does\b",
    r"\bhow\s+do\s+(?:regulators?|brokerage|broker|the\s+(?:sec|fca|cftc))\b",
    r"\bwhat(?:'s|\s+is)\s+the\s+(?:penalty|difference|role|law|regulation|rule)\b",
    r"\b(?:legal|illegal)\s*\?\s*$",
    r"\bis\s+(?:it|this)\s+(?:ever\s+)?(?:legal|legitimate)\b",
    r"\bwhy\s+(?:is|are)\b",
    r"\b(?:disclosure|compliance)\s+(?:rules?|obligations?|requirements?)\b",
    r"\bpenalt(?:y|ies)\s+(?:for|under)\b",
    r"\brequirements?\s+for\b",
    r"\bregulat(?:or|ion|ory)\b",
    r"\binvestigat\b",
    r"\b(?:how|what)\s+(?:does|do)\s+(?:the\s+)?(?:sec|fca|cftc|regulator)\b",
    r"\bthree\s+stages\b",
    r"\bplacement.*layering.*integration\b",
    r"\bpreventing\s+fraud\b",
    r"\bwhat\s+factors\s+should\b",
    r"\brisks?\s+of\b",
    r"\bhow\s+should\b",
    r"\ballocated\s+relative\b",
    r"\bhistorical\s+(?:average|annual|return)\b",
    r"\bponzi\s+scheme\b",
    r"\bred\s+flag\b",
    r"\bshort\s+selling\s+illegal\b",
    r"\b(?:aml|anti.?money)\s+(?:reporting|requirements|regulations)\b",
    r"\bwhat\s+is\s+structuring\b",
    r"\bscreening?\s+for\b",
])


def _is_educational(query_lower: str) -> bool:
    """Heuristic: does the query look educational rather than actionable?"""
    return any(p.search(query_lower) for p in _EDUCATIONAL_SIGNALS)


def check(query: str) -> SafetyResult:
    """
    Synchronous safety check. Returns a SafetyResult.

    Runs in <1ms for any input. No network calls. Deterministic.
    """
    q = query.strip()
    q_lower = q.lower()

    for category, rule in _RULES.items():
        # Step 1: Check high-confidence block patterns first
        for pattern in rule.block_patterns:
            if pattern.search(q_lower):
                # Even a block-pattern match yields if the query is clearly educational
                if _is_educational(q_lower):
                    continue
                return SafetyResult(
                    blocked=True,
                    category=category,
                    message=_REFUSAL_MESSAGES.get(category, "This request has been blocked for safety reasons."),
                )

        # Step 2: Check for topic + intent signal co-occurrence
        has_topic = any(p.search(q_lower) for p in rule.topic_patterns)
        has_intent = any(p.search(q_lower) for p in rule.intent_patterns)

        if has_topic and has_intent:
            if not _is_educational(q_lower):
                return SafetyResult(
                    blocked=True,
                    category=category,
                    message=_REFUSAL_MESSAGES.get(category, "This request has been blocked for safety reasons."),
                )

    return SafetyResult(blocked=False)
