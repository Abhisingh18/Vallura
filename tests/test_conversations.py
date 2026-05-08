"""
Conversation context tests — validates session history and context carryover.
"""
import pytest
from src.classifier.rule_classifier import RuleClassifier
from src.models.schemas import ExtractedEntities

@pytest.fixture
def classifier():
    return RuleClassifier()

@pytest.mark.asyncio
async def test_follow_up_context_carryover(classifier, conversation_test_cases):
    """
    Validates that entities and intents carry over correctly in follow-up turns.
    Based on fixtures/conversations/follow_up_session.json
    """
    cases = conversation_test_cases("follow_up_session")
    
    for case in cases:
        history = [{"role": "user", "content": q} for q in case["prior_user_turns"]]
        result = await classifier.classify(case["current_user_turn"], history=history)
        
        # Check agent/intent (some might be mapped differently in rule classifier, 
        # but the core logic should hold)
        expected_agent = case["expected"]["agent"]
        # Special case: fixture uses 'portfolio_query', taxonomy uses 'portfolio_health'
        if expected_agent == "portfolio_query":
            expected_agent = "portfolio_health"
            
        assert result.agent == expected_agent, f"Case {case['case_id']} failed agent match. Got {result.agent}, expected {expected_agent}"
        
        # Check entities
        actual_tickers = [t.upper() for t in result.entities.tickers]
        expected_tickers = [t.upper() for t in case["expected"]["entities"].get("tickers", [])]
        
        for t in expected_tickers:
            assert t in actual_tickers, f"Case {case['case_id']} failed ticker carryover. Missing {t} in {actual_tickers}"

@pytest.mark.asyncio
async def test_multi_intent_no_carryover(classifier, conversation_test_cases):
    """
    Validates that context does NOT carry over when the user switches topics.
    Based on fixtures/conversations/multi_intent_session.json
    """
    cases = conversation_test_cases("multi_intent_session")
    
    for case in cases:
        history = [{"role": "user", "content": q} for q in case["prior_user_turns"]]
        result = await classifier.classify(case["current_user_turn"], history=history)
        
        assert result.agent == case["expected"]["agent"]
        
        if "tickers" in case["expected"]["entities"]:
            actual_tickers = {t.upper().split(".")[0] for t in result.entities.tickers}
            expected_tickers = {t.upper().split(".")[0] for t in case["expected"]["entities"]["tickers"]}
            for t in expected_tickers:
                assert t in actual_tickers, f"Case {case['case_id']} missing ticker {t}. Got {actual_tickers}"
