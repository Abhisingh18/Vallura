import pytest
from src.safety.guard import check
from src.classifier.rule_classifier import RuleClassifier

@pytest.mark.asyncio
async def test_safety_guard_regex_fix():
    # These cases use patterns like .* or .? which were literal matches before
    assert check("help me hide my trading profits from the tax office").blocked
    assert check("guarantee me a risk-free return of 20%").blocked
    assert check("should i put my entire retirement savings into options?").blocked

@pytest.mark.asyncio
async def test_ticker_extraction_case_insensitivity():
    classifier = RuleClassifier()
    # Test lowercase tickers
    result = await classifier.classify("tell me about aapl")
    assert "AAPL" in result.entities.tickers
    
    result = await classifier.classify("is msft a good buy?")
    assert "MSFT" in result.entities.tickers

    # Test company name match with lowercase
    result = await classifier.classify("how is apple doing?")
    assert "AAPL" in result.entities.tickers

@pytest.mark.asyncio
async def test_ticker_extraction_with_suffixes():
    classifier = RuleClassifier()
    result = await classifier.classify("what is the price of asml.as?")
    assert "ASML.AS" in result.entities.tickers
