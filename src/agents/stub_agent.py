"""
Stub agent — structured placeholder for unimplemented specialist agents.

Returns a well-formed response indicating the agent exists in the taxonomy
but is not yet implemented. This ensures the router never crashes on
unsupported agents and the response contract is always satisfied.
"""
from __future__ import annotations

from typing import Any

from src.models.schemas import ClassifierResult, StubAgentResponse


def run(classifier_result: ClassifierResult) -> dict[str, Any]:
    """
    Return a structured stub response for an unimplemented agent.

    The response includes the classified intent and entities so the client
    can still use the classification data even though the agent didn't run.
    """
    response = StubAgentResponse(
        agent=classifier_result.agent,
        intent=classifier_result.intent,
        entities=classifier_result.entities.model_dump(exclude_none=True),
        message=(
            f"The {classifier_result.agent} agent is not implemented in this prototype build. "
            f"Your query was classified as '{classifier_result.intent}' and would be handled by "
            f"the {classifier_result.agent} agent in the full system."
        ),
    )
    return response.model_dump()
