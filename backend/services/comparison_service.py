"""Thin service wrapper around the comparison engine."""

import logging

from src.compare import (
    compare_candidates,
    generate_comparison_explanation,
)

logger = logging.getLogger(__name__)


def compare(
    job_description: str,
    session_id: str,
) -> dict:
    """Run the full candidate comparison pipeline.

    Args:
        job_description: Recruiter-supplied job requirements.
        session_id: RAG session containing Candidate B's resume.

    Returns:
        Structured comparison dict ready for the API response.
    """

    logger.info(
        "Starting candidate comparison. session_id=%s",
        session_id,
    )

    result = compare_candidates(
        job_description=job_description,
        session_id=session_id,
    )

    # Generate LLM explanation grounded in the deterministic scores
    reasoning = generate_comparison_explanation(
        job_description=job_description,
        comparison_result=result,
    )

    result["reasoning"] = reasoning

    return result
