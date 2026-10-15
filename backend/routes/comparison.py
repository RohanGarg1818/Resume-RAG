import logging

from fastapi import APIRouter, Cookie, HTTPException

from backend.schemas.comparison import ComparisonRequest, ComparisonResponse
from backend.services.comparison_service import compare
from src.session_store import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compare-candidates", tags=["Comparison"])

SESSION_COOKIE = "rag_session_id"


@router.post("", response_model=ComparisonResponse)
def compare_candidates_endpoint(
    request: ComparisonRequest,
    rag_session_id: str | None = Cookie(
        default=None,
        alias=SESSION_COOKIE,
    ),
) -> ComparisonResponse:
    """Compare Candidate A (existing resume) against Candidate B (uploaded in session).

    The recruiter provides job requirements. Candidate B must already be
    uploaded in the active RAG session.
    """
    if not rag_session_id:
        raise HTTPException(
            status_code=400,
            detail="No active session. Upload Candidate B's resume first.",
        )

    session = get_session(rag_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session expired. Please start a new session and re-upload Candidate B.",
        )

    try:
        result = compare(
            job_description=request.job_requirements,
            session_id=rag_session_id,
        )

        return ComparisonResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Candidate comparison failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to complete the candidate comparison.",
        ) from exc
