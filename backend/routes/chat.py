import logging

from fastapi import APIRouter, Cookie, HTTPException

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import chat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])

SESSION_COOKIE = "rag_session_id"


@router.post("", response_model=ChatResponse)
def chat_endpoint(
    request: ChatRequest,
    rag_session_id: str | None = Cookie(
        default=None,
        alias=SESSION_COOKIE,
    ),
) -> ChatResponse:
    """
    Ask a question about the candidate's profile.

    If a RAG session exists, temporary documents belonging to that
    session are also included in retrieval.
    """
    try:
        result = chat(
            message=request.message,
            conversation_id=request.conversation_id,
            session_id=rag_session_id,
            debug=request.debug,
            response_mode=request.response_mode,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            conversation_id=result["conversation_id"],
            debug=result.get("debug"),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Chat request failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to process your question.",
        ) from exc
