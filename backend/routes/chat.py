import logging

from fastapi import APIRouter, HTTPException

from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import chat

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Ask a question about the candidate's profile.
    """

    try:
        result = chat(
            message=request.message,
            conversation_id=request.conversation_id,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            conversation_id=result["conversation_id"],
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