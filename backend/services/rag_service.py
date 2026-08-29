import logging

from src.rag import ask_resume

logger = logging.getLogger(__name__)


def chat(
    message: str,
    conversation_id: str | None = None,
) -> dict:
    """Send a question to the existing Resume RAG engine."""

    logger.info(
        "Processing chat request. conversation_id=%s",
        conversation_id,
    )

    return ask_resume(
        question=message,
        conversation_id=conversation_id,
    )