
import logging

from src.rag import ask_resume

logger = logging.getLogger(__name__)


def chat(
    message: str,
    conversation_id: str | None = None,
    session_id: str | None = None,
    debug: bool = False,
    response_mode: str = "standard",
) -> dict:
    """Send a question to the Resume RAG engine.

    If a session_id is provided, temporary documents uploaded
    during that session are included in retrieval.
    """

    logger.info(
        "Processing chat request. conversation_id=%s session_id=%s",
        conversation_id,
        session_id,
    )

    return ask_resume(
        question=message,
        conversation_id=conversation_id,
        session_id=session_id,
        debug=debug,
        response_mode=response_mode,
    )
