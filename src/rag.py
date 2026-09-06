"""Groq-powered grounded answering on top of the retrieval pipeline.

Flow:

    question + conversation history
      -> retrieve relevant chunks (with relevance filtering)
      -> build labelled PROFILE CONTEXT
      -> Groq chat completion (system prompt + history + context/question)
      -> {"answer", "sources", "conversation_id"}

The model is configurable via the GROQ_MODEL env var and defaults to
the model this project was built with.
"""

import gc
import logging
import os

from groq import Groq

from src import conversations
from src.prompts import SYSTEM_PROMPT, build_context
from src.retrieve import DEFAULT_TOP_K, retrieve_full

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"  # model configured for this project

NOT_FOUND_ANSWER = (
    "I don't have that information in the provided profile."
)

IDENTITY_QUESTIONS = {
    "who are you",
    "what are you",
    "what do you do",
    "what can you tell me",
}


class MissingAPIKeyError(RuntimeError):
    """GROQ_API_KEY is not set."""


def get_model_name() -> str:
    """Model to use, from GROQ_MODEL env (never hardcode keys/models)."""
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


_groq_client: Groq | None = None


def is_identity_question(question: str) -> bool:
    """Return True when the user is asking about the assistant itself."""
    normalized = " ".join(question.lower().strip().split())

    return (
        normalized in IDENTITY_QUESTIONS
        or normalized.startswith("who are you")
        or normalized.startswith("what are you")
        or normalized.startswith("what do you do")
        or normalized.startswith("what can you tell me")
    )

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _build_messages(
    question: str,
    context: str,
    history: list[dict],
) -> list[dict]:
    """Assemble the chat messages: system rules, trimmed history, then the
    current question carrying the retrieved profile context."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"PROFILE CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}"
            ),
        }
    )
    return messages


def extract_sources(retrieved_chunks: list[dict]) -> list[dict]:
    """Deduplicated source citations derived from chunk metadata."""
    sources: list[dict] = []
    seen: set[tuple] = set()
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        key = (
            metadata.get("source"),
            metadata.get("section"),
            metadata.get("page"),
            metadata.get("project"),
            metadata.get("company"),
        )
        if key in seen:
            continue
        seen.add(key)
        source: dict = {
            "source": metadata.get("source", "unknown"),
            "section": metadata.get("section"),
            "page": metadata.get("page"),
        }
        if metadata.get("project"):
            source["project"] = metadata["project"]
        if metadata.get("company"):
            source["company"] = metadata["company"]
        if metadata.get("role"):
            source["role"] = metadata["role"]
        sources.append(source)
    return sources

def ask_resume(
    question: str,
    conversation_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    session_id: str | None = None,
) -> dict:

    question = (question or "").strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if not conversation_id:
        conversation_id = conversations.create_conversation()

    history = conversations.get_history(conversation_id)

    if is_identity_question(question):
        answer = (
            "I'm Rohan's Resume AI Assistant. "
            "I help users learn about Rohan's resume and LinkedIn profile, "
            "including his skills, experience, projects, education, "
            "achievements, and other professional information."
        )
        chunks = []

    else:
        retrieval = retrieve_full(
            question,
            top_k=top_k,
            session_id=session_id,
        )

        chunks = retrieval["chunks"]

        if not chunks or retrieval["confidence"] == "low":
            logger.info(
                "Low-confidence retrieval for %r - refusing without LLM call",
                question,
            )
            answer = NOT_FOUND_ANSWER

        else:
            context = build_context(chunks)

            client = _get_client()

            response = client.chat.completions.create(
                model=get_model_name(),
                messages=_build_messages(
                    question,
                    context,
                    history,
                ),
                temperature=0.2,
                max_tokens=1024,
            )

            answer = (
                response.choices[0].message.content or ""
            ).strip()

            del response
            gc.collect()

    conversations.append_messages(
        conversation_id,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    )

    return {
        "answer": answer,
        "sources": extract_sources(chunks) if chunks else [],
        "conversation_id": conversation_id,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # CLI usage convenience; the FastAPI app loads it too

    user_question = input("Ask about the candidate: ")
    result = ask_resume(user_question)
    print("\nAI:")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source['source']} | {source['section']} | page {source['page']}")
