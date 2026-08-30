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
    "I don't have that information in the provided profile. "
    "Please ask about the candidate's skills, experience, education, "
    "projects or achievements."
)


class MissingAPIKeyError(RuntimeError):
    """GROQ_API_KEY is not set."""


def get_model_name() -> str:
    """Model to use, from GROQ_MODEL env (never hardcode keys/models)."""
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


_groq_client: Groq | None = None


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
) -> dict:
    """Answer a profile question with grounded retrieval.

    Returns:
        {"answer": str,
         "sources": [ {source, section, page, project?/company?} ],
         "conversation_id": str}

    Raises:
        MissingAPIKeyError: when GROQ_API_KEY is unset.
        groq.AuthenticationError / NotFoundError / APIError: surfaced so the
            API layer can translate them into friendly HTTP responses.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if not conversation_id:
        conversation_id = conversations.create_conversation()

    history = conversations.get_history(conversation_id)
    retrieval = retrieve_full(question, top_k=top_k)
    chunks = retrieval["chunks"]

    # Guardrail: never send Groq a context we do not trust. If retrieval
    # is not confident the question relates to the profile, refuse here -
    # this is what stops "favourite football team" from hallucinating.
    if not chunks or retrieval["confidence"] == "low":
        logger.info("Low-confidence retrieval for %r - refusing without LLM call", question)
        answer = NOT_FOUND_ANSWER
    else:
        context = build_context(chunks)
        client = _get_client()
        response = client.chat.completions.create(
            model=get_model_name(),
            messages=_build_messages(question, context, history),
            temperature=0.2,
            max_tokens=1024,
        )
        answer = (response.choices[0].message.content or "").strip()
        del response
        gc.collect()

    conversations.append_messages(
        conversation_id,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    )

    return {
        "answer": answer,
        "sources": extract_sources(chunks) if answer is not NOT_FOUND_ANSWER else [],
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
