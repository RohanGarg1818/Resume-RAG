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
import re
import time

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
    response_mode: str = "standard",
) -> list[dict]:
    """Assemble the chat messages: system rules, trimmed history, then the
    current question carrying the retrieved profile context."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    mode_instruction = ""
    if response_mode == "recruiter":
        mode_instruction = (
            "\nRECRUITER MODE: Summarize only evidence relevant to the role "
            "or requirements in the question. Separate verified evidence from "
            "gaps; do not make a hiring decision."
        )
    elif response_mode == "comparison":
        mode_instruction = (
            "\nCOMPARISON MODE: Compare only the named projects or entities. "
            "Use a compact Markdown table when the retrieved evidence supports "
            "both sides, and identify missing evidence explicitly."
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"PROFILE CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}{mode_instruction}"
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
            "citation_id": chunk.get("citation_id"),
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


_ANSWER_STOPWORDS = frozenset(
    "a an the and or but is are was were be been being to of in on for with "
    "from by as at it this that these those he she rohan his her their has have "
    "had does do did can could would should will may might not no yes based "
    "information provided profile project experience skills used using includes "
    "was were is are it that this".split()
)


def validate_answer_evidence(answer: str, chunks: list[dict]) -> dict:
    """Perform a conservative post-generation evidence check.

    This is not an LLM judge. It catches an especially costly failure mode:
    an otherwise fluent answer introducing a number or rare technical term that
    is absent from the selected context.  Ambiguous answers are flagged for
    evaluation/debugging rather than silently rewritten.
    """
    if not answer:
        return {"valid": False, "support": 0.0, "reason": "empty_answer"}
    normalized = answer.lower()
    if NOT_FOUND_ANSWER.lower() in normalized:
        return {"valid": True, "support": 1.0, "reason": "controlled_refusal"}
    context = " ".join(str(chunk.get("text", "")) for chunk in chunks).lower()
    # Dots within technical names (``next.js``) are meaningful; sentence-final
    # punctuation is not.  This pattern preserves the former without making
    # ``redis.`` a different fact from ``redis``.
    technical_token = r"[a-z0-9+#]+(?:\.[a-z0-9+#]+)*"
    answer_terms = set(re.findall(technical_token, normalized)) - _ANSWER_STOPWORDS
    context_terms = set(re.findall(technical_token, context))
    answer_terms = {term for term in answer_terms if len(term) > 1 or term in {"c", "r"}}
    if not answer_terms:
        return {"valid": True, "support": 1.0, "reason": "no_checkable_claims"}
    unsupported = sorted(term for term in answer_terms if term not in context_terms)
    support = 1 - len(unsupported) / len(answer_terms)
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\+?\b", normalized))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\+?\b", context))
    unsupported_numbers = sorted(answer_numbers - context_numbers)
    # A model that invents a metric is rejected even when its surrounding text
    # mostly overlaps.  Otherwise we only reject clearly unsupported answers.
    valid = not unsupported_numbers and support >= 0.45
    return {
        "valid": valid,
        "support": round(support, 4),
        "unsupported_terms": unsupported[:20],
        "unsupported_numbers": unsupported_numbers,
        "reason": "supported" if valid else "insufficient_evidence",
    }


def append_source_citations(answer: str, chunks: list[dict]) -> str:
    """Guarantee usable citations even if a provider omits inline labels."""
    if not answer or NOT_FOUND_ANSWER.lower() in answer.lower() or not chunks:
        return answer
    labels = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        label = chunk.get("citation_id")
        if not label:
            continue
        detail = metadata.get("project") or metadata.get("company") or metadata.get("section") or metadata.get("source")
        labels.append(f"[{label}] {detail}")
    if not labels or re.search(r"\[S\d+\]", answer):
        return answer
    return f"{answer}\n\nSources: {', '.join(dict.fromkeys(labels))}"

def ask_resume(
    question: str,
    conversation_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    session_id: str | None = None,
    evaluation_mode: bool = False,
    debug: bool = False,
    response_mode: str = "standard",
) -> dict:
    question = (question or "").strip()

    if not question:
        raise ValueError("Question must not be empty.")

    if not conversation_id:
        conversation_id = conversations.create_conversation()

    history = conversations.get_history(conversation_id)

    retrieval_latency_ms = 0.0
    llm_latency_ms = 0.0
    retrieval = {
        "chunks": [],
        "confidence": "high",
    }

    if is_identity_question(question):
        retrieval_latency_ms = 0.0
        llm_latency_ms = 0.0

        retrieval = {
            "chunks": [],
            "confidence": "high",
        }

        answer = (
            "I'm Rohan's Resume AI Assistant. "
            "I help users learn about Rohan's resume and LinkedIn profile, "
            "including his skills, experience, projects, education, "
            "achievements, and other professional information."
        )

        chunks = []

    else:
        retrieval_start=time.perf_counter()


        retrieval = retrieve_full(
            question,
            top_k=top_k,
            session_id=session_id,
            debug=debug,
        )

        retrieval_latency_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

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

            llm_start = time.perf_counter()

            response = client.chat.completions.create(
                model=get_model_name(),
                messages=_build_messages(
                    question,
                    context,
                    history,
                    response_mode=response_mode,
                ),
                temperature=0.2,
                max_tokens=1024,
            )

            llm_latency_ms = (
                time.perf_counter() - llm_start
            ) * 1000

            answer = (
                response.choices[0].message.content or ""
            ).strip()

            validation = validate_answer_evidence(answer, chunks)
            if not validation["valid"]:
                logger.warning("Answer failed evidence validation: %s", validation)
                answer = NOT_FOUND_ANSWER
            else:
                answer = append_source_citations(answer, chunks)

            del response
            gc.collect()

    conversations.append_messages(
        conversation_id,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    )

    validation = locals().get("validation", {
        "valid": answer == NOT_FOUND_ANSWER or bool(chunks),
        "reason": "retrieval_refusal" if answer == NOT_FOUND_ANSWER else "identity",
    })
    result = {
        "answer": answer,
        "sources": extract_sources(chunks) if chunks else [],
        "conversation_id": conversation_id,
    }

    if evaluation_mode:
        result["evaluation"] = {
            "retrieved_chunks": chunks,
            "retrieval_confidence": retrieval.get(
                "confidence",
                "unknown",
            ),
            "retrieval_latency_ms": retrieval_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "retrieved_count": len(chunks),
            "evidence_validation": validation,
        }

    if debug:
        result["debug"] = retrieval.get("diagnostics", {})

    return result


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
