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
from src.session_store import get_candidate_name

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"  # model configured for this project

NOT_FOUND_ANSWER = (
    "I don't have that information in the provided profile."
)

IDENTITY_PATTERNS = (
    "who are you",
    "what are you",
    "what do you do",
    "what can you do",
    "what can you tell me",
)

GREETING_WORDS = frozenset(
    "hi hii hiya hello hey yo greetings howdy sup whats up "
    "hiiii heyy helloo heyyy hiya".split()
)

# Multi-word greetings that should be treated as a single greeting unit.
GREETING_PHRASES = frozenset({
    "hi there",
    "hello there",
    "hey there",
    "whats up",
    "what is up",
    "good morning",
    "good afternoon",
    "good evening",
    "good day",
})

GREETING_ANSWER = (
    "Hi! I'm Rohan's Resume AI Assistant. "
    "I can help you learn about Rohan's education, skills, projects, "
    "experience, and achievements."
)

IDENTITY_ANSWER = (
    "I'm Rohan's Resume AI Assistant. "
    "I can answer questions about Rohan's education, skills, projects, "
    "experience, and achievements from his resume and LinkedIn profile."
)


# Leading filler words that may precede an identity question.
_LEADING_FILLERS = frozenset({"so", "well", "ok", "okay", "um", "uh", "then", "now"})

# Harmless trailing words allowed after an identity phrase
# (e.g. 'who are you thanks'); anything else falls through to RAG.
_IDENTITY_TAIL_FILLERS = frozenset({
    "thanks", "thank", "you", "exactly", "actually", "really",
    "please", "today", "now", "man", "so", "far", "again",
})


def _normalize(question: str) -> str:
    """Lowercase, split on whitespace, strip punctuation from each token,
    and rejoin so 'HI!!!' becomes 'hi' and 'you???' becomes 'you'."""
    cleaned = []
    for token in question.lower().split():
        stripped = token.strip("!?.;:,\"'()[]{}<>")
        if stripped:
            cleaned.append(stripped)
    return " ".join(cleaned)


def _strip_greeting(words: list[str]) -> list[str]:
    """Remove leading greeting words/phrases from a token list."""
    changed = True
    while changed and words:
        changed = False
        if len(words) >= 2 and " ".join(words[:2]) in GREETING_PHRASES:
            words = words[2:]
            changed = True
        elif words[0] in GREETING_WORDS:
            words = words[1:]
            changed = True
    return words


def is_identity_question(question: str) -> bool:
    """True when the user asks about the assistant itself (including
    greeting + identity combos like 'hi what are you').

    Uses word-boundary matching so 'what are your skills' or
    'what can you tell me about education' are NOT treated as identity.
    """
    words = _normalize(question).split()
    if not words:
        return False

    # Strip leading greetings and filler words.
    words = _strip_greeting(words)
    while words and words[0] in _LEADING_FILLERS:
        words = words[1:]

    remainder = " ".join(words)
    if not remainder:
        return False

    for pattern in IDENTITY_PATTERNS:
        if remainder == pattern:
            return True
        # Pattern + harmless trailing fillers only (word-boundary check).
        if remainder.startswith(pattern + " "):
            tail = remainder[len(pattern) + 1:].split()
            if tail and all(w in _IDENTITY_TAIL_FILLERS for w in tail):
                return True
    return False


def is_greeting(question: str) -> bool:
    """True for a pure greeting with no real question attached."""
    normalized = _normalize(question)
    if not normalized:
        return False
    if normalized in GREETING_PHRASES:
        return True
    words = normalized.split()
    return all(w in GREETING_WORDS for w in words) and len(words) <= 3


def _greeting_prefix(question: str) -> str | None:
    """If the question starts with a greeting followed by a real question,
    return a short greeting prefix; otherwise None."""
    words = _normalize(question).split()
    if not words:
        return None

    # Determine how many leading tokens form the greeting.
    consumed = 0
    if len(words) >= 2 and " ".join(words[:2]) in GREETING_PHRASES:
        consumed = 2
    elif words[0] in GREETING_WORDS:
        consumed = 1
    if consumed == 0:
        return None

    remainder = " ".join(words[consumed:])
    if remainder and not is_identity_question(question):
        return "Hi! "
    return None


def _trim_truncated_answer(answer: str) -> str:
    """Trim a truncated LLM response so it never ends mid-word,
    mid-bullet, mid-list, or mid-sentence.

    Only called when finish_reason == "length", so the trailing content is
    known to be cut off.  Returns the longest complete prefix, or
    NOT_FOUND_ANSWER when no complete prefix exists.
    """
    if not answer:
        return answer

    # A trailing colon/dash means a list or explanation was started but cut.
    # Drop back to the last complete sentence or bullet before it.
    stripped = answer.rstrip()
    if stripped.endswith((":", ":-", "—", "-", "…")):
        answer = answer.rstrip(":-—… \t")
        # Fall through to bullet/sentence trimming on the shortened text.

    # Bullet list: the trailing bullet is incomplete (truncation cut it),
    # unless it happens to end with terminal punctuation.
    bullet_starts = [
        m.start()
        for m in re.finditer(r"(?m)^\s*(?:[-*•]|\d+\.)\s+", answer)
    ]
    if bullet_starts:
        last_bullet = bullet_starts[-1]
        last_bullet_text = answer[last_bullet:].rstrip()
        if last_bullet_text.endswith((".", "!", "?", '"', "'", "`")):
            return answer.rstrip()
        trimmed = answer[:last_bullet].rstrip()
        if trimmed:
            return trimmed

    # Prose: cut back to the last sentence boundary.  Require punctuation
    # followed by whitespace/end so a period inside a technical token
    # (next.js) is never treated as a sentence boundary.
    boundary = None
    for m in re.finditer(r"[.!?](?:\s|$)", answer):
        boundary = m
    if boundary:
        return answer[: boundary.end()].rstrip()

    # No complete sentence or bullet exists — refuse rather than return
    # a garbled fragment.
    return NOT_FOUND_ANSWER


# Leading-greeting pattern used to strip a greeting off a real question so
# retrieval only sees the profile question. Multi-word phrases come first so
# 'hi there' is consumed as a unit rather than just 'hi'.
_LEADING_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"hi there|hello there|hey there|whats up|what is up|"
    r"good (?:morning|afternoon|evening|day)|"
    r"hi+|hello+|hey+|yo|hiya|howdy|sup|greetings"
    r")\b[\s!,.;:\-]*",
    re.IGNORECASE,
)


def _strip_leading_greeting(question: str) -> str:
    """Remove a leading greeting so retrieval sees only the real question.
    'hi what is his technical stack?' -> 'what is his technical stack?'"""
    remaining = question
    for _ in range(3):  # allow repeated greetings, e.g. 'hey yo hi ...'
        match = _LEADING_GREETING_RE.match(remaining)
        if not match:
            break
        remaining = remaining[match.end():].lstrip()
    return remaining if remaining else question


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
            "or requirements in the question. Distinguish confirmed facts from "
            "missing evidence; do not make a hiring decision."
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


EXCERPT_CHARS = 700


def _similarities(query: str | None, chunks: list[dict]) -> list[float | None]:
    """Cosine similarity between the question and each chunk's text.

    Computed fresh for display because deterministic routes (contact scan,
    exact entity) carry placeholder similarities of 1.0, and cross-encoder
    logits are not calibrated on resume text. Uses the same normalised MiniLM
    embeddings as the index, so the dot product is the cosine.
    """
    if not query or not chunks:
        return [None] * len(chunks)
    try:
        from src.embeddings import generate_embeddings

        vectors = generate_embeddings([query] + [str(chunk.get("text", "")) for chunk in chunks])
        query_vector = vectors[0]
        return [
            round(max(0.0, sum(a * b for a, b in zip(query_vector, vector))), 2)
            for vector in vectors[1:]
        ]
    except Exception:
        logger.warning("Could not compute display similarities", exc_info=True)
        return [None] * len(chunks)


def extract_sources(retrieved_chunks: list[dict], query: str | None = None) -> list[dict]:
    """Deduplicated source citations derived from chunk metadata.

    Each source also lists every context label (``citation_ids``) merged into
    it, the question-to-passage cosine similarity (``score``) and the passage
    text (``excerpt``) so the UI can resolve inline [S#] markers and show the
    evidence.
    """
    sources: list[dict] = []
    by_key: dict[tuple, dict] = {}
    similarities = _similarities(query, retrieved_chunks)
    for chunk, relevance in zip(retrieved_chunks, similarities):
        metadata = chunk["metadata"]
        key = (
            metadata.get("source"),
            metadata.get("section"),
            metadata.get("page"),
            metadata.get("project"),
            metadata.get("company"),
            metadata.get("owner_id"),
        )
        existing = by_key.get(key)
        if existing is not None:
            if chunk.get("citation_id"):
                existing["citation_ids"].append(chunk["citation_id"])
            if relevance is not None and (existing["score"] is None or relevance > existing["score"]):
                existing["score"] = relevance
            continue
        text = str(chunk.get("text", "")).strip()
        source: dict = {
            "citation_id": chunk.get("citation_id"),
            "citation_ids": [chunk["citation_id"]] if chunk.get("citation_id") else [],
            "source": metadata.get("source", "unknown"),
            "section": metadata.get("section"),
            "page": metadata.get("page"),
            "score": relevance,
            "excerpt": text[:EXCERPT_CHARS] + ("…" if len(text) > EXCERPT_CHARS else ""),
        }
        by_key[key] = source
        if metadata.get("owner_name"):
            source["candidate"] = metadata["owner_name"]
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


_SUFFIXES = ("ations", "ation", "ings", "ing", "ed", "es", "s")


def _stem(term: str) -> str:
    """Very small suffix stripper used only for evidence matching."""
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            return term[: -len(suffix)]
    return term


def validate_answer_evidence(answer: str, chunks: list[dict]) -> dict:
    """Perform a conservative post-generation evidence check.

    This is not an LLM judge. It catches an especially costly failure mode:
    an otherwise fluent answer introducing a number or rare technical term that
    is absent from the selected context.  Ambiguous answers are flagged for
    evaluation/debugging rather than silently rewritten.
    """
    if not answer:
        return {"valid": False, "support": 0.0, "reason": "empty_answer"}
    # Inline citation labels ([S1]) are references, not claims.
    normalized = re.sub(r"\[s\d+\]", " ", answer.lower())
    if NOT_FOUND_ANSWER.lower() in normalized:
        return {"valid": True, "support": 1.0, "reason": "controlled_refusal"}
    # Markdown list markers ("1. ", "2) ") are formatting, not claims.  They
    # previously counted as unsupported numbers, so any correct numbered-list
    # answer was replaced by the refusal message.
    normalized = re.sub(r"(?m)^\s*\d+[.)]\s+", "", normalized)
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
    # Numbers are verified separately below; inflections ("mentored" vs
    # "mentoring") should not count as unsupported vocabulary.
    answer_terms = {term for term in answer_terms if not re.fullmatch(r"[\d.+]+", term)}
    if not answer_terms and not re.search(r"\d", normalized):
        return {"valid": True, "support": 1.0, "reason": "no_checkable_claims"}
    context_stems = {_stem(term) for term in context_terms}
    unsupported = sorted(
        term for term in answer_terms
        if term not in context_terms and _stem(term) not in context_stems
    )
    support = 1 - len(unsupported) / len(answer_terms) if answer_terms else 1.0
    answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\+?\b", normalized))
    context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\+?\b", context))
    # "100" in the answer is supported by "100+" in the context (and vice versa).
    context_number_values = {number.rstrip("+") for number in context_numbers}
    unsupported_numbers = sorted(
        number for number in answer_numbers
        if number.rstrip("+") not in context_number_values
    )
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
        owner_name = metadata.get("owner_name")
        if owner_name:
            detail = f"{owner_name} - {detail}"
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
    greeting_prefix: str | None = None
    retrieval_query = question
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

        answer = IDENTITY_ANSWER

        chunks = []

    elif is_greeting(question):
        # Pure greeting: respond directly without RAG.
        retrieval_latency_ms = 0.0
        llm_latency_ms = 0.0

        retrieval = {
            "chunks": [],
            "confidence": "high",
        }

        answer = GREETING_ANSWER

        chunks = []

    else:
        retrieval_start=time.perf_counter()

        # Detect a greeting prefix (e.g. "hi, what are his skills?")
        greeting_prefix = _greeting_prefix(question)

        # Ignore the greeting for retrieval: search with the real question.
        retrieval_query = (
            _strip_leading_greeting(question) if greeting_prefix else question
        )

        retrieval = retrieve_full(
            retrieval_query,
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
            session_candidate_name = get_candidate_name(session_id) if session_id else None
            context = build_context(chunks, session_candidate_name=session_candidate_name)

            client = _get_client()

            llm_start = time.perf_counter()

            response = client.chat.completions.create(
                model=get_model_name(),
                messages=_build_messages(
                    # Strip the greeting so the model answers the profile
                    # question directly; the "Hi! " prefix is added after.
                    retrieval_query,
                    context,
                    history,
                    response_mode=response_mode,
                ),
                temperature=0.2,
                max_tokens=4096,
            )

            llm_latency_ms = (
                time.perf_counter() - llm_start
            ) * 1000

            answer = (
                response.choices[0].message.content or ""
            ).strip()

            # Detect truncated responses (LLM hit max_tokens mid-output).
            finish_reason = (
                response.choices[0].finish_reason or ""
            )
            if finish_reason in ("length", "max_tokens"):
                logger.warning(
                    "LLM response truncated (finish_reason=%s). "
                    "Answer length=%d",
                    finish_reason,
                    len(answer),
                )
                answer = _trim_truncated_answer(answer)

            validation = validate_answer_evidence(answer, chunks)
            if not validation["valid"]:
                logger.warning("Answer failed evidence validation: %s", validation)
                answer = NOT_FOUND_ANSWER
            else:
                answer = append_source_citations(answer, chunks)

            del response
            gc.collect()

    # Prepend a brief greeting when the user started with one.
    if greeting_prefix and answer and answer != NOT_FOUND_ANSWER:
        answer = f"{greeting_prefix}{answer}"

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
        "sources": extract_sources(chunks, retrieval_query) if chunks else [],
        "conversation_id": conversation_id,
        "retrieval": {
            "chunks": len(chunks),
            "latency_ms": round(retrieval_latency_ms, 1),
            "llm_latency_ms": round(llm_latency_ms, 1),
            "confidence": retrieval.get("confidence", "unknown"),
            "route": (retrieval.get("diagnostics") or {}).get("query_type"),
        },
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
