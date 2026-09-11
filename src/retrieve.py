
"""
Advanced Hybrid Retrieval for Resume RAG.

Retrieval pipeline:

    Query analysis
          ↓
    Intent detection
          ↓
    Broad candidate retrieval
          ↓
    Semantic similarity
          +
    Lexical matching
          +
    Exact phrase matching
          +
    Numeric matching
          +
    Entity matching
          +
    Section/topic matching
          +
    Metadata matching
          ↓
    Hybrid ranking
          ↓
    CrossEncoder reranking
          ↓
    Diversity / MMR selection
          ↓
    Confidence filtering
          ↓
    Final context

Supported query types:

1. Normal factual questions
2. Profile overview questions
3. Multi-entity questions
4. Entity-list questions
5. Project selection questions
6. Candidate evaluation questions
7. Contact questions
8. Temporary-session document questions

The goal is to prevent semantic retrieval from silently dropping
important resume information, especially for:

- exact facts
- numbers
- technologies
- lists
- project names
- company names
- achievements
- section-specific queries
- multi-part questions
"""

import logging
import math
import os
import re
from typing import Any, Optional

from src.embeddings import embed_query
from src.errors import RetrievalError
from src.section_parser import SECTIONS
from src.vector_store import (
    composite_text,
    get_collection,
    query_vectors,
    get_session_collection,
    get_resume_collection,
)

from src.reranker import (
    RERANKER_CANDIDATES,
    RERANKER_ENABLED,
    RERANKER_TOP_K,
    rerank,
)

logger = logging.getLogger(__name__)

RetrievedChunk = dict[str, Any]
RetrievalResult = dict[str, Any]


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_TOP_K = int(
    os.getenv("RETRIEVAL_TOP_K", "6")
)

ABSOLUTE_FLOOR = float(
    os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.15")
)

# Candidate pool should be much larger than final top-k.
# This gives the hybrid scorer and CrossEncoder more evidence to choose from.
CANDIDATE_MULTIPLIER = int(
    os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "6")
)

SMALL_COLLECTION_THRESHOLD = int(
    os.getenv("RETRIEVAL_SMALL_COLLECTION_THRESHOLD", "50")
)

# Hybrid score weights.
LEXICAL_WEIGHT = 0.30
BM25_WEIGHT = 0.25
SECTION_TOPIC_BONUS = 0.25
ENTITY_MATCH_BONUS = 0.20
EXACT_PHRASE_BONUS = 0.20
NUMERIC_MATCH_BONUS = 0.20
METADATA_MATCH_BONUS = 0.15
AI_RELEVANCE_BONUS = 0.15
SESSION_RELEVANCE_BONUS = 0.08

# Relevance thresholds.
#
# These are intentionally less aggressive than the previous version.
# The CrossEncoder needs a reasonably broad candidate pool.
MIN_BEST_SCORE = 0.15

# Final confidence thresholds.
CONFIDENT_SCORE = 0.30
CONFIDENT_SEMANTIC_ONLY = 0.35

# CrossEncoder settings.
RERANK_CANDIDATES = max(
    DEFAULT_TOP_K,
    RERANKER_CANDIDATES,
)

RERANK_TOP_K = max(
    1,
    min(DEFAULT_TOP_K, RERANKER_TOP_K),
)

# Diversity settings.
MMR_LAMBDA = float(
    os.getenv("RETRIEVAL_MMR_LAMBDA", "0.75")
)

MAX_FINAL_CHUNKS = int(
    os.getenv("RETRIEVAL_MAX_FINAL_CHUNKS", "8")
)

# Lexical retrieval deliberately has its own bounded candidate channel.  A
# vector store can return an excellent semantic neighbour while completely
# missing an exact project name, technology, or metric that happens to embed
# poorly.  We use an in-memory BM25 pass over the (small) profile collection
# as a complementary first-stage retriever, never as a replacement for dense
# retrieval.
LEXICAL_CANDIDATE_LIMIT = int(
    os.getenv("RETRIEVAL_LEXICAL_CANDIDATE_LIMIT", "24")
)

LEXICAL_SCAN_LIMIT = int(
    os.getenv("RETRIEVAL_LEXICAL_SCAN_LIMIT", "1500")
)

_TECHNICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "api": ("rest", "restful", "backend"),
    "backend": ("api", "database", "server"),
    "database": ("postgresql", "mongodb", "redis", "sql"),
    "orm": ("prisma", "object relational mapping"),
    "realtime": ("websocket", "webrtc", "livekit"),
    "real-time": ("websocket", "webrtc", "livekit"),
    "authentication": ("auth", "jwt", "rbac", "otp"),
    "auth": ("jwt", "rbac", "otp"),
    "rag": ("retrieval", "embedding", "llm", "vector"),
    "llm": ("rag", "generative ai", "embedding"),
    "ai": ("machine learning", "llm", "rag"),
    "microservices": ("restful api", "node.js", "backend"),
    "deployment": ("render", "vercel", "docker", "cloud"),
}


# ============================================================================
# Tokenization
# ============================================================================

_TOKEN_RE = re.compile(
    r"[a-z0-9+#@._%+-]+",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\+?\b"
)

_STOPWORDS = frozenset(
    """
    a an the and are about tell me my your his her him she he they them their it
    its i you we us of in on at for with to from by is was were be been being
    am do does did done has have had having will would can could should shall
    may might must this that these those what which who whom whose when where
    why how any some please give show list also than then so such very more
    most other into over under out up down
    rohan candidate person profile resume
    """.split()
)


# ============================================================================
# Contact detection
# ============================================================================

_CONTACT_TERMS = {
    "email",
    "mail",
    "gmail",
    "emailid",
    "emailaddress",
    "phone",
    "phonenumber",
    "mobile",
    "mobilenumber",
    "contact",
    "contactdetails",
    "contactinformation",
    "reach",
    "reachme",
    "connect",
    "connection",
    "touch",
    "address",
}


def is_contact_query(query: str) -> bool:
    normalized = _normalise_text(query)

    tokens = set(
        _content_tokens(normalized)
    )

    if tokens & _CONTACT_TERMS:
        return True

    contact_phrases = (
        "get in touch",
        "in touch with",
        "how can i contact",
        "how do i contact",
        "how can i reach",
        "how do i reach",
        "how can i get in touch",
        "ways to contact",
        "ways to reach",
        "contact him",
        "contact her",
        "contact rohan",
        "reach him",
        "reach her",
        "reach rohan",
    )

    return any(
        phrase in normalized
        for phrase in contact_phrases
    )


# ============================================================================
# Query vocabulary
# ============================================================================

OVERVIEW_SECTIONS = (
    "education",
    "skills",
    "experience",
    "projects",
    "achievements",
    "competitive programming",
    "hackathons",
    "positions of responsibility",
)


_PROFILE_OVERVIEW_TERMS = frozenset(
    {
        "overview",
        "summary",
        "summarize",
        "profile",
        "background",
        "introduction",
        "introduce",
        "about",
        "describe",
        "candidate",
    }
)


_PROJECT_SELECTION_TERMS = frozenset(
    {
        "best",
        "strongest",
        "most",
        "top",
        "favorite",
        "important",
        "impressive",
        "relevant",
        "advanced",
    }
)


_SECTION_SYNONYMS: dict[str, str] = {
    # Education
    "study": "education",
    "studied": "education",
    "studies": "education",
    "school": "education",
    "university": "education",
    "college": "education",
    "degree": "education",
    "cgpa": "education",
    "gpa": "education",
    "academic": "education",
    "academics": "education",
    "graduate": "education",

    # Skills
    "skill": "skills",
    "skills": "skills",
    "stack": "skills",
    "technologies": "skills",
    "technology": "skills",
    "tech": "skills",
    "tools": "skills",
    "languages": "skills",
    "frameworks": "skills",
    "database": "skills",
    "databases": "skills",
    "backend": "skills",
    "frontend": "skills",

    # Experience
    "work": "experience",
    "worked": "experience",
    "working": "experience",
    "job": "experience",
    "jobs": "experience",
    "employer": "experience",
    "employment": "experience",
    "internship": "experience",
    "internships": "experience",
    "companies": "experience",
    "company": "experience",
    "professional": "experience",
    "roles": "experience",
    "role": "experience",
    "experience": "experience",

    # Projects
    "project": "projects",
    "projects": "projects",
    "built": "projects",
    "builds": "projects",
    "application": "projects",
    "applications": "projects",
    "app": "projects",
    "apps": "projects",

    # Achievements
    "achievement": "achievements",
    "achievements": "achievements",
    "award": "achievements",
    "awards": "achievements",
    "accomplishment": "achievements",
    "accomplishments": "achievements",

    # Competitive programming
    "hackathon": "achievements",
    "hackathons": "achievements",
    "leetcode": "competitive programming",
    "codeforces": "competitive programming",
    "competitive": "competitive programming",
    "programming": "competitive programming",
    "coding": "competitive programming",
    "dsa": "competitive programming",

    # Leadership
    "leadership": "positions of responsibility",
    "coordinator": "positions of responsibility",
    "coordination": "positions of responsibility",
}


_AI_TERMS = frozenset(
    {
        "ai",
        "artificial",
        "intelligence",
        "ml",
        "machine",
        "learning",
        "rag",
        "llm",
        "llms",
        "embedding",
        "embeddings",
        "vector",
        "vectordatabase",
        "chromadb",
        "groq",
        "generative",
        "generation",
        "semantic",
        "retrieval",
        "transformer",
        "nlp",
        "openai",
        "gemini",
    }
)


_EVALUATION_TERMS = frozenset(
    {
        "hire",
        "hiring",
        "candidate",
        "fit",
        "suitable",
        "suitability",
        "recommend",
        "recommendation",
        "qualified",
        "qualification",
        "experience",
    }
)


_LIST_TERMS = frozenset(
    {
        "list",
        "all",
        "every",
        "each",
        "technologies",
        "technology",
        "skills",
        "languages",
        "databases",
        "projects",
        "achievements",
        "experience",
        "companies",
        "roles",
        "frameworks",
    }
)


# ============================================================================
# Basic helpers
# ============================================================================

def _normalise_text(text: str) -> str:
    """
    Normalize text for matching while preserving useful symbols.
    """
    text = str(text or "").lower()

    text = text.replace("’", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _normalise(token: str) -> str:
    """
    Normalize a token.

    Keeps technical tokens such as:
        C++
        C#
        Node.js
        Next.js
        PostgreSQL
        Redis
        450+
    """
    token = str(token).lower().strip(
        "'’"
    )

    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        token = token[:-1]

    return token


def _content_tokens(text: str) -> set[str]:
    """
    Return meaningful normalized tokens.
    """
    return {
        _normalise(token)
        for token in _TOKEN_RE.findall(
            str(text).lower()
        )
        if _normalise(token) not in _STOPWORDS
        and len(_normalise(token)) > 1
    }


def _extract_numbers(text: str) -> set[str]:
    """
    Extract numeric facts such as:

        450+
        100+
        8
        75+
        9.1
    """
    return {
        value.lower()
        for value in _NUMBER_RE.findall(
            str(text)
        )
    }


def _phrase_match(
    query: str,
    candidate: str,
) -> bool:
    """
    Detect meaningful multi-word phrase overlap.
    """
    query_normalized = _normalise_text(query)
    candidate_normalized = _normalise_text(candidate)

    query_tokens = [
        token
        for token in _TOKEN_RE.findall(
            query_normalized
        )
        if token not in _STOPWORDS
    ]

    if len(query_tokens) < 2:
        return False

    # Check 2-4 token ngrams.
    for n in (4, 3, 2):
        if len(query_tokens) < n:
            continue

        for i in range(
            len(query_tokens) - n + 1
        ):
            phrase = " ".join(
                query_tokens[i:i + n]
            )

            if (
                len(phrase) >= 5
                and phrase in candidate_normalized
            ):
                return True

    return False


def _metadata_text(
    metadata: dict[str, Any],
) -> str:
    """
    Flatten metadata into searchable text.
    """
    values = []

    for key, value in metadata.items():
        if value is None:
            continue

        values.append(
            f"{key} {value}"
        )

    return " ".join(values)


def _candidate_text(
    chunk: RetrievedChunk,
) -> str:
    """
    Build the complete searchable representation of a chunk.
    """
    return composite_text(
        chunk.get("text", ""),
        chunk.get("metadata", {}),
    )


# ============================================================================
# Query intent
# ============================================================================

def is_profile_overview_query(
    query: str,
) -> bool:
    tokens = _content_tokens(query)

    normalized = _normalise_text(query)
    return bool(tokens & _PROFILE_OVERVIEW_TERMS) or any(
        phrase in normalized
        for phrase in ("tell me about", "about his profile", "about rohan")
    )


def is_project_selection_query(
    query: str,
) -> bool:
    tokens = _content_tokens(query)

    project_terms = {
        "project",
        "projects",
        "built",
        "application",
        "applications",
    }

    return bool(
        tokens & project_terms
    ) and bool(
        tokens & _PROJECT_SELECTION_TERMS
    )


def is_list_query(
    query: str,
) -> bool:
    tokens = _content_tokens(query)

    query_lower = _normalise_text(query)

    if tokens & _LIST_TERMS:
        return True

    list_phrases = (
        "what are",
        "what technologies",
        "what skills",
        "which technologies",
        "which skills",
        "what databases",
        "which databases",
        "what projects",
        "which projects",
        "tell me all",
        "list all",
        "how many projects",
    )

    return any(
        phrase in query_lower
        for phrase in list_phrases
    )


# ============================================================================
# Contact retrieval
# ============================================================================

def _retrieve_contact(
    query: str,
    session_id: Optional[str] = None,
) -> RetrievalResult:
    """
    Deterministic contact retrieval.

    Contact information should never depend purely on semantic
    similarity because exact email/phone values are better handled
    through direct scanning.
    """

    collection = get_resume_collection()

    count = collection.count()

    if count == 0:
        return {
            "chunks": [],
            "confidence": "low",
            "diagnostics": {
                "query_type": "contact",
                "reason": "resume_collection_empty",
            },
        }

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = result.get(
        "documents"
    ) or []

    metadatas = result.get(
        "metadatas"
    ) or []

    email_pattern = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    phone_pattern = re.compile(
        r"(?:\+?\d[\d\s().-]{8,}\d)"
    )

    contact_keywords = {
        "email",
        "gmail",
        "mail",
        "phone",
        "mobile",
        "contact",
        "linkedin",
        "github",
    }

    contact_chunks = []

    for index, document in enumerate(
        documents
    ):
        if not document:
            continue

        text = str(document)
        lower = text.lower()

        has_email = bool(
            email_pattern.search(text)
        )

        has_phone = bool(
            phone_pattern.search(text)
        )

        has_keyword = any(
            keyword in lower
            for keyword in contact_keywords
        )

        if not (
            has_email
            or has_phone
            or has_keyword
        ):
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            and metadatas[index]
            else {}
        )
        metadata = dict(metadata)
        # Resume headers are commonly tagged ``other`` by generic parsers.
        # Preserve the original source while exposing a stable contact route.
        metadata["section"] = "contact"
        metadata["source_label"] = "Contact Information"

        priority = 0

        if has_email:
            priority += 20

        if has_phone:
            priority += 20

        if "linkedin" in lower:
            priority += 3

        if "github" in lower:
            priority += 3

        contact_chunks.append(
            {
                "text": text,
                "metadata": metadata,
                "distance": 0.0,
                "similarity": 1.0,
                "semantic_score": 1.0,
                "lexical_score": 1.0,
                "final_score": 1.0,
                "matched_keywords": sorted(
                    contact_keywords
                    & _content_tokens(text)
                ),
                "topic_bonus": 1.0,
                "entity_bonus": 0.0,
                "exact_phrase_bonus": 1.0,
                "numeric_bonus": 0.0,
                "metadata_bonus": 1.0,
                "contact_priority": priority,
                "reasons": [
                    "exact-contact-scan"
                ],
            }
        )

    contact_chunks.sort(
        key=lambda chunk: (
            chunk["contact_priority"],
            chunk["final_score"],
        ),
        reverse=True,
    )

    if not contact_chunks:
        return {
            "chunks": [],
            "confidence": "low",
            "diagnostics": {
                "query_type": "contact",
                "reason": "no_contact_information_found",
                "resume_chunks": count,
            },
        }

    return {
        "chunks": contact_chunks[:5],
        "confidence": "high",
        "diagnostics": {
            "query_type": "contact",
            "contact_chunks_found": len(
                contact_chunks
            ),
            "retrieval_method": "exact_contact_scan",
        },
    }


# ============================================================================
# Query analysis
# ============================================================================

def analyse_query(
    query: str,
) -> dict[str, Any]:
    """
    Extract retrieval signals from the query.
    """

    query = query.strip()

    tokens = _content_tokens(query)

    topics = {
        _SECTION_SYNONYMS[token]
        for token in tokens
        if token in _SECTION_SYNONYMS
    }

    query_lower = query.lower()

    # Contact intent.
    if is_contact_query(query):
        topics.add("contact")

    # Hackathon terms should strongly indicate achievements.
    hackathon_terms = {
        "hacknite",
        "hackxtract",
        "smart india hackathon",
        "orbix hackathon",
        "hackathon",
        "hackathons",
    }

    if any(
        term in query_lower
        for term in hackathon_terms
    ):
        topics.discard("experience")
        topics.add("achievements")

    numbers = _extract_numbers(query)

    entities = []

    # We will populate exact entities later when metadata is available.
    # Keeping this field here makes diagnostics consistent.
    if is_list_query(query):
        query_type = "list"
    elif is_profile_overview_query(query):
        query_type = "overview"
    elif is_evaluation_query(query):
        query_type = "evaluation"
    elif is_project_selection_query(query):
        query_type = "project_selection"
    else:
        query_type = "normal"

    return {
        "tokens": tokens,
        "topics": sorted(topics),
        "numbers": numbers,
        "entities": entities,
        "is_list": is_list_query(query),
        "raw": query,
        "query_type": query_type,
    }


def expand_query(query: str, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return conservative, deterministic expansions for technical queries.

    Expansion is intentionally limited to aliases that improve recall without
    changing the question's subject.  Entity names are not expanded here: they
    are handled by the exact entity candidate channel below.
    """
    analysis = analysis or analyse_query(query)
    tokens = set(analysis.get("tokens", set()))
    expansions: set[str] = set()
    for token in tokens:
        expansions.update(_TECHNICAL_EXPANSIONS.get(token, ()))

    # A query about a named item benefits from retaining its exact spelling in
    # the lexical channel.  This field is populated after metadata inspection.
    entities = analysis.get("entities", [])
    expansions.update(str(entity) for entity in entities)
    expansions.difference_update(tokens)

    return {
        "terms": sorted(expansions),
        "semantic_query": " ".join([query, *sorted(expansions)]).strip(),
    }


def _bm25_scores(
    query_tokens: set[str],
    documents: list[str],
) -> list[float]:
    """Small dependency-free BM25 implementation for first-stage retrieval."""
    if not query_tokens or not documents:
        return [0.0] * len(documents)

    tokenized = [list(_content_tokens(document)) for document in documents]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(len(lengths), 1)
    document_frequency = {
        token: sum(token in tokens for tokens in tokenized)
        for token in query_tokens
    }
    total_documents = len(documents)
    scores: list[float] = []
    k1, b = 1.5, 0.75
    for tokens, length in zip(tokenized, lengths):
        score = 0.0
        for token in query_tokens:
            frequency = tokens.count(token)
            if not frequency:
                continue
            idf = math.log(1 + (total_documents - document_frequency[token] + 0.5)
                           / (document_frequency[token] + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * length / max(average_length, 1.0)
            )
            score += idf * (frequency * (k1 + 1) / denominator)
        scores.append(score)
    return scores


def _lexical_candidates(
    collection: Any,
    query: str,
    analysis: dict[str, Any],
    limit: int = LEXICAL_CANDIDATE_LIMIT,
) -> list[RetrievedChunk]:
    """Fetch BM25, phrase, number, entity and section candidates.

    Chroma's dense query is deliberately kept as a separate candidate source.
    This scan is bounded for uploaded/document-heavy sessions and tags each
    returned chunk so debug output makes the retrieval channel auditable.
    """
    try:
        count = collection.count()
        if not count:
            return []
        result = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("Lexical candidate generation failed: %s", exc)
        return []

    pairs = list(zip(result.get("documents", []) or [], result.get("metadatas", []) or []))
    pairs = pairs[:LEXICAL_SCAN_LIMIT]
    composites = [composite_text(str(document or ""), dict(metadata or {}))
                  for document, metadata in pairs]
    expanded = expand_query(query, analysis)
    lexical_tokens = _content_tokens(query) | _content_tokens(" ".join(expanded["terms"]))
    scores = _bm25_scores(lexical_tokens, composites)
    query_numbers = _extract_numbers(query)
    entities = [str(entity).lower() for entity in analysis.get("entities", [])]
    sections = set(analysis.get("topics", []))

    candidates: list[RetrievedChunk] = []
    for index, ((document, metadata), bm25) in enumerate(zip(pairs, scores)):
        metadata = dict(metadata or {})
        text = str(document or "")
        searchable = composites[index].lower()
        phrase = _phrase_match(query, searchable)
        number_hit = bool(query_numbers & _extract_numbers(searchable))
        entity_hit = any(entity in searchable for entity in entities)
        section_hit = str(metadata.get("section", "")).lower() in sections
        # Small deterministic boosts make exact facts eligible even if their
        # body is short.  Hybrid scoring still decides final relevance.
        channel_score = bm25 + (2.0 if phrase else 0.0) + (2.0 if entity_hit else 0.0)
        channel_score += 1.5 if number_hit else 0.0
        channel_score += 0.6 if section_hit else 0.0
        if channel_score <= 0:
            continue
        candidates.append({
            "text": text,
            "metadata": metadata,
            "distance": 1.0,
            "similarity": 0.0,
            "lexical_retrieval_score": round(channel_score, 4),
            "candidate_channels": ["bm25"] + (["exact_phrase"] if phrase else [])
            + (["entity"] if entity_hit else []) + (["numeric"] if number_hit else [])
            + (["section"] if section_hit else []),
        })
    candidates.sort(key=lambda chunk: chunk["lexical_retrieval_score"], reverse=True)
    return candidates[:max(1, limit)]


def _merge_candidates(*candidate_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate candidate channels while retaining every retrieval reason."""
    merged: dict[tuple[str, tuple[tuple[str, str], ...]], RetrievedChunk] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            key = (candidate.get("text", ""), tuple(sorted((str(k), str(v)) for k, v in metadata.items())))
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(candidate)
                continue
            existing["similarity"] = max(float(existing.get("similarity", 0.0)), float(candidate.get("similarity", 0.0)))
            existing["distance"] = min(float(existing.get("distance", 1.0)), float(candidate.get("distance", 1.0)))
            existing["lexical_retrieval_score"] = max(float(existing.get("lexical_retrieval_score", 0.0)), float(candidate.get("lexical_retrieval_score", 0.0)))
            existing["candidate_channels"] = sorted(set(existing.get("candidate_channels", [])) | set(candidate.get("candidate_channels", [])))
    return list(merged.values())


def _debug_chunks(chunks: list[RetrievedChunk], limit: int = 24) -> list[dict[str, Any]]:
    """Return a safe, compact retrieval trace instead of serialising chunks twice."""
    trace: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks[:limit], start=1):
        trace.append({
            "rank": rank,
            "metadata": chunk.get("metadata", {}),
            "channels": chunk.get("candidate_channels", []),
            "semantic_score": chunk.get("semantic_score", chunk.get("similarity", 0.0)),
            "bm25_score": chunk.get("bm25_score", chunk.get("lexical_retrieval_score", 0.0)),
            "hybrid_score": chunk.get("final_score", 0.0),
            "reranker_score": chunk.get("reranker_score"),
            "reasons": chunk.get("reasons", []),
            "text_preview": str(chunk.get("text", "")).replace("\n", " ")[:360],
        })
    return trace


# ============================================================================
# Collection retrieval
# ============================================================================

def _optimal_candidate_count(
    top_k: int,
    collection_count: int,
    min_candidates: int = 0,
) -> int:
    """
    Dynamically determine candidate pool size.

    Small collections:
        retrieve everything.

    Larger collections:
        retrieve a broad semantic pool before hybrid ranking.
    """

    if collection_count <= 0:
        return 0

    if collection_count <= SMALL_COLLECTION_THRESHOLD:
        return collection_count

    requested = max(
        top_k * CANDIDATE_MULTIPLIER,
        top_k,
        min_candidates,
    )

    return min(
        collection_count,
        requested,
    )


def _query_collection(
    collection,
    query_embedding,
    n_results: int,
) -> list[RetrievedChunk]:
    """
    Query ChromaDB and normalize results.
    """

    count = collection.count()

    if count == 0:
        return []

    n_results = min(
        count,
        max(1, n_results),
    )

    raw = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = raw.get(
        "documents",
        [[]],
    )[0]

    metadatas = raw.get(
        "metadatas",
        [[]],
    )[0]

    distances = raw.get(
        "distances",
        [[]],
    )[0]

    return [
        {
            "text": document,
            "metadata": dict(
                metadata or {}
            ),
            "distance": distance,
            "similarity": max(
                0.0,
                1.0 - float(distance),
            ),
        }
        for document, metadata, distance
        in zip(
            documents,
            metadatas,
            distances,
        )
    ]


def _raw_to_candidates(
    raw: dict,
) -> list[RetrievedChunk]:
    """
    Convert a Chroma query response to our chunk format.
    """

    documents = raw.get(
        "documents",
        [[]],
    )[0]

    metadatas = raw.get(
        "metadatas",
        [[]],
    )[0]

    distances = raw.get(
        "distances",
        [[]],
    )[0]

    return [
        {
            "text": document,
            "metadata": dict(
                metadata or {}
            ),
            "distance": distance,
            "similarity": max(
                0.0,
                1.0 - float(distance),
            ),
        }
        for document, metadata, distance
        in zip(
            documents,
            metadatas,
            distances,
        )
    ]


# ============================================================================
# Entity extraction
# ============================================================================

def extract_query_entities(
    query: str,
) -> list[str]:
    """
    Detect exact project/company/role/event names
    from the permanent metadata.

    This is intentionally metadata-driven rather than
    relying on an LLM.
    """

    query_lower = query.lower()

    try:
        collection = get_collection()

        result = collection.get(
            include=[
                "metadatas"
            ]
        )

        entities: set[str] = set()

        for metadata in result.get(
            "metadatas",
            [],
        ):
            if not metadata:
                continue

            for key in (
                "project",
                "company",
                "role",
                "event",
            ):
                value = metadata.get(key)

                if not value:
                    continue

                value_str = str(value)

                if (
                    value_str.lower()
                    in query_lower
                ):
                    entities.add(
                        value_str
                    )

        return sorted(
            entities,
            key=len,
            reverse=True,
        )

    except Exception as exc:
        logger.warning(
            "Entity extraction failed: %s",
            exc,
        )

        return []


# ============================================================================
# Hybrid scoring
# ============================================================================

def _score_candidates(
    candidates: list[RetrievedChunk],
    analysis: dict[str, Any],
) -> list[RetrievedChunk]:
    """
    Advanced hybrid scoring.

    Score components:

        semantic similarity
        lexical coverage
        exact phrase match
        numeric match
        section/topic match
        metadata match
        entity match
        AI relevance
        session relevance
    """

    tokens = analysis["tokens"]
    topics = set(
        analysis["topics"]
    )
    raw_query = analysis["raw"]
    query_numbers = analysis.get(
        "numbers",
        set(),
    )

    total_tokens = max(
        len(tokens),
        1,
    )

    query_lower = raw_query.lower()
    max_bm25 = max(
        (float(chunk.get("lexical_retrieval_score", 0.0)) for chunk in candidates),
        default=0.0,
    )

    for chunk in candidates:

        metadata = chunk.get(
            "metadata",
            {},
        )

        text = chunk.get(
            "text",
            "",
        )

        composite = _candidate_text(
            chunk
        )

        composite_lower = (
            composite.lower()
        )

        candidate_tokens = _content_tokens(
            composite
        )

        candidate_numbers = _extract_numbers(
            composite
        )

        # ------------------------------------------------------------
        # Semantic score
        # ------------------------------------------------------------

        semantic = float(
            chunk.get(
                "similarity",
                0.0,
            )
        )

        # ------------------------------------------------------------
        # Lexical score
        # ------------------------------------------------------------

        matched = (
            tokens
            & candidate_tokens
        )

        lexical_coverage = (
            len(matched)
            / total_tokens
        )
        bm25_score = float(chunk.get("lexical_retrieval_score", 0.0))
        bm25_normalized = bm25_score / max_bm25 if max_bm25 else 0.0

        # ------------------------------------------------------------
        # Topic / section match
        # ------------------------------------------------------------

        section = str(
            metadata.get(
                "section",
                "",
            )
        ).lower()

        topic_hit = (
            section in topics
        )

        # ------------------------------------------------------------
        # Entity match
        # ------------------------------------------------------------

        entity_names = [
            str(metadata[key])
            for key in (
                "project",
                "company",
                "role",
                "event",
            )
            if metadata.get(key)
        ]

        entity_hits = [
            name
            for name in entity_names
            if name.lower()
            in query_lower
        ]

        entity_hit = bool(
            entity_hits
        )

        # ------------------------------------------------------------
        # Exact phrase match
        # ------------------------------------------------------------

        exact_phrase_hit = _phrase_match(
            raw_query,
            composite,
        )

        # ------------------------------------------------------------
        # Numeric match
        # ------------------------------------------------------------

        matched_numbers = (
            query_numbers
            & candidate_numbers
        )

        numeric_hit = bool(
            query_numbers
            and matched_numbers
        )

        # ------------------------------------------------------------
        # Metadata match
        # ------------------------------------------------------------

        metadata_string = (
            _metadata_text(
                metadata
            ).lower()
        )

        metadata_tokens = _content_tokens(
            metadata_string
        )

        metadata_matches = (
            tokens
            & metadata_tokens
        )

        metadata_hit = bool(
            metadata_matches
        )

        # ------------------------------------------------------------
        # AI relevance
        # ------------------------------------------------------------

        ai_query = bool(
            tokens & _AI_TERMS
        )

        ai_matches = (
            candidate_tokens
            & _AI_TERMS
        )

        ai_hit = (
            ai_query
            and bool(ai_matches)
        )

        # ------------------------------------------------------------
        # Session relevance
        # ------------------------------------------------------------

        session_hit = bool(
            metadata.get(
                "session_id"
            )
        )

        # ------------------------------------------------------------
        # List query handling
        # ------------------------------------------------------------

        list_bonus = 0.0

        if analysis.get(
            "is_list"
        ):
            if topic_hit:
                list_bonus += 0.10

            if (
                metadata.get("project")
                or metadata.get("company")
                or metadata.get("role")
                or metadata.get("event")
            ):
                list_bonus += 0.05

        # ------------------------------------------------------------
        # Calculate hybrid score
        # ------------------------------------------------------------

        final_score = (
            semantic
            + LEXICAL_WEIGHT
            * lexical_coverage
            + BM25_WEIGHT
            * bm25_normalized
            + (
                SECTION_TOPIC_BONUS
                if topic_hit
                else 0.0
            )
            + (
                ENTITY_MATCH_BONUS
                if entity_hit
                else 0.0
            )
            + (
                EXACT_PHRASE_BONUS
                if exact_phrase_hit
                else 0.0
            )
            + (
                NUMERIC_MATCH_BONUS
                if numeric_hit
                else 0.0
            )
            + (
                METADATA_MATCH_BONUS
                if metadata_hit
                else 0.0
            )
            + (
                AI_RELEVANCE_BONUS
                if ai_hit
                else 0.0
            )
            + (
                SESSION_RELEVANCE_BONUS
                if session_hit
                else 0.0
            )
            + list_bonus
        )

        # ------------------------------------------------------------
        # Explainability
        # ------------------------------------------------------------

        reasons: list[str] = []

        if matched:
            reasons.append(
                "keywords:"
                + ",".join(
                    sorted(
                        matched
                    )[:12]
                )
            )

        if bm25_score:
            reasons.append(f"bm25:{bm25_score:.2f}")

        if topic_hit:
            reasons.append(
                f"section:{metadata.get('section')}"
            )

        if entity_hit:
            reasons.append(
                "entity:"
                + entity_hits[0]
            )

        if exact_phrase_hit:
            reasons.append(
                "exact-phrase"
            )

        if numeric_hit:
            reasons.append(
                "numbers:"
                + ",".join(
                    sorted(
                        matched_numbers
                    )
                )
            )

        if metadata_hit:
            reasons.append(
                "metadata-match"
            )

        if ai_hit:
            reasons.append(
                "ai:"
                + ",".join(
                    sorted(
                        ai_matches
                    )[:8]
                )
            )

        if session_hit:
            reasons.append(
                "temporary-session"
            )

        if list_bonus:
            reasons.append(
                "list-query"
            )

        if not reasons:
            reasons.append(
                "semantic-only"
            )

        chunk.update(
            {
                "semantic_score": round(
                    semantic,
                    4,
                ),
                "lexical_score": round(
                    lexical_coverage,
                    4,
                ),
                "bm25_score": round(bm25_score, 4),
                "bm25_normalized": round(bm25_normalized, 4),
                "matched_keywords": sorted(
                    matched
                ),
                "topic_bonus": (
                    SECTION_TOPIC_BONUS
                    if topic_hit
                    else 0.0
                ),
                "entity_bonus": (
                    ENTITY_MATCH_BONUS
                    if entity_hit
                    else 0.0
                ),
                "exact_phrase_bonus": (
                    EXACT_PHRASE_BONUS
                    if exact_phrase_hit
                    else 0.0
                ),
                "numeric_bonus": (
                    NUMERIC_MATCH_BONUS
                    if numeric_hit
                    else 0.0
                ),
                "metadata_bonus": (
                    METADATA_MATCH_BONUS
                    if metadata_hit
                    else 0.0
                ),
                "ai_bonus": (
                    AI_RELEVANCE_BONUS
                    if ai_hit
                    else 0.0
                ),
                "session_bonus": (
                    SESSION_RELEVANCE_BONUS
                    if session_hit
                    else 0.0
                ),
                "matched_numbers": sorted(
                    matched_numbers
                ),
                "entity_hits": entity_hits,
                "metadata_matches": sorted(
                    metadata_matches
                ),
                "final_score": round(
                    final_score,
                    4,
                ),
                "reasons": reasons,
            }
        )

    candidates.sort(
        key=lambda chunk: (
            chunk.get(
                "final_score",
                0.0,
            ),
            chunk.get(
                "semantic_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    return candidates


# ============================================================================
# Relevance filtering
# ============================================================================

def _filter_relevant(
    scored: list[RetrievedChunk],
    min_similarity: float,
) -> list[RetrievedChunk]:
    """
    Soft relevance filtering.

    IMPORTANT:
    We do not aggressively remove candidates based only on hybrid
    score. CrossEncoder reranking needs a broad candidate pool.

    A candidate survives if:

        semantic similarity is above the floor

    OR

        it contains strong lexical/entity/numeric evidence.
    """

    if not scored:
        return []

    best = scored[0].get(
        "final_score",
        0.0,
    )

    if best < MIN_BEST_SCORE:
        return []

    filtered = []

    for chunk in scored:

        semantic = float(
            chunk.get(
                "semantic_score",
                0.0,
            )
        )

        final_score = float(
            chunk.get(
                "final_score",
                0.0,
            )
        )

        strong_evidence = bool(
            chunk.get(
                "matched_keywords"
            )
            or chunk.get(
                "entity_bonus"
            )
            or chunk.get(
                "numeric_bonus"
            )
            or chunk.get(
                "exact_phrase_bonus"
            )
        )

        if (
            semantic >= min_similarity
            or strong_evidence
            or final_score
            >= best - 0.30
        ):
            filtered.append(
                chunk
            )

    return filtered


# ============================================================================
# Confidence
# ============================================================================

def _confidence(
    best_chunk: RetrievedChunk | None,
) -> str:
    """
    Estimate retrieval confidence.

    Confidence is based on both semantic evidence and explicit
    retrieval signals.
    """

    if not best_chunk:
        return "none"

    final_score = float(
        best_chunk.get(
            "final_score",
            0.0,
        )
    )

    semantic_score = float(
        best_chunk.get(
            "semantic_score",
            0.0,
        )
    )

    lexical_coverage = float(best_chunk.get("lexical_score", 0.0))
    strong_evidence = bool(
        lexical_coverage >= 0.5
        or best_chunk.get(
            "entity_bonus"
        )
        or best_chunk.get(
            "numeric_bonus"
        )
        or best_chunk.get(
            "exact_phrase_bonus"
        )
    )

    if final_score < CONFIDENT_SCORE:
        return "low"

    if (
        strong_evidence
        or semantic_score
        >= CONFIDENT_SEMANTIC_ONLY
    ):
        return "high"

    return "low"


# ============================================================================
# CrossEncoder + diversity
# ============================================================================

def _token_similarity(
    a: str,
    b: str,
) -> float:
    """
    Lightweight lexical similarity used by MMR.

    This prevents multiple final chunks from being almost identical.
    """

    tokens_a = _content_tokens(a)
    tokens_b = _content_tokens(b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(
        tokens_a & tokens_b
    )

    union = len(
        tokens_a | tokens_b
    )

    if union == 0:
        return 0.0

    return intersection / union


def _apply_diversity(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Apply a lightweight MMR-style selection.

    Relevance remains dominant, but highly redundant chunks
    are penalized.

    This is particularly useful for:

        profile overview
        candidate evaluation
        skills
        achievements
        project questions
    """

    if len(chunks) <= top_k:
        return chunks[:top_k]

    selected: list[RetrievedChunk] = []

    remaining = list(
        chunks
    )
    reranker_values = [float(chunk.get("reranker_score", chunk.get("final_score", 0.0))) for chunk in chunks]
    hybrid_values = [float(chunk.get("final_score", 0.0)) for chunk in chunks]
    reranker_low, reranker_high = min(reranker_values), max(reranker_values)
    hybrid_low, hybrid_high = min(hybrid_values), max(hybrid_values)

    def normalise_score(value: float, low: float, high: float) -> float:
        return (value - low) / (high - low) if high > low else 1.0

    while (
        remaining
        and len(selected) < top_k
    ):
        best_chunk = None
        best_mmr = float("-inf")

        for chunk in remaining:

            reranker_score = float(
                chunk.get(
                    "reranker_score",
                    0.0,
                )
            )

            hybrid_score = float(
                chunk.get(
                    "final_score",
                    0.0,
                )
            )

            # Cross-encoder logits and hybrid scores use different scales.
            # Rank-normalising each candidate set avoids a negative logit
            # overwhelming explicit entity/section/numeric evidence.
            relevance = (
                0.60 * normalise_score(reranker_score, reranker_low, reranker_high)
                + 0.40 * normalise_score(hybrid_score, hybrid_low, hybrid_high)
            )

            redundancy = 0.0

            if selected:
                redundancy = max(
                    _token_similarity(
                        chunk.get(
                            "text",
                            "",
                        ),
                        previous.get(
                            "text",
                            "",
                        ),
                    )
                    for previous
                    in selected
                )

            mmr = (
                MMR_LAMBDA
                * relevance
                - (
                    1.0
                    - MMR_LAMBDA
                )
                * redundancy
            )

            if mmr > best_mmr:
                best_mmr = mmr
                best_chunk = chunk

        if best_chunk is None:
            break

        selected.append(
            best_chunk
        )

        remaining.remove(
            best_chunk
        )

    return selected


def _rerank_candidates(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    CrossEncoder reranking followed by diversity selection.
    """

    if not candidates:
        return []

    candidate_pool = candidates[
        :max(
            RERANK_CANDIDATES,
            top_k,
        )
    ]

    if (
        RERANKER_ENABLED
        and candidate_pool
    ):
        try:
            reranked = rerank(
                query=query,
                chunks=candidate_pool,
                # Keep the whole reranked pool for MMR.  Truncating to final
                # K first makes diversity selection a no-op and can preserve
                # a semantically generic top result over exact evidence.
                top_k=len(candidate_pool),
            )
        except Exception:
            logger.exception(
                "CrossEncoder reranking failed; "
                "falling back to hybrid ranking"
            )

            reranked = candidate_pool

    else:
        reranked = candidate_pool

    final = _apply_diversity(
        query=query,
        chunks=reranked,
        top_k=top_k,
    )

    return final


# ============================================================================
# Normal retrieval
# ============================================================================

def _retrieve_combined(
    query: str,
    session_id: str | None,
    top_k: int,
    analysis: dict[str, Any],
    min_similarity: float,
) -> tuple[
    list[RetrievedChunk],
    str,
]:
    """
    Retrieve from permanent Resume/LinkedIn data and temporary
    session documents.
    """

    logger.info(
        "Starting advanced retrieval. "
        "query_length=%d session_id=%s",
        len(query),
        session_id,
    )

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    try:
        expansion = expand_query(query, analysis)
        query_embedding = embed_query(expansion["semantic_query"])
    except Exception:
        logger.exception(
            "QUERY EMBEDDING FAILED"
        )
        raise

    # ------------------------------------------------------------------
    # Candidate count
    # ------------------------------------------------------------------

    candidate_count = max(
        top_k * CANDIDATE_MULTIPLIER,
        top_k,
        RERANK_CANDIDATES,
    )

    # ------------------------------------------------------------------
    # Permanent collection
    # ------------------------------------------------------------------

    try:
        resume_collection = (
            get_resume_collection()
        )

        resume_count = (
            resume_collection.count()
        )

        dense_resume_candidates = (
            _query_collection(
                resume_collection,
                query_embedding,
                candidate_count,
            )
        )
        for candidate in dense_resume_candidates:
            candidate["candidate_channels"] = ["dense"]
        lexical_resume_candidates = _lexical_candidates(
            resume_collection,
            query,
            analysis,
        )
        resume_candidates = _merge_candidates(
            dense_resume_candidates,
            lexical_resume_candidates,
        )

        logger.info(
            "Resume candidates=%d "
            "collection_size=%d",
            len(resume_candidates),
            resume_count,
        )

    except Exception:
        logger.exception(
            "PERMANENT RESUME RETRIEVAL FAILED"
        )
        raise

    # ------------------------------------------------------------------
    # Session collection
    # ------------------------------------------------------------------

    session_candidates: list[RetrievedChunk] = []

    if session_id:

        try:
            session_collection = (
                get_session_collection(
                    session_id
                )
            )

            session_count = (
                session_collection.count()
            )

            if session_count > 0:
                dense_session_candidates = (
                    _query_collection(
                        session_collection,
                        query_embedding,
                        candidate_count,
                    )
                )
                for candidate in dense_session_candidates:
                    candidate["candidate_channels"] = ["dense"]
                session_candidates = _merge_candidates(
                    dense_session_candidates,
                    _lexical_candidates(
                        session_collection,
                        query,
                        analysis,
                    ),
                )

            logger.info(
                "Session candidates=%d "
                "collection_size=%d",
                len(session_candidates),
                session_count,
            )

        except Exception:
            logger.exception(
                "SESSION RETRIEVAL FAILED"
            )

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    candidates = (
        resume_candidates
        + session_candidates
    )

    if not candidates:
        return [], "none"

    # ------------------------------------------------------------------
    # Hybrid scoring
    # ------------------------------------------------------------------

    scored = _score_candidates(
        candidates,
        analysis,
    )
    analysis["candidate_trace"] = _debug_chunks(scored)

    # ------------------------------------------------------------------
    # Soft relevance filter
    # ------------------------------------------------------------------

    relevant = _filter_relevant(
        scored,
        min_similarity,
    )

    if not relevant:
        return [], "none"

    # ------------------------------------------------------------------
    # CrossEncoder reranking
    # ------------------------------------------------------------------

    final_chunks = _rerank_candidates(
        query=query,
        candidates=relevant,
        top_k=min(
            top_k,
            MAX_FINAL_CHUNKS,
        ),
    )
    analysis["final_trace"] = _debug_chunks(final_chunks, limit=MAX_FINAL_CHUNKS)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    confidence = _confidence(
        final_chunks[0]
        if final_chunks
        else None
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        logger.info(
            "FINAL #%d | section=%s | "
            "project=%s | semantic=%.3f | "
            "hybrid=%.3f | reranker=%.3f | "
            "numbers=%s | reasons=%s",
            index,
            metadata.get(
                "section"
            ),
            metadata.get(
                "project"
            ),
            chunk.get(
                "semantic_score",
                0.0,
            ),
            chunk.get(
                "final_score",
                0.0,
            ),
            chunk.get(
                "reranker_score",
                0.0,
            ),
            chunk.get(
                "matched_numbers",
                [],
            ),
            chunk.get(
                "reasons",
                [],
            ),
        )

    return (
        final_chunks,
        confidence,
    )


# ============================================================================
# Evaluation query retrieval
# ============================================================================

def is_evaluation_query(
    query: str,
) -> bool:
    """
    Detect candidate evaluation questions.
    """

    tokens = _content_tokens(
        query
    )

    query_lower = query.lower()

    direct_patterns = (
        "should we hire",
        "should i hire",
        "would you hire",
        "good candidate",
        "good fit",
        "strong candidate",
        "suitable candidate",
        "fit for",
        "suitable for",
        "qualified for",
        "recommend hiring",
        "why should we hire",
        "does he have experience",
    )

    return (
        bool(
            tokens
            & _EVALUATION_TERMS
        )
        or any(
            pattern in query_lower
            for pattern
            in direct_patterns
        )
    )


def _retrieve_evaluation(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Retrieve diverse evidence for candidate evaluation.
    """

    collection = get_collection()

    count = collection.count()

    if count == 0:
        return []

    analysis = analyse_query(query)
    analysis["entities"] = extract_query_entities(query)
    query_embedding = embed_query(expand_query(query, analysis)["semantic_query"])

    raw = query_vectors(
        query_embedding,
        n_results=_optimal_candidate_count(
            top_k=max(
                top_k,
                6,
            ),
            collection_count=count,
            min_candidates=30,
        ),
    )

    dense_candidates = _raw_to_candidates(raw)
    for candidate in dense_candidates:
        candidate["candidate_channels"] = ["dense"]
    candidates = _merge_candidates(
        dense_candidates,
        _lexical_candidates(collection, query, analysis),
    )

    scored = _score_candidates(
        candidates,
        analysis,
    )

    preferred_sections = {
        "skills",
        "experience",
        "projects",
        "achievements",
        "education",
        "positions of responsibility",
    }

    preferred = [
        chunk
        for chunk in scored
        if str(
            chunk["metadata"].get(
                "section",
                "",
            )
        ).lower()
        in preferred_sections
    ]

    others = [
        chunk
        for chunk in scored
        if chunk not in preferred
    ]

    selected_pool = (
        preferred
        + others
    )

    selected = _rerank_candidates(
        query=query,
        candidates=selected_pool,
        top_k=top_k,
    )

    return selected


# ============================================================================
# Overview retrieval
# ============================================================================

def _retrieve_overview(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Retrieve representative evidence across major sections.

    Unlike plain semantic search, this intentionally preserves
    section diversity.
    """

    collection = get_collection()

    count = collection.count()

    if count == 0:
        return []

    analysis = analyse_query(query)
    analysis["entities"] = extract_query_entities(query)
    query_embedding = embed_query(expand_query(query, analysis)["semantic_query"])

    candidate_count = (
        _optimal_candidate_count(
            top_k=max(
                top_k,
                8,
            ),
            collection_count=count,
            min_candidates=30,
        )
    )

    raw = query_vectors(
        query_embedding,
        n_results=candidate_count,
    )

    dense_candidates = _raw_to_candidates(raw)
    for candidate in dense_candidates:
        candidate["candidate_channels"] = ["dense"]
    candidates = _merge_candidates(
        dense_candidates,
        _lexical_candidates(collection, query, analysis),
    )

    scored = _score_candidates(
        candidates,
        analysis,
    )

    # Prefer one representative chunk from each major section.
    selected: list[
        RetrievedChunk
    ] = []

    sections_seen: set[
        str
    ] = set()

    for chunk in scored:

        section = str(
            chunk["metadata"].get(
                "section",
                "",
            )
        ).lower()

        if (
            section in OVERVIEW_SECTIONS
            and section not in sections_seen
        ):
            selected.append(
                chunk
            )

            sections_seen.add(
                section
            )

    # Fill remaining candidates.
    for chunk in scored:

        if len(selected) >= max(
            top_k * 2,
            8,
        ):
            break

        if chunk not in selected:
            selected.append(
                chunk
            )

    return _rerank_candidates(
        query=query,
        candidates=selected,
        top_k=min(
            top_k,
            MAX_FINAL_CHUNKS,
        ),
    )


# ============================================================================
# Entity retrieval
# ============================================================================

def _retrieve_entity(
    entity: str,
    top_k: int = 4,
    query: str | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks belonging to an exact entity.

    First attempt:
        metadata exact match

    Fallback:
        semantic retrieval + hybrid scoring
    """

    collection = get_collection()

    count = collection.count()

    if count == 0:
        return []

    entity_lower = (
        entity.lower().strip()
    )

    # ---------------------------------------------------------------
    # Exact metadata retrieval
    # ---------------------------------------------------------------

    exact_chunks: list[
        RetrievedChunk
    ] = []

    try:
        result = collection.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        for index, document in enumerate(
            documents
        ):

            if not document:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            matched = any(
                metadata.get(key)
                and str(
                    metadata[key]
                ).lower().strip()
                == entity_lower
                for key in (
                    "project",
                    "company",
                    "role",
                    "event",
                )
            )

            if not matched:
                continue

            exact_chunks.append(
                {
                    "text": document,
                    "metadata": dict(
                        metadata or {}
                    ),
                    "distance": 0.0,
                    "similarity": 1.0,
                    "semantic_score": 1.0,
                    "final_score": 1.5,
                    "entity_bonus": ENTITY_MATCH_BONUS,
                    "exact_phrase_bonus": EXACT_PHRASE_BONUS,
                    "matched_keywords": [],
                    "reasons": [
                        f"exact-entity:{entity}"
                    ],
                }
            )

    except Exception:
        logger.exception(
            "Exact entity retrieval failed for %s",
            entity,
        )

    if exact_chunks:
        entity_analysis = analyse_query(query or entity)
        entity_analysis["entities"] = [entity]
        exact_chunks = _score_candidates(exact_chunks, entity_analysis)
        return _rerank_candidates(
            query=query or entity,
            candidates=exact_chunks,
            top_k=top_k,
        )

    # ---------------------------------------------------------------
    # Semantic fallback
    # ---------------------------------------------------------------

    raw = query_vectors(
        embed_query(entity),
        n_results=min(
            count,
            max(
                top_k * 5,
                15,
            ),
        ),
    )

    candidates = _raw_to_candidates(
        raw
    )

    analysis = analyse_query(query or entity)
    analysis["entities"] = [entity]

    scored = _score_candidates(
        candidates,
        analysis,
    )

    entity_results = []

    for chunk in scored:

        metadata = chunk[
            "metadata"
        ]

        matches = any(
            metadata.get(key)
            and str(
                metadata[key]
            ).lower()
            == entity_lower
            for key in (
                "project",
                "company",
                "role",
                "event",
            )
        )

        if matches:
            entity_results.append(
                chunk
            )

    if entity_results:
        return _rerank_candidates(
            query=query or entity,
            candidates=entity_results,
            top_k=top_k,
        )

    return scored[:top_k]


# ============================================================================
# Section/entity-list retrieval
# ============================================================================

def _retrieve_all_section_entities(
    section: str,
) -> list[RetrievedChunk]:
    """
    Retrieve all chunks belonging to a section.

    Used for list-oriented questions where semantic search
    can easily omit valid entries.
    """

    try:
        collection = get_collection()

        result = collection.get(
            where={
                "section": section
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        chunks = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):

            chunks.append(
                {
                    "text": document,
                    "metadata": dict(
                        metadata or {}
                    ),
                    "similarity": 1.0,
                    "distance": 0.0,
                    "final_score": 1.0,
                    "semantic_score": 1.0,
                    "lexical_score": 0.0,
                    "topic_bonus": SECTION_TOPIC_BONUS,
                    "entity_bonus": 0.0,
                    "matched_keywords": [],
                    "reasons": [
                        f"section:{section}"
                    ],
                }
            )

        return chunks

    except Exception as exc:

        raise RetrievalError(
            f"Failed to retrieve section "
            f"'{section}': {exc}"
        ) from exc


def _detect_list_section(
    query: str,
    analysis: dict[str, Any],
) -> Optional[str]:
    """
    Detect whether a list query maps strongly to one section.
    """

    topics = set(
        analysis.get(
            "topics",
            [],
        )
    )

    query_lower = query.lower()

    # Explicit mappings first.
    if (
        "database" in query_lower
        or "databases" in query_lower
        or "language" in query_lower
        or "languages" in query_lower
        or "technology" in query_lower
        or "technologies" in query_lower
        or "skills" in query_lower
    ):
        return "skills"

    if (
        "leetcode" in query_lower
        or "dsa" in query_lower
        or "competitive" in query_lower
        or "achievement" in query_lower
        or "achievements" in query_lower
        or "hackathon" in query_lower
    ):
        return "achievements"

    if (
        "projects" in query_lower
        or "projects" in topics
    ):
        return "projects"

    if (
        "experience" in query_lower
        or "work" in query_lower
        or "companies" in query_lower
        or "roles" in query_lower
    ):
        return "experience"

    if "education" in topics:
        return "education"

    return None


def _retrieve_list(
    query: str,
    top_k: int,
    analysis: dict[str, Any],
) -> list[RetrievedChunk]:
    """
    Retrieve list-oriented questions using section-aware
    retrieval rather than only top semantic matches.
    """

    section = _detect_list_section(
        query,
        analysis,
    )

    if section:

        section_chunks = (
            _retrieve_all_section_entities(
                section
            )
        )

        if section_chunks:

            scored = _score_candidates(
                section_chunks,
                analysis,
            )

            # Rerank only when the list is reasonably small.
            # For very broad lists, preserving coverage is more
            # important than aggressively cutting candidates.
            if len(scored) <= 20:

                return _rerank_candidates(
                    query=query,
                    candidates=scored,
                    top_k=min(
                        max(
                            top_k,
                            8,
                        ),
                        MAX_FINAL_CHUNKS,
                    ),
                )

            return scored[
                :min(
                    max(
                        top_k,
                        8,
                    ),
                    MAX_FINAL_CHUNKS,
                )
            ]

    # Generic list fallback.
    return _retrieve_combined(
        query=query,
        session_id=None,
        top_k=top_k,
        analysis=analysis,
        min_similarity=ABSOLUTE_FLOOR,
    )[0]


# ============================================================================
# Project retrieval
# ============================================================================

def _retrieve_project_candidates(
    query: str,
    session_id: str | None,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Retrieve project-related evidence from permanent and
    temporary collections.
    """

    logger.info(
        "Starting project retrieval. "
        "session_id=%s",
        session_id,
    )

    analysis = analyse_query(query)
    analysis["entities"] = extract_query_entities(query) if not session_id else []
    query_embedding = embed_query(expand_query(query, analysis)["semantic_query"])

    collections = [
        (
            "resume",
            get_resume_collection(),
        )
    ]

    if session_id:

        try:
            collections.append(
                (
                    "session",
                    get_session_collection(
                        session_id
                    ),
                )
            )

        except Exception:
            logger.exception(
                "Failed to access session "
                "collection for project retrieval"
            )

    all_candidates: list[
        RetrievedChunk
    ] = []

    for source_name, collection in (
        collections
    ):

        try:

            count = collection.count()

            if count == 0:
                continue

            candidate_count = (
                _optimal_candidate_count(
                    top_k=max(
                        top_k,
                        8,
                    ),
                    collection_count=count,
                    min_candidates=20,
                )
            )

            dense_candidates = (
                _query_collection(
                    collection,
                    query_embedding,
                    candidate_count,
                )
            )
            for candidate in dense_candidates:
                candidate[
                    "_retrieval_source"
                ] = source_name
                candidate["candidate_channels"] = ["dense"]

            candidates = _merge_candidates(
                dense_candidates,
                _lexical_candidates(collection, query, analysis),
            )
            for candidate in candidates:
                candidate["_retrieval_source"] = source_name
            all_candidates.extend(candidates)

        except Exception:

            logger.exception(
                "Project retrieval failed "
                "for source=%s",
                source_name,
            )

            if source_name == "session":
                continue

            raise

    if not all_candidates:
        return []

    scored = _score_candidates(
        all_candidates,
        analysis,
    )

    return _rerank_candidates(
        query=query,
        candidates=scored,
        top_k=min(
            max(
                top_k,
                8,
            ),
            MAX_FINAL_CHUNKS,
        ),
    )


# ============================================================================
# Main retrieval router
# ============================================================================

def retrieve_full(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = ABSOLUTE_FLOOR,
    session_id: str | None = None,
    debug: bool = False,
) -> RetrievalResult:
    """
    Main retrieval entry point.
    """

    if not query or not query.strip():
        raise RetrievalError(
            "Cannot retrieve with an empty query."
        )

    query = query.strip()

    analysis = analyse_query(
        query
    )

    analysis["raw"] = query

    # ---------------------------------------------------------------
    # 1. Contact
    # ---------------------------------------------------------------

    if is_contact_query(
        query
    ):
        return _retrieve_contact(
            query,
            session_id=session_id,
        )

    # ---------------------------------------------------------------
    # 2. Multi-entity
    # ---------------------------------------------------------------

    entities = []

    if not session_id:
        entities = (
            extract_query_entities(
                query
            )
        )

    analysis["entities"] = entities

    if len(entities) > 1:

        all_chunks: list[
            RetrievedChunk
        ] = []

        for entity in entities:

            all_chunks.extend(
                _retrieve_entity(
                    entity,
                    top_k=4,
                    query=query,
                )
            )

        unique: dict[
            tuple,
            RetrievedChunk,
        ] = {}

        for chunk in all_chunks:

            metadata = chunk[
                "metadata"
            ]

            key = (
                metadata.get(
                    "project"
                ),
                metadata.get(
                    "company"
                ),
                metadata.get(
                    "role"
                ),
                metadata.get(
                    "event"
                ),
                chunk.get(
                    "text",
                    "",
                ),
            )

            unique[key] = chunk

        chunks = list(
            unique.values()
        )

        # Rerank all entity evidence together.
        chunks = _rerank_candidates(
            query=query,
            candidates=chunks,
            top_k=min(
                max(
                    top_k,
                    len(entities) * 2,
                ),
                MAX_FINAL_CHUNKS,
            ),
        )

        return {
            "chunks": chunks,
            "confidence": (
                "high"
                if chunks
                else "none"
            ),
            "diagnostics": {
                "query_type": "multi_entity",
                "entities": entities,
                "candidates_considered": len(
                    all_chunks
                ),
            },
        }

    # A named project/company/event is more specific than generic list intent
    # (for example, "technologies used in JugaadHub").  Route it to exact
    # entity evidence before the broad skills/list router can dilute context.
    if len(entities) == 1 and not session_id:
        chunks = _retrieve_entity(
            entities[0],
            top_k=max(top_k, 4),
            query=query,
        )
        return {
            "chunks": chunks,
            "confidence": "high" if chunks else "none",
            "diagnostics": {
                "query_type": "exact_entity",
                "entities": entities,
                "topics": analysis["topics"],
                "query_tokens": sorted(analysis["tokens"]),
                "debug": {
                    "intent": "exact_entity",
                    "entities": entities,
                    "routed_sections": analysis["topics"],
                    "final_context": _debug_chunks(chunks),
                } if debug else None,
            },
        }

    # ---------------------------------------------------------------
    # 3. Profile overview
    # ---------------------------------------------------------------

    if (
        is_profile_overview_query(
            query
        )
        and not session_id
    ):

        chunks = _retrieve_overview(
            query,
            top_k=top_k,
        )

        return {
            "chunks": chunks,
            "confidence": (
                "high"
                if chunks
                else "none"
            ),
            "diagnostics": {
                "query_type":
                    "profile_overview",
                "query_tokens":
                    sorted(
                        analysis[
                            "tokens"
                        ]
                    ),
                "topics":
                    analysis[
                        "topics"
                    ],
                "candidates_considered":
                    len(chunks),
            },
        }

    # ---------------------------------------------------------------
    # 4. Candidate evaluation
    # ---------------------------------------------------------------

    if (
        is_evaluation_query(
            query
        )
        and not session_id
    ):

        chunks = _retrieve_evaluation(
            query,
            top_k=max(
                top_k,
                6,
            ),
        )

        return {
            "chunks": chunks,
            "confidence": (
                "high"
                if chunks
                else "none"
            ),
            "diagnostics": {
                "query_type":
                    "evaluation",
                "query_tokens":
                    sorted(
                        analysis[
                            "tokens"
                        ]
                    ),
                "topics":
                    analysis[
                        "topics"
                    ],
                "candidates_considered":
                    len(chunks),
            },
        }

    # ---------------------------------------------------------------
    # 5. Explicit list query
    # ---------------------------------------------------------------

    if (
        is_list_query(
            query
        )
        and not session_id
    ):

        chunks = _retrieve_list(
            query=query,
            top_k=top_k,
            analysis=analysis,
        )

        if chunks:

            return {
                "chunks": chunks,
                "confidence": "high",
                "diagnostics": {
                    "query_type":
                        "entity_list",
                    "section":
                        _detect_list_section(
                            query,
                            analysis,
                        ),
                    "query_tokens":
                        sorted(
                            analysis[
                                "tokens"
                            ]
                        ),
                    "topics":
                        analysis[
                            "topics"
                        ],
                    "candidates_considered":
                        len(chunks),
                },
            }

    # ---------------------------------------------------------------
    # 6. Project queries
    # ---------------------------------------------------------------

    if (
        (
            "projects"
            in analysis["topics"]
            or is_project_selection_query(
                query
            )
        )
        and not entities
    ):

        project_chunks = (
            _retrieve_project_candidates(
                query=query,
                session_id=session_id,
                top_k=max(
                    top_k,
                    8,
                ),
            )
        )

        if project_chunks:

            return {
                "chunks":
                    project_chunks,
                "confidence":
                    "high",
                "diagnostics": {
                    "query_type":
                        "project_list",
                    "query_tokens":
                        sorted(
                            analysis[
                                "tokens"
                            ]
                        ),
                    "topics":
                        analysis[
                            "topics"
                        ],
                    "session_id":
                        session_id,
                    "candidates_considered":
                        len(
                            project_chunks
                        ),
                },
            }

    # ---------------------------------------------------------------
    # 7. Normal hybrid retrieval
    # ---------------------------------------------------------------

    kept, confidence = (
        _retrieve_combined(
            query=query,
            session_id=session_id,
            top_k=top_k,
            analysis=analysis,
            min_similarity=min_similarity,
        )
    )

    diagnostics: dict[str, Any] = {
        "query_type": "normal",
        "query_tokens": sorted(analysis["tokens"]),
        "topics": analysis["topics"],
        "entities": entities,
        "numbers": sorted(analysis.get("numbers", set())),
        "candidates_considered": len(analysis.get("candidate_trace", [])),
        "scores": {
            "absolute_floor": min_similarity,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "reranker_enabled": RERANKER_ENABLED,
            "rerank_candidates": RERANK_CANDIDATES,
            "rerank_top_k": RERANK_TOP_K,
            "mmr_lambda": MMR_LAMBDA,
        },
    }
    if debug:
        diagnostics["debug"] = {
            "intent": analysis["query_type"],
            "expanded_terms": expand_query(query, analysis)["terms"],
            "routed_sections": analysis["topics"],
            "entities": entities,
            "candidates": analysis.get("candidate_trace", []),
            "final_context": analysis.get("final_trace", _debug_chunks(kept)),
        }

    return {
        "chunks": kept,
        "confidence": confidence,
        "diagnostics": diagnostics,
    }


# ============================================================================
# Backwards-compatible helpers
# ============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = ABSOLUTE_FLOOR,
    session_id: str | None = None,
) -> list[RetrievedChunk]:
    """
    Backwards-compatible retrieval helper.
    """

    return retrieve_full(
        query,
        top_k=top_k,
        min_similarity=min_similarity,
        session_id=session_id,
    )["chunks"]


def retrieve_documents(
    query: str,
    n_results: int = DEFAULT_TOP_K,
) -> list[str]:
    """
    Return only retrieved chunk text.
    """

    return [
        chunk["text"]
        for chunk in retrieve(
            query,
            top_k=n_results,
        )
    ]


# ============================================================================
# CLI diagnostics
# ============================================================================

if __name__ == "__main__":

    question = input(
        "Ask something about the resume: "
    ).strip()

    if not question:
        raise SystemExit(
            "Please enter a question."
        )

    result = retrieve_full(
        question
    )

    diagnostics = result.get(
        "diagnostics",
        {},
    )

    print(
        "\n========================================"
    )
    print(
        "RETRIEVAL DIAGNOSTICS"
    )
    print(
        "========================================"
    )

    print(
        f"Query: {question}"
    )

    print(
        f"Query type: "
        f"{diagnostics.get('query_type')}"
    )

    print(
        f"Confidence: "
        f"{result.get('confidence')}"
    )

    print(
        f"Topics: "
        f"{diagnostics.get('topics', [])}"
    )

    print(
        f"Entities: "
        f"{diagnostics.get('entities', [])}"
    )
