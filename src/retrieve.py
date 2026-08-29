"""Hybrid retrieval for Resume RAG.

Retrieval supports four query types:

1. Normal questions
   -> semantic + lexical + section + entity scoring

2. Profile overview questions
   -> representative chunks from major resume sections

3. Multi-entity questions
   -> retrieve each named project/company independently

4. Entity-list questions
   -> retrieve all entities belonging to a section

The goal is to prevent semantic search from silently dropping
important resume information.
"""

import logging
import os
import re
from typing import Any

from src.embeddings import embed_query
from src.errors import RetrievalError
from src.section_parser import SECTIONS
from src.vector_store import (
    composite_text,
    get_collection,
    query_vectors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))

ABSOLUTE_FLOOR = float(
    os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.15")
)

RELATIVE_GAP = 0.15
MIN_BEST_SCORE = 0.20

LEXICAL_WEIGHT = 0.25
SECTION_TOPIC_BONUS = 0.15
ENTITY_MATCH_BONUS = 0.10

CANDIDATE_MULTIPLIER = 4

CONFIDENT_SCORE = 0.45
CONFIDENT_SEMANTIC_ONLY = 0.40


# ---------------------------------------------------------------------------
# Query vocabulary
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9+#]+")

_STOPWORDS = frozenset("""
a an and are about tell me my your his her him she he they them their it
its i you we us of in on at for with to from by is was were be been being
am do does did done has have had having will would can could should shall
may might must this that these those what which who whom whose when where
why how any some please give show list also than then so such very more
most other into over under out up down
rohan candidate person profile resume
""".split())


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

    # Competitive programming / hackathons
    "hackathon": "hackathons",
    "hackathons": "hackathons",
    "leetcode": "competitive programming",
    "codeforces": "competitive programming",
    "competitive": "competitive programming",
    "programming": "competitive programming",
    "coding": "competitive programming",
    "dsa": "competitive programming",

    # Certifications / leadership
    "certification": "certifications",
    "certifications": "certifications",
    "certificate": "certifications",
    "leadership": "positions of responsibility",
    "coordinator": "positions of responsibility",
}

_EVALUATION_TERMS = frozenset({
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
})


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RetrievedChunk = dict[str, Any]
RetrievalResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Basic text helpers
# ---------------------------------------------------------------------------

def _normalise(token: str) -> str:
    """Lowercase and remove simple plural endings."""
    token = token.lower().strip("'’")

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]

    return token


def is_evaluation_query(query: str) -> bool:
    """
    Detect questions that ask us to evaluate the candidate
    using evidence already present in the profile.
    """
    tokens = _content_tokens(query)

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
        bool(tokens & _EVALUATION_TERMS)
        or any(pattern in query_lower for pattern in direct_patterns)
    )


def _retrieve_evaluation(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Retrieve diverse evidence for candidate-evaluation questions.
    """

    query_embedding = embed_query(query)

    raw = query_vectors(
        query_embedding,
        n_results=max(top_k * CANDIDATE_MULTIPLIER, 20),
    )

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    candidates: list[RetrievedChunk] = [
        {
            "text": document,
            "metadata": dict(metadata or {}),
            "distance": distance,
            "similarity": 1.0 - distance,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]

    analysis = analyse_query(query)
    analysis["raw"] = query.strip()

    scored = _score_candidates(candidates, analysis)

    # Prefer evidence from sections useful for candidate evaluation.
    preferred_sections = {
        "skills",
        "experience",
        "projects",
        "achievements",
        "education",
        "positions of responsibility",
    }

    selected: list[RetrievedChunk] = []
    sections_seen: set[str] = set()

    # First: get diverse evidence.
    for chunk in scored:
        section = str(chunk["metadata"].get("section", ""))

        if (
            section in preferred_sections
            and section not in sections_seen
        ):
            selected.append(chunk)
            sections_seen.add(section)

        if len(selected) >= top_k:
            break

    # Second: fill remaining slots by score.
    for chunk in scored:
        if len(selected) >= top_k:
            break

        if chunk not in selected:
            selected.append(chunk)

    return selected[:top_k]

def _content_tokens(text: str) -> set[str]:
    """Return meaningful normalised tokens."""
    return {
        _normalise(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------

def analyse_query(query: str) -> dict[str, Any]:
    """Extract retrieval signals from a question."""

    tokens = _content_tokens(query)

    topics = {
        _SECTION_SYNONYMS[token]
        for token in tokens
        if token in _SECTION_SYNONYMS
    }

    query_lower = query.lower()

    # Explicit hackathon terms should be treated as achievements,
    # even when generic words such as "work" also appear.
    hackathon_terms = {
        "hacknite",
        "hackxtract",
        "smart india hackathon",
        "orbix hackathon",
        "hackathon",
        "hackathons",
    }

    if any(term in query_lower for term in hackathon_terms):
        topics.discard("experience")
        topics.add("achievements")

    return {
        "tokens": tokens,
        "topics": sorted(topics),
        "raw": query.strip(),       # <-- IMPORTANT
    }

def is_profile_overview_query(query: str) -> bool:
    """Detect broad profile-summary questions."""

    tokens = _content_tokens(query)

    return bool(tokens & _PROFILE_OVERVIEW_TERMS)

def _confidence(best_chunk: RetrievedChunk | None) -> str:
    """Verdict used by the RAG layer to refuse weak-context questions."""
    if not best_chunk:
        return "none"

    has_evidence = bool(
        best_chunk["matched_keywords"]
        or best_chunk["topic_bonus"]
        or best_chunk["entity_bonus"]
    )

    semantic = best_chunk["semantic_score"]
    final = best_chunk["final_score"]

    # Strong semantic match by itself is enough.
    if semantic >= CONFIDENT_SEMANTIC_ONLY:
        return "high"

    # Otherwise require actual structured/lexical evidence
    # and a reasonably strong final score.
    if has_evidence and final >= CONFIDENT_SCORE:
        return "high"

    if final >= MIN_BEST_SCORE:
        return "low"

    return "none"


def is_entity_list_query(query: str) -> bool:
    """Detect questions asking for all projects/companies/etc."""

    tokens = _content_tokens(query)

    list_words = {
        "list",
        "all",
        "projects",
        "project",
        "companies",
        "company",
        "roles",
        "role",
        "hackathons",
        "achievements",
    }

    return bool(tokens & list_words)


# ---------------------------------------------------------------------------
# Entity detection
# ---------------------------------------------------------------------------

def extract_query_entities(query: str) -> list[str]:
    """Find project/company/role names explicitly mentioned in a query."""

    query_lower = query.lower()

    try:
        collection = get_collection()

        result = collection.get(
            include=["metadatas"]
        )

        entities: set[str] = set()

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue

            for key in ("project", "company", "role","event"):
                value = metadata.get(key)

                if value and str(value).lower() in query_lower:
                    entities.add(str(value))

        return sorted(entities)

    except Exception as exc:
        logger.warning(
            "Entity extraction failed: %s",
            exc,
        )

        return []


# ---------------------------------------------------------------------------
# Candidate conversion
# ---------------------------------------------------------------------------

def _raw_to_candidates(raw: dict) -> list[RetrievedChunk]:
    """Convert Chroma response into our internal chunk format."""

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    return [
        {
            "text": document,
            "metadata": dict(metadata or {}),
            "distance": distance,
            "similarity": 1.0 - distance,
        }
        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        )
    ]


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def _score_candidates(
    candidates: list[RetrievedChunk],
    analysis: dict[str, Any],
) -> list[RetrievedChunk]:
    """Apply semantic, lexical, topic and entity scoring."""

    tokens = analysis["tokens"]
    topics = set(analysis["topics"])
    raw_query = analysis["raw"]

    total_tokens = max(len(tokens), 1)

    for chunk in candidates:
        metadata = chunk["metadata"]

        composite = composite_text(
            chunk["text"],
            metadata,
        )

        semantic = chunk["similarity"]

        matched = tokens & _content_tokens(composite)

        lexical_coverage = (
            len(matched) / total_tokens
        )

        topic_hit = (
            str(metadata.get("section", "")) in topics
        )

        entity_names = [
            str(metadata[key])
            for key in ("project", "company", "role","event")
            if metadata.get(key)
        ]

        entity_hit = [
            name
            for name in entity_names
            if name.lower() in raw_query.lower()
        ]

        final_score = (
            semantic
            + LEXICAL_WEIGHT * lexical_coverage
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
        )

        reasons: list[str] = []

        if matched:
            reasons.append(
                "keywords:" + ",".join(sorted(matched))
            )

        if topic_hit:
            reasons.append(
                f"section:{metadata.get('section')}"
            )

        if entity_hit:
            reasons.append(
                f"entity:{entity_hit[0]}"
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
                "matched_keywords": sorted(matched),
                "final_score": round(
                    final_score,
                    4,
                ),
                "reasons": reasons or ["semantic-only"],
            }
        )

    candidates.sort(
        key=lambda chunk: chunk["final_score"],
        reverse=True,
    )

    return candidates


# ---------------------------------------------------------------------------
# Relevance + confidence
# ---------------------------------------------------------------------------

def _filter_relevant(
    scored: list[RetrievedChunk],
    min_similarity: float,
) -> list[RetrievedChunk]:
    """Filter weak chunks using absolute and relative thresholds."""

    if not scored:
        return []

    best = scored[0]["final_score"]

    if best < MIN_BEST_SCORE:
        return []

    cutoff = max(
        min_similarity,
        best - RELATIVE_GAP,
    )

    return [
        chunk
        for chunk in scored
        if chunk["final_score"] >= cutoff
    ]


def _confidence(
    best_chunk: RetrievedChunk | None,
) -> str:
    """Determine whether retrieved context is trustworthy."""

    if not best_chunk:
        return "none"

    has_evidence = bool(
        best_chunk.get("matched_keywords")
        or best_chunk.get("topic_bonus")
        or best_chunk.get("entity_bonus")
    )

    if best_chunk["final_score"] < CONFIDENT_SCORE:
        return "low"

    if (
        has_evidence
        or best_chunk["semantic_score"]
        >= CONFIDENT_SEMANTIC_ONLY
    ):
        return "high"

    return "low"


# ---------------------------------------------------------------------------
# Overview retrieval
# ---------------------------------------------------------------------------

def _retrieve_overview(
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Retrieve representative chunks across major resume sections."""

    collection_count = get_collection().count()

    if collection_count == 0:
        return []

    query_embedding = embed_query(query)

    candidate_count = min(
        collection_count,
        max(top_k * CANDIDATE_MULTIPLIER, 30),
    )

    raw = query_vectors(
        query_embedding,
        n_results=candidate_count,
    )

    candidates = _raw_to_candidates(raw)

    candidates.sort(
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )

    selected: list[RetrievedChunk] = []
    sections_seen: set[str] = set()

    # First pass: guarantee section diversity.
    for chunk in candidates:
        section = str(
            chunk["metadata"].get("section", "")
        )

        if (
            section in OVERVIEW_SECTIONS
            and section not in sections_seen
        ):
            selected.append(chunk)
            sections_seen.add(section)

    # Second pass: fill remaining slots by relevance.
    for chunk in candidates:
        if len(selected) >= top_k:
            break

        if chunk not in selected:
            selected.append(chunk)

    return selected[:top_k]


# ---------------------------------------------------------------------------
# Entity-specific retrieval
# ---------------------------------------------------------------------------

def _retrieve_entity(
    entity: str,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """Retrieve chunks belonging to one exact entity."""

    collection_count = get_collection().count()

    if collection_count == 0:
        return []

    raw = query_vectors(
        embed_query(entity),
        n_results=min(
            collection_count,
            max(top_k * 3, 10),
        ),
    )

    candidates = _raw_to_candidates(raw)

    entity_lower = entity.lower()

    results = []

    for chunk in candidates:
        metadata = chunk["metadata"]

        matches = any(
            metadata.get(key)
            and str(metadata[key]).lower()
            == entity_lower
            for key in (
                "project",
                "company",
                "role",
                "event"
            )
        )

        if matches:
            results.append(chunk)

    return results[:top_k]


# ---------------------------------------------------------------------------
# Entity-list retrieval
# ---------------------------------------------------------------------------

def _retrieve_all_section_entities(
    section: str,
) -> list[RetrievedChunk]:
    """Retrieve every chunk belonging to a section."""

    try:
        collection = get_collection()

        result = collection.get(
            where={"section": section},
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

        return [
            {
                "text": document,
                "metadata": dict(metadata or {}),
                "similarity": 1.0,
                "distance": 0.0,
                "final_score": 1.0,
                "semantic_score": 1.0,
                "lexical_score": 1.0,
                "topic_bonus": 0.0,
                "entity_bonus": 0.0,
                "matched_keywords": [],
                "reasons": [
                    f"section:{section}"
                ],
            }
            for document, metadata in zip(
                documents,
                metadatas,
            )
        ]

    except Exception as exc:
        raise RetrievalError(
            f"Failed to retrieve section '{section}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_full(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = ABSOLUTE_FLOOR,
) -> RetrievalResult:
    """Retrieve relevant resume/LinkedIn context with diagnostics."""

    if not query or not query.strip():
        raise RetrievalError(
            "Cannot retrieve with an empty query."
        )

    query = query.strip()

    analysis = analyse_query(query)

    # Keep the original query available to the scoring layer.
    analysis["raw"] = query

    # ---------------------------------------------------------------
    # 1. Multi-entity query
    # ---------------------------------------------------------------

    entities = extract_query_entities(query)

    if len(entities) > 1:
        all_chunks: list[RetrievedChunk] = []

        for entity in entities:
            all_chunks.extend(
                _retrieve_entity(
                    entity,
                    top_k=3,
                )
            )

        unique: dict[tuple, RetrievedChunk] = {}

        for chunk in all_chunks:
            metadata = chunk["metadata"]

            key = (
                metadata.get("project"),
                metadata.get("company"),
                metadata.get("role"),
                metadata.get("event"),
                chunk["text"],
            )

            unique[key] = chunk

        chunks = list(unique.values())

        chunks.sort(
            key=lambda chunk: chunk["similarity"],
            reverse=True,
        )

        return {
            "chunks": chunks[:max(
                top_k,
                len(entities) * 2,
            )],
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

    # ---------------------------------------------------------------
    # 2. Profile overview
    # ---------------------------------------------------------------

    if is_profile_overview_query(query):
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
                "query_type": "profile_overview",
                "query_tokens": sorted(
                    analysis["tokens"]
                ),
                "topics": analysis["topics"],
                "candidates_considered": len(
                    chunks
                ),
            },
        }

    # ---------------------------------------------------------------
    # 3. Candidate evaluation query
    #
    # Examples:
    #   "Should we hire Rohan?"
    #   "Is he a good backend candidate?"
    #   "Would he be suitable for a backend role?"
    #
    # These questions require evidence from multiple resume sections.
    # We therefore use broader, diverse retrieval instead of relying
    # only on the normal similarity threshold.
    # ---------------------------------------------------------------

    if is_evaluation_query(query):
        chunks = _retrieve_evaluation(
            query,
            top_k=max(top_k, 6),
        )

        return {
            "chunks": chunks,
            "confidence": (
                "high"
                if chunks
                else "none"
            ),
            "diagnostics": {
                "query_type": "evaluation",
                "query_tokens": sorted(
                    analysis["tokens"]
                ),
                "topics": analysis["topics"],
                "entities": entities,
                "candidates_considered": len(
                    chunks
                ),
            },
        }

    # ---------------------------------------------------------------
    # 4. Project/entity list query
    # ---------------------------------------------------------------

    if (
        "projects" in analysis["topics"]
        and not entities
    ):
        project_chunks = (
            _retrieve_all_section_entities(
                "projects"
            )
        )

        if project_chunks:
            return {
                "chunks": project_chunks,
                "confidence": "high",
                "diagnostics": {
                    "query_type": "project_list",
                    "query_tokens": sorted(
                        analysis["tokens"]
                    ),
                    "topics": analysis["topics"],
                    "candidates_considered": len(
                        project_chunks
                    ),
                },
            }

    # ---------------------------------------------------------------
    # 5. Normal hybrid retrieval
    # ---------------------------------------------------------------

    collection_count = get_collection().count()

    if collection_count == 0:
        return {
            "chunks": [],
            "confidence": "none",
            "diagnostics": {
                "query_type": "normal",
                "query_tokens": sorted(
                    analysis["tokens"]
                ),
                "topics": analysis["topics"],
                "entities": entities,
                "candidates_considered": 0,
            },
        }

    candidate_count = min(
        collection_count,
        max(
            top_k * CANDIDATE_MULTIPLIER,
            top_k,
        ),
    )

    raw = query_vectors(
        embed_query(query),
        n_results=candidate_count,
    )

    candidates = _raw_to_candidates(raw)

    scored = _score_candidates(
        candidates,
        analysis,
    )

    kept = _filter_relevant(
        scored,
        min_similarity,
    )

    confidence = _confidence(
        kept[0]
        if kept
        else None
    )

    logger.info(
        "Retrieval: %d/%d chunks kept, "
        "confidence=%s, best_final=%.3f",
        len(kept),
        len(candidates),
        confidence,
        kept[0]["final_score"]
        if kept
        else 0.0,
    )

    return {
        "chunks": kept,
        "confidence": confidence,
        "diagnostics": {
            "query_type": "normal",
            "query_tokens": sorted(
                analysis["tokens"]
            ),
            "topics": analysis["topics"],
            "entities": entities,
            "candidates_considered": len(
                candidates
            ),
            "scores": {
                "absolute_floor": min_similarity,
                "relative_gap": RELATIVE_GAP,
                "min_best_score": MIN_BEST_SCORE,
            },
        },
    }

# ---------------------------------------------------------------------------
# Backwards-compatible helpers
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = ABSOLUTE_FLOOR,
) -> list[RetrievedChunk]:
    """Return only ranked chunks."""

    return retrieve_full(
        query,
        top_k=top_k,
        min_similarity=min_similarity,
    )["chunks"]


def retrieve_documents(
    query: str,
    n_results: int = DEFAULT_TOP_K,
) -> list[str]:
    """Return only chunk text."""

    return [
        chunk["text"]
        for chunk in retrieve(
            query,
            top_k=n_results,
        )
    ]


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = input(
        "Ask something about the resume: "
    )

    result = retrieve_full(question)

    print(
        f"\nQuery type: "
        f"{result['diagnostics'].get('query_type')}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}"
    )

    print(
        f"Topics: "
        f"{result['diagnostics'].get('topics', [])}"
    )

    for chunk in result["chunks"]:
        metadata = chunk["metadata"]

        print(
            f"\n[final="
            f"{chunk.get('final_score', 0):.3f}"
            f" sem="
            f"{chunk.get('semantic_score', 0):.3f}] "
            f"{metadata.get('section')} "
            f"p{metadata.get('page')} "
            f"{metadata.get('project') or metadata.get('company') or ''}"
        )

        print(
            "  why: "
            + ", ".join(
                chunk.get(
                    "reasons",
                    [],
                )
            )
        )

        print(
            "  "
            + chunk["text"][:250]
        )