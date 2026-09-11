"""Cross-encoder reranking for Resume RAG.

The reranker is a second-stage ranker. The existing retrieval system first
uses ChromaDB + hybrid scoring to find relevant candidates. This module then
uses a CrossEncoder to improve the ordering of those candidates.

Pipeline:
    ChromaDB -> hybrid scoring -> relevance filtering -> CrossEncoder -> top-K

The model is loaded lazily so importing this module does not immediately load
another PyTorch model. If the reranker is disabled, unavailable, or fails,
the caller can safely continue with the hybrid-ranked results.
"""

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

RERANKER_ENABLED = (
    os.getenv("RERANKER_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

RERANKER_CANDIDATES = max(
    1,
    int(os.getenv("RERANKER_CANDIDATES", "12")),
)

RERANKER_TOP_K = max(
    1,
    int(os.getenv("RERANKER_TOP_K", "6")),
)

RERANKER_BATCH_SIZE = max(
    1,
    int(os.getenv("RERANKER_BATCH_SIZE", "8")),
)


# Lazy-loaded CrossEncoder instance.
_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _get_model():
    """Load and cache the CrossEncoder model when first needed."""
    global _model

    if _model is not None:
        return _model

    if not RERANKER_ENABLED:
        logger.info("Cross-encoder reranking is disabled.")
        return None

    try:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading cross-encoder model: %s",
            RERANKER_MODEL,
        )

        _model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
        )

        logger.info(
            "Cross-encoder loaded successfully: %s",
            RERANKER_MODEL,
        )

        return _model

    except Exception:
        logger.exception("Failed to load cross-encoder model.")
        _model = None
        return None


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = RERANKER_TOP_K,
) -> list[dict[str, Any]]:
    """Rerank already-retrieved chunks using a CrossEncoder.

    Args:
        query: Original user query.
        chunks: Hybrid-ranked candidate chunks. Each chunk should contain
            a ``text`` field and may contain retrieval diagnostics.
        top_k: Number of final chunks to return.

    Returns:
        A new list of chunks ordered by CrossEncoder relevance. Original
        chunk dictionaries are copied before reranker fields are attached.
        If reranking cannot run, the original order is preserved.
    """

    if not chunks:
        return []

    top_k = max(1, int(top_k))

    if not query or not query.strip():
        return chunks[:top_k]

    if not RERANKER_ENABLED:
        return chunks[:top_k]

    # Only send a bounded candidate pool to the CrossEncoder.
    candidate_limit = max(top_k, RERANKER_CANDIDATES)
    candidates = chunks[:candidate_limit]

    model = _get_model()
    if model is None:
        logger.warning(
            "Cross-encoder unavailable. Falling back to hybrid ranking."
        )
        return candidates[:top_k]

    pairs: list[tuple[str, str]] = []
    valid_chunks: list[dict[str, Any]] = []

    for chunk in candidates:
        text = str(chunk.get("text", ""))
        if not text.strip():
            continue

        pairs.append((query, text))
        valid_chunks.append(chunk)

    if not valid_chunks:
        logger.warning("Cross-encoder received no valid chunks.")
        return []

    try:
        logger.info(
            "Starting cross-encoder reranking | candidates=%d | query=%s",
            len(valid_chunks),
            query[:120],
        )

        scores = model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []

        for chunk, score in zip(valid_chunks, scores):
            updated = dict(chunk)
            reranker_score = float(score)

            updated["reranker_score"] = round(reranker_score, 4)
            updated["hybrid_score"] = chunk.get("final_score", 0.0)
            updated["reranker_reason"] = "cross-encoder"

            # Preserve existing retrieval diagnostics.
            updated["reasons"] = list(
                chunk.get("reasons", [])
            )
            updated["reasons"].append("reranked:cross-encoder")

            reranked.append(updated)

        reranked.sort(
            key=lambda chunk: chunk.get(
                "reranker_score",
                float("-inf"),
            ),
            reverse=True,
        )

        final_chunks = reranked[:top_k]

        for index, chunk in enumerate(final_chunks, start=1):
            metadata = chunk.get("metadata", {})

            logger.info(
                "Reranked #%d | reranker=%.4f | hybrid=%.4f | "
                "semantic=%.4f | section=%s | project=%s | source=%s",
                index,
                chunk.get("reranker_score", 0.0),
                chunk.get("hybrid_score", 0.0),
                chunk.get("semantic_score", 0.0),
                metadata.get("section"),
                metadata.get("project"),
                metadata.get("source"),
            )

        logger.info(
            "Cross-encoder reranking completed | returned=%d",
            len(final_chunks),
        )

        return final_chunks

    except Exception:
        logger.exception(
            "Cross-encoder reranking failed. Falling back to hybrid ranking."
        )
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Memory / health helpers
# ---------------------------------------------------------------------------


def unload_reranker() -> None:
    """Release the loaded CrossEncoder and run garbage collection."""
    global _model

    if _model is None:
        return

    try:
        del _model
    except Exception:
        pass

    _model = None
    gc.collect()

    logger.info("Cross-encoder model unloaded from memory.")



def is_reranker_available() -> bool:
    """Return True when the CrossEncoder can be loaded."""
    if not RERANKER_ENABLED:
        return False

    return _get_model() is not None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Resume RAG Cross-Encoder Reranker")
    print(f"Model: {RERANKER_MODEL}")
    print(f"Enabled: {RERANKER_ENABLED}")
    print(f"Candidate count: {RERANKER_CANDIDATES}")
    print(f"Top K: {RERANKER_TOP_K}")

    model = _get_model()

    if model is None:
        print("\nReranker is unavailable or disabled.")
    else:
        print("\nReranker model loaded successfully.")

        query = input("\nEnter a test query: ").strip()

        if query:
            sample_chunks = [
                {
                    "text": (
                        "Rohan mentored 100+ students during the DSA Bootcamp."
                    ),
                    "metadata": {
                        "section": "Work Experience",
                    },
                    "final_score": 0.70,
                    "semantic_score": 0.68,
                    "reasons": ["semantic-only"],
                },
                {
                    "text": (
                        "Rohan built CollabSphere using Redis and PostgreSQL."
                    ),
                    "metadata": {
                        "section": "Projects",
                        "project": "CollabSphere",
                    },
                    "final_score": 0.65,
                    "semantic_score": 0.63,
                    "reasons": ["semantic-only"],
                },
            ]

            results = rerank(
                query=query,
                chunks=sample_chunks,
                top_k=2,
            )

            print("\nReranked results:")

            for index, chunk in enumerate(results, start=1):
                print(f"\n#{index}")
                print("Reranker score:", chunk.get("reranker_score"))
                print("Hybrid score:", chunk.get("hybrid_score"))
                print("Text:", chunk.get("text", ""))
