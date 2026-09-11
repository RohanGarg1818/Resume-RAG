
"""Lightweight MiniLM embeddings for the Resume RAG pipeline.

The embedding model is loaded lazily and cached so it is initialized only
once per process. CPU inference is used because the application runs
on a small Render instance without GPU acceleration.

Torch thread limits are configured defensively. PyTorch may reject
set_num_interop_threads() after parallel work has already started, so
thread configuration failures are logged and do not crash the application.
"""

import gc
import logging
from typing import Any

import torch

from src.errors import EmbeddingError


logger = logging.getLogger(__name__)


MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
BATCH_SIZE = 8

_model: Any = None
_threads_configured = False


def _configure_torch_threads() -> None:
    """Configure conservative CPU thread limits once per process.

    PyTorch only allows inter-op thread configuration before parallel work
    has started. Therefore, failures are intentionally ignored and logged
    instead of breaking embedding model initialization.
    """

    global _threads_configured

    if _threads_configured:
        return

    try:
        torch.set_num_threads(1)
    except RuntimeError as exc:
        logger.warning(
            "Could not set PyTorch intra-op threads: %s",
            exc,
        )

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        logger.warning(
            "Could not set PyTorch inter-op threads; "
            "parallel work may already have started: %s",
            exc,
        )

    _threads_configured = True


def get_model() -> Any:
    """Load and cache the embedding model."""

    global _model

    if _model is not None:
        return _model

    try:
        # Configure threading before importing SentenceTransformer/model code.
        _configure_torch_threads()

        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model: %s",
            MODEL_NAME,
        )

        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )

        # Prevent unnecessary gradient tracking during inference.
        _model.eval()

        logger.info(
            "Embedding model loaded successfully: %s",
            MODEL_NAME,
        )

        return _model

    except Exception as exc:
        _model = None
        gc.collect()

        raise EmbeddingError(
            f"Failed to load embedding model "
            f"'{MODEL_NAME}': {exc}"
        ) from exc


def generate_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Generate normalized MiniLM embeddings for multiple texts."""

    if not texts:
        raise EmbeddingError(
            "Cannot embed an empty list of texts."
        )

    try:
        model = get_model()

        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        return [
            vector.tolist()
            for vector in vectors
        ]

    except EmbeddingError:
        raise

    except Exception as exc:
        raise EmbeddingError(
            f"Embedding generation failed: {exc}"
        ) from exc


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a single query."""

    query = query.strip()

    if not query:
        raise EmbeddingError(
            "Cannot embed an empty query."
        )

    return generate_embeddings([query])[0]


def embed_documents(
    documents: list[str],
) -> list[list[float]]:
    """Generate normalized embeddings for uploaded documents."""

    if not documents:
        return []

    try:
        model = get_model()

        vectors = model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        return [
            vector.tolist()
            for vector in vectors
        ]

    except EmbeddingError:
        raise

    except Exception as exc:
        raise EmbeddingError(
            f"Document embedding generation failed: {exc}"
        ) from exc


def unload_model() -> None:
    """Unload the cached embedding model and release memory."""

    global _model

    if _model is not None:
        del _model
        _model = None
        gc.collect()

        logger.info(
            "Embedding model unloaded."
        )


if __name__ == "__main__":
    sentences = [
        "I am a React developer",
        "I build frontend applications using React",
        "I love playing cricket",
    ]

    vectors = generate_emb_
