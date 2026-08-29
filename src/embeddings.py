"""MiniLM embeddings via sentence-transformers.

The model is loaded lazily on first use (not at import time) so that
importing this module stays cheap for tests and for the FastAPI app,
and so a missing/corrupt model produces a clear EmbeddingError instead
of crashing at import.
"""

from typing import Any

from src.errors import EmbeddingError

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: Any = None


def get_model() -> Any:
    """Load and cache the MiniLM model (idempotent)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model '{MODEL_NAME}': {exc}"
            ) from exc
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with MiniLM.

    Raises:
        EmbeddingError: if encoding fails or the input is empty.
    """
    if not texts:
        raise EmbeddingError("Cannot embed an empty list of texts.")
    try:
        vectors = get_model().encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return generate_embeddings([query])[0]


if __name__ == "__main__":
    sentences = [
        "I am a React developer",
        "I build frontend applications using React",
        "I love playing cricket",
    ]
    vectors = generate_embeddings(sentences)
    print(f"{len(vectors)} embeddings of dimension {len(vectors[0])}")
