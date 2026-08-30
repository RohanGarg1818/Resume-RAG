
"""Lightweight MiniLM embeddings for the Resume RAG pipeline.

The embedding model is loaded lazily and cached so it is initialized only
once per process. CPU inference is used because the application runs on
a small Render instance without GPU acceleration.

Torch thread limits are set to 1 to reduce per-thread memory arenas,
which is critical for staying within Render's 512 MB memory limit.
"""

from typing import Any

from src.errors import EmbeddingError

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

_model: Any = None


def get_model() -> Any:
    """Load and cache the embedding model."""
    global _model

    if _model is None:
        try:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(
                MODEL_NAME,
                device="cpu",
            )

        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load embedding model "
                f"'{MODEL_NAME}': {exc}"
            ) from exc

    return _model


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


if __name__ == "__main__":
    sentences = [
        "I am a React developer",
        "I build frontend applications using React",
        "I love playing cricket",
    ]

    vectors = generate_embeddings(sentences)

    print(
        f"{len(vectors)} embeddings "
        f"of dimension {len(vectors[0])}"
    )
