
"""ChromaDB vector storage for the Resume + LinkedIn RAG.

The same ChromaDB collection stores both resume and LinkedIn
documents. Metadata is included in the composite embedding text so
that semantic and lexical retrieval can use fields such as:

    section
    project
    company
    role
    event
    technologies
    source

Temporary uploaded documents are stored in isolated ChromaDB
collections, one collection per RAG session.

The persistent client and collections are created lazily so importing
this module never touches disk.
"""

import logging
import os
import re

import chromadb

from src.chunk import Chunk, chunk_id
from src.embeddings import generate_embeddings
from src.errors import RetrievalError


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./chroma_db",
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "resume",
)

SESSION_COLLECTION_PREFIX = "session_"


# -------------------------------------------------------------------
# Lazy ChromaDB state
# -------------------------------------------------------------------

_client: chromadb.api.ClientAPI | None = None
_collection = None


def get_client() -> chromadb.api.ClientAPI:
    """Return the shared persistent ChromaDB client."""

    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

    return _client


def get_collection():
    """
    Return the shared ChromaDB collection.

    Resume and LinkedIn documents are stored in the same collection.
    Cosine distance is used because the project uses normalized
    MiniLM embeddings.
    """

    global _collection

    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    return _collection


def get_resume_collection():
    """
    Return the permanent Resume + LinkedIn collection.

    This collection is never modified by temporary session uploads.
    """

    return get_collection()


# -------------------------------------------------------------------
# Session collection helpers
# -------------------------------------------------------------------

def _validate_session_id(
    session_id: str,
) -> None:
    """Validate that the session ID is a UUID-like value."""

    if not session_id:
        raise ValueError(
            "session_id is required"
        )

    if not re.fullmatch(
        r"[a-f0-9-]{36}",
        session_id,
        re.IGNORECASE,
    ):
        raise ValueError(
            "Invalid session_id"
        )


def _get_session_collection_name(
    session_id: str,
) -> str:
    """Build the isolated Chroma collection name for a session."""

    _validate_session_id(
        session_id
    )

    return (
        f"{SESSION_COLLECTION_PREFIX}"
        f"{session_id.replace('-', '')}"
    )


def get_session_collection(
    session_id: str,
):
    """
    Return the isolated ChromaDB collection for one session.

    Each session receives its own collection:

        session_<uuid-without-hyphens>

    Temporary documents remain completely separate from the
    permanent Resume + LinkedIn collection.

    Embeddings are supplied manually by the MiniLM embedding
    pipeline, so no Chroma embedding function is required.
    """

    collection_name = (
        _get_session_collection_name(
            session_id
        )
    )

    client = get_client()

    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "type": "temporary_session",
            "session_id": session_id,
        },
    )


def delete_session_collection(
    session_id: str,
) -> None:
    """
    Permanently delete all vector data belonging to a session.

    The permanent Resume + LinkedIn collection is never touched.
    """

    if not session_id:
        return

    try:
        collection_name = (
            _get_session_collection_name(
                session_id
            )
        )

        client = get_client()

        client.delete_collection(
            name=collection_name
        )

        logger.info(
            "Deleted temporary session collection '%s'",
            collection_name,
        )

    except Exception as exc:
        logger.debug(
            "Session collection deletion skipped for "
            "%s: %s",
            session_id,
            exc,
        )


# -------------------------------------------------------------------
# Session indexing
# -------------------------------------------------------------------

def add_session_documents(
    session_id: str,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
    embeddings: list[list[float]],
) -> int:
    """
    Store temporary uploaded document chunks in the session's
    isolated ChromaDB collection.

    These documents are never inserted into the permanent Resume
    collection.
    """

    if not documents:
        return 0

    if not (
        len(documents)
        == len(metadatas)
        == len(ids)
        == len(embeddings)
    ):
        raise RetrievalError(
            "Session document data lengths do not match."
        )

    try:
        collection = get_session_collection(
            session_id
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info(
            "Stored %d temporary chunks for session %s",
            len(documents),
            session_id,
        )

        return len(documents)

    except RetrievalError:
        raise

    except Exception as exc:
        raise RetrievalError(
            "Failed to store temporary session "
            f"documents in ChromaDB: {exc}"
        ) from exc


# -------------------------------------------------------------------
# Collection management
# -------------------------------------------------------------------

def reset_collection() -> None:
    """
    Delete and recreate the permanent collection.

    Temporary session collections are not affected.
    """

    global _collection

    client = get_client()

    try:
        client.delete_collection(
            COLLECTION_NAME
        )

        logger.info(
            "Deleted existing collection '%s'",
            COLLECTION_NAME,
        )

    except Exception:
        pass

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


def count_documents() -> int:
    """Return the number of chunks stored in the permanent collection."""

    return get_collection().count()


# -------------------------------------------------------------------
# Retrieval helpers
# -------------------------------------------------------------------

def query_all() -> dict:
    """
    Return all documents and metadata from the permanent collection.

    Mainly useful for debugging, entity extraction, and inspection.
    """

    try:
        return get_collection().get(
            include=[
                "documents",
                "metadatas",
            ]
        )

    except Exception as exc:
        raise RetrievalError(
            f"ChromaDB full retrieval failed: {exc}"
        ) from exc


def query_vectors(
    query_embedding: list[float],
    n_results: int,
) -> dict:
    """
    Perform raw vector similarity search against the permanent
    Resume + LinkedIn collection.

    Returns documents, metadata, and ChromaDB distances.
    """

    try:
        collection = get_collection()

        collection_count = collection.count()

        if collection_count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        n_results = min(
            n_results,
            collection_count,
        )

        return collection.query(
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

    except Exception as exc:
        raise RetrievalError(
            f"ChromaDB query failed: {exc}"
        ) from exc


# -------------------------------------------------------------------
# Embedding text
# -------------------------------------------------------------------

def composite_text(
    text: str,
    metadata: dict,
) -> str:
    """
    Build the text used for embeddings and lexical retrieval.

    Metadata is explicitly included so queries can match information
    such as projects, companies, roles, events and technologies.
    """

    context_bits: list[str] = []

    metadata_fields = (
        "source",
        "section",
        "project",
        "company",
        "role",
        "event",
        "technologies",
        "date",
    )

    for key in metadata_fields:
        value = metadata.get(key)

        if value:
            context_bits.append(
                str(value)
            )

    return " | ".join(
        [
            *context_bits,
            text,
        ]
    )


def embedding_input(
    chunk: Chunk,
) -> str:
    """Return the exact text that will be embedded for a chunk."""

    return composite_text(
        chunk["text"],
        chunk["metadata"],
    )


# -------------------------------------------------------------------
# Permanent indexing
# -------------------------------------------------------------------

def add_chunks(
    chunks: list[Chunk],
) -> int:
    """
    Embed and store chunks in the permanent ChromaDB collection.

    Works for both resume and LinkedIn chunks.

    Deterministic IDs mean repeated ingestion performs an upsert
    instead of creating duplicate documents.
    """

    if not chunks:
        return 0

    try:
        embedding_inputs = [
            embedding_input(chunk)
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            embedding_inputs
        )

        ids = [
            chunk_id(
                chunk,
                index,
            )
            for index, chunk in enumerate(chunks)
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            chunk["metadata"]
            for chunk in chunks
        ]

        get_collection().upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d chunks in ChromaDB",
            len(chunks),
        )

        return len(chunks)

    except RetrievalError:
        raise

    except Exception as exc:
        raise RetrievalError(
            f"Failed to store chunks in ChromaDB: {exc}"
        ) from exc


# -------------------------------------------------------------------
# Debug / manual indexing
# -------------------------------------------------------------------

if __name__ == "__main__":
    from src.chunk import chunk_resume

    reset_collection()

    stored = add_chunks(
        chunk_resume(
            "data/Resume.pdf"
        )
    )

    print(
        f"Stored {stored} resume chunks."
    )

    print(
        f"Collection now has "
        f"{count_documents()} documents."
    )
