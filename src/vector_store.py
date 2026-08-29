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

The persistent client and collection are created lazily so importing
this module never touches disk.
"""

import logging
import os

import chromadb

from src.chunk import Chunk, chunk_id
from src.embeddings import generate_embeddings
from src.errors import RetrievalError

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "resume",
)


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
    Cosine distance is used because the project uses MiniLM embeddings.
    """

    global _collection

    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    return _collection


# -------------------------------------------------------------------
# Collection management
# -------------------------------------------------------------------

def reset_collection() -> None:
    """
    Delete and recreate the collection.

    Useful when completely rebuilding the RAG index from the resume
    and LinkedIn data.
    """

    global _collection

    client = get_client()

    try:
        client.delete_collection(COLLECTION_NAME)

        logger.info(
            "Deleted existing collection '%s'",
            COLLECTION_NAME,
        )

    except Exception:
        # Collection may not exist yet.
        pass

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )


def count_documents() -> int:
    """Return the number of chunks stored in ChromaDB."""

    return get_collection().count()


# -------------------------------------------------------------------
# Retrieval helpers
# -------------------------------------------------------------------

def query_all() -> dict:
    """
    Return all documents and metadata from the collection.

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
    Perform raw vector similarity search.

    Returns documents, metadata, and ChromaDB distances.
    """

    try:
        collection_count = get_collection().count()

        if collection_count == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        # Chroma cannot request more results than exist.
        n_results = min(
            n_results,
            collection_count,
        )

        return get_collection().query(
            query_embeddings=[query_embedding],
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

    Metadata normally isn't part of the actual document body, so
    explicitly including it here allows queries such as:

        "What projects has he built?"
        "What did he do at Hacknite?"
        "What technologies does he know?"
        "Tell me about CollabSphere"

    to match metadata-backed information.

    Supported metadata:

        source
        section
        project
        company
        role
        event
        technologies
        date
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
    """
    Return the exact text that will be embedded for a chunk.
    """

    return composite_text(
        chunk["text"],
        chunk["metadata"],
    )


# -------------------------------------------------------------------
# Indexing
# -------------------------------------------------------------------

def add_chunks(
    chunks: list[Chunk],
) -> int:
    """
    Embed and store chunks in ChromaDB.

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
            chunk_id(chunk, index)
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