
import hashlib
from typing import BinaryIO

from src.compare import extract_candidate_name
from src.document_loader import load_document
from src.embeddings import embed_documents
from src.session_store import set_candidate_name
from src.vector_store import add_session_documents


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Keep embedding batches small to reduce peak memory usage on Render.
EMBED_BATCH_SIZE = 8
    

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _generate_embeddings_in_batches(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings in small batches.

    Batching prevents a large uploaded document from creating a
    large temporary memory spike on small Render instances.
    """

    embeddings: list[list[float]] = []

    for start in range(
        0,
        len(texts),
        EMBED_BATCH_SIZE,
    ):
        batch = texts[
            start:start + EMBED_BATCH_SIZE
        ]

        batch_embeddings = embed_documents(
            batch
        )

        embeddings.extend(
            batch_embeddings
        )

    return embeddings


def ingest_file(
    session_id: str,
    file: BinaryIO,
    filename: str,
) -> dict:
    """Read, chunk, embed and store a temporary uploaded document."""

    pages = load_document(
        file,
        filename,
    )

    all_chunks = []
    all_metadatas = []

    chunk_index = 0

    for page_data in pages:

        text = page_data["text"]

        chunks = chunk_text(text)

        for chunk in chunks:

            chunk_id_source = (
                f"{session_id}:"
                f"{filename}:"
                f"{chunk_index}:"
                f"{chunk}"
            )

            chunk_id = hashlib.sha256(
                chunk_id_source.encode("utf-8")
            ).hexdigest()

            metadata = {
                "source": "session",
                "session_id": session_id,
                "filename": filename,
                "file_type": filename.rsplit(
                    ".",
                    1,
                )[-1].lower(),
                "chunk_index": chunk_index,
                "owner_id": "uploaded_candidate",
            }

            if page_data.get("page") is not None:
                metadata["page"] = page_data["page"]

            if page_data.get("row") is not None:
                metadata["row"] = page_data["row"]

            all_chunks.append(chunk)
            all_metadatas.append(metadata)

            chunk_index += 1

    if not all_chunks:
        raise ValueError(
            "The uploaded file contains no readable text."
        )

    # ---------------------------------------------------------------
    # Extract candidate name from the document header.
    # ---------------------------------------------------------------

    candidate_name = extract_candidate_name(all_chunks)
    set_candidate_name(session_id, candidate_name)

    # Add candidate_name and owner metadata to all chunk metadata
    # so retrieval can label which candidate each chunk belongs to.
    for metadata in all_metadatas:
        metadata["candidate_name"] = candidate_name
        metadata["owner_name"] = candidate_name
        metadata["owner_type"] = "candidate"

    # ---------------------------------------------------------------
    # Generate embeddings in small batches.
    # ---------------------------------------------------------------

    embeddings = _generate_embeddings_in_batches(
        all_chunks
    )

    if len(embeddings) != len(all_chunks):
        raise ValueError(
            "Embedding count does not match chunk count."
        )

    # ---------------------------------------------------------------
    # Generate deterministic IDs.
    # ---------------------------------------------------------------

    ids = []

    for metadata, chunk in zip(
        all_metadatas,
        all_chunks,
    ):
        chunk_id_source = (
            f"{session_id}:"
            f"{metadata['filename']}:"
            f"{metadata['chunk_index']}:"
            f"{chunk}"
        )

        ids.append(
            hashlib.sha256(
                chunk_id_source.encode("utf-8")
            ).hexdigest()
        )

    # ---------------------------------------------------------------
    # Store only in this session's temporary collection.
    # ---------------------------------------------------------------

    chunks_added = add_session_documents(
        session_id=session_id,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return {
        "filename": filename,
        "chunks_added": chunks_added,
    }
