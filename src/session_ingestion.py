import hashlib
from typing import BinaryIO

from src.document_loader import load_document
from src.embeddings import embed_documents
from src.vector_store import add_session_documents


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:

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


def ingest_file(
    session_id: str,
    file: BinaryIO,
    filename: str,
) -> dict:

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

    embeddings = embed_documents(
        all_chunks
    )

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

    add_session_documents(
        session_id=session_id,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return {
        "filename": filename,
        "chunks_added": len(all_chunks),
    }