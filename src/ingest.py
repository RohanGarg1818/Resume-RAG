"""Ingest resume and LinkedIn data into ChromaDB."""

from src.chunk import chunk_resume
from src.linkedin_parser import parse_linkedin_csv
from src.vector_store import add_chunks, count_documents, reset_collection


RESUME_PATH = "data/Resume.pdf"
LINKEDIN_PATH = "data/linkedin/Rich_Media.csv"


def ingest_resume() -> int:
    """Parse and index the resume."""
    chunks = chunk_resume(RESUME_PATH)
    print(f"Resume chunks: {len(chunks)}")

    stored = add_chunks(chunks)
    print(f"Resume chunks indexed: {stored}")

    return stored


def ingest_linkedin() -> int:
    """Parse and index LinkedIn posts."""
    documents = parse_linkedin_csv(LINKEDIN_PATH)

    print(f"LinkedIn posts: {len(documents)}")

    stored = add_chunks(documents)
    print(f"LinkedIn posts indexed: {stored}")

    return stored


def main() -> None:
    """Rebuild the complete RAG index."""
    print("Resetting ChromaDB...")
    reset_collection()

    print("\nIndexing resume...")
    resume_count = ingest_resume()

    print("\nIndexing LinkedIn...")
    linkedin_count = ingest_linkedin()

    print("\n--------------------------------")
    print("Ingestion complete")
    print("--------------------------------")
    print(f"Resume chunks: {resume_count}")
    print(f"LinkedIn posts: {linkedin_count}")
    print(f"Total ChromaDB chunks: {count_documents()}")


if __name__ == "__main__":
    main()