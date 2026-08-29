import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.chunk import chunk_resume
from src.vector_store import add_chunks, count_documents, reset_collection

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
) -> dict:
    """
    Upload and ingest a resume PDF.

    The existing RAG ingestion pipeline is reused:
    
    PDF
    ↓
    extraction
    ↓
    section-aware chunking
    ↓
    MiniLM embeddings
    ↓
    ChromaDB
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temporary_path: Path | None = None

    try:
        # Create a temporary file rather than permanently storing
        # an uploaded document automatically.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temporary_file:

            temporary_path = Path(temporary_file.name)

            shutil.copyfileobj(
                file.file,
                temporary_file,
            )

        # Validate and chunk the uploaded PDF.
        chunks = chunk_resume(str(temporary_path))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No usable content was found in the PDF.",
            )

        # This application currently represents one profile at a time.
        reset_collection()

        stored = add_chunks(chunks)

        logger.info(
            "Resume ingestion completed: %d chunks stored",
            stored,
        )

        return {
            "message": "Resume uploaded and indexed successfully.",
            "filename": file.filename,
            "chunks_created": stored,
            "collection_size": count_documents(),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Resume ingestion failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to process the resume.",
        ) from exc

    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)