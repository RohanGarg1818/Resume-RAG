from fastapi import (
    APIRouter,
    Cookie,
    File,
    HTTPException,
    Response,
    UploadFile,
)

from src.session_store import (
    create_session,
    delete_session,
    get_session,
    list_files,
    register_file,
    touch_session,
    unregister_file,
)
import logging

from src.session_ingestion import ingest_file

from src.vector_store import delete_session_collection, delete_session_file




logger=logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/session",
    tags=["session"],
)


SESSION_COOKIE = "rag_session_id"


@router.post("/start")
def start_session(
    response: Response,
):

    session_id = create_session()

    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60,
    )

    return {
        "success": True,
        "session_id": session_id,
    }


@router.get("/status")
def session_status(
    rag_session_id: str | None = Cookie(
        default=None
    ),
):

    if not rag_session_id:
        return {
            "active": False,
            "files": [],
        }

    session = get_session(
        rag_session_id
    )

    if not session:
        return {
            "active": False,
            "files": [],
        }

    touch_session(
        rag_session_id
    )

    return {
        "active": True,
        "session_id": rag_session_id,
        "files": list_files(rag_session_id),
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    rag_session_id: str | None = Cookie(
        default=None
    ),
):

    if not rag_session_id:
        raise HTTPException(
            status_code=400,
            detail="No active session.",
        )

    session = get_session(
        rag_session_id
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session expired.",
        )

    allowed = {
        ".pdf",
        ".docx",
        ".md",
        ".markdown",
        ".csv",
    }

    filename = file.filename or ""

    extension = ""

    if "." in filename:
        extension = (
            "." +
            filename.rsplit(
                ".",
                1,
            )[-1].lower()
        )

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, DOCX, MD and CSV "
                "files are supported."
            ),
        )

    try:

        result = ingest_file(
            session_id=rag_session_id,
            file=file.file,
            filename=filename,
        )

        touch_session(
            rag_session_id
        )
        register_file(
            rag_session_id,
            filename,
            result["chunks_added"],
        )

        return {
            "success": True,
            **result,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Session upload failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process file."
            ),
        ) from exc


@router.post("/end")
def end_session(
    response: Response,
    rag_session_id: str | None = Cookie(
        default=None
    ),
):

    if rag_session_id:

        delete_session_collection(
            rag_session_id
        )

        delete_session(
            rag_session_id
        )

    response.delete_cookie(
        SESSION_COOKIE
    )

    return {
        "success": True,
    }


@router.delete("/files/{filename}")
def remove_document(
    filename: str,
    rag_session_id: str | None = Cookie(default=None),
):
    """Remove one Dynamic RAG document without ending the session."""
    if not rag_session_id or not get_session(rag_session_id):
        raise HTTPException(status_code=404, detail="Session expired.")
    try:
        deleted = delete_session_file(rag_session_id, filename)
        if not unregister_file(rag_session_id, filename):
            raise HTTPException(status_code=404, detail="File not found in this session.")
        return {"success": True, "filename": filename, "chunks_removed": deleted}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Session document deletion failed")
        raise HTTPException(status_code=500, detail="Failed to remove file.") from exc
