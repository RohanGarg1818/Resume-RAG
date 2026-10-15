
import logging


from backend.evaluation.retrieval_eval import (
    HISTORY_PATH,
    LATEST_PATH,
    run_retrieval_evaluation,
)


import asyncio

from backend.session_cleanup import (
    cleanup_expired_sessions,
)


import os
from backend.routes.session import router as session_router
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.chat import router as chat_router
from backend.routes.comparison import router as comparison_router
from backend.routes.document import router as documents_router
from backend.routes.voice import router as voice_router

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

async def session_cleanup_loop():

    while True:

        try:
            cleanup_expired_sessions()

        except Exception as exc:
            print(
                "Session cleanup failed:",
                exc,
            )

        await asyncio.sleep(
            60
        )

app = FastAPI(
    title="Resume RAG API",
    description=(
        "AI-powered API for asking questions about "
        "a candidate's professional profile."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

# Browsers only accept a credentialed (cookie) response when the server
# echoes the exact requesting origin, so origins must be listed explicitly;
# a "*" wildcard is not allowed together with allow_credentials=True.
#
#   CORS_ALLOWED_ORIGINS  comma-separated list of extra allowed origins
#   FRONTEND_URL          the deployed frontend origin
#
# Local development origins (Next.js on 3000/3001, Vite on 5173, via both
# localhost and 127.0.0.1) are allowed unless CORS_ALLOW_LOCALHOST=false,
# which production deployments should set.

DEFAULT_FRONTEND_URL = "https://rohan-resume-three.vercel.app"

LOCAL_DEV_ORIGINS = tuple(
    f"http://{host}:{port}"
    for host in ("localhost", "127.0.0.1")
    for port in (3000, 3001, 5173)
)


def _normalise_origin(origin: str) -> str:
    # An origin never has a path or trailing slash; "https://x.app/" in an
    # env var would otherwise silently fail to match the browser's Origin.
    return origin.strip().rstrip("/")


def get_allowed_origins() -> list[str]:
    origins: list[str] = []
    candidates = [
        os.getenv("FRONTEND_URL", DEFAULT_FRONTEND_URL),
        *os.getenv("CORS_ALLOWED_ORIGINS", "").split(","),
    ]
    if os.getenv("CORS_ALLOW_LOCALHOST", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }:
        candidates.extend(LOCAL_DEV_ORIGINS)
    for candidate in candidates:
        origin = _normalise_origin(candidate)
        if not origin:
            continue
        if origin == "*":
            raise RuntimeError(
                "CORS_ALLOWED_ORIGINS must list explicit origins; '*' cannot be "
                "used with cookie-based sessions."
            )
        if origin not in origins:
            origins.append(origin)
    return origins


allowed_origins = get_allowed_origins()
logger.info("CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(chat_router)
app.include_router(comparison_router)
app.include_router(voice_router)
app.include_router(documents_router)
app.include_router(
    session_router
)







# ---------------------------------------------------------
# Health
# ---------------------------------------------------------


@app.get("/api/evaluation")
def get_evaluation():
    """Return the most recent deterministic retrieval run and its history."""
    if not LATEST_PATH.exists():
        return {
            "message": "No retrieval-only evaluation has been run yet."
        }

    import json

    with open(
        LATEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        latest = json.load(file)

    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            latest["history"] = json.load(file)
    else:
        latest["history"] = []
    return latest


@app.post("/api/evaluation/retrieval")
def run_retrieval_evaluation_endpoint():
    """Run retrieval metrics locally; this endpoint never calls Groq."""
    return run_retrieval_evaluation()



@app.get("/api/index/status", tags=["Health"])
def index_status() -> dict:
    """Read-only facts about the permanent index for the UI's status panel."""
    from src.embeddings import MODEL_NAME
    from src.rag import get_model_name
    from src.reranker import RERANKER_ENABLED, RERANKER_MODEL
    from src.vector_store import collection_snapshot, get_collection

    snapshot = collection_snapshot(get_collection())
    by_source: dict[str, int] = {}
    for metadata in snapshot["metadatas"]:
        source = str(metadata.get("source", "unknown"))
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "chunks": len(snapshot["ids"]),
        "by_source": by_source,
        "embedding_model": MODEL_NAME,
        "reranker_model": RERANKER_MODEL if RERANKER_ENABLED else None,
        "llm_model": get_model_name(),
        "llm_location": "cloud",
    }


@app.get(
    "/api/health",
    tags=["Health"],
)
def health_check() -> dict:
    """Basic API health check."""

    return {
        "status": "ok",
        "service": "resume-rag-api",
    }


@app.get(
    "/",
    tags=["Health"],
)
def root() -> dict:
    """Root endpoint."""

    return {
        "message": "Resume RAG API is running.",
        "docs": "/docs",
    }


@app.on_event("startup")
async def startup():

    asyncio.create_task(
        session_cleanup_loop()
    )
