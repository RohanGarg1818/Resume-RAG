
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

frontend_url = os.getenv(
    "FRONTEND_URL",
    "https://rohan-resume-three.vercel.app",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://rohan-resume-three.vercel.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(chat_router)
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
