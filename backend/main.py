
import logging
import os

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


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

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
