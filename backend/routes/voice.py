import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from groq import Groq

router = APIRouter(prefix="/api/voice", tags=["Voice"])


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured",
        )

    return Groq(api_key=api_key)


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    Convert uploaded audio into text using Groq Whisper.
    """

    if not audio.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required",
        )

    try:
        audio_bytes = await audio.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file",
            )

        client = get_groq_client()

        transcription = client.audio.transcriptions.create(
            file=(
                audio.filename,
                audio_bytes,
            ),
            model="whisper-large-v3-turbo",
            response_format="json",
        )

        return {
            "text": transcription.text.strip()
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        ) from exc