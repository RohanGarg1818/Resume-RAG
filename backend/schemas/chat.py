from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the candidate.",
    )

    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID for follow-up questions.",
    )


class Source(BaseModel):
    """A source used to answer the question."""

    source: str
    section: str | None = None
    page: int | None = None
    project: str | None = None
    company: str | None = None
    role: str | None = None


class ChatResponse(BaseModel):
    """Response returned by the chat endpoint."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    conversation_id: str