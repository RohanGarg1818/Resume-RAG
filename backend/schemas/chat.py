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

    debug: bool = Field(
        default=False,
        description="Return compact retrieval diagnostics for this request.",
    )

    response_mode: str = Field(
        default="standard",
        pattern="^(standard|recruiter|comparison)$",
        description="Answer presentation mode; all modes remain grounded.",
    )


class Source(BaseModel):
    """A source used to answer the question."""

    source: str
    citation_id: str | None = None
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
    debug: dict | None = None
