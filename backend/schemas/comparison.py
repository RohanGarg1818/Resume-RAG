from pydantic import BaseModel, Field


class ComparisonRequest(BaseModel):
    """Request body for the candidate comparison endpoint."""

    job_requirements: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Job description or hiring requirements text.",
    )


class RequirementMatch(BaseModel):
    """One requirement's match status for a single candidate."""

    requirement: str
    met: bool
    evidence: str = ""
    confidence: str = "unknown"
    candidate: str


class RequirementComparison(BaseModel):
    """Side-by-side comparison of one requirement."""

    requirement: str
    candidate_a: RequirementMatch
    candidate_b: RequirementMatch
    better_match: str = "tie"


class CandidateScore(BaseModel):
    """Fit score breakdown for one candidate."""

    overall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized overall fit score 0-1.",
    )
    required_skills: float = Field(default=0.0, ge=0.0, le=1.0)
    experience: float = Field(default=0.0, ge=0.0, le=1.0)
    projects: float = Field(default=0.0, ge=0.0, le=1.0)
    preferred_skills: float = Field(default=0.0, ge=0.0, le=1.0)
    education: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateAnalysis(BaseModel):
    """Full analysis for one candidate."""

    name: str
    score: CandidateScore
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    unique_skills: list[str] = Field(default_factory=list)
    shared_strengths: list[str] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    """Structured comparison response."""

    recommendation: str
    reasoning: str
    score_a: CandidateScore
    score_b: CandidateScore
    requirement_comparisons: list[RequirementComparison] = Field(
        default_factory=list
    )
    candidate_a: CandidateAnalysis
    candidate_b: CandidateAnalysis
    missing_requirements: list[str] = Field(default_factory=list)
    ambiguous_requirements: list[str] = Field(default_factory=list)
    is_tie: bool = False
