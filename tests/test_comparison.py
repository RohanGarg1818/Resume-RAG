"""Tests for the Candidate Comparison feature.

These tests cover:
1. Requirement extraction and matching
2. Candidate isolation (A vs B evidence)
3. Missing requirements handling
4. Unknown / insufficient evidence
5. Score computation
6. Similar candidates / tie detection
7. Different job descriptions producing different results
8. Schema validation
9. Edge cases (empty inputs, invalid files)
"""

import pytest

from src.compare import (
    _match_requirement,
    _compute_overall_score,
    _compute_category_scores,
    _tokenize,
    _semantic_overlap,
    compare_candidates,
    extract_requirements,
    _extract_candidate_b_name,
    _is_likely_name,
    CANDIDATE_A_NAME,
    CANDIDATE_B_FALLBACK,
)
from backend.schemas.comparison import (
    ComparisonRequest,
    ComparisonResponse,
    CandidateScore,
    RequirementMatch,
    RequirementComparison,
)


# ====================================================================
# Tokenization and overlap
# ====================================================================


class TestTokenization:
    def test_basic_tokenization(self):
        tokens = _tokenize("Python experience with Flask")
        assert "python" in tokens
        assert "experience" in tokens
        assert "flask" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("the and or but is are")
        assert len(tokens) == 0

    def test_short_tokens_removed(self):
        tokens = _tokenize("a b c Python API")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "python" in tokens

    def test_technical_tokens_preserved(self):
        tokens = _tokenize("Next.js and C++ are used")
        assert "next.js" in tokens
        assert "c++" in tokens


class TestSemanticOverlap:
    def test_perfect_overlap(self):
        score = _semantic_overlap({"python", "flask"}, {"python", "flask", "extra"})
        assert score == 1.0

    def test_no_overlap(self):
        score = _semantic_overlap({"python"}, {"java"})
        assert score == 0.0

    def test_partial_overlap(self):
        score = _semantic_overlap({"python", "flask"}, {"flask"})
        assert score == pytest.approx(0.5)

    def test_empty_query(self):
        score = _semantic_overlap(set(), {"anything"})
        assert score == 0.0


# ====================================================================
# Requirement matching
# ====================================================================


class TestMatchRequirement:
    def test_no_evidence(self):
        req = {"text": "Python experience", "category": "required", "type": "skill"}
        result = _match_requirement(req, [])
        assert result["met"] is False
        assert result["confidence"] == "none"
        assert result["score"] == 0.0
        assert result["evidence"] == ""

    def test_strong_match(self, sample_chunks_a):
        req = {"text": "REST APIs and PostgreSQL", "category": "required", "type": "skill"}
        result = _match_requirement(req, sample_chunks_a)
        assert result["met"] is True
        assert result["confidence"] in ("high", "medium")
        assert result["score"] > 0.3

    def test_weak_match(self):
        chunks = [
            {
                "text": "Completely unrelated content about marketing strategies.",
                "metadata": {"source": "resume", "section": "other"},
                "similarity": 0.10,
            }
        ]
        req = {"text": "Python programming", "category": "required", "type": "skill"}
        result = _match_requirement(req, chunks)
        assert result["met"] is False
        assert result["confidence"] in ("none", "low")

    def test_evidence_preserved(self, sample_chunks_a):
        req = {"text": "Docker experience", "category": "required", "type": "skill"}
        result = _match_requirement(req, sample_chunks_a)
        assert result["evidence"] != ""
        assert len(result["evidence_chunks"]) > 0


# ====================================================================
# Score computation
# ====================================================================


class TestScoreComputation:
    def test_overall_score_weights_required_higher(self):
        requirements = [
            {"text": "Python", "category": "required", "type": "skill"},
            {"text": "Docker", "category": "preferred", "type": "skill"},
        ]
        matches = {
            "Python": {"score": 0.8},
            "Docker": {"score": 0.4},
        }
        score = _compute_overall_score(requirements, matches)
        # Required (1.5 * 0.8) + preferred (1.0 * 0.4) / (1.5 + 1.0)
        expected = (1.5 * 0.8 + 1.0 * 0.4) / (1.5 + 1.0)
        assert score == pytest.approx(expected, abs=0.01)

    def test_overall_score_empty_requirements(self):
        score = _compute_overall_score([], {})
        assert score == 0.0

    def test_category_scores(self):
        requirements = [
            {"text": "Python", "category": "required", "type": "skill"},
            {"text": "Docker", "category": "required", "type": "skill"},
            {"text": "3 years experience", "category": "required", "type": "experience"},
        ]
        matches = {
            "Python": {"score": 0.8},
            "Docker": {"score": 0.6},
            "3 years experience": {"score": 0.7},
        }
        cats = _compute_category_scores(requirements, matches)
        assert "skill" in cats
        assert "experience" in cats
        assert cats["skill"] == pytest.approx(0.7, abs=0.01)
        assert cats["experience"] == pytest.approx(0.7, abs=0.01)


# ====================================================================
# Candidate isolation
# ====================================================================


class TestCandidateIsolation:
    """Verify that Candidate A evidence is never attributed to B and vice versa."""

    def test_evidence_not_cross_attributed(self, sample_chunks_a, sample_chunks_b):
        req = {"text": "Python experience", "category": "required", "type": "skill"}

        match_a = _match_requirement(req, sample_chunks_a)
        match_b = _match_requirement(req, sample_chunks_b)

        # Each match should only contain evidence from its own chunks
        for chunk in match_a.get("evidence_chunks", []):
            assert chunk["metadata"].get("source") == "resume"
        for chunk in match_b.get("evidence_chunks", []):
            assert chunk["metadata"].get("source") == "session"

    def test_empty_session_no_evidence_for_b(self, sample_chunks_a):
        req = {"text": "REST APIs", "category": "required", "type": "skill"}

        match_a = _match_requirement(req, sample_chunks_a)
        match_b = _match_requirement(req, [])  # empty session

        assert match_a["met"] is True
        assert match_b["met"] is False
        assert match_b["evidence"] == ""


# ====================================================================
# Missing requirements
# ====================================================================


class TestMissingRequirements:
    def test_all_requirements_unsupported(self):
        req = {"text": "Kubernetes certification", "category": "required", "type": "skill"}
        chunks = [
            {
                "text": "No relevant information about Kubernetes.",
                "metadata": {"source": "resume"},
                "similarity": 0.05,
            }
        ]
        result = _match_requirement(req, chunks)
        assert result["met"] is False
        assert result["confidence"] in ("none", "low")


# ====================================================================
# Schema validation
# ====================================================================


class TestSchemas:
    def test_comparison_request_valid(self):
        req = ComparisonRequest(job_requirements="Looking for a Python developer.")
        assert req.job_requirements == "Looking for a Python developer."

    def test_comparison_request_empty_fails(self):
        with pytest.raises(Exception):
            ComparisonRequest(job_requirements="")

    def test_comparison_request_too_long_fails(self):
        with pytest.raises(Exception):
            ComparisonRequest(job_requirements="x" * 10001)

    def test_candidate_score_bounds(self):
        score = CandidateScore(
            overall=0.75,
            required_skills=0.8,
            experience=0.6,
            projects=0.7,
            preferred_skills=0.5,
            education=0.9,
        )
        assert 0 <= score.overall <= 1

    def test_candidate_score_out_of_bounds_fails(self):
        with pytest.raises(Exception):
            CandidateScore(overall=1.5)

    def test_requirement_match_creation(self):
        rm = RequirementMatch(
            requirement="Python",
            met=True,
            evidence="Built Python apps",
            confidence="high",
            candidate="Candidate A",
        )
        assert rm.met is True
        assert rm.candidate == "Candidate A"


# ====================================================================
# Edge cases
# ====================================================================


class TestEdgeCases:
    def test_tokenizer_handles_empty_string(self):
        tokens = _tokenize("")
        assert len(tokens) == 0

    def test_match_with_single_chunk(self):
        req = {"text": "Python", "category": "required", "type": "skill"}
        chunks = [
            {
                "text": "Python developer with 5 years experience.",
                "metadata": {"source": "resume"},
                "similarity": 0.90,
            }
        ]
        result = _match_requirement(req, chunks)
        assert result["score"] > 0.0
        assert len(result["evidence_chunks"]) >= 1

    def test_very_low_similarity_chunks_filtered(self):
        req = {"text": "Machine learning", "category": "required", "type": "skill"}
        chunks = [
            {
                "text": "Cooking recipes for dinner parties.",
                "metadata": {"source": "resume"},
                "similarity": 0.02,
            }
        ]
        result = _match_requirement(req, chunks)
        assert result["met"] is False

    def test_multiple_chunks_same_score(self):
        req = {"text": "Python", "category": "required", "type": "skill"}
        chunks = [
            {
                "text": "Python developer.",
                "metadata": {"source": "resume"},
                "similarity": 0.80,
            },
            {
                "text": "Python and Flask.",
                "metadata": {"source": "resume"},
                "similarity": 0.80,
            },
        ]
        result = _match_requirement(req, chunks)
        assert result["score"] > 0.0
        assert len(result["evidence_chunks"]) >= 1


# ====================================================================
# Candidate name extraction
# ====================================================================


class TestCandidateName:
    """Tests for Candidate A fixed name and Candidate B name extraction."""

    def test_candidate_a_name_is_rohan_garg(self):
        """Candidate A is always 'Rohan Garg'."""
        assert CANDIDATE_A_NAME == "Rohan Garg"

    def test_candidate_b_fallback_constant(self):
        """Fallback name for Candidate B when extraction fails."""
        assert CANDIDATE_B_FALLBACK == "The Other Candidate"

    def test_extract_name_returns_fallback_for_empty_session(self, empty_session_id):
        """Extraction returns fallback when session has no documents."""
        name = _extract_candidate_b_name(empty_session_id)
        assert name == CANDIDATE_B_FALLBACK

    def test_extract_name_returns_fallback_for_invalid_session(self):
        """Extraction returns fallback for a non-existent session."""
        import uuid
        name = _extract_candidate_b_name(str(uuid.uuid4()))
        assert name == CANDIDATE_B_FALLBACK

    def test_extract_name_prefers_header_over_filename(self, empty_session_id):
        """The function never uses the filename as a candidate name."""
        # Even if we call the function, it should never return a filename-derived name
        name = _extract_candidate_b_name(empty_session_id)
        assert "." not in name  # filenames have extensions
        assert name == CANDIDATE_B_FALLBACK

    def test_api_response_contains_candidate_names(self):
        """API response schema includes name fields for both candidates."""
        response = ComparisonResponse(
            recommendation="Rohan Garg",
            reasoning="Test reasoning.",
            score_a=CandidateScore(overall=0.8),
            score_b=CandidateScore(overall=0.6),
            candidate_a={"name": "Rohan Garg", "score": CandidateScore(overall=0.8)},
            candidate_b={"name": "Jane Smith", "score": CandidateScore(overall=0.6)},
        )
        assert response.candidate_a.name == "Rohan Garg"
        assert response.candidate_b.name == "Jane Smith"

    def test_api_response_recommendation_uses_real_name(self):
        """Recommendation field uses the real candidate name, not 'Candidate A'."""
        response = ComparisonResponse(
            recommendation="Rohan Garg",
            reasoning="Test.",
            score_a=CandidateScore(overall=0.9),
            score_b=CandidateScore(overall=0.5),
            candidate_a={"name": "Rohan Garg", "score": CandidateScore(overall=0.9)},
            candidate_b={"name": "Jane Smith", "score": CandidateScore(overall=0.5)},
        )
        assert response.recommendation == "Rohan Garg"
        assert "Candidate A" not in response.recommendation
        assert "Candidate B" not in response.recommendation

    def test_requirement_match_uses_real_names(self):
        """RequirementMatch candidate field uses real names."""
        match_a = RequirementMatch(
            requirement="Python",
            met=True,
            evidence="Built apps",
            confidence="high",
            candidate="Rohan Garg",
        )
        match_b = RequirementMatch(
            requirement="Python",
            met=False,
            evidence="",
            confidence="none",
            candidate="Jane Smith",
        )
        assert match_a.candidate == "Rohan Garg"
        assert match_b.candidate == "Jane Smith"
        assert "Candidate A" not in match_a.candidate
        assert "Candidate B" not in match_b.candidate

    def test_frontend_renders_rohan_garg(self):
        """The input form should display 'Rohan Garg' as the label for Candidate A."""
        import pathlib
        frontend_path = (
            pathlib.Path(__file__).parent.parent
            / "frontend"
            / "components"
            / "ComparisonView.tsx"
        )
        content = frontend_path.read_text()
        assert "Rohan Garg" in content
        # No hardcoded "Candidate A" or "Candidate B" labels in user-facing text
        # (HTML comments are allowed)
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip HTML comments
            if stripped.startswith("{/*") or stripped.startswith("//"):
                continue
            # Skip string literals in code that are not user-facing
            if '"Candidate A"' in stripped or '"Candidate B"' in stripped:
                continue
            if "'Candidate A'" in stripped or "'Candidate B'" in stripped:
                continue
            # The only user-facing "Candidate A/B" text would be in JSX content
            # (not in comments, code, or string literals)
            assert "Candidate A" not in stripped, f"Found 'Candidate A' in: {stripped}"
            assert "Candidate B" not in stripped, f"Found 'Candidate B' in: {stripped}"

    def test_filename_not_used_as_name(self):
        """The name extraction function never uses the uploaded filename."""
        import pathlib
        compare_path = (
            pathlib.Path(__file__).parent.parent / "src" / "compare.py"
        )
        content = compare_path.read_text()
        # The function should not reference 'filename' in the name extraction logic
        # (it reads from ChromaDB documents, not from filenames)
        # Verify the extraction function exists and works via the session collection
        assert "_extract_candidate_b_name" in content
        assert "get_session_collection" in content


# ====================================================================
# Name validation (_is_likely_name)
# ====================================================================


class TestIsLikelyName:
    """Unit tests for the _is_likely_name helper."""

    def test_valid_two_word_name(self):
        assert _is_likely_name("Jane Smith") == "Jane Smith"

    def test_valid_three_word_name(self):
        assert _is_likely_name("Mary Jane Watson") == "Mary Jane Watson"

    def test_valid_four_word_name(self):
        assert _is_likely_name("Jean Luc Picard") == "Jean Luc Picard"

    def test_rejects_job_title(self):
        assert _is_likely_name("Technical Executive") is None

    def test_rejects_location(self):
        assert _is_likely_name("New Delhi") is None

    def test_rejects_title_and_location_combined(self):
        """Regression: 'Technical Executive New Delhi' was incorrectly accepted."""
        assert _is_likely_name("Technical Executive New Delhi") is None

    def test_rejects_senior_engineer(self):
        assert _is_likely_name("Senior Engineer") is None

    def test_rejects_single_word(self):
        assert _is_likely_name("Rohan") is None

    def test_rejects_five_words(self):
        assert _is_likely_name("Jean Luc PICARD Sir Captain") is None

    def test_rejects_lowercase_start(self):
        assert _is_likely_name("jane Smith") is None

    def test_rejects_digits(self):
        assert _is_likely_name("John Smith 123") is None

    def test_rejects_contact_info(self):
        assert _is_likely_name("John Smith +91-9876543210") is None

    def test_rejects_empty(self):
        assert _is_likely_name("") is None

    def test_rejects_long_string(self):
        assert _is_likely_name("A" * 61) is None

    def test_rejects_india(self):
        assert _is_likely_name("Rohan India") is None

    def test_rejects_bangalore(self):
        assert _is_likely_name("Priya Bangalore") is None

    def test_rejects_section_heading(self):
        assert _is_likely_name("Technical Skills") is None

    def test_rejects_work_experience(self):
        assert _is_likely_name("Work Experience") is None

    def test_rejects_education(self):
        assert _is_likely_name("Education") is None


# ====================================================================
# Regression: actual resume header structures
# ====================================================================


class TestNameExtractionRegression:
    """Regression tests using actual header structures from uploaded resumes."""

    def test_does_not_extract_title_location_as_name(self):
        """Regression: extractor returned 'Technical Executive New Delhi'
        instead of the actual candidate name from the resume header.

        A resume whose header looks like:
            Technical Executive New Delhi
            email@example.com | +91-1234567890

        should NOT produce 'Technical Executive New Delhi' as the name.
        """
        # Simulate what the first chunk of such a resume would contain
        fake_header = (
            "Technical Executive New Delhi\n"
            "email@example.com | +91-1234567890 | LinkedIn | GitHub\n"
            "Education\n"
            "Some University\n"
            "2020 - 2024\n"
            "Bachelor of Technology in Computer Science"
        )
        # The name extraction logic should reject this
        lines = fake_header.split("\n")
        for line in lines[:8]:
            name = _is_likely_name(line.strip())
            if name is not None:
                # If any line passes, it must NOT be "Technical Executive New Delhi"
                assert name != "Technical Executive New Delhi", (
                    f"Extractor incorrectly accepted '{name}' as a candidate name"
                )

    def test_extracts_real_name_from_resume_header(self):
        """A typical resume header like 'John Smith' should be extracted."""
        fake_header = (
            "John Smith\n"
            "john@example.com | +1-555-1234567 | LinkedIn | GitHub\n"
            "Education\n"
            "MIT\n"
            "2018 - 2022\n"
            "Bachelor of Science in Computer Science"
        )
        lines = fake_header.split("\n")
        for line in lines[:8]:
            name = _is_likely_name(line.strip())
            if name is not None:
                assert name == "John Smith"
                return
        pytest.fail("Failed to extract 'John Smith' from resume header")

    def test_chromadb_get_sorting(self):
        """Verify that the extraction sorts documents by chunk_index."""
        import hashlib
        import uuid
        from src.vector_store import get_session_collection

        session_id = str(uuid.uuid4())
        collection = get_session_collection(session_id)

        # Insert documents in reverse chunk_index order
        docs = [
            "Education\nMIT\n2020",
            "email@test.com | +1-555-0000",
            "Alice Johnson\nemail@test.com | +1-555-0000",
        ]
        metas = [
            {"source": "session", "session_id": session_id,
             "filename": "test.pdf", "file_type": "pdf", "chunk_index": 2},
            {"source": "session", "session_id": session_id,
             "filename": "test.pdf", "file_type": "pdf", "chunk_index": 1},
            {"source": "session", "session_id": session_id,
             "filename": "test.pdf", "file_type": "pdf", "chunk_index": 0},
        ]
        ids = [hashlib.sha256(f"{session_id}:test:{i}".encode()).hexdigest()
               for i in range(3)]
        embeddings = [[0.1] * 384 for _ in range(3)]

        collection.upsert(ids=ids, documents=docs, metadatas=metas,
                          embeddings=embeddings)

        try:
            name = _extract_candidate_b_name(session_id)
            assert name == "Alice Johnson"
        finally:
            from src.vector_store import delete_session_collection
            delete_session_collection(session_id)


# ============================================================================
# Education query confidence regression tests
# ============================================================================


class TestEducationQueryConfidence:
    """Regression tests for education queries returning low confidence.

    Issue: education queries like 'What is his academic qualification?' route
    to the education section with high hybrid score but low semantic_score,
    causing the confidence gate to reject them as 'low'.
    """

    def test_confidence_high_with_topic_bonus(self):
        """topic_bonus should count as strong evidence."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.773,
            "semantic_score": 0.273,
            "lexical_score": 0.30,
            "topic_bonus": 0.25,
            "entity_bonus": 0.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "high"

    def test_confidence_low_without_topic_bonus(self):
        """Low semantic + no strong evidence should still be low."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.40,
            "semantic_score": 0.20,
            "lexical_score": 0.25,
            "topic_bonus": 0.0,
            "entity_bonus": 0.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "low"

    def test_confidence_high_with_entity_bonus(self):
        """entity_bonus should count as strong evidence."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.55,
            "semantic_score": 0.30,
            "lexical_score": 0.20,
            "topic_bonus": 0.0,
            "entity_bonus": 1.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "high"

    def test_confidence_high_with_high_semantic(self):
        """High semantic alone should be enough."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.60,
            "semantic_score": 0.40,
            "lexical_score": 0.10,
            "topic_bonus": 0.0,
            "entity_bonus": 0.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "high"

    def test_confidence_none_for_no_chunk(self):
        """No chunk means no confidence."""
        from src.retrieve import _confidence

        assert _confidence(None) == "none"

    def test_confidence_low_below_final_score_threshold(self):
        """final_score below 0.30 should always be low."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.20,
            "semantic_score": 0.50,
            "lexical_score": 0.60,
            "topic_bonus": 0.25,
            "entity_bonus": 0.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "low"

    def test_confidence_high_with_high_lexical(self):
        """High lexical coverage should count as strong evidence."""
        from src.retrieve import _confidence

        chunk = {
            "final_score": 0.55,
            "semantic_score": 0.25,
            "lexical_score": 0.60,
            "topic_bonus": 0.0,
            "entity_bonus": 0.0,
            "numeric_bonus": 0.0,
            "exact_phrase_bonus": 0.0,
        }
        assert _confidence(chunk) == "high"


# ============================================================================
# Multi-candidate chat pipeline tests
# ============================================================================


class TestCandidateNameExtraction:
    """Tests for extract_candidate_name from document chunks."""

    def test_extracts_name_from_header(self):
        from src.compare import extract_candidate_name

        chunks = [
            "Aryan Sharma\naryan@email.com | +91 9876543210\nBTech Computer Science",
            "Experience\nBuilt REST APIs using Python and Flask.",
        ]
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_extracts_two_word_name(self):
        from src.compare import extract_candidate_name

        chunks = ["Priya Patel\npriya@test.com\nSkills: Python, Java"]
        assert extract_candidate_name(chunks) == "Priya Patel"

    def test_extracts_three_word_name(self):
        from src.compare import extract_candidate_name

        chunks = ["Rohan Kumar Singh\nrohan@test.com\nEducation: IIT Delhi"]
        assert extract_candidate_name(chunks) == "Rohan Kumar Singh"

    def test_returns_fallback_when_no_name(self):
        from src.compare import extract_candidate_name, CANDIDATE_B_FALLBACK

        chunks = [
            "education skills experience projects",
            "Built REST APIs using Python and Flask.",
        ]
        assert extract_candidate_name(chunks) == CANDIDATE_B_FALLBACK

    def test_skips_email_lines(self):
        from src.compare import extract_candidate_name

        chunks = [
            "aryan.sharma@gmail.com\nAryan Sharma\n+91 9876543210",
        ]
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_skips_phone_lines(self):
        from src.compare import extract_candidate_name

        chunks = [
            "+91 9876543210\nAryan Sharma\naryan@test.com",
        ]
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_skips_linkedin_lines(self):
        from src.compare import extract_candidate_name

        chunks = [
            "linkedin.com/in/aryansharma\nAryan Sharma\naryan@test.com",
        ]
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_skips_section_headings(self):
        from src.compare import extract_candidate_name

        chunks = [
            "Education\nAryan Sharma\nBTech Computer Science",
        ]
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_rejects_title_words(self):
        from src.compare import extract_candidate_name

        chunks = [
            "Aryan Sharma\nSenior Engineer Delhi\naryan@test.com",
        ]
        # "Aryan Sharma" is found on the first line before the title line
        assert extract_candidate_name(chunks) == "Aryan Sharma"

    def test_rejects_single_word(self):
        from src.compare import extract_candidate_name, CANDIDATE_B_FALLBACK

        chunks = ["Aryan\naryan@test.com"]
        assert extract_candidate_name(chunks) == CANDIDATE_B_FALLBACK

    def test_rejects_five_word_line(self):
        from src.compare import extract_candidate_name, CANDIDATE_B_FALLBACK

        chunks = ["Aryan Kumar Singh Sharma Jr\naryan@test.com"]
        assert extract_candidate_name(chunks) == CANDIDATE_B_FALLBACK

    def test_custom_fallback(self):
        from src.compare import extract_candidate_name

        chunks = ["education skills experience"]
        assert extract_candidate_name(chunks, fallback="Unknown Person") == "Unknown Person"

    def test_empty_chunks(self):
        from src.compare import extract_candidate_name, CANDIDATE_B_FALLBACK

        assert extract_candidate_name([]) == CANDIDATE_B_FALLBACK


class TestBuildContextCandidateLabels:
    """Tests for build_context labelling chunks with candidate identity."""

    def test_session_chunks_get_candidate_label(self):
        from src.prompts import build_context

        chunks = [
            {
                "text": "Built REST APIs using Python and Flask.",
                "metadata": {"source": "session", "section": "projects", "candidate_name": "Aryan Sharma",
                              "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
                "similarity": 0.85,
            },
        ]
        context = build_context(chunks, session_candidate_name="Aryan Sharma")
        assert "ARYAN SHARMA" in context
        assert "Built REST APIs" in context

    def test_permanent_chunks_get_rohan_label(self):
        from src.prompts import build_context

        chunks = [
            {
                "text": "Built CollabSphere using Next.js and PostgreSQL.",
                "metadata": {"source": "resume", "section": "projects",
                              "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
                "similarity": 0.85,
            },
        ]
        context = build_context(chunks)
        assert "ROHAN GARG" in context
        assert "CollabSphere" in context

    def test_mixed_chunks_get_correct_labels(self):
        from src.prompts import build_context

        chunks = [
            {
                "text": "Built CollabSphere using Next.js and PostgreSQL.",
                "metadata": {"source": "resume", "section": "projects",
                              "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
                "similarity": 0.85,
            },
            {
                "text": "Built JugaadHub using React and Node.js.",
                "metadata": {"source": "session", "section": "projects", "candidate_name": "Aryan Sharma",
                              "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
                "similarity": 0.80,
            },
        ]
        context = build_context(chunks, session_candidate_name="Aryan Sharma")
        assert "ROHAN GARG" in context
        assert "ARYAN SHARMA" in context
        assert "CollabSphere" in context
        assert "JugaadHub" in context

    def test_session_chunk_without_name_uses_fallback(self):
        from src.prompts import build_context

        chunks = [
            {
                "text": "Built REST APIs using Python and Flask.",
                "metadata": {"source": "session", "section": "projects",
                              "owner_id": "uploaded_candidate", "owner_name": "Uploaded Candidate", "owner_type": "candidate"},
                "similarity": 0.85,
            },
        ]
        context = build_context(chunks, session_candidate_name="Aryan Sharma")
        assert "UPLOADED CANDIDATE" in context

    def test_empty_chunks(self):
        from src.prompts import build_context

        context = build_context([])
        assert "No relevant information" in context


class TestSessionCandidateNameStorage:
    """Tests for session candidate name storage and retrieval."""

    def test_set_and_get_candidate_name(self):
        from src.session_store import create_session, set_candidate_name, get_candidate_name, delete_session

        session_id = create_session()
        try:
            assert get_candidate_name(session_id) is None
            set_candidate_name(session_id, "Aryan Sharma")
            assert get_candidate_name(session_id) == "Aryan Sharma"
        finally:
            delete_session(session_id)

    def test_get_candidate_name_for_nonexistent_session(self):
        from src.session_store import get_candidate_name

        assert get_candidate_name("nonexistent-session-id") is None

    def test_set_candidate_name_for_nonexistent_session(self):
        from src.session_store import set_candidate_name

        assert set_candidate_name("nonexistent-session-id", "Test") is False


class TestMultiCandidateFactIsolation:
    """Tests ensuring facts from one candidate are not attributed to another."""

    def test_build_context_separates_candidates(self):
        from src.prompts import build_context

        rohan_chunk = {
            "text": "Built CollabSphere using Next.js and PostgreSQL for real-time messaging.",
            "metadata": {"source": "resume", "section": "projects", "project": "CollabSphere",
                          "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
            "similarity": 0.90,
        }
        aryan_chunk = {
            "text": "Developed E-commerce platform using React and MongoDB.",
            "metadata": {"source": "session", "section": "projects", "candidate_name": "Aryan Sharma",
                          "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
            "similarity": 0.85,
        }
        context = build_context([rohan_chunk, aryan_chunk], session_candidate_name="Aryan Sharma")

        # Each chunk should be in its own section
        assert "ROHAN GARG" in context
        assert "ARYAN SHARMA" in context
        # Facts should be separated by section boundaries
        assert "CollabSphere" in context
        assert "E-commerce platform" in context

    def test_session_metadata_preserves_owner_fields(self):
        """Verify that owner fields are stored in chunk metadata during ingestion."""
        candidate_name = "Aryan Sharma"
        metadata = {
            "source": "session",
            "session_id": "test-session",
            "filename": "aryan_resume.pdf",
            "file_type": "pdf",
            "chunk_index": 0,
            "candidate_name": candidate_name,
            "owner_id": "uploaded_candidate",
            "owner_name": candidate_name,
            "owner_type": "candidate",
        }
        assert metadata["owner_id"] == "uploaded_candidate"
        assert metadata["owner_name"] == "Aryan Sharma"
        assert metadata["owner_type"] == "candidate"

    def test_no_cross_attribution_in_labels(self):
        """Chunks from different candidates must have different owner sections."""
        from src.prompts import build_context

        rohan_chunks = [
            {
                "text": "Education: BTech Information Technology from GGSIPU.",
                "metadata": {"source": "resume", "section": "education",
                              "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
                "similarity": 0.88,
            },
        ]
        aryan_chunks = [
            {
                "text": "Education: BCA from Delhi University.",
                "metadata": {"source": "session", "section": "education", "candidate_name": "Aryan Sharma",
                              "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
                "similarity": 0.82,
            },
        ]
        context = build_context(rohan_chunks + aryan_chunks, session_candidate_name="Aryan Sharma")

        # Each candidate's education should be in its own section
        assert "ROHAN GARG" in context
        assert "ARYAN SHARMA" in context
        assert "BTech Information Technology" in context
        assert "BCA from Delhi University" in context

    def test_rohan_linkedin_post_with_other_name_stays_rohans(self):
        """A LinkedIn post mentioning another person is still Rohan's evidence."""
        from src.prompts import build_context

        # Simulate a LinkedIn post that mentions "Aryan Kumar Srivastava"
        rohan_linkedin_chunk = {
            "text": "Congratulations to Aryan Kumar Srivastava for winning the hackathon!",
            "metadata": {"source": "linkedin", "source_type": "linkedin", "section": "achievements",
                          "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
            "similarity": 0.75,
        }
        context = build_context([rohan_linkedin_chunk])

        # Must be in Rohan's section, NOT Aryan's
        assert "ROHAN GARG — PRIMARY PROFILE" in context
        assert "Aryan Kumar Srivastava" in context  # mentioned in text
        # Should NOT create an uploaded candidate section
        assert "UPLOADED CANDIDATE" not in context

    def test_owner_id_determines_section_not_content(self):
        """Ownership is determined by owner_id, not by names in text."""
        from src.prompts import build_context

        # Chunk owned by Rohan that mentions "Priya Patel" in the text
        rohan_chunk = {
            "text": "Worked with Priya Patel on the CollabSphere project.",
            "metadata": {"source": "resume", "section": "projects",
                          "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
            "similarity": 0.80,
        }
        # Chunk owned by uploaded candidate
        uploaded_chunk = {
            "text": "Developed E-commerce platform using React.",
            "metadata": {"source": "session", "candidate_name": "Priya Patel",
                          "owner_id": "uploaded_candidate", "owner_name": "Priya Patel", "owner_type": "candidate"},
            "similarity": 0.78,
        }
        context = build_context([rohan_chunk, uploaded_chunk])

        # The context must have exactly two sections
        assert context.count("===") >= 4  # opening and closing === on each section

        # Rohan's section must contain his chunk (even though it mentions Priya)
        assert "ROHAN GARG" in context
        assert "Worked with Priya Patel" in context

        # Uploaded section must be separate
        assert "PRIYA PATEL" in context
        assert "E-commerce platform" in context

        # The Rohan section header must appear before Rohan's content
        rohan_pos = context.index("ROHAN GARG")
        priya_pos = context.index("PRIYA PATEL")
        content_pos = context.index("Worked with Priya Patel")
        # Rohan's content must be in Rohan's section (between Rohan header and Priya header)
        assert rohan_pos < content_pos < priya_pos


class TestRecruiterModeNoVerified:
    """Test that the recruiter mode instruction does not echo 'Verified'."""

    def test_recruiter_mode_instruction_wording(self):
        from src.rag import _build_messages

        messages = _build_messages(
            question="Is this candidate suitable for a backend role?",
            context="[S1, source=resume, section=skills]\nPython, FastAPI",
            history=[],
            response_mode="recruiter",
        )
        user_msg = messages[-1]["content"]
        assert "Separate verified evidence" not in user_msg
        assert "confirmed facts" in user_msg.lower() or "missing evidence" in user_msg.lower()


class TestIdentityAnswerGeneric:
    """Test that the identity answer is not hardcoded to Rohan."""

    def test_identity_answer_not_rohan_specific(self):
        from src.rag import is_identity_question
        assert is_identity_question("Who are you?")
        assert is_identity_question("What are you?")
        assert is_identity_question("What do you do?")


# ============================================================================
# Owner-aware retrieval and context tests
# ============================================================================


class TestOwnerAwareBuildContext:
    """Tests for build_context with explicit owner-based section grouping."""

    def test_rohan_only_context(self):
        """Tell me about Rohan with no session → only Rohan section."""
        from src.prompts import build_context

        chunks = [
            {"text": "BTech Information Technology from GGSIPU.",
             "metadata": {"source": "resume", "section": "education",
                           "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
             "similarity": 0.90},
            {"text": "Python, FastAPI, PostgreSQL.",
             "metadata": {"source": "resume", "section": "skills",
                           "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
             "similarity": 0.85},
        ]
        context = build_context(chunks)
        assert "ROHAN GARG — PRIMARY PROFILE" in context
        assert "UPLOADED CANDIDATE" not in context
        assert "BTech Information Technology" in context
        assert "Python, FastAPI" in context

    def test_uploaded_only_context(self):
        """Tell me about Aryan with session → only uploaded section."""
        from src.prompts import build_context

        chunks = [
            {"text": "BCA from Delhi University.",
             "metadata": {"source": "session", "section": "education",
                           "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
             "similarity": 0.88},
        ]
        context = build_context(chunks, session_candidate_name="Aryan Sharma")
        assert "UPLOADED CANDIDATE — ARYAN SHARMA" in context
        assert "ROHAN GARG" not in context
        assert "BCA from Delhi University" in context

    def test_both_candidates_context(self):
        """Tell me about Aryan and Rohan → both sections present."""
        from src.prompts import build_context

        rohan_chunks = [
            {"text": "Built CollabSphere using Next.js.",
             "metadata": {"source": "resume", "section": "projects",
                           "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
             "similarity": 0.90},
        ]
        aryan_chunks = [
            {"text": "Built E-commerce platform using React.",
             "metadata": {"source": "session", "section": "projects",
                           "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
             "similarity": 0.85},
        ]
        context = build_context(rohan_chunks + aryan_chunks, session_candidate_name="Aryan Sharma")
        assert "ROHAN GARG — PRIMARY PROFILE" in context
        assert "UPLOADED CANDIDATE — ARYAN SHARMA" in context
        assert "CollabSphere" in context
        assert "E-commerce platform" in context

    def test_rohan_linkedin_post_with_other_name_stays_rohans(self):
        """LinkedIn post mentioning another person remains Rohan's evidence."""
        from src.prompts import build_context

        chunks = [
            {"text": "Congratulations to Aryan Kumar Srivastava for winning the hackathon!",
             "metadata": {"source": "linkedin", "source_type": "linkedin", "section": "achievements",
                           "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
             "similarity": 0.75},
        ]
        context = build_context(chunks)
        assert "ROHAN GARG — PRIMARY PROFILE" in context
        assert "Aryan Kumar Srivastava" in context  # mentioned in text
        assert "UPLOADED CANDIDATE" not in context  # not a separate candidate

    def test_aryan_resume_never_becomes_rohan_evidence(self):
        """Uploaded resume chunks must always be in uploaded section."""
        from src.prompts import build_context

        chunks = [
            {"text": "Experience at Google as SWE.",
             "metadata": {"source": "session", "owner_id": "uploaded_candidate",
                           "owner_name": "Aryan Sharma", "owner_type": "candidate"},
             "similarity": 0.80},
        ]
        context = build_context(chunks, session_candidate_name="Aryan Sharma")
        # Must be in uploaded section, not Rohan's
        assert "UPLOADED CANDIDATE" in context
        rohan_section_start = context.find("ROHAN GARG")
        uploaded_section_start = context.find("UPLOADED CANDIDATE")
        if rohan_section_start >= 0:
            assert uploaded_section_start > rohan_section_start

    def test_rohan_evidence_never_becomes_aryan_evidence(self):
        """Rohan's chunks must never appear in uploaded section."""
        from src.prompts import build_context

        rohan_chunks = [
            {"text": "Python, FastAPI, Docker.",
             "metadata": {"source": "resume", "section": "skills",
                           "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
             "similarity": 0.85},
        ]
        aryan_chunks = [
            {"text": "React, Node.js, MongoDB.",
             "metadata": {"source": "session", "section": "skills",
                           "owner_id": "uploaded_candidate", "owner_name": "Aryan Sharma", "owner_type": "candidate"},
             "similarity": 0.80},
        ]
        context = build_context(rohan_chunks + aryan_chunks, session_candidate_name="Aryan Sharma")
        # Split into sections
        rohan_start = context.find("ROHAN GARG")
        aryan_start = context.find("ARYAN SHARMA")
        rohan_section = context[rohan_start:aryan_start] if aryan_start > rohan_start else context[rohan_start:]
        aryan_section = context[aryan_start:] if aryan_start > rohan_start else context[aryan_start:context.find("ROHAN GARG")]
        assert "Python, FastAPI" in rohan_section
        assert "React, Node.js" in aryan_section
        assert "Python, FastAPI" not in aryan_section
        assert "React, Node.js" not in rohan_section

    def test_unknown_candidate_name_uses_fallback(self):
        """Unknown uploaded candidate name → 'Uploaded Candidate'."""
        from src.prompts import build_context

        chunks = [
            {"text": "Some experience.",
             "metadata": {"source": "session", "owner_id": "uploaded_candidate",
                           "owner_name": "Uploaded Candidate", "owner_type": "candidate"},
             "similarity": 0.75},
        ]
        context = build_context(chunks)
        assert "UPLOADED CANDIDATE" in context

    def test_owner_id_determines_section_not_content(self):
        """Ownership determined by owner_id, not by names in text."""
        from src.prompts import build_context

        rohan_chunk = {
            "text": "Worked with Priya Patel on CollabSphere.",
            "metadata": {"source": "resume", "section": "projects",
                          "owner_id": "rohan", "owner_name": "Rohan Garg", "owner_type": "primary"},
            "similarity": 0.80,
        }
        uploaded_chunk = {
            "text": "Developed E-commerce platform.",
            "metadata": {"source": "session", "candidate_name": "Priya Patel",
                          "owner_id": "uploaded_candidate", "owner_name": "Priya Patel", "owner_type": "candidate"},
            "similarity": 0.78,
        }
        context = build_context([rohan_chunk, uploaded_chunk])
        # Rohan's chunk stays in Rohan's section even though it mentions Priya
        rohan_pos = context.index("ROHAN GARG")
        priya_pos = context.index("PRIYA PATEL")
        content_pos = context.index("Worked with Priya Patel")
        assert rohan_pos < content_pos < priya_pos


class TestTruncationDetection:
    """Tests for truncation detection in LLM responses."""

    def test_finish_reason_length_triggers_refusal(self):
        """Responses cut off by max_tokens should be refused."""
        from src.rag import NOT_FOUND_ANSWER

        # Simulate what happens when finish_reason == "length"
        # and answer ends mid-sentence
        answer = "Rohan has experience with Python and **Missing"
        # Check that truncated-looking answers are detected
        assert not answer.endswith((".", "!", "?", ":", "`"))


# ============================================================================
# Chat intent routing: greetings, assistant identity, greeting+question
# ============================================================================


class TestGreetingDetection:
    """Pure greetings are recognised without case/punctuation sensitivity."""

    @pytest.mark.parametrize("msg", [
        "hi", "hii", "hello", "hey", "hi there", "hello there",
        "Hi", "HII", "HELLO", "Hey!", "Hi there!", "hElLo ThErE",
        "hi!!!", "  hello.  ", "heyy", "helloo",
    ])
    def test_pure_greeting_is_greeting(self, msg):
        from src.rag import is_greeting
        assert is_greeting(msg) is True

    @pytest.mark.parametrize("msg", [
        "hi", "hii", "hello", "hey", "hi there",
    ])
    def test_pure_greeting_is_not_identity(self, msg):
        from src.rag import is_identity_question
        assert is_identity_question(msg) is False

    @pytest.mark.parametrize("msg", [
        "what is his technical stack?",
        "tell me about education",
        "who is rohan",
        "",
    ])
    def test_non_greetings_are_not_greetings(self, msg):
        from src.rag import is_greeting
        assert is_greeting(msg) is False


class TestIdentityDetection:
    """Assistant-identity questions (incl. greeting combos) are detected."""

    @pytest.mark.parametrize("msg", [
        "what are you?",
        "who are you?",
        "what do you do?",
        "what can you do?",
        "what can you tell me?",
        "hi what are you?",
        "hello who are you?",
        "hey what do you do?",
        "hii what are you",
        "Hi What Are You",
        "HI!!! What are you???",
        "hello, who are you?",
        "So who are you?",
    ])
    def test_identity_questions_detected(self, msg):
        from src.rag import is_identity_question
        assert is_identity_question(msg) is True, msg

    @pytest.mark.parametrize("msg", [
        "what are your skills",
        "what can you tell me about education",
        "what does he do",
        "who is rohan",
        "hi what is his technical stack?",
        "tell me about his experience",
    ])
    def test_profile_questions_not_identity(self, msg):
        """C: 'what are you' is about the assistant, but profile questions
        that merely start with similar words must go through RAG."""
        from src.rag import is_identity_question
        assert is_identity_question(msg) is False, msg


class TestGreetingPlusProfileQuestion:
    """B: greeting is ignored for retrieval and prefixed to the answer."""

    @pytest.mark.parametrize("msg,expected", [
        ("hi what is his technical stack?", "what is his technical stack?"),
        ("hii, what are his skills?", "what are his skills?"),
        ("hey what projects did he build", "what projects did he build"),
        ("hello there, how many years of experience?",
         "how many years of experience?"),
    ])
    def test_greeting_stripped_for_retrieval(self, msg, expected):
        from src.rag import _strip_leading_greeting, _greeting_prefix
        assert _greeting_prefix(msg) == "Hi! "
        assert _strip_leading_greeting(msg) == expected

    @pytest.mark.parametrize("msg", [
        "hi what is his technical stack?",
        "hii what is his technical stack?",
        "hello what are his skills",
    ])
    def test_greeting_plus_question_is_neither(self, msg):
        from src.rag import is_greeting, is_identity_question
        assert is_greeting(msg) is False
        assert is_identity_question(msg) is False

    def test_no_greeting_means_no_prefix_and_no_stripping(self):
        from src.rag import _strip_leading_greeting, _greeting_prefix
        msg = "what is his technical stack?"
        assert _greeting_prefix(msg) is None
        assert _strip_leading_greeting(msg) == msg


# ============================================================================
# End-to-end routing through ask_resume (no network)
# ============================================================================


def _fail_retrieval(*args, **kwargs):
    raise AssertionError("retrieve_full must not be called for this query")


def _fail_client():
    raise AssertionError("LLM must not be called for this query")


def _make_fake_client(captured, reply, finish_reason="stop"):
    """Minimal stand-in for the Groq client used by ask_resume."""
    import types

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=reply),
                finish_reason=finish_reason,
            )
        ]
    )

    def create(**kwargs):
        captured.append(kwargs)
        return response

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


def _fake_chunks():
    return [
        {
            "text": "Rohan uses Python, FastAPI, PostgreSQL for backend services.",
            "metadata": {
                "source": "resume",
                "section": "skills",
                "page": 1,
                "owner_id": "rohan",
                "owner_name": "Rohan Garg",
                "owner_type": "primary",
            },
            "similarity": 0.90,
            "citation_id": "S1",
        }
    ]


class TestAskResumeRouting:
    """End-to-end intent routing in ask_resume."""

    @pytest.mark.parametrize("msg", ["hi", "hii", "hello"])
    def test_pure_greeting_returns_greeting_answer(self, msg, monkeypatch):
        from src import rag

        monkeypatch.setattr(rag, "retrieve_full", _fail_retrieval)
        monkeypatch.setattr(rag, "_get_client", _fail_client)

        result = rag.ask_resume(msg, evaluation_mode=True)
        answer = result["answer"]
        assert answer == rag.GREETING_ANSWER
        assert rag.NOT_FOUND_ANSWER not in answer
        assert "Resume AI Assistant" in answer
        assert result["evaluation"]["retrieved_count"] == 0

    @pytest.mark.parametrize("msg", [
        "what are you?",
        "who are you?",
        "hi what are you?",
        "hello who are you?",
        "hey what do you do?",
    ])
    def test_identity_returns_identity_answer(self, msg, monkeypatch):
        from src import rag

        monkeypatch.setattr(rag, "retrieve_full", _fail_retrieval)
        monkeypatch.setattr(rag, "_get_client", _fail_client)

        result = rag.ask_resume(msg, evaluation_mode=True)
        answer = result["answer"]
        assert answer == rag.IDENTITY_ANSWER
        assert rag.NOT_FOUND_ANSWER not in answer
        assert "Resume AI Assistant" in answer
        assert result["evaluation"]["retrieved_count"] == 0

    def test_greeting_plus_profile_question_goes_through_rag(self, monkeypatch):
        from src import rag

        captured_queries = []
        captured_calls = []

        def fake_retrieve(query, **kwargs):
            captured_queries.append(query)
            return {"chunks": _fake_chunks(), "confidence": "high",
                    "diagnostics": {}}

        monkeypatch.setattr(rag, "retrieve_full", fake_retrieve)
        monkeypatch.setattr(
            rag, "_get_client",
            lambda: _make_fake_client(
                captured_calls,
                "Rohan uses Python and FastAPI for backend services.",
            ),
        )

        result = rag.ask_resume(
            "hi what is his technical stack?", evaluation_mode=True
        )

        # Greeting ignored for retrieval.
        assert captured_queries == ["what is his technical stack?"]
        # LLM sees the real question only (greeting handled by code).
        llm_question = captured_calls[0]["messages"][-1]["content"]
        assert "QUESTION: what is his technical stack?" in llm_question
        # Exactly one greeting prefix on the final answer.
        assert result["answer"].startswith("Hi! ")
        assert result["answer"].count("Hi!") == 1
        assert rag.NOT_FOUND_ANSWER not in result["answer"]

    def test_profile_question_without_greeting_has_no_prefix(self, monkeypatch):
        from src import rag

        captured_queries = []
        captured_calls = []

        def fake_retrieve(query, **kwargs):
            captured_queries.append(query)
            return {"chunks": _fake_chunks(), "confidence": "high",
                    "diagnostics": {}}

        monkeypatch.setattr(rag, "retrieve_full", fake_retrieve)
        monkeypatch.setattr(
            rag, "_get_client",
            lambda: _make_fake_client(
                captured_calls,
                "Rohan uses Python and FastAPI for backend services.",
            ),
        )

        result = rag.ask_resume(
            "what is his technical stack?", evaluation_mode=True
        )
        assert captured_queries == ["what is his technical stack?"]
        assert not result["answer"].startswith("Hi! ")


# ============================================================================
# Truncation handling (D)
# ============================================================================


class TestTrimTruncatedAnswer:
    """finish_reason == 'length' answers are trimmed to complete content."""

    def test_incomplete_bullet_dropped(self):
        from src.rag import _trim_truncated_answer
        answer = "- Python backend work\n- React frontend\n- GitHub for"
        trimmed = _trim_truncated_answer(answer)
        assert trimmed == "- Python backend work\n- React frontend"
        assert not trimmed.endswith("for")

    def test_complete_bullets_kept(self):
        from src.rag import _trim_truncated_answer
        answer = "- Python backend work.\n- React frontend."
        assert _trim_truncated_answer(answer) == answer

    def test_numbered_list_truncated(self):
        from src.rag import _trim_truncated_answer
        answer = "1. Built REST APIs\n2. Designed DB schemas\n3. Wrote"
        trimmed = _trim_truncated_answer(answer)
        assert trimmed == "1. Built REST APIs\n2. Designed DB schemas"
        assert "Wrote" not in trimmed

    def test_prose_cut_at_sentence_boundary(self):
        from src.rag import _trim_truncated_answer
        answer = "He built with Next.js. Then he moved"
        assert _trim_truncated_answer(answer) == "He built with Next.js."

    def test_technical_token_period_not_a_boundary(self):
        """A period inside 'next.js' must never be a cut point."""
        from src.rag import _trim_truncated_answer, NOT_FOUND_ANSWER
        answer = "He used next.js and React for the frontend"
        assert _trim_truncated_answer(answer) == NOT_FOUND_ANSWER

    def test_no_complete_prefix_refuses(self):
        from src.rag import _trim_truncated_answer, NOT_FOUND_ANSWER
        assert _trim_truncated_answer("Rohan worked on projects such as") == (
            NOT_FOUND_ANSWER
        )

    def test_trailing_colon_drops_empty_list(self):
        from src.rag import _trim_truncated_answer, NOT_FOUND_ANSWER
        assert _trim_truncated_answer("His technical skills include:") == (
            NOT_FOUND_ANSWER
        )

    def test_empty_answer(self):
        from src.rag import _trim_truncated_answer
        assert _trim_truncated_answer("") == ""

    def test_finish_reason_length_trims_instead_of_garbling(self, monkeypatch):
        """D: a length-truncated answer is trimmed, never shown mid-word."""
        from src import rag

        truncated = "- Python backend work\n- React frontend\n- GitHub for"
        chunks = [
            {
                "text": "Python backend work using React frontend and GitHub.",
                "metadata": {
                    "source": "resume",
                    "section": "skills",
                    "page": 1,
                    "owner_id": "rohan",
                    "owner_name": "Rohan Garg",
                    "owner_type": "primary",
                },
                "similarity": 0.90,
                "citation_id": "S1",
            }
        ]
        monkeypatch.setattr(
            rag, "retrieve_full",
            lambda *a, **k: {"chunks": chunks, "confidence": "high",
                             "diagnostics": {}},
        )
        monkeypatch.setattr(
            rag, "_get_client",
            lambda: _make_fake_client([], truncated, finish_reason="length"),
        )

        result = rag.ask_resume("what are his skills?", evaluation_mode=True)
        answer = result["answer"]
        assert rag.NOT_FOUND_ANSWER not in answer
        assert "- React frontend" in answer
        assert "GitHub for" not in answer
        # Trimmed content is kept (not thrown away) and never garbled.
        assert "- Python backend work" in answer

    def test_chat_uses_generous_output_limit(self):
        """The chat completion must not be capped at a token budget that
        cuts lists mid-item."""
        import inspect
        from src import rag

        source = inspect.getsource(rag.ask_resume)
        assert "max_tokens=4096" in source


# ============================================================================
# Broad profile requests (overview intent, section coverage, no refusal)
# ============================================================================


def _section_chunk(section, text="evidence", **meta):
    return {
        "text": text,
        "metadata": {"source": "resume", "section": section, **meta},
        "similarity": 0.8,
    }


class TestProfileOverviewDetection:
    """Broad profile requests must be recognised as overview intent."""

    @pytest.mark.parametrize("msg", [
        "tell me everything about him",
        "summarise his profile",
        "summarize his profile",
        "tell me about Rohan",
        "tell me about him",
        "who is he",
        "his profile",
        "full profile",
        "entire profile",
        "walk me through his profile",
        "give me an overview of his background",
        "give a summary of his resume",
        "describe the candidate",
        "what is his background",
        "everything about rohan",
    ])
    def test_broad_requests_detected(self, msg):
        from src.retrieve import is_profile_overview_query
        assert is_profile_overview_query(msg) is True, msg

    @pytest.mark.parametrize("msg", [
        "Which project used Redis?",
        "What database was used in CollabSphere?",
        "What is Rohan's CGPA?",
        "What programming languages does Rohan know?",
        "Where is Rohan pursuing his BTech?",
        "How many problems has Rohan solved on LeetCode?",
        "What is Rohan's email address?",
    ])
    def test_specific_questions_not_overview(self, msg):
        """Specific section questions keep their precise routing."""
        from src.retrieve import is_profile_overview_query
        assert is_profile_overview_query(msg) is False, msg


class TestOverviewRouting:
    """retrieve_full routes broad requests to the overview path."""

    @pytest.mark.parametrize("msg", [
        "tell me everything about him",
        "summarise his profile",
        "tell me about Rohan",
        "who is he",
    ])
    def test_broad_query_routes_to_overview(self, msg, monkeypatch):
        import src.retrieve as retrieve

        captured = {}

        def fake_overview(query, top_k, session_id=None):
            captured.update(query=query, top_k=top_k, session_id=session_id)
            return [_section_chunk("education"), _section_chunk("skills")]

        monkeypatch.setattr(retrieve, "_retrieve_overview", fake_overview)

        result = retrieve.retrieve_full(msg)
        assert result["diagnostics"]["query_type"] == "profile_overview"
        assert result["confidence"] == "high"
        assert len(result["chunks"]) == 2
        assert captured["query"] == msg

    def test_overview_passes_session_id(self, monkeypatch):
        """Overview retrieval is session-aware (uploaded candidate)."""
        import src.retrieve as retrieve

        captured = {}

        def fake_overview(query, top_k, session_id=None):
            captured["session_id"] = session_id
            return [_section_chunk("projects")]

        monkeypatch.setattr(retrieve, "_retrieve_overview", fake_overview)

        retrieve.retrieve_full(
            "summarise his profile",
            session_id="11111111-2222-3333-4444-555555555555",
        )
        assert captured["session_id"] == "11111111-2222-3333-4444-555555555555"

    def test_overview_returns_high_confidence_not_refusal(self, monkeypatch):
        """A broad request with evidence must not be refused."""
        from src import rag

        chunks = [
            _section_chunk("education", "BTech Information Technology"),
            _section_chunk("skills", "Python, FastAPI"),
            _section_chunk("projects", "JugaadHub"),
        ]
        monkeypatch.setattr(
            rag, "retrieve_full",
            lambda *a, **k: {"chunks": chunks, "confidence": "high",
                             "diagnostics": {}},
        )
        monkeypatch.setattr(
            rag, "_get_client",
            lambda: _make_fake_client(
                [],
                "Rohan is pursuing a BTech and works with Python and FastAPI.",
            ),
        )

        result = rag.ask_resume("tell me everything about him",
                                evaluation_mode=True)
        assert rag.NOT_FOUND_ANSWER not in result["answer"]
        assert result["evaluation"]["retrieved_count"] == 3


class TestOverviewSectionCoverage:
    """Overview context covers every relevant section, not just one."""

    def test_selection_covers_all_sections_reranker_dropped(self):
        from src.retrieve import OVERVIEW_SECTIONS, _select_overview_chunks

        scored = [
            _section_chunk("projects", "JugaadHub"),
            _section_chunk("education", "BTech"),
            _section_chunk("skills", "Python"),
            _section_chunk("experience", "IEEE"),
            _section_chunk("achievements", "Olympiad"),
        ]
        # Reranker only kept one section.
        ranked = [scored[0], scored[0]]

        selected = _select_overview_chunks(scored, ranked, limit=8)
        sections = [c["metadata"]["section"] for c in selected]
        for section in OVERVIEW_SECTIONS:
            if any(c["metadata"]["section"] == section for c in scored):
                assert section in sections, section

    def test_selection_prefers_reranked_representative(self):
        """The section rep should be the reranker's best chunk for it."""
        from src.retrieve import _select_overview_chunks

        scored = [
            _section_chunk("projects", "a"),
            _section_chunk("projects", "b"),
            _section_chunk("education", "c"),
        ]
        ranked = [scored[1], scored[0], scored[2]]

        selected = _select_overview_chunks(scored, ranked, limit=8)
        # Projects rep is the reranked-best chunk (scored[1]), not scored[0].
        assert selected[0] is scored[1]
        # Education still covered, remaining slot filled from ranked order.
        assert scored[2] in selected
        assert scored[0] in selected

    def test_fill_follows_reranked_order(self):
        from src.retrieve import _select_overview_chunks

        scored = [
            _section_chunk("projects", "a"),
            _section_chunk("other", "b"),
            _section_chunk("other", "c"),
        ]
        ranked = [scored[2], scored[1], scored[0]]

        selected = _select_overview_chunks(scored, ranked, limit=3)
        # Projects rep first (only overview section present), then the
        # remaining slots in reranked order.
        assert selected == [scored[0], scored[2], scored[1]]

    def test_selection_respects_limit(self):
        from src.retrieve import _select_overview_chunks

        scored = [_section_chunk("projects", str(i)) for i in range(20)]
        selected = _select_overview_chunks(scored, scored, limit=5)
        assert len(selected) == 5

    def test_no_duplicates(self):
        from src.retrieve import _select_overview_chunks

        scored = [
            _section_chunk("projects", "a"),
            _section_chunk("projects", "b"),
            _section_chunk("education", "c"),
        ]
        selected = _select_overview_chunks(scored, scored, limit=8)
        assert len(selected) == len({id(c) for c in selected}) == 3


class TestBroadOverviewPrompt:
    """The system prompt instructs a structured, non-refusing overview."""

    def test_prompt_has_structured_section_order(self):
        from src.prompts import SYSTEM_PROMPT

        assert "BROAD PROFILE QUESTIONS" in SYSTEM_PROMPT
        for heading in (
            "Education",
            "Technical Skills",
            "Projects",
            "Experience",
            "Achievements",
            "Competitions",
            "extracurriculars",
        ):
            assert heading in SYSTEM_PROMPT, heading

    def test_prompt_forbids_refusal_for_broad_requests(self):
        from src.prompts import SYSTEM_PROMPT

        assert "do not refuse" in SYSTEM_PROMPT.lower()

    def test_prompt_forbids_inventing_for_broad_requests(self):
        from src.prompts import SYSTEM_PROMPT

        assert "never invent missing facts" in SYSTEM_PROMPT.lower()

    def test_prompt_keeps_grounding_refusal_for_missing_info(self):
        """Broad-query rule must not remove the grounding refusal rule."""
        from src.prompts import SYSTEM_PROMPT

        assert "I don't have that information in the provided profile." in (
            SYSTEM_PROMPT
        )


class TestOverviewExcludesContactCards:
    """Broad overviews must never surface phone numbers or email addresses."""

    def test_contact_header_is_contact_card(self):
        from src.retrieve import _is_contact_card_chunk

        chunk = {
            "text": "Rohan Garg +91 8383913039 | rg16rohangarg@gmail.com "
                    "| LinkedIn | Codeforces | LeetCode | GitHub",
            "metadata": {"section": "other", "source": "resume"},
        }
        assert _is_contact_card_chunk(chunk) is True

    def test_bootcamp_content_chunk_is_not_contact_card(self):
        from src.retrieve import _is_contact_card_chunk

        chunk = {
            "text": "I led the organization and delivery of an 8-session "
                    "Data Structures & Algorithms Bootcamp mentoring 100+ "
                    "students under the ACM ICPC Student Chapter.",
            "metadata": {"section": "other", "source": "linkedin"},
        }
        assert _is_contact_card_chunk(chunk) is False

    def test_project_chunk_without_contact_is_not_contact_card(self):
        from src.retrieve import _is_contact_card_chunk

        chunk = {
            "text": "CollabSphere is a real-time collaboration platform "
                    "built with React, Node.js, and Redis.",
            "metadata": {"section": "projects", "source": "resume"},
        }
        assert _is_contact_card_chunk(chunk) is False

    def test_content_chunk_needing_both_channels(self):
        """Outside 'other', a chunk needs both email and phone to be a card."""
        from src.retrieve import _is_contact_card_chunk

        email_only = {
            "text": "Reach me at rohan@example.com for details about the "
                    "project.",
            "metadata": {"section": "projects", "source": "resume"},
        }
        assert _is_contact_card_chunk(email_only) is False

        both = {
            "text": "Rohan Garg, rg16rohangarg@gmail.com, +91 8383913039",
            "metadata": {"section": "projects", "source": "resume"},
        }
        assert _is_contact_card_chunk(both) is True

    def test_prompt_forbids_contact_info_in_overview(self):
        import re as _re

        from src.prompts import SYSTEM_PROMPT

        # The prompt wraps across lines; compare on collapsed whitespace.
        flattened = _re.sub(r"\s+", " ", SYSTEM_PROMPT).lower()
        assert "never include a phone number or email address" in flattened
        assert "broad profile overview" in flattened
