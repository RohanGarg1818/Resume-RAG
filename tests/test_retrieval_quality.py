"""Regression tests for retrieval quality and answer grounding.

Unit tests run without models or ChromaDB.  The integration tests at the
bottom need the ingested index (``python -m src.ingest``) and the local
embedding/cross-encoder models; they are skipped automatically otherwise.
"""

import json
from pathlib import Path

import pytest

from src import retrieve
from src.chunk import chunk_pages, split_category_lines
from src.rag import validate_answer_evidence


# ---------------------------------------------------------------------------
# Query routing / intent
# ---------------------------------------------------------------------------


class TestContactRouting:
    @pytest.mark.parametrize("query", [
        "Which round did he reach in Smart India Hackathon?",
        "How did he address scalability in CollabSphere?",
        "Did he connect frontend and backend in VideoTube?",
    ])
    def test_ambiguous_verbs_are_not_contact(self, query):
        assert not retrieve.is_contact_query(query)

    @pytest.mark.parametrize("query", [
        "What is Rohan's email address?",
        "What is his phone number?",
        "How can I reach him?",
        "How do I get in touch with Rohan?",
    ])
    def test_real_contact_queries(self, query):
        assert retrieve.is_contact_query(query)


class TestIntentVocabulary:
    def test_plural_vocabulary_matches_normalised_tokens(self):
        # "technologies"/"languages" are normalised to singular-ish tokens;
        # the vocab must be normalised the same way.
        analysis = retrieve.analyse_query("Which tools and languages does he use?")
        assert "skills" in analysis["topics"]

    def test_experience_with_technology_is_not_evaluation_or_list(self):
        query = "Does Rohan have experience with Docker?"
        assert not retrieve.is_evaluation_query(query)
        assert not retrieve.is_list_query(query)
        assert "experience" not in retrieve.analyse_query(query)["topics"]

    def test_work_experience_is_still_list(self):
        assert retrieve.is_list_query("What is his work experience?")

    def test_hiring_question_is_evaluation(self):
        assert retrieve.is_evaluation_query("Should we hire him for a backend role?")

    def test_degree_abbreviation_maps_to_education(self):
        analysis = retrieve.analyse_query("Where is he pursuing his BTech?")
        assert "education" in analysis["topics"]
        assert "bachelor of technology" in retrieve.expand_query(
            "Where is he pursuing his BTech?", analysis
        )["terms"]


class TestEntityAliases:
    def test_acronym_and_parts(self):
        aliases = retrieve._entity_aliases("Association for Computing Machinery - ICPC")
        assert {"acm", "icpc", "association for computing machinery"} <= aliases

    def test_short_aliases_are_not_generated(self):
        assert all(len(alias) >= 3 for alias in retrieve._entity_aliases("Orbix Hackathon"))

    def test_alias_requires_word_boundary(self):
        assert retrieve._alias_in_text("acm", "role at acm icpc")
        assert not retrieve._alias_in_text("acm", "the acme corp")


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------


class TestBm25:
    def test_term_frequency_counts(self):
        # Regression: documents were tokenised into sets, so TF was always 1.
        scores = retrieve._bm25_from_tokens(
            {"redis"},
            [["redis", "redis", "cache"], ["redis", "cache", "queue"]],
        )
        assert scores[0] > scores[1]


class TestWeightedCoverage:
    def test_rare_term_outweighs_generic_term(self):
        index = {"size": 20, "document_frequency": {"experience": 12, "docker": 1}}
        rare = retrieve._weighted_coverage({"experience", "docker"}, {"docker"}, index)
        generic = retrieve._weighted_coverage({"experience", "docker"}, {"experience"}, index)
        assert rare > 0.5 > generic


class TestFusedRelevance:
    def test_noise_between_negative_logits_does_not_beat_hybrid(self):
        strong = {"final_score": 0.66, "reranker_score": -11.1}
        noise = {"final_score": 0.17, "reranker_score": -10.2}
        pool = [noise, strong]
        assert retrieve._fused_relevance(strong, pool) > retrieve._fused_relevance(noise, pool)

    def test_confident_cross_encoder_wins(self):
        confident = {"final_score": 0.5, "reranker_score": 4.0}
        other = {"final_score": 0.6, "reranker_score": -9.0}
        pool = [confident, other]
        assert retrieve._fused_relevance(confident, pool) > retrieve._fused_relevance(other, pool)

    def test_diversity_orders_small_pools_by_fused_relevance(self):
        weak = {"text": "linkedin post", "final_score": 0.17, "reranker_score": -10.2}
        strong = {"text": "Tools: Docker", "final_score": 0.66, "reranker_score": -11.0}
        assert retrieve._apply_diversity("q", [weak, strong], top_k=6)[0] is strong


class TestConfidence:
    def test_context_confidence_uses_leading_chunks(self):
        weak = {"final_score": 0.32, "semantic_score": 0.32}
        strong = {"final_score": 0.8, "semantic_score": 0.2, "topic_bonus": 0.25}
        assert retrieve._confidence(weak) == "low"
        assert retrieve._context_confidence([weak, strong]) == "high"

    def test_specific_term_is_strong_evidence(self):
        chunk = {"final_score": 0.66, "semantic_score": 0.30, "lexical_score": 0.38,
                 "specific_terms": ["docker"]}
        assert retrieve._confidence(chunk) == "high"

    def test_empty_context(self):
        assert retrieve._context_confidence([]) == "none"


class TestContextSelection:
    def test_drops_tail_without_evidence(self):
        chunks = [
            {"text": "a", "final_score": 1.0, "reranker_score": 3.0},
            {"text": "b", "final_score": 0.9, "reranker_score": -10.0},   # close hybrid
            {"text": "c", "final_score": 0.3, "reranker_score": -1.0},    # CE plausible
            {"text": "d", "final_score": 0.3, "reranker_score": -11.0},   # no evidence
        ]
        kept = [chunk["text"] for chunk in retrieve.select_context(chunks)]
        assert kept == ["a", "b", "c"]

    def test_always_keeps_first(self):
        chunks = [{"text": "a", "final_score": 0.2, "reranker_score": -11.0},
                  {"text": "b", "final_score": 0.01, "reranker_score": -11.0}]
        assert [c["text"] for c in retrieve.select_context(chunks)] == ["a"]


class TestContactCardExclusion:
    def test_header_chunk_removed(self):
        header = {"text": "Rohan Garg\n+91 8383913039 | someone@gmail.com",
                  "metadata": {"section": "other"}}
        body = {"text": "CGPA: 8.3", "metadata": {"section": "education"}}
        assert retrieve._exclude_contact_cards([header, body]) == [body]


class TestExplicitEvidenceGate:
    def test_preference_without_evidence_is_low(self):
        result = {"chunks": [{"text": "Languages: C, C++, Python", "metadata": {}}],
                  "confidence": "high", "diagnostics": {}}
        gated = retrieve._apply_explicit_evidence_gate(
            "What is his favorite programming language?", result)
        assert gated["confidence"] == "low"

    def test_preference_with_evidence_is_kept(self):
        result = {"chunks": [{"text": "My favorite language is Python", "metadata": {}}],
                  "confidence": "high", "diagnostics": {}}
        gated = retrieve._apply_explicit_evidence_gate(
            "What is his favorite language?", result)
        assert gated["confidence"] == "high"

    def test_non_preference_query_untouched(self):
        result = {"chunks": [{"text": "x", "metadata": {}}], "confidence": "high"}
        assert retrieve._apply_explicit_evidence_gate("What is his CGPA?", result)["confidence"] == "high"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestSkillCategoryChunking:
    PAGES = [{"page": 1, "text": (
        "Technical Skills\n"
        "Languages: C, C++, Python\n"
        "Backend & Databases: Node.js, PostgreSQL, Redis\n"
        "Core Coursework: Data Structures, Operating Systems, Computer\n"
        "Networks\n"
        "Projects\n"
        "Alpha | Thing | Python | Live\n"
        "• Built alpha.\n"
    )}]

    def test_one_chunk_per_category(self):
        skills = [c for c in chunk_pages(self.PAGES) if c["metadata"]["section"] == "skills"]
        assert [c["metadata"]["skill_category"] for c in skills] == [
            "Languages", "Backend & Databases", "Core Coursework"]

    def test_wrapped_line_stays_with_category(self):
        skills = [c for c in chunk_pages(self.PAGES) if c["metadata"]["section"] == "skills"]
        assert skills[-1]["text"].endswith("Networks")

    def test_plain_block_unchanged(self):
        assert split_category_lines("just one line\nand another") == [
            (None, "just one line\nand another")]


# ---------------------------------------------------------------------------
# Answer grounding validator
# ---------------------------------------------------------------------------


class TestEvidenceValidation:
    CHUNKS = [{"text": "CollabSphere | Next.js, Redis, PostgreSQL\n• mentoring 100+ students"}]

    def test_numbered_list_answer_is_not_rejected(self):
        answer = "Projects:\n1. CollabSphere uses Redis and PostgreSQL.\n2. It uses Next.js."
        assert validate_answer_evidence(answer, self.CHUNKS)["valid"]

    def test_plus_suffix_numbers_match(self):
        assert validate_answer_evidence("He mentored 100 students.", self.CHUNKS)["valid"]

    def test_invented_number_rejected(self):
        result = validate_answer_evidence("He mentored 250 students.", self.CHUNKS)
        assert not result["valid"] and result["unsupported_numbers"] == ["250"]

    def test_invented_technologies_rejected(self):
        answer = "He built it with Kubernetes, Kafka and Terraform on AWS."
        assert not validate_answer_evidence(answer, self.CHUNKS)["valid"]


# ---------------------------------------------------------------------------
# Integration: retrieval metrics must not regress (needs index + models)
# ---------------------------------------------------------------------------


def _index_available() -> bool:
    try:
        from src.vector_store import count_documents
        from src.embeddings import get_model
        if count_documents() == 0:
            return False
        get_model()
        return True
    except Exception:
        return False


integration = pytest.mark.skipif(
    not _index_available(),
    reason="ChromaDB index or embedding model unavailable (run python -m src.ingest)",
)

EVAL_DIR = Path(__file__).resolve().parents[1] / "backend" / "evaluation"


@integration
class TestRetrievalRegression:
    # Floors are set below the measured values (see report) so normal
    # variation passes while real regressions fail.
    @pytest.mark.parametrize("dataset, floors", [
        ("dataset.json", {"hit_at_1": 0.89, "hit_at_3": 0.96, "mrr": 0.93,
                          "f1_at_k": 0.85, "refusal_accuracy": 1.0}),
        ("holdout.json", {"hit_at_1": 0.94, "hit_at_3": 0.94, "mrr": 0.94,
                          "f1_at_k": 0.83, "refusal_accuracy": 1.0}),
    ])
    def test_metric_floors(self, dataset, floors):
        from backend.evaluation.retrieval_eval import run_retrieval_evaluation
        summary = run_retrieval_evaluation(save=False, dataset_path=EVAL_DIR / dataset)["summary"]
        for metric, floor in floors.items():
            assert summary[metric] >= floor, (metric, summary[metric])

    @pytest.mark.parametrize("query", [
        "Where is Rohan pursuing his BTech?",
        "Does Rohan have experience with Docker?",
        "Does Rohan have experience with Redis?",
        "What was Rohan's role at ACM ICPC?",
    ])
    def test_answerable_questions_are_not_refused(self, query):
        result = retrieve.retrieve_full(query)
        assert result["confidence"] == "high"
        assert not any(retrieve._is_contact_card_chunk(c) for c in result["chunks"])

    def test_education_question_ranks_education_first(self):
        chunks = retrieve.retrieve_full("Where is Rohan pursuing his BTech?")["chunks"]
        assert chunks[0]["metadata"]["section"] == "education"

    def test_docker_question_ranks_tools_first(self):
        chunks = retrieve.retrieve_full("Does Rohan have experience with Docker?")["chunks"]
        assert "Docker" in chunks[0]["text"]

    @pytest.mark.parametrize("query", [
        "What is Rohan's favorite football team?",
        "What is Rohan's favorite programming language?",
        "What is Rohan's date of birth?",
    ])
    def test_unsupported_questions_are_low_confidence(self, query):
        assert retrieve.retrieve_full(query)["confidence"] in {"low", "none"}

    def test_hackathon_round_is_not_contact(self):
        result = retrieve.retrieve_full("Which round did he reach in Smart India Hackathon?")
        assert result["diagnostics"].get("query_type") != "contact"
        assert any("Round 2" in chunk["text"] for chunk in result["chunks"])
