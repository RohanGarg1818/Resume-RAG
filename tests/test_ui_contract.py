"""Contract tests for the fields the redesigned workspace UI relies on.

All fields are additive and optional, so older clients keep working.
"""

from fastapi.testclient import TestClient

from backend import main
from backend.schemas.chat import ChatResponse
from src import rag


def _chunk(text, citation_id, **metadata):
    return {"text": text, "citation_id": citation_id, "metadata": {"source": "resume", **metadata}}


class TestExtractSources:
    def test_merged_sources_keep_every_label_and_excerpt(self, monkeypatch):
        monkeypatch.setattr(rag, "_similarities", lambda query, chunks: [0.42, 0.61])
        chunks = [
            _chunk("CollabSphere uses Redis TTL.", "S1", section="projects", project="CollabSphere"),
            _chunk("CollabSphere uses PostgreSQL.", "S2", section="projects", project="CollabSphere"),
        ]
        sources = rag.extract_sources(chunks, "Which project used Redis?")
        assert len(sources) == 1
        assert sources[0]["citation_ids"] == ["S1", "S2"]
        assert sources[0]["score"] == 0.61  # best of the merged chunks
        assert sources[0]["excerpt"] == "CollabSphere uses Redis TTL."

    def test_long_excerpt_is_truncated(self, monkeypatch):
        monkeypatch.setattr(rag, "_similarities", lambda query, chunks: [None])
        sources = rag.extract_sources([_chunk("x" * 2000, "S1", section="skills")], "q")
        assert sources[0]["excerpt"].endswith("…")
        assert len(sources[0]["excerpt"]) == rag.EXCERPT_CHARS + 1
        assert sources[0]["score"] is None

    def test_similarity_failure_is_not_fatal(self, monkeypatch):
        import src.embeddings as embeddings

        def boom(texts):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(embeddings, "generate_embeddings", boom)
        assert rag._similarities("q", [{"text": "a"}]) == [None]


class TestInlineCitations:
    def test_citation_labels_do_not_count_as_unsupported_terms(self):
        chunks = [{"text": "Built RESTful APIs with Node.js and MongoDB at IEEE."}]
        answer = "At IEEE he built RESTful APIs with Node.js and MongoDB [S1]."
        result = rag.validate_answer_evidence(answer, chunks)
        assert result["valid"], result
        assert "s1" not in result.get("unsupported_terms", [])

    def test_prompt_requests_inline_labels_and_no_sources_section(self):
        from src.prompts import SYSTEM_PROMPT

        assert "directly after the claim" in SYSTEM_PROMPT
        assert 'NEVER add a separate "Sources"' in SYSTEM_PROMPT


class TestResponseSchema:
    def test_retrieval_summary_is_optional(self):
        legacy = ChatResponse(answer="a", sources=[], conversation_id="c")
        assert legacy.retrieval is None

    def test_full_payload_round_trips(self):
        payload = ChatResponse(
            answer="a [S1]",
            sources=[{"source": "resume", "citation_id": "S1", "citation_ids": ["S1"], "score": 0.5, "excerpt": "x"}],
            conversation_id="c",
            retrieval={"chunks": 1, "latency_ms": 12.5, "llm_latency_ms": 900.0, "confidence": "high", "route": "normal"},
        ).model_dump()
        assert payload["sources"][0]["citation_ids"] == ["S1"]
        assert payload["retrieval"]["route"] == "normal"


def test_index_status_endpoint(monkeypatch):
    import src.vector_store as vector_store

    monkeypatch.setattr(
        vector_store,
        "collection_snapshot",
        lambda collection: {"ids": ["a", "b", "c"], "metadatas": [{"source": "resume"}, {"source": "resume"}, {"source": "linkedin"}], "documents": [], "key": None},
    )
    monkeypatch.setattr(vector_store, "get_collection", lambda: object())
    body = TestClient(main.app).get("/api/index/status").json()
    assert body["chunks"] == 3
    assert body["by_source"] == {"resume": 2, "linkedin": 1}
    assert body["embedding_model"]
    assert body["llm_model"]
