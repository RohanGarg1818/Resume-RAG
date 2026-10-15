"""Deterministic retrieval evaluation, independent of any LLM provider."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.retrieve import retrieve_full

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
LATEST_PATH = BASE_DIR / "latest_retrieval.json"
HISTORY_PATH = BASE_DIR / "history.json"


def _normalise(value: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9+#.@ -]+", " ", str(value or "").lower()).strip()


def _canonical_section(section: str) -> str:
    value = _normalise(section)
    if "skill" in value or "technical" in value:
        return "skills"
    if "work" in value or "experience" in value:
        return "experience"
    if "project" in value:
        return "projects"
    if "education" in value or "academic" in value:
        return "education"
    if "achievement" in value or "competitive" in value or "hackathon" in value:
        return "achievements"
    if "contact" in value or "header" in value:
        return "contact"
    return value


def _chunk_text(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return _normalise(" ".join([str(chunk.get("text", "")), *map(str, metadata.values())]))


def _matches_label(chunk: dict[str, Any], label: str) -> bool:
    label = _normalise(label)
    return bool(label) and label in _chunk_text(chunk)


def _section_matches(chunk: dict[str, Any], expected: str) -> bool:
    actual = _canonical_section((chunk.get("metadata", {}) or {}).get("section", ""))
    return actual == _canonical_section(expected)


def _is_relevant(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
    """Gold-free relevance rule documented by each dataset item's labels.

    The dataset does not yet contain chunk IDs, so relevance is evaluated from
    the explicit expected source/entity, section, and fact labels.  This is
    intentionally transparent and can be replaced by gold IDs later.
    """
    expected_sources = item.get("expected_sources", [])
    expected_sections = item.get("expected_sections", [])
    keywords = [_normalise(value) for value in item.get("expected_keywords", [])]
    source_hit = any(_matches_label(chunk, source) for source in expected_sources)
    section_hit = any(_section_matches(chunk, section) for section in expected_sections)
    keyword_hit = any(keyword and keyword in _chunk_text(chunk) for keyword in keywords)
    if not (expected_sources or expected_sections or keywords):
        return False
    # A correctly routed labelled source/section is relevant evidence even if
    # a multi-fact answer spans several chunks.  Keyword-only labels require
    # an actual fact match to avoid rewarding generic routing.
    return source_hit or (section_hit and keyword_hit) or (not (expected_sources or expected_sections) and keyword_hit)


def _coverage(chunks: list[dict[str, Any]], item: dict[str, Any]) -> float:
    scores: list[float] = []
    for keyword in item.get("expected_keywords", []):
        scores.append(float(any(_normalise(keyword) in _chunk_text(chunk) for chunk in chunks)))
    for source in item.get("expected_sources", []):
        scores.append(float(any(_matches_label(chunk, source) for chunk in chunks)))
    for section in item.get("expected_sections", []):
        scores.append(float(any(_section_matches(chunk, section) for chunk in chunks)))
    return sum(scores) / len(scores) if scores else 1.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_retrieval_evaluation(
    top_k: int = 6,
    save: bool = True,
    dataset_path: Path | str = DATASET_PATH,
) -> dict[str, Any]:
    """Evaluate retrieval only; Groq is never called from this function.

    ``dataset_path`` lets the same harness score the held-out set
    (``holdout.json``), which is never used for tuning and therefore acts as
    an overfitting check for the main dataset.
    """
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    question_results: list[dict[str, Any]] = []
    precision_at: dict[int, list[float]] = {k: [] for k in (1, 3, 5, top_k)}
    recall_at: dict[int, list[float]] = {k: [] for k in (1, 3, 5, top_k)}
    hits: dict[int, list[float]] = {k: [] for k in (1, 3, 5)}
    mrrs: list[float] = []
    source_accuracy: list[float] = []
    section_accuracy: list[float] = []
    refusal: list[float] = []
    latency: list[float] = []
    context_sizes: list[float] = []

    for item in dataset:
        started = time.perf_counter()
        result = retrieve_full(item["question"], top_k=top_k, debug=True)
        elapsed = (time.perf_counter() - started) * 1000
        chunks = result.get("chunks", [])
        unsupported = item.get("query_type") == "unsupported"
        relevant = [_is_relevant(chunk, item) for chunk in chunks]

        if unsupported:
            refused = not chunks or result.get("confidence") in {"none", "low"}
            refusal.append(float(refused))
        else:
            first_rank = next((index + 1 for index, value in enumerate(relevant) if value), None)
            mrrs.append(1 / first_rank if first_rank else 0.0)
            for k in precision_at:
                ranked = chunks[:k]
                precision_at[k].append(sum(_is_relevant(chunk, item) for chunk in ranked) / len(ranked) if ranked else 0.0)
                recall_at[k].append(_coverage(ranked, item))
            for k in hits:
                hits[k].append(float(any(relevant[:k])))
            if chunks and item.get("expected_sources"):
                source_accuracy.append(float(any(_matches_label(chunks[0], label) for label in item["expected_sources"])))
            if chunks and item.get("expected_sections"):
                section_accuracy.append(float(any(_section_matches(chunks[0], label) for label in item["expected_sections"])))

        question_results.append({
            "id": item["id"],
            "question": item["question"],
            "query_type": item.get("query_type", "normal"),
            "confidence": result.get("confidence"),
            "retrieval_latency_ms": round(elapsed, 2),
            "precision_at_k": {str(k): round(precision_at[k][-1], 4) for k in precision_at if not unsupported},
            "recall_at_k": {str(k): round(recall_at[k][-1], 4) for k in recall_at if not unsupported},
            "first_relevant_rank": next((index + 1 for index, value in enumerate(relevant) if value), None),
            "refused": (not chunks or result.get("confidence") in {"none", "low"}) if unsupported else None,
            "retrieved_sections": [chunk.get("metadata", {}).get("section") for chunk in chunks],
            "debug": result.get("diagnostics", {}).get("debug", {}),
        })
        latency.append(elapsed)
        if not unsupported:
            context_sizes.append(float(len(chunks)))

    primary_precision = _mean(precision_at[top_k])
    primary_recall = _mean(recall_at[top_k])
    f1 = 2 * primary_precision * primary_recall / (primary_precision + primary_recall) if primary_precision + primary_recall else 0.0
    summary = {
        "evaluation_type": "retrieval_only",
        "llm_calls": 0,
        "total_questions": len(dataset),
        "supported_questions": len(dataset) - len(refusal),
        "unsupported_questions": len(refusal),
        "precision_at_k": {str(k): round(_mean(values), 4) for k, values in precision_at.items()},
        "recall_at_k": {str(k): round(_mean(values), 4) for k, values in recall_at.items()},
        "f1_at_k": round(f1, 4),
        "f1_at_k_by_k": {
            str(k): round(
                2 * _mean(precision_at[k]) * _mean(recall_at[k])
                / (_mean(precision_at[k]) + _mean(recall_at[k])), 4)
            if _mean(precision_at[k]) + _mean(recall_at[k]) else 0.0
            for k in precision_at
        },
        "avg_context_chunks": round(_mean(context_sizes), 2),
        "hit_at_1": round(_mean(hits[1]), 4),
        "hit_at_3": round(_mean(hits[3]), 4),
        "hit_at_5": round(_mean(hits[5]), 4),
        "mrr": round(_mean(mrrs), 4),
        "source_accuracy": round(_mean(source_accuracy), 4),
        "section_accuracy": round(_mean(section_accuracy), 4),
        "refusal_accuracy": round(_mean(refusal), 4),
        "avg_retrieval_latency_ms": round(_mean(latency), 2),
    }
    payload = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "questions": question_results,
    }
    if save:
        _save_json(LATEST_PATH, payload)
        history = _load_history()
        history.append({"run_id": payload["run_id"], "created_at": payload["created_at"], "summary": summary})
        _save_json(HISTORY_PATH, history[-50:])
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval-only evaluation")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    is_main_dataset = Path(args.dataset).resolve() == DATASET_PATH.resolve()
    outcome = run_retrieval_evaluation(
        top_k=args.top_k,
        # Only the main dataset updates latest/history so held-out runs do
        # not overwrite the tracked baseline.
        save=not args.no_save and is_main_dataset,
        dataset_path=args.dataset,
    )
    print(json.dumps(outcome["summary"], indent=2))
