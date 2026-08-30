"""Retrieval quality evaluation: semantic-only vs hybrid scoring.

Runs the ingestion pipeline once into an isolated ChromaDB directory,
then evaluates each question under two retrieval strategies:

  semantic - the ORIGINAL behaviour: rank and filter purely on MiniLM
             cosine similarity (floor/gap/gate on similarity).
  hybrid   - the current engine: semantic + lexical + topic + entity
             signals, same three-tier filter applied to final_score.

A question passes when:
  answerable -> retrieval is confident AND kept chunks include at least
                one from an acceptable section
  junk       -> retrieval refuses (nothing confident reaches the LLM)

Usage:
    python -m evaluation.run_eval --mode both
    python -m evaluation.run_eval --mode hybrid --verbose
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# Must be set before src.vector_store reads its configuration.
EVAL_CHROMA_PATH = "/tmp/opencode/rrag_eval_chroma"
os.environ["CHROMA_PATH"] = EVAL_CHROMA_PATH

from src.chunk import chunk_resume
from src.embeddings import embed_query
from src.retrieve import ABSOLUTE_FLOOR, CANDIDATE_MULTIPLIER, MIN_BEST_SCORE, RELATIVE_GAP, retrieve_full
from src.vector_store import add_chunks, query_vectors, reset_collection  # noqa: E402


def load_questions() -> list[dict]:
    path = Path(__file__).parent / "questions.json"
    return json.loads(path.read_text())["questions"]


def setup_index() -> int:
    """Ingest the resume fresh; return the number of stored chunks."""
    reset_collection()
    return add_chunks(chunk_resume("data/Resume.pdf"))


def semantic_only_retrieve(query: str, top_k: int = 6) -> list[dict]:
    """Recreation of the pre-hybrid pipeline for comparison:
    candidates ranked and filtered purely on cosine similarity."""
    candidate_count = max(
        top_k * CANDIDATE_MULTIPLIER,
        top_k
    )

    raw = query_vectors(
        embed_query(query),
        n_results=candidate_count
    )
    candidates = [
        {
            "text": doc,
            "metadata": dict(meta or {}),
            "similarity": 1.0 - dist,
        }
        for doc, meta, dist in zip(
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        )
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    best = candidates[0]["similarity"]
    if best < MIN_BEST_SCORE:
        return []
    cutoff = max(ABSOLUTE_FLOOR, best - RELATIVE_GAP)
    return [c for c in candidates if c["similarity"] >= cutoff]


def evaluate_question(question: dict, mode: str) -> dict:
    """Return pass/fail plus detail for one question in one mode."""
    acceptable = set(question["acceptable_sections"])

    if mode == "hybrid":
        result = retrieve_full(question["question"])
        kept = result["chunks"]
        confident = result["confidence"] == "high"
    else:
        kept = semantic_only_retrieve(question["question"])
        confident = bool(kept)

    sections_hit = [
        chunk["metadata"].get("section")
        for chunk in kept
        if chunk["metadata"].get("section") in acceptable
    ]

    if question["must_answer"]:
        passed = confident and bool(sections_hit)
        detail = "" if passed else (
            "no chunks retrieved" if not kept
            else "retrieved but low confidence" if not confident
            else f"kept {kept and [c['metadata'].get('section') for c in kept]}, none acceptable"
        )
    else:
        leaked = confident and bool(kept)
        passed = not leaked
        detail = "LEAKED weak context toward LLM" if leaked else ""

    return {
        "id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "passed": passed,
        "top1_section": kept[0]["metadata"].get("section") if kept else None,
        "kept_sections": [c["metadata"].get("section") for c in kept],
        "detail": detail,
    }


def run_mode(mode: str, questions: list[dict], verbose: bool) -> dict:
    results = [evaluate_question(q, mode) for q in questions]

    print(f"\n{'=' * 76}\nMODE: {mode.upper()}\n{'=' * 76}")
    print(f"{'id':<5} {'pass':<6} {'top-1 section':<24} question")
    print("-" * 76)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{r['id']:<5} {mark:<6} {str(r['top1_section']):<24} {r['question']}")
        if verbose or not r["passed"]:
            note = r["detail"] or f"kept={r['kept_sections']}"
            print(f"      -> {note}")

    answered = [
        r for q, r in zip(questions, results) if q["must_answer"]
    ]
    refusals = [
        r for q, r in zip(questions, results) if not q["must_answer"]
    ]
    ans_pass = sum(r["passed"] for r in answered)
    ref_pass = sum(r["passed"] for r in refusals)
    total = sum(r["passed"] for r in results)

    print("-" * 76)
    print(
        f"TOTAL: {total}/{len(results)} ({total / len(results):.0%}) | "
        f"answerable: {ans_pass}/{len(answered)} | refusals: {ref_pass}/{len(refusals)}"
    )
    return {"mode": mode, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["semantic", "hybrid", "both"], default="both")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    shutil.rmtree(EVAL_CHROMA_PATH, ignore_errors=True)
    questions = load_questions()
    stored = setup_index()
    print(f"Ingested {stored} chunks into isolated index at {EVAL_CHROMA_PATH}")

    modes = ["semantic", "hybrid"] if args.mode == "both" else [args.mode]
    summary = {}
    for mode in modes:
        summary[mode] = run_mode(mode, questions, args.verbose)

    if len(summary) == 2:
        sem = sum(r["passed"] for r in summary["semantic"]["results"])
        hyb = sum(r["passed"] for r in summary["hybrid"]["results"])
        print(f"\nIMPROVEMENT: semantic-only {sem} -> hybrid {hyb} of {len(questions)} questions")


if __name__ == "__main__":
    main()
