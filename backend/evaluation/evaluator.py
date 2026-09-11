
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Callable

from src.rag import ask_resume

from .metrics import (
    retrieval_precision,
    retrieval_recall,
    retrieval_f1,
    answer_relevance,
    grounding_score,
    refusal_accuracy,
    average,
)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
RESULTS_PATH = BASE_DIR / "results.json"


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

def load_dataset() -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from dataset.json.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any]) -> None:
    """
    Save evaluation results to results.json.
    """

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def run_evaluation(
    rag_function: Callable[..., Dict[str, Any]] = ask_resume,
) -> Dict[str, Any]:
    """
    Run the complete evaluation dataset against the actual RAG pipeline.

    Each question is evaluated using:

    - Retrieval Precision
    - Retrieval Recall
    - Retrieval F1
    - Answer Relevance
    - Grounding
    - Refusal Accuracy
    - Retrieval Latency
    - LLM Latency

    Retrieval evaluation uses:

    - expected_keywords
    - expected_sources
    - expected_sections
    """

    dataset = load_dataset()

    question_results: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Metric accumulators
    # -----------------------------------------------------------------------

    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []

    relevance_scores: List[float] = []
    grounding_scores: List[float] = []

    # Only unsupported questions contribute to the meaningful
    # refusal accuracy calculation.
    refusal_scores: List[float] = []

    retrieval_latencies: List[float] = []
    llm_latencies: List[float] = []

    grounding_failures = 0

    supported_questions = 0

    print()
    print("=" * 60)
    print("STARTING RAG EVALUATION")
    print("=" * 60)
    print(f"Questions: {len(dataset)}")
    print()

    # -----------------------------------------------------------------------
    # RUN EACH QUESTION
    # -----------------------------------------------------------------------

    for index, item in enumerate(dataset, start=1):

        question = item["question"]

        expected_answer = item.get(
            "expected_answer"
        )

        expected_keywords = item.get(
            "expected_keywords",
            [],
        )

        expected_sources = item.get(
            "expected_sources",
            [],
        )

        expected_sections = item.get(
            "expected_sections",
            [],
        )

        query_type = item.get(
            "query_type",
            "normal",
        )

        print(
            f"[{index}/{len(dataset)}] {question}"
        )

        # -------------------------------------------------------------------
        # RUN RAG
        # -------------------------------------------------------------------

        start_time = time.perf_counter()

        try:

            result = rag_function(
                question=question,
                conversation_id=None,
                session_id=None,
                evaluation_mode=True,
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            question_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "query_type": query_type,
                    "error": str(error),
                    "score": 0.0,
                }
            )

            continue

        total_latency = (
            time.perf_counter() - start_time
        ) * 1000

        # -------------------------------------------------------------------
        # RAG OUTPUT
        # -------------------------------------------------------------------

        answer = result.get(
            "answer",
            "",
        )

        evaluation_data = result.get(
            "evaluation",
            {},
        )

        retrieved_chunks = evaluation_data.get(
            "retrieved_chunks",
            [],
        )

        retrieval_latency = evaluation_data.get(
            "retrieval_latency_ms",
            0.0,
        )

        llm_latency = evaluation_data.get(
            "llm_latency_ms",
            0.0,
        )

        retrieval_confidence = evaluation_data.get(
            "retrieval_confidence",
            "unknown",
        )

        # -------------------------------------------------------------------
        # METRICS
        # -------------------------------------------------------------------

        precision = retrieval_precision(
            retrieved_chunks,
            expected_keywords,
            expected_sources,
            expected_sections,
        )

        recall = retrieval_recall(
            retrieved_chunks,
            expected_keywords,
            expected_sources,
            expected_sections,
        )

        f1 = retrieval_f1(
            precision,
            recall,
        )

        relevance = answer_relevance(
            answer,
            expected_keywords,
        )

        grounding = grounding_score(
            answer,
            retrieved_chunks,
        )

        refusal = refusal_accuracy(
            answer,
            query_type,
        )

        # -------------------------------------------------------------------
        # UNSUPPORTED QUESTIONS
        # -------------------------------------------------------------------

        if query_type == "unsupported":

            # Correct refusal should not be considered a grounding failure.
            if refusal == 1.0:
                grounding = 1.0

            # Only unsupported questions contribute to refusal accuracy.
            refusal_scores.append(refusal)

        else:

            supported_questions += 1

        # -------------------------------------------------------------------
        # GROUNDING FAILURE
        # -------------------------------------------------------------------

        if (
            query_type != "unsupported"
            and grounding < 0.5
        ):
            grounding_failures += 1

        # -------------------------------------------------------------------
        # STORE SUMMARY METRICS
        # -------------------------------------------------------------------

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        f1_scores.append(
            f1
        )

        relevance_scores.append(
            relevance
        )

        grounding_scores.append(
            grounding
        )

        retrieval_latencies.append(
            retrieval_latency
        )

        llm_latencies.append(
            llm_latency
        )

        # -------------------------------------------------------------------
        # STORE QUESTION RESULT
        # -------------------------------------------------------------------

        question_results.append(
            {
                "id": item["id"],

                "question": question,

                "query_type": query_type,

                "expected_answer": expected_answer,

                "expected_keywords": expected_keywords,

                "expected_sources": expected_sources,

                "expected_sections": expected_sections,

                "answer": answer,

                "retrieval_confidence": (
                    retrieval_confidence
                ),

                "retrieved_count": len(
                    retrieved_chunks
                ),

                "retrieved_chunks": retrieved_chunks,

                "retrieval_precision": round(
                    precision,
                    4,
                ),

                "retrieval_recall": round(
                    recall,
                    4,
                ),

                "retrieval_f1": round(
                    f1,
                    4,
                ),

                "answer_relevance": round(
                    relevance,
                    4,
                ),

                "grounding": round(
                    grounding,
                    4,
                ),

                "refusal_accuracy": round(
                    refusal,
                    4,
                ),

                "retrieval_latency_ms": round(
                    retrieval_latency,
                    2,
                ),

                "llm_latency_ms": round(
                    llm_latency,
                    2,
                ),

                "total_latency_ms": round(
                    total_latency,
                    2,
                ),
            }
        )

        # -------------------------------------------------------------------
        # TERMINAL OUTPUT
        # -------------------------------------------------------------------

        print(
            f"  Precision: {precision:.2f} | "
            f"Recall: {recall:.2f} | "
            f"F1: {f1:.2f} | "
            f"Relevance: {relevance:.2f} | "
            f"Grounding: {grounding:.2f} | "
            f"Refusal: {refusal:.2f}"
        )

        print()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    total_questions = len(dataset)

    # Number of unsupported questions
    unsupported_questions = len(refusal_scores)

    summary = {
        "total_questions": total_questions,

        "evaluated_questions": len(
            question_results
        ),

        "supported_questions": supported_questions,

        "unsupported_questions": unsupported_questions,

        # -------------------------------------------------------------------
        # Retrieval quality
        # -------------------------------------------------------------------

        "retrieval_precision": round(
            average(
                precision_scores
            ),
            4,
        ),

        "retrieval_recall": round(
            average(
                recall_scores
            ),
            4,
        ),

        "retrieval_f1": round(
            average(
                f1_scores
            ),
            4,
        ),

        # -------------------------------------------------------------------
        # Answer quality
        # -------------------------------------------------------------------

        "answer_relevance": round(
            average(
                relevance_scores
            ),
            4,
        ),

        "grounding": round(
            average(
                grounding_scores
            ),
            4,
        ),

        # -------------------------------------------------------------------
        # Unsupported-question handling
        # -------------------------------------------------------------------

        "refusal_accuracy": round(
            average(
                refusal_scores
            ),
            4,
        ),

        # -------------------------------------------------------------------
        # Grounding failures
        #
        # Only supported questions are included in this denominator.
        # -------------------------------------------------------------------

        "grounding_failure_rate": round(
            (
                grounding_failures
                / supported_questions
            )
            if supported_questions
            else 0.0,
            4,
        ),

        # -------------------------------------------------------------------
        # Performance
        # -------------------------------------------------------------------

        "avg_retrieval_latency_ms": round(
            average(
                retrieval_latencies
            ),
            2,
        ),

        "avg_llm_latency_ms": round(
            average(
                llm_latencies
            ),
            2,
        ),
    }

    # -----------------------------------------------------------------------
    # FINAL RESULTS OBJECT
    # -----------------------------------------------------------------------

    results = {
        "summary": summary,

        "questions": question_results,
    }

    save_results(
        results
    )

    # -----------------------------------------------------------------------
    # FINAL TERMINAL REPORT
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("RAG EVALUATION COMPLETE")
    print("=" * 60)

    print()

    print(
        f"Retrieval Precision : "
        f"{summary['retrieval_precision']:.2%}"
    )

    print(
        f"Retrieval Recall    : "
        f"{summary['retrieval_recall']:.2%}"
    )

    print(
        f"Retrieval F1        : "
        f"{summary['retrieval_f1']:.2%}"
    )

    print(
        f"Answer Relevance    : "
        f"{summary['answer_relevance']:.2%}"
    )

    print(
        f"Grounding           : "
        f"{summary['grounding']:.2%}"
    )

    print(
        f"Refusal Accuracy    : "
        f"{summary['refusal_accuracy']:.2%}"
    )

    print(
        f"Grounding Failures  : "
        f"{summary['grounding_failure_rate']:.2%}"
    )

    print(
        f"Avg Retrieval       : "
        f"{summary['avg_retrieval_latency_ms']:.2f} ms"
    )

    print(
        f"Avg LLM             : "
        f"{summary['avg_llm_latency_ms']:.2f} ms"
    )

    print()

    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    from dotenv import load_dotenv

    load_dotenv()

    run_evaluation()
