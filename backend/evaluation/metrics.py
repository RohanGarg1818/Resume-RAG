
from typing import List, Dict, Any
import re


# ---------------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize text for evaluation.

    Keeps useful characters such as:
    - letters
    - numbers
    - email symbols
    - programming symbols
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+#.@\- ]+",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# METADATA HELPERS
# ---------------------------------------------------------------------------

def get_chunk_metadata(chunk: Dict[str, Any]) -> str:
    """
    Convert all chunk metadata into normalized searchable text.

    This allows the evaluator to work with different metadata structures.
    """

    metadata = chunk.get("metadata", {})

    if isinstance(metadata, dict):
        values = []

        for key, value in metadata.items():
            values.append(str(key))
            values.append(str(value))

        return normalize(" ".join(values))

    return normalize(str(metadata))


def metadata_matches(
    chunk: Dict[str, Any],
    expected_sources: List[str],
    expected_sections: List[str],
) -> bool:
    """
    Determine whether a retrieved chunk belongs to an expected
    source or section.

    Matching is case-insensitive and normalization-aware.
    """

    metadata = get_chunk_metadata(chunk)

    if not metadata:
        return False

    # Check expected sources
    for source in expected_sources:
        normalized_source = normalize(source)

        if normalized_source and normalized_source in metadata:
            return True

    # Check expected sections
    for section in expected_sections:
        normalized_section = normalize(section)

        if normalized_section and normalized_section in metadata:
            return True

    return False


# ---------------------------------------------------------------------------
# KEYWORD MATCHING
# ---------------------------------------------------------------------------

def keyword_match_score(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Measure how many expected keywords appear in the answer.
    """

    if not expected_keywords:
        return 1.0

    normalized_answer = normalize(answer)

    matched = 0

    for keyword in expected_keywords:
        normalized_keyword = normalize(keyword)

        if (
            normalized_keyword
            and normalized_keyword in normalized_answer
        ):
            matched += 1

    return matched / len(expected_keywords)


# ---------------------------------------------------------------------------
# RETRIEVAL PRECISION
# ---------------------------------------------------------------------------

def retrieval_precision(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    expected_sources: List[str] | None = None,
    expected_sections: List[str] | None = None,
) -> float:
    """
    Source-aware retrieval precision.

    A chunk is considered relevant when:

    1. Its metadata matches an expected source or section, OR
    2. Its text/metadata contains an expected keyword.

    Metadata is preferred because it provides stronger evidence
    that the correct document/section was retrieved.
    """

    if not retrieved_chunks:
        return 0.0

    expected_sources = expected_sources or []
    expected_sections = expected_sections or []

    # Unsupported questions should not be penalized for retrieving
    # nothing useful.
    if (
        not expected_keywords
        and not expected_sources
        and not expected_sections
    ):
        return 1.0

    normalized_keywords = [
        normalize(keyword)
        for keyword in expected_keywords
        if normalize(keyword)
    ]

    relevant_chunks = 0

    for chunk in retrieved_chunks:

        # ---------------------------------------------------------------
        # 1. Strong signal: source / section metadata
        # ---------------------------------------------------------------

        if metadata_matches(
            chunk,
            expected_sources,
            expected_sections,
        ):
            relevant_chunks += 1
            continue

        # ---------------------------------------------------------------
        # 2. Secondary signal: keyword presence
        # ---------------------------------------------------------------

        document = normalize(
            chunk.get("text", "")
        )

        metadata = get_chunk_metadata(chunk)

        content = f"{document} {metadata}"

        if any(
            keyword in content
            for keyword in normalized_keywords
        ):
            relevant_chunks += 1

    return relevant_chunks / len(retrieved_chunks)


# ---------------------------------------------------------------------------
# RETRIEVAL RECALL
# ---------------------------------------------------------------------------

def retrieval_recall(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    expected_sources: List[str] | None = None,
    expected_sections: List[str] | None = None,
) -> float:
    """
    Source-aware retrieval recall.

    Recall checks whether the retrieved context contains the
    information/source/section required by the evaluation question.

    A keyword is considered retrieved when it appears in the retrieved
    text or metadata.

    Expected sources and sections are also checked independently.
    """

    expected_sources = expected_sources or []
    expected_sections = expected_sections or []

    if (
        not expected_keywords
        and not expected_sources
        and not expected_sections
    ):
        return 1.0

    if not retrieved_chunks:
        return 0.0

    # -------------------------------------------------------------------
    # Build retrieved context
    # -------------------------------------------------------------------

    retrieved_text = " ".join(
        normalize(
            chunk.get("text", "")
        )
        for chunk in retrieved_chunks
    )

    retrieved_metadata = " ".join(
        get_chunk_metadata(chunk)
        for chunk in retrieved_chunks
    )

    retrieved_content = (
        f"{retrieved_text} {retrieved_metadata}"
    )

    # -------------------------------------------------------------------
    # Keyword recall
    # -------------------------------------------------------------------

    keyword_scores = []

    for keyword in expected_keywords:

        normalized_keyword = normalize(keyword)

        if not normalized_keyword:
            continue

        keyword_scores.append(
            1.0
            if normalized_keyword in retrieved_content
            else 0.0
        )

    keyword_recall = (
        sum(keyword_scores) / len(keyword_scores)
        if keyword_scores
        else 1.0
    )

    # -------------------------------------------------------------------
    # Source recall
    # -------------------------------------------------------------------

    source_scores = []

    for source in expected_sources:

        normalized_source = normalize(source)

        if not normalized_source:
            continue

        found = any(
            normalized_source in get_chunk_metadata(chunk)
            for chunk in retrieved_chunks
        )

        source_scores.append(
            1.0 if found else 0.0
        )

    source_recall = (
        sum(source_scores) / len(source_scores)
        if source_scores
        else 1.0
    )

    # -------------------------------------------------------------------
    # Section recall
    # -------------------------------------------------------------------

    section_scores = []

    for section in expected_sections:

        normalized_section = normalize(section)

        if not normalized_section:
            continue

        found = any(
            normalized_section in get_chunk_metadata(chunk)
            for chunk in retrieved_chunks
        )

        section_scores.append(
            1.0 if found else 0.0
        )

    section_recall = (
        sum(section_scores) / len(section_scores)
        if section_scores
        else 1.0
    )

    # -------------------------------------------------------------------
    # Combine signals
    #
    # Source and section metadata are stronger signals than keywords.
    # -------------------------------------------------------------------

    components = []

    if expected_keywords:
        components.append(keyword_recall)

    if expected_sources:
        components.append(source_recall)

    if expected_sections:
        components.append(section_recall)

    return sum(components) / len(components)


# ---------------------------------------------------------------------------
# RETRIEVAL F1
# ---------------------------------------------------------------------------

def retrieval_f1(
    precision: float,
    recall: float,
) -> float:
    """
    Harmonic mean of retrieval precision and recall.
    """

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ---------------------------------------------------------------------------
# ANSWER RELEVANCE
# ---------------------------------------------------------------------------

def answer_relevance(
    answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Estimate answer relevance using expected keyword coverage.

    This is still a lightweight heuristic.
    """

    return keyword_match_score(
        answer,
        expected_keywords,
    )


# ---------------------------------------------------------------------------
# GROUNDING
# ---------------------------------------------------------------------------

def grounding_score(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Estimate grounding using token overlap between the generated
    answer and retrieved context.

    This is a lightweight heuristic, NOT a true faithfulness metric.
    """

    if not answer:
        return 0.0

    if not retrieved_chunks:
        return 0.0

    context = " ".join(
        chunk.get("text", "")
        for chunk in retrieved_chunks
    )

    normalized_answer = normalize(answer)
    normalized_context = normalize(context)

    answer_tokens = set(
        normalized_answer.split()
    )

    context_tokens = set(
        normalized_context.split()
    )

    if not answer_tokens:
        return 0.0

    overlap = (
        answer_tokens.intersection(
            context_tokens
        )
    )

    return len(overlap) / len(answer_tokens)


# ---------------------------------------------------------------------------
# REFUSAL ACCURACY
# ---------------------------------------------------------------------------

NOT_FOUND_ANSWER = (
    "I don't have that information in the provided profile."
)


def refusal_accuracy(
    answer: str,
    query_type: str,
) -> float:
    """
    Measure whether unsupported questions were correctly refused.

    Supported questions are not scored using this metric.
    """

    if query_type != "unsupported":
        return 1.0

    normalized_answer = normalize(answer)

    normalized_refusal = normalize(
        NOT_FOUND_ANSWER
    )

    return (
        1.0
        if normalized_answer == normalized_refusal
        else 0.0
    )


# ---------------------------------------------------------------------------
# AVERAGE
# ---------------------------------------------------------------------------

def average(values: List[float]) -> float:
    """
    Return the arithmetic mean.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)
