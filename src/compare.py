"""Candidate comparison engine.

Pipeline:
    job requirements
      -> extract structured requirements (LLM)
      -> for each requirement:
           retrieve evidence from Candidate A (permanent collection)
           retrieve evidence from Candidate B (session collection)
           score match deterministically
      -> aggregate scores
      -> LLM explains comparison (evidence-grounded)
      -> structured result

Candidate A lives in the permanent resume collection.
Candidate B lives in an isolated session collection.
Evidence is never cross-attributed.
"""

import gc
import json
import logging
import os
import re
from typing import Any

from groq import Groq

from src.embeddings import embed_query
from src.reranker import rerank
from src.vector_store import (
    get_collection,
    get_session_collection,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"

CANDIDATE_A_NAME = "Rohan Garg"
CANDIDATE_B_FALLBACK = "The Other Candidate"

# Owner identity constants — the single source of truth for document ownership.
OWNER_ID_ROHAN = "rohan"
OWNER_ID_UPLOADED = "uploaded_candidate"
OWNER_TYPE_PRIMARY = "primary"
OWNER_TYPE_CANDIDATE = "candidate"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Requirement extraction
# ---------------------------------------------------------------------------

REQUIREMENT_EXTRACTION_PROMPT = """\
Extract a structured list of hiring requirements from the job description below.

Categorize each requirement as:
- "required": must-have skills/qualifications
- "preferred": nice-to-have skills/qualifications

For each requirement, provide:
- "text": the requirement as a concise phrase
- "category": "required" or "preferred"
- "type": one of "skill", "experience", "education", "responsibility", "soft_skill"

Return a JSON array. Do not include anything outside the JSON array.
Example:
[
  {{"text": "3+ years of experience with Python", "category": "required", "type": "experience"}},
  {{"text": "Experience with AWS or GCP", "category": "preferred", "type": "skill"}}
]

JOB DESCRIPTION:
{job_description}
"""


def extract_requirements(job_description: str) -> list[dict]:
    """Use LLM to extract structured requirements from job description."""

    client = _get_client()

    response = client.chat.completions.create(
        model=_get_model(),
        messages=[
            {
                "role": "user",
                "content": REQUIREMENT_EXTRACTION_PROMPT.format(
                    job_description=job_description
                ),
            }
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    text = (response.choices[0].message.content or "").strip()
    del response
    gc.collect()

    # Extract JSON from possible markdown code block
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not json_match:
        logger.warning("Failed to extract JSON from LLM response: %s", text[:200])
        return [{"text": job_description, "category": "required", "type": "skill"}]

    try:
        requirements = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse requirements JSON")
        return [{"text": job_description, "category": "required", "type": "skill"}]

    return requirements if isinstance(requirements, list) else []


# ---------------------------------------------------------------------------
# Evidence retrieval for a single requirement
# ---------------------------------------------------------------------------

def _retrieve_evidence(
    query: str,
    collection,
    top_k: int = 6,
) -> list[dict]:
    """Retrieve evidence chunks from a specific ChromaDB collection."""

    try:
        count = collection.count()
    except Exception:
        return []

    if count == 0:
        return []

    try:
        embedding = embed_query(query)
    except Exception:
        logger.warning("Failed to embed query: %s", query[:80])
        return []

    n_results = min(top_k, count)

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("ChromaDB query failed: %s", exc)
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "metadata": meta if isinstance(meta, dict) else {},
            "similarity": 1.0 - dist,
        })

    return chunks


def _retrieve_evidence_reranked(
    query: str,
    collection,
    top_k: int = 6,
) -> list[dict]:
    """Retrieve and rerank evidence from a collection."""

    candidate_count = min(top_k * 4, 24)
    candidates = _retrieve_evidence(query, collection, top_k=candidate_count)

    if not candidates:
        return []

    reranked = rerank(query=query, chunks=candidates, top_k=top_k)
    return reranked


# ---------------------------------------------------------------------------
# Deterministic requirement matching
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but is are was were be been being to of in on for with "
    "from by as at it this that these those has have had does do did can could "
    "would should will may might not no yes the their his her its".split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase token set excluding stopwords and short tokens."""
    tokens = set(re.findall(r"[a-z0-9+#.]+", text.lower()))
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def _semantic_overlap(query_tokens: set[str], text_tokens: set[str]) -> float:
    """Jaccard-like overlap ratio."""
    if not query_tokens:
        return 0.0
    intersection = query_tokens & text_tokens
    return len(intersection) / len(query_tokens)


def _match_requirement(
    requirement: dict,
    chunks: list[dict],
) -> dict:
    """Score how well retrieved chunks match a single requirement.

    Returns a dict with met (bool), evidence, confidence, score, and
    evidence chunks.  This is deterministic -- no LLM call.
    """

    req_text = requirement.get("text", "")
    req_tokens = _tokenize(req_text)

    if not chunks:
        return {
            "met": False,
            "evidence": "",
            "confidence": "none",
            "score": 0.0,
            "evidence_chunks": [],
        }

    best_score = 0.0
    best_evidence = ""
    best_chunks: list[dict] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        text_tokens = _tokenize(text)

        # Semantic similarity from retrieval
        semantic = chunk.get("similarity", 0.0)

        # Lexical overlap
        overlap = _semantic_overlap(req_tokens, text_tokens)

        # Combined score
        score = 0.5 * semantic + 0.5 * overlap

        if score > best_score:
            best_score = score
            best_evidence = text
            best_chunks = [chunk]
        elif score > best_score * 0.8:
            best_chunks.append(chunk)

    # Confidence thresholds
    if best_score >= 0.45:
        confidence = "high"
    elif best_score >= 0.30:
        confidence = "medium"
    elif best_score >= 0.15:
        confidence = "low"
    else:
        confidence = "none"

    met = confidence in ("high", "medium")

    # Combine top evidence texts (max 3 chunks)
    evidence_texts = [c.get("text", "") for c in best_chunks[:3]]
    evidence = " | ".join(t for t in evidence_texts if t)

    return {
        "met": met,
        "evidence": evidence,
        "confidence": confidence,
        "score": round(best_score, 4),
        "evidence_chunks": best_chunks,
    }


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def _compute_category_scores(
    requirements: list[dict],
    matches: dict[str, dict],
) -> dict[str, float]:
    """Compute normalized scores per requirement category."""

    categories: dict[str, list[dict]] = {}
    for req in requirements:
        cat = req.get("type", "skill")
        categories.setdefault(cat, []).append(req)

    scores: dict[str, float] = {}
    for cat, reqs in categories.items():
        cat_matches = []
        for req in reqs:
            key = req.get("text", "")
            m = matches.get(key, {})
            cat_matches.append(m.get("score", 0.0))
        scores[cat] = sum(cat_matches) / len(cat_matches) if cat_matches else 0.0

    return scores


def _compute_overall_score(
    requirements: list[dict],
    matches: dict[str, dict],
) -> float:
    """Weighted overall score.

    Required skills are weighted more heavily than preferred.
    """

    if not requirements:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for req in requirements:
        weight = 1.5 if req.get("category") == "required" else 1.0
        key = req.get("text", "")
        m = matches.get(key, {})
        score = m.get("score", 0.0)
        weighted_sum += weight * score
        total_weight += weight

    return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0


# ---------------------------------------------------------------------------
# Candidate name extraction
# ---------------------------------------------------------------------------

# Words that indicate a line is a job title, location, section header,
# or other non-name text.  A line containing any of these (when it also
# has 2-4 words) is almost certainly not a person's name.
_TITLE_LOCATION_WORDS = frozenset(
    # Job titles / roles
    "executive engineer developer manager director analyst intern associate "
    "senior junior lead principal staff coordinator specialist consultant "
    "architect supervisor officer technician scientist professor lecturer "
    "trainee adviser mentor founder co-founder head vice "
    # Locations / geography
    "delhi bangalore mumbai chennai hyderabad pune kolkata india usa us "
    "uk canada australia germany france singapore japan china london "
    "new york california texas toronto ontario paris tokyo berlin "
    # Degree / education
    "bachelor master doctorate phd mba bca mca btech mtech bsc msc "
    "graduate postgraduate undergraduate diploma"
    # Section headers
    "education skills experience projects achievements certifications "
    "summary objective profile about contact references hobbies "
    "interests languages publications patents awards"
    .split()
)

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

_PHONE_PATTERN = re.compile(
    r"[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,}"
)

_LINKEDIN_PATTERN = re.compile(
    r"linkedin\.com/in/[a-zA-Z0-9_-]+",
    re.IGNORECASE,
)

_SECTION_HEADING_RE = re.compile(
    r"^[\s#*>\-]*(?:education|skills|experience|projects|achievements|"
    r"certifications|summary|objective|profile|about|contact|references|"
    r"hobbies|interests|languages|publications|patents|awards|"
    r"technical skills|work experience|internships?|education "
    r"(?:background|history))[\s:]*$",
    re.IGNORECASE,
)


def _is_likely_name(line: str) -> str | None:
    """Return the line if it looks like a person's name, else None.

    A valid resume-header name is typically 2-4 alphabetic words that
    do NOT contain job titles, locations, section headers, or contact info.
    """
    line = line.strip()
    if not line or len(line) > 60:
        return None

    # Must be only alphabetic characters and spaces (no digits, symbols, etc.)
    if not re.fullmatch(r"[A-Za-z\s.'-]+", line):
        return None

    words = line.split()
    if not (2 <= len(words) <= 4):
        return None

    # Each word must start with an uppercase letter
    if not all(w[0].isupper() for w in words):
        return None

    # Reject if any word is a known title/location/section word
    lower_words = {w.lower() for w in words}
    if lower_words & _TITLE_LOCATION_WORDS:
        return None

    # Reject common section headings and false positives
    if _SECTION_HEADING_RE.match(line):
        return None

    return line


def extract_candidate_name(texts: list[str], fallback: str = CANDIDATE_B_FALLBACK) -> str:
    """Extract a person's name from a list of document text chunks.

    Examines the first few chunks (the header region of a resume) and
    returns the first line that looks like a person's name, or *fallback*.
    """
    for doc in texts[:3]:
        for line in doc.split("\n")[:8]:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if _EMAIL_PATTERN.search(line_stripped):
                continue
            if _PHONE_PATTERN.search(line_stripped):
                continue
            if _LINKEDIN_PATTERN.search(line_stripped):
                continue
            if _SECTION_HEADING_RE.match(line_stripped):
                continue
            name = _is_likely_name(line_stripped)
            if name:
                return name
            if re.fullmatch(r"[A-Za-z\s.'/-]+", line_stripped):
                break
    return fallback


def _extract_candidate_b_name(session_id: str) -> str:
    """Extract Candidate B's full name from the session collection.

    Reads the first documents from the session collection (which contain
    the resume header/contact section) and looks for a name on the first
    meaningful line.

    Returns the extracted name or CANDIDATE_B_FALLBACK.
    """
    try:
        collection = get_session_collection(session_id)
        count = collection.count()
    except Exception:
        return CANDIDATE_B_FALLBACK

    if count == 0:
        return CANDIDATE_B_FALLBACK

    try:
        results = collection.get(
            include=["documents", "metadatas"],
            limit=min(count, 5),
        )
    except Exception:
        return CANDIDATE_B_FALLBACK

    documents = results.get("documents", [])
    if not documents:
        return CANDIDATE_B_FALLBACK

    return extract_candidate_name(documents)


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def compare_candidates(
    job_description: str,
    session_id: str,
    candidate_a_name: str = CANDIDATE_A_NAME,
    candidate_b_name: str | None = None,
) -> dict:
    """Full comparison pipeline.

    Args:
        job_description: Recruiter-supplied job requirements text.
        session_id: RAG session containing Candidate B's resume.
        candidate_a_name: Always "Rohan Garg" (the permanent resume).
        candidate_b_name: Extracted from resume; falls back to CANDIDATE_B_FALLBACK.

    Returns:
        Structured comparison result dict.
    """

    # ---------------------------------------------------------------
    # 1. Resolve Candidate B name from session collection
    # ---------------------------------------------------------------

    if candidate_b_name is None:
        candidate_b_name = _extract_candidate_b_name(session_id)

    logger.info(
        "Comparing %s vs %s",
        candidate_a_name,
        candidate_b_name,
    )

    # ---------------------------------------------------------------
    # 2. Extract requirements
    # ---------------------------------------------------------------

    requirements = extract_requirements(job_description)

    if not requirements:
        raise ValueError("Could not extract requirements from the job description.")

    logger.info("Extracted %d requirements from job description", len(requirements))

    # ---------------------------------------------------------------
    # 3. Get collections
    # ---------------------------------------------------------------

    collection_a = get_collection()  # permanent resume collection
    collection_b = get_session_collection(session_id)  # Candidate B session

    # ---------------------------------------------------------------
    # 4. Retrieve and match evidence for each requirement
    # ---------------------------------------------------------------

    matches_a: dict[str, dict] = {}
    matches_b: dict[str, dict] = {}

    for req in requirements:
        req_text = req.get("text", "")
        query = f"{req_text} {job_description[:200]}"

        # Retrieve from Candidate A (permanent)
        evidence_a = _retrieve_evidence_reranked(query, collection_a, top_k=6)
        matches_a[req_text] = _match_requirement(req, evidence_a)

        # Retrieve from Candidate B (session)
        evidence_b = _retrieve_evidence_reranked(query, collection_b, top_k=6)
        matches_b[req_text] = _match_requirement(req, evidence_b)

    # ---------------------------------------------------------------
    # 5. Compute scores
    # ---------------------------------------------------------------

    score_a_overall = _compute_overall_score(requirements, matches_a)
    score_b_overall = _compute_overall_score(requirements, matches_b)

    cat_a = _compute_category_scores(requirements, matches_a)
    cat_b = _compute_category_scores(requirements, matches_b)

    # ---------------------------------------------------------------
    # 6. Build requirement comparisons
    # ---------------------------------------------------------------

    requirement_comparisons = []
    for req in requirements:
        key = req.get("text", "")
        m_a = matches_a.get(key, {})
        m_b = matches_b.get(key, {})

        if m_a.get("score", 0) > m_b.get("score", 0):
            better = candidate_a_name
        elif m_b.get("score", 0) > m_a.get("score", 0):
            better = candidate_b_name
        else:
            better = "tie"

        requirement_comparisons.append({
            "requirement": key,
            "candidate_a": {
                "requirement": key,
                "met": m_a.get("met", False),
                "evidence": m_a.get("evidence", ""),
                "confidence": m_a.get("confidence", "none"),
                "candidate": candidate_a_name,
            },
            "candidate_b": {
                "requirement": key,
                "met": m_b.get("met", False),
                "evidence": m_b.get("evidence", ""),
                "confidence": m_b.get("confidence", "none"),
                "candidate": candidate_b_name,
            },
            "better_match": better,
        })

    # ---------------------------------------------------------------
    # 7. Collect strengths, gaps, unique skills
    # ---------------------------------------------------------------

    strengths_a, gaps_a = [], []
    strengths_b, gaps_b = [], []
    unique_a, unique_b = [], []
    shared = []

    for req in requirements:
        key = req.get("text", "")
        m_a = matches_a.get(key, {})
        m_b = matches_b.get(key, {})
        category = req.get("type", "skill")
        req_label = f"{req.get('text', '')} ({category})"

        a_met = m_a.get("met", False)
        b_met = m_b.get("met", False)

        if a_met:
            strengths_a.append(req_label)
        else:
            gaps_a.append(req_label)

        if b_met:
            strengths_b.append(req_label)
        else:
            gaps_b.append(req_label)

        if a_met and b_met:
            shared.append(req_label)
        elif a_met and not b_met:
            unique_a.append(req_label)
        elif b_met and not a_met:
            unique_b.append(req_label)

    # ---------------------------------------------------------------
    # 8. Identify missing / ambiguous requirements
    # ---------------------------------------------------------------

    missing = []
    ambiguous = []
    for req in requirements:
        key = req.get("text", "")
        m_a = matches_a.get(key, {})
        m_b = matches_b.get(key, {})
        if m_a.get("confidence") == "none" and m_b.get("confidence") == "none":
            missing.append(req.get("text", ""))
        elif m_a.get("confidence") in ("low",) or m_b.get("confidence") in ("low",):
            ambiguous.append(req.get("text", ""))

    # ---------------------------------------------------------------
    # 9. Gather evidence for each candidate
    # ---------------------------------------------------------------

    evidence_a_all = []
    evidence_b_all = []
    seen_texts_a: set[str] = set()
    seen_texts_b: set[str] = set()

    for req in requirements:
        key = req.get("text", "")
        for chunk in matches_a.get(key, {}).get("evidence_chunks", []):
            text = chunk.get("text", "")
            if text and text not in seen_texts_a:
                seen_texts_a.add(text)
                evidence_a_all.append({
                    "text": text,
                    "section": chunk.get("metadata", {}).get("section", ""),
                    "relevance": round(chunk.get("similarity", 0.0), 4),
                })
        for chunk in matches_b.get(key, {}).get("evidence_chunks", []):
            text = chunk.get("text", "")
            if text and text not in seen_texts_b:
                seen_texts_b.add(text)
                evidence_b_all.append({
                    "text": text,
                    "section": chunk.get("metadata", {}).get("section", ""),
                    "relevance": round(chunk.get("similarity", 0.0), 4),
                })

    # ---------------------------------------------------------------
    # 10. Determine recommendation
    # ---------------------------------------------------------------

    score_diff = score_a_overall - score_b_overall
    is_tie = abs(score_diff) < 0.05

    if is_tie:
        recommendation = "Tie / Both suitable"
    elif score_a_overall > score_b_overall:
        recommendation = candidate_a_name
    else:
        recommendation = candidate_b_name

    # ---------------------------------------------------------------
    # 11. Build result
    # ---------------------------------------------------------------

    result = {
        "recommendation": recommendation,
        "reasoning": "",  # filled by LLM in service layer
        "score_a": {
            "overall": score_a_overall,
            "required_skills": cat_a.get("skill", 0.0),
            "experience": cat_a.get("experience", 0.0),
            "projects": cat_a.get("responsibility", 0.0),
            "preferred_skills": cat_a.get("skill", 0.0),
            "education": cat_a.get("education", 0.0),
        },
        "score_b": {
            "overall": score_b_overall,
            "required_skills": cat_b.get("skill", 0.0),
            "experience": cat_b.get("experience", 0.0),
            "projects": cat_b.get("responsibility", 0.0),
            "preferred_skills": cat_b.get("skill", 0.0),
            "education": cat_b.get("education", 0.0),
        },
        "requirement_comparisons": requirement_comparisons,
        "candidate_a": {
            "name": candidate_a_name,
            "score": {
                "overall": score_a_overall,
                "required_skills": cat_a.get("skill", 0.0),
                "experience": cat_a.get("experience", 0.0),
                "projects": cat_a.get("responsibility", 0.0),
                "preferred_skills": cat_a.get("skill", 0.0),
                "education": cat_a.get("education", 0.0),
            },
            "strengths": strengths_a,
            "gaps": gaps_a,
            "evidence": evidence_a_all[:20],
            "unique_skills": unique_a,
            "shared_strengths": shared,
        },
        "candidate_b": {
            "name": candidate_b_name,
            "score": {
                "overall": score_b_overall,
                "required_skills": cat_b.get("skill", 0.0),
                "experience": cat_b.get("experience", 0.0),
                "projects": cat_b.get("responsibility", 0.0),
                "preferred_skills": cat_b.get("skill", 0.0),
                "education": cat_b.get("education", 0.0),
            },
            "strengths": strengths_b,
            "gaps": gaps_b,
            "evidence": evidence_b_all[:20],
            "unique_skills": unique_b,
            "shared_strengths": shared,
        },
        "missing_requirements": missing,
        "ambiguous_requirements": ambiguous,
        "is_tie": is_tie,
        "_debug_requirements": requirements,
    }

    return result


# ---------------------------------------------------------------------------
# LLM explanation (called after deterministic scoring)
# ---------------------------------------------------------------------------

COMPARISON_EXPLANATION_PROMPT = """\
You are a recruiting analyst. You are comparing two candidates for a job.

The scores and requirement matches below are DETERMINISTIC and GROUNDED in
actual resume evidence. You must explain and summarize them -- do NOT override,
re-rank, or contradict the scores.

Rules:
- Never invent skills, experience, education, achievements, or qualifications.
- Every conclusion must be supported by the evidence provided.
- Distinguish "not found in the resume" from "candidate does not have this skill."
- If evidence is insufficient, say "Unknown / Insufficient evidence."
- Do not allow {name_a}'s evidence to be attributed to {name_b} or vice versa.
- Project/work evidence is stronger than a skill appearing only in a skills list.
- If both are similarly qualified, you may say "Tie / Both suitable."

JOB DESCRIPTION:
{job_description}

{name_a} SCORE: {score_a}
{name_b} SCORE: {score_b}

REQUIREMENT-BY-REQUIREMENT MATCHES:
{requirement_details}

{name_a} STRENGTHS: {strengths_a}
{name_a} GAPS: {gaps_a}

{name_b} STRENGTHS: {strengths_b}
{name_b} GAPS: {gaps_b}

{name_a} EVIDENCE:
{evidence_a}

{name_b} EVIDENCE:
{evidence_b}

Provide:
1. A concise "reasoning" paragraph (3-5 sentences) explaining why one candidate
   is recommended over the other, grounded in the evidence above. Reference
   specific requirements and evidence.

Return JSON:
{{
  "reasoning": "Your explanation here."
}}
"""


def generate_comparison_explanation(
    job_description: str,
    comparison_result: dict,
) -> str:
    """Use LLM to explain the comparison result.

    The LLM is given the deterministic scores and evidence and asked to
    explain them -- not to re-evaluate or re-rank.
    """

    score_a = comparison_result["score_a"]["overall"]
    score_b = comparison_result["score_b"]["overall"]
    name_a = comparison_result["candidate_a"]["name"]
    name_b = comparison_result["candidate_b"]["name"]

    # Build requirement details string
    req_lines = []
    for rc in comparison_result.get("requirement_comparisons", []):
        a_met = "Yes" if rc["candidate_a"]["met"] else "No"
        b_met = "Yes" if rc["candidate_b"]["met"] else "No"
        req_lines.append(
            f"- {rc['requirement']}: {name_a}={a_met}, {name_b}={b_met}"
        )
    requirement_details = "\n".join(req_lines) if req_lines else "(none)"

    strengths_a = ", ".join(comparison_result["candidate_a"]["strengths"][:5]) or "(none)"
    gaps_a = ", ".join(comparison_result["candidate_a"]["gaps"][:5]) or "(none)"
    strengths_b = ", ".join(comparison_result["candidate_b"]["strengths"][:5]) or "(none)"
    gaps_b = ", ".join(comparison_result["candidate_b"]["gaps"][:5]) or "(none)"

    # Limit evidence text to avoid token limits
    evidence_a_texts = [
        e["text"][:200] for e in comparison_result["candidate_a"]["evidence"][:6]
    ]
    evidence_b_texts = [
        e["text"][:200] for e in comparison_result["candidate_b"]["evidence"][:6]
    ]
    evidence_a = "\n".join(f"- {t}" for t in evidence_a_texts) or "(none)"
    evidence_b = "\n".join(f"- {t}" for t in evidence_b_texts) or "(none)"

    prompt = COMPARISON_EXPLANATION_PROMPT.format(
        job_description=job_description[:3000],
        score_a=f"{score_a:.2%}",
        score_b=f"{score_b:.2%}",
        name_a=name_a,
        name_b=name_b,
        requirement_details=requirement_details,
        strengths_a=strengths_a,
        gaps_a=gaps_a,
        strengths_b=strengths_b,
        gaps_b=gaps_b,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        text = (response.choices[0].message.content or "").strip()
        del response
        gc.collect()

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("reasoning", text)

        return text

    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        if comparison_result["is_tie"]:
            return (
                "Both candidates appear similarly qualified for this role. "
                "Review the requirement-by-requirement breakdown for details."
            )
        winner = comparison_result["recommendation"]
        return (
            f"{winner} is recommended based on the deterministic scoring "
            "of resume evidence against the stated job requirements."
        )
