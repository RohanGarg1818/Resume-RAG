"""Section-aware parsing helpers for resumes.

A resume is not free-flowing prose: it is organised into headings
(Education, Projects, ...) and repeated entry patterns such as

    CollabSphere | Real-Time Messaging Platform | Next.js, Redis ...

The chunker uses the helpers here to tag every chunk with its section
and, where possible, the project or company it belongs to.
"""

import re

# Canonical section names used in ChromaDB metadata.
SECTIONS = [
    "header",
    "education",
    "skills",
    "experience",
    "projects",
    "achievements",
    "competitive programming",
    "hackathons",
    "certifications",
    "positions of responsibility",
    "other",
]

# Each canonical section maps to one or more accepted heading spellings.
# A heading must match a whole line (anchored) so body text mentioning
# the word "projects" is never mistaken for a heading.
_HEADING_VARIANTS: dict[str, tuple[str, ...]] = {
    "education": (r"education", r"academic(?:\s+background)?"),
    "skills": (
        r"technical\s+skills",
        r"skills(?:\s*&(?:amp;)?\s*(?:technologies|tools))?",
        r"technical\s+(?:knowledge|expertise)",
    ),
    "experience": (
        r"(?:work|professional|relevant)?\s*experience",
        r"internships?(?:\s+experience)?",
        r"employment(?:\s+history)?",
    ),
    "projects": (r"projects?", r"(?:major|selected|key)\s+projects"),
    "achievements": (
        r"achievements?(?:\s*&(?:amp;)?\s*(?:awards?|competitive\s+programming))*",
        r"awards(?:\s*&(?:amp;)?\s*achievements?)?",
        r"accomplishments",
    ),
    "competitive programming": (
        r"competitive\s+programming(?:\s*&(?:amp;)?\s*achievements?)*",
        r"(?:coding|programming)\s+(?:profile|profiles)",
    ),
    "hackathons": (r"hackathons?",),
    "certifications": (
        r"certifications?",
        r"licenses?\s*&(?:amp;)?\s*certifications?",
        r"courses?(?:\s*&(?:amp;)?\s*certifications?)?",
    ),
    "positions of responsibility": (
        r"positions?\s+of\s+responsibility",
        r"leadership(?:\s+(?:roles?|experience))?",
        r"extra[- ]?curricular(?:\s+activities)?",
    ),
}

_HEADING_RES: dict[str, tuple[re.Pattern, ...]] = {
    section: tuple(
        re.compile(rf"^\s*(?:{variant})\s*:?\s*$", re.IGNORECASE)
        for variant in variants
    )
    for section, variants in _HEADING_VARIANTS.items()
}

# "CollabSphere | Real-Time Messaging Platform | Next.js, Redis ..." style
_PROJECT_LINE = re.compile(r"^\s*([A-Z][\w .+#/&'-]{1,40})\s*\|[^|]+\|")

# Date ranges like "Jul 2025 - Jun 2029" mark organisation headers in the
# experience section (org name is the line just above them).
_DATE_RANGE = re.compile(
    r"^[A-Za-z]{3,9}\.?['’]?\s*\d{4}\s*(?:-|–|to)\s*(?:[A-Za-z]{3,9}\.?['’]?\s*\d{4}|\d{4}|Present|Current)",
    re.IGNORECASE,
)

MAX_HEADING_LENGTH = 60


def detect_section_heading(line: str) -> str | None:
    """Return the canonical section name if `line` itself is a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_LENGTH:
        return None
    for section, patterns in _HEADING_RES.items():
        if any(pattern.match(stripped) for pattern in patterns):
            return section
    return None


def detect_section(text: str) -> str:
    """Best-effort section guess for an arbitrary block of text.

    Used as a fallback when no explicit heading was seen (e.g. legacy
    chunk flows). Checks which canonical section word occurs earliest.
    """
    lower = text.lower()
    best: tuple[int, str] | None = None
    for section in SECTIONS:
        if section == "other":
            continue
        index = lower.find(section.split()[-1])
        if index != -1 and (best is None or index < best[0]):
            best = (index, section)
    return best[1] if best else "other"


def detect_project_name(line: str) -> str | None:
    """Extract a project title from 'Name | subtitle | tech | ...' lines."""
    match = _PROJECT_LINE.match(line)
    return match.group(1).strip() if match else None


def is_date_range(line: str) -> bool:
    """True for lines like 'Oct 2025 - Jun 2026' or '2023 - 2025'."""
    return bool(_DATE_RANGE.match(line.strip()))
