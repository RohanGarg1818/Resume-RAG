"""Section-aware chunking for resumes.

Instead of blindly slicing raw text every N characters, the pipeline is:

    pages (from extract.py)
      -> walk line by line, tracking the current section
         and current project/company entry
      -> group lines into coherent blocks
      -> split oversized blocks at line boundaries with overlap

Every chunk carries metadata:

    {"source": "resume", "page": 2, "section": "projects",
     "project": "JugaadHub"}            # when detectable
    {"source": "resume", "page": 1, "section": "experience",
     "company": "IEEE", "role": "Tech Team Member"}
"""

import hashlib
import re
from typing import Any

from src.extract import extract_pages, PageDict
from src.section_parser import (
    detect_project_name,
    detect_section,
    detect_section_heading,
    is_date_range,
)

Chunk = dict[str, Any]  # {"text": str, "metadata": dict}

DEFAULT_CHUNK_SIZE = 700   # characters per chunk target
DEFAULT_OVERLAP = 100      # characters repeated between consecutive chunks

_BULLET_PREFIXES = ("•", "-", "*", "◦", "‣")

# Owner identity for the permanent resume (Rohan Garg).
_OWNER_ID = "rohan"
_OWNER_NAME = "Rohan Garg"
_OWNER_TYPE = "primary"


def build_blocks(pages: list[PageDict]) -> list[dict[str, Any]]:
    """Group resume lines into blocks sharing one section/entity context.

    Returns a list of:
        {"section": str, "entity": dict, "items": [(page_number, line), ...]}

    Heuristics (tuned for common resume layouts, including this repo's
    data/Resume.pdf):
      * a whole-line heading switches the current section,
      * 'Name | subtitle | tech' lines start a new project entity,
      * in experience-like sections the line above a date range is the
        company and the first short non-bullet line below it is the role.
    """
    blocks: list[dict[str, Any]] = []

    def flush(section: str, entity: dict, items: list[tuple[int, str]]) -> None:
        if items:
            blocks.append({"section": section, "entity": dict(entity), "items": items})

    section = "header"
    entity: dict[str, Any] = {}
    items: list[tuple[int, str]] = []
    carry: list[tuple[int, str]] = []  # lines awaiting a company entity
    pending_org: str | None = None
    awaiting_role = False

    def flush_items() -> None:
        """Emit the current entity block (carry is handled separately)."""
        nonlocal items
        if items:
            blocks.append({"section": section, "entity": dict(entity), "items": items})
            items = []

    def flush_all() -> None:
        """Emit everything: current entity block plus any carried lines."""
        nonlocal carry
        flush_items()
        if carry:
            blocks.append({"section": section, "entity": {}, "items": list(carry)})
            carry = []

    for page in pages:
        page_number = page["page"]

        for raw_line in page["text"].split("\n"):
            line = raw_line.rstrip()
            stripped = line.strip()

            heading = detect_section_heading(line)
            if heading:
                flush_all()
                section, entity = heading, {}
                pending_org, awaiting_role = None, False
                continue

            if section == "projects":
                project_name = detect_project_name(line)
                if project_name:
                    flush_all()
                    entity = {"project": project_name}

            if section in ("experience", "positions of responsibility"):
                is_bullet = stripped.startswith(_BULLET_PREFIXES)
                if is_date_range(line):
                    flush_items()  # close the previous company block
                    entity = {"company": pending_org} if pending_org else {}
                    items = [*carry, (page_number, line)]
                    carry, pending_org, awaiting_role = [], None, True
                    continue
                if awaiting_role and stripped and not is_bullet and len(stripped) <= 40:
                    # First short non-bullet line after the dates is the
                    # role; later short lines (e.g. location) are ignored.
                    entity["role"] = stripped
                    awaiting_role = False
                    items.append((page_number, line))
                    continue
                if stripped and not is_bullet:
                    # A non-bullet line can be (a) a wrapped continuation of
                    # a bullet (starts lowercase), (b) the next organisation
                    # header, or (c) a location/detail of the current job.
                    starts_like_name = stripped[0].isupper()
                    previous_has_bullets = any(
                        item_text.lstrip().startswith(_BULLET_PREFIXES)
                        for _, item_text in items
                    )
                    if starts_like_name and (
                        previous_has_bullets or not entity.get("company")
                    ):
                        pending_org = stripped
                        carry.append((page_number, line))
                    else:
                        items.append((page_number, line))
                    continue

            if stripped:
                items.append((page_number, line))

    flush_all()
    return blocks


# "Backend & Databases: Node.js, Express.js, ..." style category lines.
_CATEGORY_LINE = re.compile(r"^\s*([A-Z][A-Za-z0-9 &/+.,()-]{1,40}?)\s*:\s*\S")

# Sections whose body is usually a set of "Category: item, item" lines.
_CATEGORY_SECTIONS = ("skills",)


def split_category_lines(text: str) -> list[tuple[str | None, str]]:
    """Split a skills-style block into one piece per labelled category line.

    A single skills chunk mixing languages, databases, tools and coursework
    is a poor retrieval unit: the cross-encoder scores long comma lists
    badly and every skills question receives the whole block.  One chunk per
    category ("Tools & Infrastructure: Git, Docker, ...") is precise while a
    broad "what are his skills" question still gets every category through
    section-level list retrieval.

    Wrapped continuation lines (no "Label:" prefix) stay with the previous
    category.  Returns ``[(category_or_None, text), ...]``; a block without
    at least two category lines is returned unchanged as one piece.
    """
    pieces: list[tuple[str | None, list[str]]] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        match = _CATEGORY_LINE.match(line)
        if match:
            pieces.append((match.group(1).strip(), [line]))
        elif pieces:
            pieces[-1][1].append(line)
        else:
            pieces.append((None, [line]))
    if sum(1 for category, _ in pieces if category) < 2:
        return [(None, text.strip())] if text.strip() else []
    return [(category, "\n".join(lines).strip()) for category, lines in pieces]


def _split_block(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a block of text into <=chunk_size pieces at line boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}".strip()
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            # Carry over the tail so consecutive chunks share context.
            tail = current[-overlap:]
            current = f"{tail}\n{line}".strip() if overlap else line
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_resume(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Extract + chunk a resume PDF into metadata-rich chunks."""
    return chunk_pages(extract_pages(pdf_path), chunk_size=chunk_size, overlap=overlap)


def chunk_pages(
    pages: list[PageDict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk already-extracted pages. Kept separate from extraction so it
    can be unit-tested without a PDF file."""
    chunks: list[Chunk] = []
    for block in build_blocks(pages):
        text_items = block["items"]
        block_text = "\n".join(line for _, line in text_items)
        page_number = min(page for page, _ in text_items)

        fallback_section = (
            detect_section(block_text) if block["section"] == "header" else block["section"]
        )
        if fallback_section in _CATEGORY_SECTIONS:
            categorised = split_category_lines(block_text)
        else:
            categorised = [(None, block_text)]

        for category, category_text in categorised:
            for piece in _split_block(category_text, chunk_size, overlap):
                metadata: dict[str, Any] = {
                    "source": "resume",
                    "section": fallback_section,
                    "page": page_number,
                    "owner_id": _OWNER_ID,
                    "owner_name": _OWNER_NAME,
                    "owner_type": _OWNER_TYPE,
                }
                if category:
                    metadata["skill_category"] = category
                metadata.update(block["entity"])
                chunks.append({"text": piece, "metadata": metadata})
    return chunks


def chunk_id(chunk: Chunk, index: int) -> str:
    """Deterministic ID so re-ingesting the same document upserts instead
    of duplicating rows in ChromaDB."""
    digest = hashlib.sha1(chunk["text"].encode("utf-8")).hexdigest()[:16]
    return f"{chunk['metadata'].get('source', 'doc')}-{index}-{digest}"


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Backward-compatible plain-text chunking used by older scripts."""
    return _split_block(text, chunk_size, overlap)


if __name__ == "__main__":
    for i, chunk in enumerate(chunk_resume("data/Resume.pdf")):
        print(f"\n--- CHUNK {i} {chunk['metadata']} ---")
        print(chunk["text"])
