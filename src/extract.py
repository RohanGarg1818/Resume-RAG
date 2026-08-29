"""PDF text extraction with page preservation.

The original version joined every page into one string, which made it
impossible to know which page a chunk came from. `extract_pages` now
returns one entry per page so page numbers can travel all the way into
ChromaDB metadata and source citations.

`extract_text_from_pdf` is kept for backward compatibility.
"""

from pathlib import Path
from typing import Any

import pymupdf

from src.errors import DocumentError

# A single extracted page: {"page": 1-based page number, "text": page text}
PageDict = dict[str, Any]


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Extract text from a PDF, preserving page boundaries.

    Returns:
        List of {"page": <1-based page number>, "text": <page text>} dicts.

    Raises:
        DocumentError: if the file does not exist or cannot be opened.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise DocumentError(f"PDF file not found: {path}")

    try:
        document = pymupdf.open(path)
    except Exception as exc:  # malformed/corrupt PDF
        raise DocumentError(f"Could not open PDF '{path.name}': {exc}") from exc

    pages: list[dict] = []
    try:
        for index, page in enumerate(document):
            text = page.get_text().strip()
            if text:
                pages.append({"page": index + 1, "text": text})
    except Exception as exc:
        raise DocumentError(f"Failed while reading '{path.name}': {exc}") from exc
    finally:
        document.close()

    if not pages:
        raise DocumentError(
            f"No extractable text found in '{path.name}'. "
            "The PDF may be empty, corrupted, or a scanned image."
        )
    return pages


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Backward-compatible helper: full PDF text as a single string."""
    return "\n\n".join(page["text"] for page in extract_pages(pdf_path))


if __name__ == "__main__":
    for page_data in extract_pages("data/Resume.pdf"):
        print(f"===== PAGE {page_data['page']} =====")
        print(page_data["text"])
