from pathlib import Path
from typing import BinaryIO

import pandas as pd
from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".md",
    ".markdown",
    ".csv",
}


def load_pdf(file: BinaryIO) -> list[dict]:
    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages


def load_docx(file: BinaryIO) -> list[dict]:
    document = Document(file)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(
                {
                    "text": text,
                    "page": None,
                }
            )

    return paragraphs


def load_markdown(file: BinaryIO) -> list[dict]:
    content = file.read()

    if isinstance(content, bytes):
        content = content.decode(
            "utf-8",
            errors="ignore",
        )

    return [
        {
            "text": content,
            "page": None,
        }
    ]


def load_csv(file: BinaryIO) -> list[dict]:
    df = pd.read_csv(file)

    rows = []

    for index, row in df.iterrows():

        parts = []

        for column in df.columns:

            value = row[column]

            if pd.isna(value):
                continue

            parts.append(
                f"{column}: {value}"
            )

        if parts:
            rows.append(
                {
                    "text": "\n".join(parts),
                    "page": None,
                    "row": index + 1,
                }
            )

    return rows


def load_document(
    file: BinaryIO,
    filename: str,
) -> list[dict]:

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        return load_pdf(file)

    if extension in {".doc", ".docx"}:
        return load_docx(file)

    if extension in {".md", ".markdown"}:
        return load_markdown(file)

    if extension == ".csv":
        return load_csv(file)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )