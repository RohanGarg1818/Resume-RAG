import csv
import re
from pathlib import Path
from typing import Any


def clean_linkedin_text(text: str) -> str:
    """Clean LinkedIn post text for RAG indexing."""

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove LinkedIn's replacement/question-mark artifacts
    text = text.replace("?", " ")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_linkedin_posts(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load meaningful LinkedIn posts from Rich_Media.csv.

    Media-only rows are ignored.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"LinkedIn CSV not found: {csv_path}"
        )

    posts = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            description = (row.get("Media Description") or "").strip()
            date_time = (row.get("Date/Time") or "").strip()
            media_link = (row.get("Media Link") or "").strip()

            # Ignore media-only rows.
            if not description or description == "-":
                continue

            cleaned = clean_linkedin_text(description)

            if not cleaned:
                continue

            posts.append(
                {
                    "text": cleaned,
                    "date": date_time,
                    "media_link": media_link or None,
                    "source_type": "linkedin",
                }
            )

    return posts


def infer_linkedin_metadata(text: str) -> dict[str, Any]:
    """
    Infer useful resume-style metadata from a LinkedIn post.
    """

    lower = text.lower()

    metadata: dict[str, Any] = {
        "source": "linkedin",
        "source_type": "linkedin",
        "owner_id": "rohan",
        "owner_name": "Rohan Garg",
        "owner_type": "primary",
    }

    # Projects
    project_names = [
        "JugaadHub",
        "CollabSphere",
        "VideoTube",
    ]

    for project in project_names:
        if project.lower() in lower:
            metadata["project"] = project
            metadata["section"] = "projects"
            break

    # Hackathons
    hackathon_names = [
        "Hacknite",
        "HackXtract",
        "Smart India Hackathon",
        "Orbix Hackathon",
    ]

    for event in hackathon_names:
        if event.lower() in lower:
            metadata["event"] = event
            metadata["section"] = "achievements"
            break

    # Technical topics
    technical_terms = [
        "Next.js",
        "TypeScript",
        "Redis",
        "PostgreSQL",
        "WebSockets",
        "WebRTC",
        "Node.js",
        "MongoDB",
        "Python",
        "React",
        "Gemini",
    ]

    matched_technologies = [
        tech
        for tech in technical_terms
        if tech.lower() in lower
    ]

    if matched_technologies:
        metadata["technologies"] = ", ".join(
            matched_technologies
        )

        if "section" not in metadata:
            metadata["section"] = "skills"

    if "section" not in metadata:
        metadata["section"] = "other"

    return metadata



def parse_linkedin_csv(
    csv_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Parse LinkedIn CSV into RAG-ready documents.
    """

    posts = load_linkedin_posts(csv_path)

    documents = []

    for index, post in enumerate(posts):

        metadata = infer_linkedin_metadata(
            post["text"]
        )

        metadata["linkedin_post_id"] = f"linkedin_{index}"

        if post.get("date"):
            metadata["date"] = post["date"]

        documents.append(
            {
                "text": post["text"],
                "metadata": metadata,
            }
        )

    return documents