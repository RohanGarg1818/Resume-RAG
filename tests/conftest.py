"""Shared test fixtures for the comparison feature tests."""

import hashlib
import uuid

import pytest


@pytest.fixture
def sample_job_description():
    return (
        "Senior Backend Engineer\n\n"
        "Requirements:\n"
        "- 3+ years of experience with Python or Go\n"
        "- Experience building REST APIs\n"
        "- Knowledge of PostgreSQL or similar RDBMS\n"
        "- Experience with cloud platforms (AWS, GCP, or Azure)\n"
        "- Familiarity with Docker and containerization\n\n"
        "Preferred:\n"
        "- Experience with FastAPI or similar frameworks\n"
        "- Knowledge of Redis or caching systems\n"
        "- contributions to open source projects\n"
    )


@pytest.fixture
def minimal_job_description():
    return (
        "Data Analyst\n"
        "Must know SQL and Python.\n"
        "Experience with pandas is preferred.\n"
    )


@pytest.fixture
def sample_chunks_a():
    """Simulated Candidate A evidence chunks."""
    return [
        {
            "text": "Built REST APIs using FastAPI and PostgreSQL for a real-time messaging platform.",
            "metadata": {"source": "resume", "section": "projects", "project": "CollabSphere"},
            "similarity": 0.82,
        },
        {
            "text": "Experience with Docker and Kubernetes for container orchestration.",
            "metadata": {"source": "resume", "section": "skills"},
            "similarity": 0.75,
        },
        {
            "text": "Deployed applications on AWS EC2 and RDS.",
            "metadata": {"source": "resume", "section": "projects", "project": "VideoTube"},
            "similarity": 0.70,
        },
    ]


@pytest.fixture
def sample_chunks_b():
    """Simulated Candidate B evidence chunks."""
    return [
        {
            "text": "Developed backend services in Python using Flask and SQLAlchemy.",
            "metadata": {"source": "session", "section": "projects"},
            "similarity": 0.78,
        },
        {
            "text": "Worked with PostgreSQL for data storage and complex queries.",
            "metadata": {"source": "session", "section": "experience"},
            "similarity": 0.72,
        },
    ]


@pytest.fixture
def empty_session_id():
    """A valid-format session ID that has no collection."""
    return str(uuid.uuid4())
