"""Domain-specific exceptions shared by the RAG engine and the API layer.

Keeping them in one small module lets the FastAPI layer map each error
to a clean HTTP status code without leaking stack traces to users.
"""


class RagError(Exception):
    """Base class for all RAG pipeline errors."""


class DocumentError(RagError):
    """Raised when a PDF is missing, unreadable, or contains no text."""


class ConfigurationError(RagError):
    """Raised when required environment configuration is missing/invalid."""


class EmbeddingError(RagError):
    """Raised when the embedding model fails to load or encode."""


class RetrievalError(RagError):
    """Raised when ChromaDB queries fail."""


class LLMError(RagError):
    """Base class for Groq/LLM failures."""


class MissingAPIKeyError(ConfigurationError):
    """GROQ_API_KEY is not set."""


class InvalidAPIKeyError(LLMError):
    """Groq rejected the API key (401)."""


class ModelUnavailableError(LLMError):
    """The configured Groq model does not exist or is unavailable."""


class RateLimitError(LLMError):
    """Groq rate limit hit."""
