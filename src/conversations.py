"""In-memory conversation store.

Conversation history is deliberately kept OUT of the vector database:
retrieved chunks answer "what does the resume say", history answers
"what are we talking about". The LLM receives both separately, and the
system prompt forbids history from overriding the profile context.

The store is process-local (fine for a single-instance demo app); swap
`_conversations` for Redis/a DB later without touching the RAG engine.
"""

import threading
import uuid
from collections import deque

MAX_TURNS_KEPT = 8  # (user+assistant) message pairs retained per conversation

Message = dict  # {"role": "user"|"assistant", "content": str}

_lock = threading.Lock()
_conversations: dict[str, deque] = {}


def create_conversation() -> str:
    """Create an empty conversation and return its ID."""
    conversation_id = uuid.uuid4().hex
    with _lock:
        _conversations[conversation_id] = deque(maxlen=MAX_TURNS_KEPT * 2)
    return conversation_id


def get_history(conversation_id: str | None) -> list[Message]:
    """Return a copy of the stored messages (empty list if unknown id)."""
    if not conversation_id:
        return []
    with _lock:
        conversation = _conversations.get(conversation_id)
        return list(conversation) if conversation else []


def append_messages(conversation_id: str, *messages: Message) -> None:
    """Append user/assistant messages to a conversation."""
    with _lock:
        conversation = _conversations.setdefault(
            conversation_id, deque(maxlen=MAX_TURNS_KEPT * 2)
        )
        for message in messages:
            conversation.append(message)


def clear_conversation(conversation_id: str) -> bool:
    """Drop all messages for a conversation. True if it existed."""
    with _lock:
        return _conversations.pop(conversation_id, None) is not None
