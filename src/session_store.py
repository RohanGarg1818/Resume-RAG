import time
import uuid
import threading
from dataclasses import dataclass, field


SESSION_TTL_SECONDS = 30 * 60


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    files: dict[str, dict] = field(default_factory=dict)


_sessions: dict[str, Session] = {}

_lock = threading.Lock()


def create_session() -> str:
    session_id = str(uuid.uuid4())

    with _lock:
        _sessions[session_id] = Session(
            session_id=session_id
        )

    return session_id


def get_session(session_id: str) -> Session | None:
    with _lock:
        return _sessions.get(session_id)


def touch_session(session_id: str) -> bool:
    with _lock:
        session = _sessions.get(session_id)

        if not session:
            return False

        session.last_activity = time.time()

        return True


def delete_session(session_id: str) -> bool:
    with _lock:
        return _sessions.pop(session_id, None) is not None


def session_exists(session_id: str) -> bool:
    with _lock:
        return session_id in _sessions


def register_file(session_id: str, filename: str, chunks_added: int) -> bool:
    """Record session upload metadata without retaining document contents."""
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return False
        session.files[filename] = {
            "filename": filename,
            "chunks_added": chunks_added,
            "uploaded_at": time.time(),
        }
        session.last_activity = time.time()
        return True


def list_files(session_id: str) -> list[dict]:
    with _lock:
        session = _sessions.get(session_id)
        return list(session.files.values()) if session else []


def unregister_file(session_id: str, filename: str) -> bool:
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return False
        removed = session.files.pop(filename, None) is not None
        if removed:
            session.last_activity = time.time()
        return removed


def get_expired_sessions() -> list[str]:
    now = time.time()

    with _lock:
        return [
            session_id
            for session_id, session in _sessions.items()
            if now - session.last_activity > SESSION_TTL_SECONDS
        ]
