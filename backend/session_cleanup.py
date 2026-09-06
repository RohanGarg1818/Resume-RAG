from src.session_store import (
    get_expired_sessions,
    delete_session,
)

from src.vector_store import (
    delete_session_collection,
)


def cleanup_expired_sessions():

    expired = (
        get_expired_sessions()
    )

    for session_id in expired:

        try:

            delete_session_collection(
                session_id
            )

        finally:

            delete_session(
                session_id
            )