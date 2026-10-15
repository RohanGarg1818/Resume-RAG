import { apiFetch } from "@/lib/config";

export async function startRagSession() {
  const response = await apiFetch("/api/session/start", {
    method: "POST",
    // Required so the browser stores the HttpOnly rag_session_id cookie.
    credentials: "include",
  });

  if (!response.ok) {
    const errorText = await response.text();

    console.error("Failed to start RAG session:", {
      status: response.status,
      body: errorText,
      url: response.url,
    });

    throw new Error(
      `Failed to start RAG session (${response.status}): ${errorText}`
    );
  }

  return response.json();
}

export async function endRagSession() {
  const response = await apiFetch("/api/session/end", {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    const errorText = await response.text();

    console.error("Failed to end RAG session:", {
      status: response.status,
      body: errorText,
    });

    throw new Error(
      `Failed to end RAG session (${response.status}): ${errorText}`
    );
  }

  return response.json();
}
