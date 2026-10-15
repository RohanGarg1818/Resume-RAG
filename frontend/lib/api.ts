import { apiFetch } from "@/lib/config";
import type { ChatRequest, ChatResponse, IndexStatus } from "@/types/chat";

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const response = await apiFetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },

    // Sends the rag_session_id cookie so the backend can include this
    // browser session's uploaded documents in retrieval.
    credentials: "include",

    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let message = "Failed to communicate with the backend.";

    try {
      const error = await response.json();

      if (typeof error.detail === "string") {
        message = error.detail;
      }
    } catch {
      // Keep default error message.
    }

    throw new Error(message);
  }

  return response.json();
}

export async function fetchIndexStatus(): Promise<IndexStatus | null> {
  try {
    const response = await apiFetch("/api/index/status");
    return response.ok ? response.json() : null;
  } catch {
    return null;
  }
}
