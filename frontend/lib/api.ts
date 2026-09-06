
import type { ChatRequest, ChatResponse } from "@/types/chat";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

if (!API_URL) {
  throw new Error("NEXT_PUBLIC_API_URL is not configured.");
}

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },

    // Important:
    // Sends the rag_session_id cookie to the backend so the
    // backend knows which temporary RAG collection belongs
    // to this browser session.
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