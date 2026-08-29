import type { ChatRequest, ChatResponse } from "@/types/chat";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let message = "Failed to communicate with the backend.";

    try {
      const error = await response.json();
      message = error.detail || message;
    } catch {
      // Keep default error message.
    }

    throw new Error(message);
  }

  return response.json();
}