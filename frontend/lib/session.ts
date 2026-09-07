const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function startRagSession() {
  const response = await fetch(`${API_URL}/api/session/start`, {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    const errorText = await response.text();

    console.error("Failed to start RAG session:", {
      status: response.status,
      body: errorText,
      url: `${API_URL}/api/session/start`,
    });

    throw new Error(
      `Failed to start RAG session (${response.status}): ${errorText}`
    );
  }

  return response.json();
}

export async function endRagSession() {
  const response = await fetch(`${API_URL}/api/session/end`, {
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