export async function startRagSession() {

  const response = await fetch(
    "http://localhost:8000/api/session/start",
    {
      method: "POST",
      credentials: "include",
    }
  );

  if (!response.ok) {
    throw new Error(
      "Failed to start RAG session"
    );
  }

  return response.json();
}