export async function endRagSession() {

  const response = await fetch(
    "http://localhost:8000/api/session/end",
    {
      method: "POST",
      credentials: "include",
      keepalive: true,
    }
  );

  if (!response.ok) {
    throw new Error(
      "Failed to end session"
    );
  }

  return response.json();
}