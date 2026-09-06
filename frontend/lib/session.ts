export async function startRagSession() {

  const response = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/session/start`, 
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