export async function endRagSession() {

  const response = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/session/end`, 
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