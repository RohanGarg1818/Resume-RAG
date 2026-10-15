import { apiFetch } from "@/lib/config";

export async function endRagSession() {

  const response = await apiFetch(
    "/api/session/end",
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