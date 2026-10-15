import type { ComparisonRequest, ComparisonResponse } from "@/types/comparison";

import { apiFetch } from "@/lib/config";

export async function compareCandidates(
  request: ComparisonRequest
): Promise<ComparisonResponse> {
  const response = await apiFetch("/api/compare-candidates", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    credentials: "include",
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let message = "Failed to compare candidates.";

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
