/**
 * Single source of truth for the backend base URL.
 *
 * NEXT_PUBLIC_API_URL is inlined at build/dev-server start, so it must be
 * read with this literal property access, and `npm run dev` must be
 * restarted after changing frontend/.env.local.
 */
const RAW_API_URL = process.env.NEXT_PUBLIC_API_URL;

export const API_URL = (RAW_API_URL ?? "").trim().replace(/\/+$/, "");

export function apiUrl(path: string): string {
  if (!API_URL) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not configured. Set it in frontend/.env.local " +
        "(for example http://localhost:8000) and restart the Next.js dev server."
    );
  }
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

/**
 * fetch() wrapper that turns the browser's opaque "TypeError: Failed to
 * fetch" into an actionable message.  That error means no usable HTTP
 * response reached the page: the backend is not running at API_URL, or it
 * answered without CORS headers for this page's origin.
 */
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = apiUrl(path);
  try {
    return await fetch(url, init);
  } catch (error) {
    const origin = typeof window !== "undefined" ? window.location.origin : "server";
    throw new Error(
      `Cannot reach the backend at ${url}. Check that the FastAPI server is ` +
        `running and that its CORS allowed origins include ${origin}. ` +
        `(${error instanceof Error ? error.message : String(error)})`
    );
  }
}
