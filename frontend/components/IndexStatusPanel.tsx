"use client";

import type { IndexStatus } from "@/types/chat";

const KNURL =
  "M90.0 48.0L93.5 48.0M89.8 51.7L93.3 52.0M89.4 55.3L92.8 55.9M88.6 58.9L91.9 59.8M87.5 62.4L90.8 63.6M86.1 65.7L89.2 67.2M84.4 69.0L87.4 70.8M82.4 72.1L85.3 74.1M80.2 75.0L82.9 77.2M77.7 77.7L80.2 80.2M75.0 80.2L77.2 82.9M72.1 82.4L74.1 85.3M69.0 84.4L70.8 87.4M65.7 86.1L67.2 89.2M62.4 87.5L63.6 90.8M58.9 88.6L59.8 91.9M55.3 89.4L55.9 92.8M51.7 89.8L52.0 93.3M48.0 90.0L48.0 93.5M44.3 89.8L44.0 93.3M40.7 89.4L40.1 92.8M37.1 88.6L36.2 91.9M33.6 87.5L32.4 90.8M30.3 86.1L28.8 89.2M27.0 84.4L25.3 87.4M23.9 82.4L21.9 85.3M21.0 80.2L18.8 82.9M18.3 77.7L15.8 80.2M15.8 75.0L13.1 77.2M13.6 72.1L10.7 74.1M11.6 69.0L8.6 70.8M9.9 65.7L6.8 67.2M8.5 62.4L5.2 63.6M7.4 58.9L4.1 59.8M6.6 55.3L3.2 55.9M6.2 51.7L2.7 52.0M6.0 48.0L2.5 48.0M6.2 44.3L2.7 44.0M6.6 40.7L3.2 40.1M7.4 37.1L4.1 36.2M8.5 33.6L5.2 32.4M9.9 30.3L6.8 28.8M11.6 27.0L8.6 25.2M13.6 23.9L10.7 21.9M15.8 21.0L13.1 18.8M18.3 18.3L15.8 15.8M21.0 15.8L18.8 13.1M23.9 13.6L21.9 10.7M27.0 11.6L25.2 8.6M30.3 9.9L28.8 6.8M33.6 8.5L32.4 5.2M37.1 7.4L36.2 4.1M40.7 6.6L40.1 3.2M44.3 6.2L44.0 2.7M48.0 6.0L48.0 2.5M51.7 6.2L52.0 2.7M55.3 6.6L55.9 3.2M58.9 7.4L59.8 4.1M62.4 8.5L63.6 5.2M65.7 9.9L67.2 6.8M69.0 11.6L70.8 8.6M72.1 13.6L74.1 10.7M75.0 15.8L77.2 13.1M77.7 18.3L80.2 15.8M80.2 21.0L82.9 18.8M82.4 23.9L85.3 21.9M84.4 27.0L87.4 25.2M86.1 30.3L89.2 28.8M87.5 33.6L90.8 32.4M88.6 37.1L91.9 36.2M89.4 40.7L92.8 40.1M89.8 44.3L93.3 44.0";
const SCALE =
  "M24.0 72.0L27.5 68.5M20.1 67.5L22.6 65.8M17.2 62.4L19.9 61.1M15.2 56.8L20.0 55.5M14.1 51.0L17.1 50.7M14.1 45.0L17.1 45.3M15.2 39.2L20.0 40.5M17.2 33.6L19.9 34.9M20.1 28.5L22.6 30.2M24.0 24.0L27.5 27.5M28.5 20.1L30.2 22.6M33.6 17.2L34.9 19.9M39.2 15.2L40.5 20.0M45.0 14.1L45.3 17.1M51.0 14.1L50.7 17.1M56.8 15.2L55.5 20.0M62.4 17.2L61.1 19.9M67.5 20.1L65.8 22.6M72.0 24.0L68.5 27.5M75.9 28.5L73.4 30.2M78.8 33.6L76.1 34.9M80.8 39.2L76.0 40.5M81.9 45.0L78.9 45.3M81.9 51.0L78.9 50.7M80.8 56.8L76.0 55.5M78.8 62.4L76.1 61.1M75.9 67.5L73.4 65.8M72.0 72.0L68.5 68.5";

const fmt = (value: number) => value.toLocaleString("en-US");

/**
 * Skeuomorphic index status: a machined rotary dial plus an LED bar.
 * Upload progress is not reported by the backend, so while a file is being
 * indexed the dial shows an indeterminate sweep and the LEDs pulse; they never
 * fake a percentage. Pulse and glow appear only while indexing.
 */
export default function IndexStatusPanel({
  status,
  sessionChunks,
  sessionFiles,
  indexing,
  error,
}: {
  status: IndexStatus | null;
  sessionChunks: number;
  sessionFiles: number;
  indexing: string | null;
  error: string | null;
}) {
  const live = Boolean(indexing);
  const offline = !status;
  const total = (status?.chunks ?? 0) + sessionChunks;
  const label = error ? "Error" : live ? "Indexing" : offline ? "Offline" : "Synced";
  const dot = error ? "err" : live ? "live" : offline ? "" : "ok";

  return (
    <section
      aria-label="Index status"
      style={{
        flex: "none",
        borderTop: "1px solid var(--line)",
        padding: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
        background: "linear-gradient(#0B0F15, #090C11)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <svg width="80" height="80" viewBox="0 0 96 96" role="img" aria-label={`Index ${label.toLowerCase()}`}>
          <circle cx="48" cy="48" r="47" fill="#050608" />
          <path d={KNURL} stroke="#2A313C" strokeWidth="1.2" />
          <circle cx="48" cy="48" r="41" fill="url(#rr-bezel)" />
          <circle cx="48" cy="48" r="39" fill="url(#rr-face)" stroke="#000" strokeOpacity="0.6" />
          <path d={SCALE} stroke="#4A5363" strokeWidth="1" />
          <circle cx="48" cy="48" r="26" fill="none" stroke="#06080B" strokeWidth="4" strokeDasharray="122.5 200" transform="rotate(135 48 48)" strokeLinecap="round" />
          {live ? (
            <g className="rr-sweep">
              <circle cx="48" cy="48" r="26" fill="none" stroke="#4D8DFF" strokeWidth="3" strokeDasharray="28 200" strokeLinecap="round" className="rr-arc-live" />
            </g>
          ) : (
            !offline && (
              <circle cx="48" cy="48" r="26" fill="none" stroke={error ? "#E5484D" : "#2B6BFF"} strokeWidth="3" strokeDasharray="122.5 200" transform="rotate(135 48 48)" strokeLinecap="round" />
            )
          )}
          <circle cx="48" cy="48" r="20" fill="url(#rr-knob)" stroke="#000" strokeOpacity="0.7" />
          <circle cx="48" cy="48" r="20" fill="none" stroke="url(#rr-spec)" strokeWidth="1" />
          <g transform={`rotate(${offline ? 135 : 45} 48 48)`} style={{ transition: "transform 300ms cubic-bezier(.2,.8,.2,1)" }}>
            <line x1="48" y1="48" x2="64" y2="48" stroke="#E8EDF5" strokeOpacity="0.85" strokeWidth="1.6" strokeLinecap="round" />
          </g>
          <circle cx="48" cy="48" r="2.5" fill="#0A0D12" stroke="#4A5363" strokeWidth="0.8" />
          <rect x="0" y="0" width="96" height="96" filter="url(#rr-noise)" opacity="0.6" style={{ mixBlendMode: "overlay" }} />
        </svg>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="rr-lbl" style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--text)" }}>
            <span className={`rr-dot ${dot}`} />
            {label}
          </div>
          <div className="rr-mono" style={{ fontSize: 20, letterSpacing: "-0.02em", color: "var(--text)" }}>
            {offline ? "—" : fmt(total)}
          </div>
          <div className="rr-lbl" style={{ fontSize: 10, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={indexing ?? undefined}>
            {live ? indexing : "chunks indexed"}
          </div>
        </div>
      </div>

      <div className={`rr-ledbar${live ? " live" : ""}`} aria-hidden="true">
        {Array.from({ length: 16 }, (_, i) => (
          <span
            key={i}
            className={`rr-led${error && i === 15 ? " err" : offline ? "" : live ? (i < 10 ? " on pulse" : "") : " on"}`}
          />
        ))}
      </div>

      {error && (
        <p className="rr-mono" style={{ margin: 0, fontSize: 11, color: "var(--err)", lineHeight: 1.5 }}>{error}</p>
      )}

      <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "8px 12px" }}>
        {Object.entries(status?.by_source ?? {}).map(([source, count]) => (
          <div key={source}>
            <dt className="rr-lbl" style={{ fontSize: 10 }}>{source}</dt>
            <dd className="rr-mono" style={{ margin: 0, color: "var(--text)" }}>{fmt(count)}</dd>
          </div>
        ))}
        <div>
          <dt className="rr-lbl" style={{ fontSize: 10 }}>Session</dt>
          <dd className="rr-mono" style={{ margin: 0, color: "var(--text)" }}>{sessionFiles} · {fmt(sessionChunks)}</dd>
        </div>
        <div style={{ gridColumn: "1 / -1" }}>
          <dt className="rr-lbl" style={{ fontSize: 10 }}>Embeddings</dt>
          <dd className="rr-mono" style={{ margin: 0, color: "var(--text)", fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {status?.embedding_model.replace("sentence-transformers/", "") ?? "—"}
          </dd>
        </div>
      </dl>
    </section>
  );
}
