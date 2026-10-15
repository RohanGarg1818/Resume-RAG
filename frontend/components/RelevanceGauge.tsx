/**
 * Skeuomorphic relevance meter: a small analog gauge whose needle and arc show
 * the question-to-passage cosine similarity on a linear 0-1 scale. Blue arc =
 * cited in the answer, grey arc = context only. Empty dial when unscored.
 */
const TICKS =
  "M6.0 40.0L11.0 40.0M7.5 30.7L10.3 31.7M11.7 22.4L14.2 24.1M18.4 15.7L20.1 18.2M26.7 11.5L27.7 14.3M36.0 10.0L36.0 15.0M45.3 11.5L44.3 14.3M53.6 15.7L51.9 18.2M60.3 22.4L57.8 24.1M64.5 30.7L61.7 31.7M66.0 40.0L61.0 40.0";

export function GaugeDefs() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
      <defs>
        <radialGradient id="rr-face" cx="50%" cy="35%" r="75%">
          <stop offset="0" stopColor="#1B222C" />
          <stop offset="0.7" stopColor="#0D1117" />
          <stop offset="1" stopColor="#080A0E" />
        </radialGradient>
        <linearGradient id="rr-bezel" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#4A5363" />
          <stop offset="0.45" stopColor="#1A1F28" />
          <stop offset="1" stopColor="#050608" />
        </linearGradient>
        <radialGradient id="rr-knob" cx="38%" cy="30%" r="80%">
          <stop offset="0" stopColor="#3A4352" />
          <stop offset="0.35" stopColor="#1E242E" />
          <stop offset="1" stopColor="#0A0D12" />
        </radialGradient>
        <linearGradient id="rr-spec" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffffff" stopOpacity="0.16" />
          <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
        <filter id="rr-noise" x="0" y="0" width="100%" height="100%">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
          <feComponentTransfer>
            <feFuncA type="linear" slope="0.09" />
          </feComponentTransfer>
        </filter>
      </defs>
    </svg>
  );
}

export default function RelevanceGauge({
  score,
  cited,
}: {
  score: number | null | undefined;
  cited: boolean;
}) {
  const has = typeof score === "number";
  const frac = has ? Math.max(0, Math.min(1, score as number)) : 0;
  const angle = -90 + frac * 180;

  return (
    <div style={{ flex: "none", display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
      <svg
        width="72"
        height="46"
        viewBox="0 0 72 46"
        role="img"
        aria-label={has ? `Similarity ${(score as number).toFixed(2)}` : "Similarity not available"}
      >
        <path d="M2 42 A34 34 0 0 1 70 42 Z" fill="url(#rr-bezel)" />
        <path d="M4.5 41.5 A31.5 31.5 0 0 1 67.5 41.5 Z" fill="url(#rr-face)" stroke="#000" strokeOpacity="0.7" />
        <path d={TICKS} stroke="#4A5363" strokeWidth="0.9" />
        <path d="M12 40 A24 24 0 0 1 60 40" fill="none" stroke="#06080B" strokeWidth="3" />
        {has && (
          <path
            d="M12 40 A24 24 0 0 1 60 40"
            fill="none"
            stroke={cited ? "#2B6BFF" : "#7A8596"}
            strokeWidth="2"
            strokeDasharray={`${(frac * 75.4).toFixed(1)} 200`}
            style={{ transition: "stroke-dasharray 300ms cubic-bezier(.2,.8,.2,1)" }}
          />
        )}
        <g
          transform={`rotate(${has ? angle.toFixed(1) : -90} 36 40)`}
          style={{ opacity: has ? 1 : 0.35 }}
        >
          <line x1="36" y1="40" x2="36" y2="15" stroke="#E8EDF5" strokeWidth="1.2" strokeLinecap="round" />
        </g>
        <circle cx="36" cy="40" r="3" fill="url(#rr-knob)" stroke="#000" strokeOpacity="0.8" />
        <path d="M6 38 A30 30 0 0 1 66 38 L60 30 A24 24 0 0 0 12 30 Z" fill="url(#rr-spec)" opacity="0.5" />
        <rect x="0" y="0" width="72" height="46" filter="url(#rr-noise)" opacity="0.7" style={{ mixBlendMode: "overlay" }} />
      </svg>
      <span className="rr-mono" style={{ color: has ? "var(--text)" : "var(--muted)" }}>
        {has ? (score as number).toFixed(2) : "—"}
      </span>
    </div>
  );
}
