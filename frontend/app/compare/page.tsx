"use client";

import ComparisonView from "@/components/ComparisonView";
import TopBar from "@/components/TopBar";

export default function ComparePage() {
  return (
    <div style={{ position: "relative", height: "100dvh", display: "flex", flexDirection: "column", background: "var(--bg)", color: "var(--text)" }}>
      <div className="rr-grid-band rr-bluegrid" />
      <TopBar current="compare" />

      <main className="rr-dotgrid" style={{ position: "relative", flex: 1, minHeight: 0, overflow: "hidden" }}>
        <ComparisonView />
      </main>
    </div>
  );
}
