"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import Icon from "@/components/Icon";
import { fetchIndexStatus } from "@/lib/api";
import type { IndexStatus } from "@/types/chat";

type Page = "chat" | "compare" | "evaluation";

export default function TopBar({
  current,
  status,
}: {
  current: Page;
  /** Pass when the page already loads index status; otherwise it is fetched. */
  status?: IndexStatus | null;
}) {
  const [ownStatus, setOwnStatus] = useState<IndexStatus | null>(null);
  useEffect(() => {
    if (status !== undefined) return;
    let cancelled = false;
    fetchIndexStatus().then((result) => {
      if (!cancelled) setOwnStatus(result);
    });
    return () => {
      cancelled = true;
    };
  }, [status]);
  const shown = status !== undefined ? status : ownStatus;
  const nav: { page: Page; href: string; label: string; icon: "scan" | "compare" | "chart" }[] = [
    { page: "chat", href: "/", label: "Workspace", icon: "scan" },
    { page: "compare", href: "/compare", label: "Compare", icon: "compare" },
    { page: "evaluation", href: "/evaluation", label: "Evaluation", icon: "chart" },
  ];

  return (
    <header
      className="rr-glass"
      style={{
        position: "relative",
        zIndex: 5,
        minHeight: 52,
        flex: "none",
        display: "flex",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 16,
        padding: "8px 16px",
        borderWidth: "0 0 1px 0",
      }}
    >
      <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--text)" }}>
        <span
          aria-hidden="true"
          style={{
            width: 22, height: 22, borderRadius: 6, background: "var(--blue)", color: "#fff",
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 500,
            boxShadow: "inset 0 1px 0 rgba(255,255,255,.25)",
          }}
        >
          R
        </span>
        <span className="rr-h" style={{ fontSize: 14 }}>Resume Rag</span>
      </Link>
      <span style={{ color: "var(--faint)" }} aria-hidden="true">/</span>
      <span style={{ fontSize: 13, color: "var(--text)" }}>Rohan Garg · profile</span>

      <nav aria-label="Primary" style={{ display: "flex", gap: 4, marginLeft: "auto", flexWrap: "wrap" }}>
        {nav.map((item) => (
          <Link
            key={item.page}
            href={item.href}
            className="rr-btn rr-btn-ghost"
            aria-current={current === item.page ? "page" : undefined}
          >
            <Icon name={item.icon} />
            {item.label}
          </Link>
        ))}
      </nav>

      <div
        className="rr-lbl rr-hide-sm"
        title={shown ? `Embeddings: ${shown.embedding_model}` : undefined}
        style={{ display: "flex", alignItems: "center", gap: 8, height: 28, padding: "0 10px", border: "1px solid var(--line)", borderRadius: 8 }}
      >
        <Icon name="cloud" size={14} />
        <span>{shown ? (shown.llm_location === "local" ? "Local" : "Cloud") : "Model"}</span>
        <span style={{ color: "var(--faint)" }} aria-hidden="true">·</span>
        <span style={{ color: "var(--text)", textTransform: "none", letterSpacing: 0 }}>
          {shown?.llm_model ?? "—"}
        </span>
      </div>
    </header>
  );
}
