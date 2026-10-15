"use client";

import { useMemo, useState } from "react";

import Icon from "@/components/Icon";
import RelevanceGauge from "@/components/RelevanceGauge";
import { citedLabels, sourceLocation, sourceTitle } from "@/lib/citations";
import type { ChatMessage, TraceChunk } from "@/types/chat";

export type PanelTab = "sources" | "trace";
type Filter = "all" | "cited" | "context";

function num(value: unknown, digits = 2): string {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function chunkName(chunk: TraceChunk): string {
  const meta = (chunk.metadata ?? {}) as Record<string, unknown>;
  const name = meta.project || meta.company || meta.event || meta.filename || meta.section || meta.source;
  return String(name ?? "chunk");
}

function TraceTable({ rows }: { rows: TraceChunk[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: 11 }}>
        <thead>
          <tr className="rr-lbl" style={{ fontSize: 10 }}>
            {["#", "chunk", "sem", "bm25", "hyb", "ce"].map((head) => (
              <th key={head} style={{ textAlign: head === "chunk" ? "left" : "right", padding: "4px 4px", fontWeight: 500 }}>{head}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.rank}-${row.text_preview?.slice(0, 20)}`} style={{ borderTop: "1px solid var(--line)", color: "var(--muted)" }}>
              <td style={{ padding: "5px 4px", textAlign: "right" }}>{row.rank}</td>
              <td style={{ padding: "5px 4px", color: "var(--text-2)", maxWidth: 110, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={row.text_preview}>
                {chunkName(row)}
              </td>
              <td style={{ padding: "5px 4px", textAlign: "right" }}>{num(row.semantic_score)}</td>
              <td style={{ padding: "5px 4px", textAlign: "right" }}>{num(row.bm25_score, 1)}</td>
              <td style={{ padding: "5px 4px", textAlign: "right", color: "var(--text)" }}>{num(row.hybrid_score)}</td>
              <td style={{ padding: "5px 4px", textAlign: "right" }}>{num(row.reranker_score, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function SourcesPanel({
  message,
  tab,
  onTab,
  activeSource,
  onSelectSource,
}: {
  message: ChatMessage | null;
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  activeSource: number | null;
  onSelectSource: (index: number | null) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const [open, setOpen] = useState<Record<number, boolean>>({});

  const sources = useMemo(() => message?.sources ?? [], [message]);
  const cited = useMemo(() => new Set(citedLabels(message?.content ?? "")), [message]);
  const isCited = (index: number) => {
    const source = sources[index];
    const labels = [source.citation_id, ...(source.citation_ids ?? [])].filter(Boolean) as string[];
    return labels.some((label) => cited.has(label));
  };
  const usedCount = sources.filter((_, index) => isCited(index)).length;

  const rows = sources
    .map((source, index) => ({ source, index, used: isCited(index) }))
    .filter((row) => filter === "all" || (filter === "cited" ? row.used : !row.used));

  const trace = message?.trace;
  const debug = trace?.debug ?? null;

  return (
    <aside
      className="rr-pane"
      aria-label="Sources inspector"
      style={{ width: 320, flex: "none", height: "100%", display: "flex", flexDirection: "column", background: "var(--surface)" }}
    >
      <div style={{ flex: "none", padding: "14px 16px 12px", display: "flex", flexDirection: "column", gap: 12, borderBottom: "1px solid var(--line)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <div className="rr-seg" role="tablist" aria-label="Inspector view">
            <button type="button" role="tab" aria-selected={tab === "sources"} onClick={() => onTab("sources")}>Sources</button>
            <button type="button" role="tab" aria-selected={tab === "trace"} onClick={() => onTab("trace")}>Trace</button>
          </div>
          <span className="rr-lbl" style={{ fontSize: 10 }}>
            {message ? `${usedCount} cited · ${sources.length} retrieved` : "no answer yet"}
          </span>
        </div>
        {tab === "sources" && sources.length > 0 && (
          <div className="rr-seg" role="group" aria-label="Filter sources" style={{ width: "100%" }}>
            {([
              ["all", "All"],
              ["cited", "Cited"],
              ["context", "Context only"],
            ] as [Filter, string][]).map(([value, label]) => (
              <button key={value} type="button" style={{ flex: 1 }} aria-pressed={filter === value} onClick={() => setFilter(value)}>
                {label}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="rr-scroll" style={{ flex: 1, minHeight: 0, padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
        {tab === "sources" && (
          <>
            {!message || sources.length === 0 ? (
              <>
                {[1, 0.6, 0.3].map((opacity) => (
                  <div key={opacity} className="rr-dash" style={{ height: 76, borderRadius: 12, opacity }} />
                ))}
                <p className="rr-mono" style={{ fontSize: 11, lineHeight: 1.7, margin: "8px 4px 0", color: "var(--muted)" }}>
                  {message
                    ? "This answer did not use retrieved context."
                    : "Retrieved chunks appear here, ranked by relevance, with the ones cited in the answer marked."}
                </p>
              </>
            ) : (
              rows.map(({ source, index, used }) => {
                const isOpen = Boolean(open[index]);
                return (
                  <div
                    key={`${source.citation_id}-${index}`}
                    className={`rr-src${activeSource === index ? " on" : ""}`}
                    style={{ padding: "10px 10px 10px 12px", display: "flex", flexDirection: "column", gap: 8 }}
                  >
                    <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                      <button
                        type="button"
                        onClick={() => onSelectSource(activeSource === index ? null : index)}
                        style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6, background: "none", border: 0, padding: 0, textAlign: "left", cursor: "pointer", color: "inherit" }}
                        aria-pressed={activeSource === index}
                      >
                        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span className="rr-lbl" style={{ fontSize: 10 }}>#{index + 1}</span>
                          <span className="rr-mono" style={{ color: "var(--text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {sourceTitle(source)}
                          </span>
                        </span>
                        <span className="rr-lbl" style={{ fontSize: 10, textTransform: "none", letterSpacing: "0.02em" }}>{sourceLocation(source)}</span>
                        <span>
                          {used ? (
                            <span className="rr-tag used rr-fade-in">Cited · [{index + 1}]</span>
                          ) : (
                            <span className="rr-tag">Context only</span>
                          )}
                        </span>
                      </button>
                      <RelevanceGauge score={source.score} cited={used} />
                    </div>
                    {isOpen && source.excerpt && (
                      <div className="rr-well rr-fade-in" style={{ borderRadius: 10, padding: "10px 12px", fontSize: 13, lineHeight: 1.6, color: "var(--text-2)", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
                        {source.excerpt}
                      </div>
                    )}
                    {source.excerpt && (
                      <button
                        type="button"
                        className="rr-btn rr-btn-ghost"
                        aria-expanded={isOpen}
                        onClick={() => setOpen((state) => ({ ...state, [index]: !isOpen }))}
                        style={{ height: 24, padding: "0 6px", alignSelf: "flex-start", fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase" }}
                      >
                        <Icon name="chevronRight" size={12} style={{ transform: `rotate(${isOpen ? 90 : 0}deg)`, transition: "transform 200ms" }} />
                        {isOpen ? "Collapse" : "Expand"}
                      </button>
                    )}
                  </div>
                );
              })
            )}
          </>
        )}

        {tab === "trace" && (
          <>
            {!trace ? (
              <p className="rr-mono" style={{ fontSize: 11, lineHeight: 1.7, margin: 4, color: "var(--muted)" }}>
                Ask a question to see how it was routed, which candidates were scored and what reached the model.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <dl style={{ margin: 0, display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "10px 12px" }}>
                  <div>
                    <dt className="rr-lbl" style={{ fontSize: 10 }}>Route</dt>
                    <dd className="rr-mono" style={{ margin: 0, color: "var(--text)" }}>{trace.query_type ?? "—"}</dd>
                  </div>
                  <div>
                    <dt className="rr-lbl" style={{ fontSize: 10 }}>Retrieval</dt>
                    <dd className="rr-mono" style={{ margin: 0, color: "var(--text)" }}>
                      {message?.retrieval ? `${Math.round(message.retrieval.latency_ms)} ms` : "—"}
                    </dd>
                  </div>
                  <div style={{ gridColumn: "1 / -1" }}>
                    <dt className="rr-lbl" style={{ fontSize: 10 }}>Sections · entities</dt>
                    <dd className="rr-mono" style={{ margin: 0, color: "var(--text-2)", fontSize: 11 }}>
                      {[...(trace.topics ?? []), ...(trace.entities ?? [])].join(", ") || "none detected"}
                    </dd>
                  </div>
                  {debug?.expanded_terms && debug.expanded_terms.length > 0 && (
                    <div style={{ gridColumn: "1 / -1" }}>
                      <dt className="rr-lbl" style={{ fontSize: 10 }}>Expanded terms</dt>
                      <dd style={{ margin: "4px 0 0", display: "flex", flexWrap: "wrap", gap: 4 }}>
                        {debug.expanded_terms.map((term) => <span key={term} className="rr-tag">{term}</span>)}
                      </dd>
                    </div>
                  )}
                </dl>

                {debug?.final_context && debug.final_context.length > 0 && (
                  <section style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <div className="rr-lbl" style={{ fontSize: 10, color: "var(--blue-hi)" }}>Final context · sent to model</div>
                    <TraceTable rows={debug.final_context} />
                  </section>
                )}

                {debug?.candidates && debug.candidates.length > 0 && (
                  <section style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <div className="rr-lbl" style={{ fontSize: 10 }}>Candidates · hybrid ranked ({debug.candidates.length})</div>
                    <TraceTable rows={debug.candidates} />
                  </section>
                )}

                {!debug && (
                  <p className="rr-mono" style={{ fontSize: 11, lineHeight: 1.7, margin: 0, color: "var(--muted)" }}>
                    Per-chunk scores are recorded for direct and named-entity questions. This route ({trace.query_type ?? "unknown"}) reports its summary only.
                  </p>
                )}
                <p className="rr-mono" style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--muted)" }}>
                  sem = cosine · bm25 = lexical · hyb = fused score · ce = cross-encoder logit
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
