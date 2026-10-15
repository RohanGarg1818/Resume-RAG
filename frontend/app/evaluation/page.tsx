"use client";

import { useCallback, useEffect, useState } from "react";

import Icon from "@/components/Icon";
import TopBar from "@/components/TopBar";
import { apiFetch } from "@/lib/config";

type EvaluationSummary = {
  total_questions: number;
  supported_questions?: number;
  unsupported_questions?: number;
  evaluation_type?: string;
  precision_at_k?: Record<string, number>;
  recall_at_k?: Record<string, number>;
  f1_at_k?: number;
  f1_at_k_by_k?: Record<string, number>;
  hit_at_1?: number;
  hit_at_3?: number;
  hit_at_5?: number;
  mrr?: number;
  source_accuracy?: number;
  section_accuracy?: number;
  refusal_accuracy?: number;
  avg_retrieval_latency_ms: number;
  avg_context_chunks?: number;
};

type QuestionResult = {
  id: string;
  question: string;
  query_type?: string;
  expected_answer?: string | null;
  answer?: string;
  confidence?: string;
  precision_at_k?: Record<string, number>;
  recall_at_k?: Record<string, number>;
  first_relevant_rank?: number | null;
  refused?: boolean | null;
  retrieval_latency_ms?: number;
  llm_latency_ms?: number;
  retrieved_sections?: (string | null)[];
  error?: string;
};

type HistoryRun = { run_id: string; created_at: string; summary: EvaluationSummary };

type EvaluationData = {
  run_id?: string;
  created_at?: string;
  summary: EvaluationSummary;
  questions: QuestionResult[];
  history?: HistoryRun[];
};

function pct(value: number | undefined | null, digits = 1) {
  return typeof value === "number" ? `${(value * 100).toFixed(digits)}%` : "—";
}

function f1(precision?: number, recall?: number) {
  if (typeof precision !== "number" || typeof recall !== "number" || precision + recall === 0) return undefined;
  return (2 * precision * recall) / (precision + recall);
}

function Delta({ now, before, invert = false }: { now?: number; before?: number; invert?: boolean }) {
  if (typeof now !== "number" || typeof before !== "number") return null;
  const diff = now - before;
  if (Math.abs(diff) < 0.0005) return <span className="rr-mono" style={{ fontSize: 11, color: "var(--muted)" }}>±0</span>;
  const better = invert ? diff < 0 : diff > 0;
  return (
    <span className="rr-mono" style={{ fontSize: 11, color: better ? "var(--blue-hi)" : "var(--err)" }}>
      {diff > 0 ? "↑" : "↓"} {invert ? Math.abs(diff).toFixed(0) : (Math.abs(diff) * 100).toFixed(1)}
    </span>
  );
}

function Stat({
  label,
  value,
  sub,
  delta,
  primary = false,
}: {
  label: string;
  value: string;
  sub?: string;
  delta?: React.ReactNode;
  primary?: boolean;
}) {
  return (
    <div
      className="rr-src"
      style={{
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        minWidth: 0,
        borderColor: primary ? "var(--blue-line)" : undefined,
      }}
    >
      <div className="rr-lbl" style={{ fontSize: 10, display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span>{label}</span>
        {delta}
      </div>
      <div className="rr-mono" style={{ fontSize: 26, letterSpacing: "-0.02em", color: "var(--text)" }}>{value}</div>
      {sub && <div className="rr-lbl" style={{ fontSize: 10, textTransform: "none", letterSpacing: "0.02em" }}>{sub}</div>}
    </div>
  );
}

/** Thin inline meter for a 0-1 value; blue only when the value is perfect. */
function Meter({ value }: { value?: number }) {
  if (typeof value !== "number") return <span className="rr-mono" style={{ color: "var(--muted)" }}>—</span>;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, minWidth: 92 }}>
      <span
        aria-hidden="true"
        style={{ position: "relative", width: 44, height: 4, borderRadius: 2, background: "#06080B", boxShadow: "inset 0 1px 1px rgba(0,0,0,.8)" }}
      >
        <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${value * 100}%`, borderRadius: 2, background: value >= 0.999 ? "var(--blue)" : "#3A4352" }} />
      </span>
      <span className="rr-mono" style={{ color: value >= 0.999 ? "var(--text)" : "var(--text-2)" }}>{(value * 100).toFixed(0)}%</span>
    </span>
  );
}

const K_OPTIONS = ["1", "3", "5", "6"];

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [k, setK] = useState("6");
  const [typeFilter, setTypeFilter] = useState("all");

  const loadEvaluation = useCallback(async () => {
    try {
      setLoading(true);
      setError("");

      const response = await apiFetch("/api/evaluation");

      if (!response.ok) {
        throw new Error("Failed to load evaluation");
      }

      const result = await response.json();

      if (result.message) {
        setData(null);
        setError(result.message);
        return;
      }

      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial load; state updates happen after the request resolves.
    const timer = window.setTimeout(loadEvaluation, 0);
    return () => window.clearTimeout(timer);
  }, [loadEvaluation]);

  const shell = (children: React.ReactNode) => (
    <div style={{ minHeight: "100vh", background: "var(--bg)", position: "relative" }}>
      <div className="rr-grid-band rr-bluegrid" />
      <TopBar current="evaluation" />
      <main className="rr-dotgrid" style={{ position: "relative", minHeight: "calc(100vh - 52px)", padding: "28px clamp(16px, 4vw, 32px) 56px" }}>
        <div style={{ maxWidth: 1240, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 16, flexWrap: "wrap" }}>
            <div style={{ minWidth: 0 }}>
              <div className="rr-lbl" style={{ color: "var(--blue-hi)" }}>Retrieval only · no LLM calls</div>
              <h1 className="rr-h" style={{ margin: "6px 0 0", fontSize: 28, letterSpacing: "-0.03em" }}>Evaluation</h1>
              <p style={{ margin: "6px 0 0", fontSize: 14, lineHeight: 1.6, color: "var(--muted)", maxWidth: "70ch" }}>
                Deterministic retrieval metrics on the labelled question set. Groq is never called, so provider rate limits cannot affect these numbers.
              </p>
            </div>
            <button type="button" onClick={loadEvaluation} disabled={loading} className="rr-btn rr-btn-neu">
              {loading ? <span className="rr-dot live" /> : <Icon name="scan" />}
              {loading ? "Loading" : "Refresh"}
            </button>
          </div>
          {children}
        </div>
      </main>
    </div>
  );

  if (loading && !data) {
    return shell(
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }} aria-busy="true">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="rr-src" style={{ height: 96, padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
            <span style={{ width: "40%", height: 8, borderRadius: 2, background: "var(--raised)" }} />
            <span style={{ width: "60%", height: 18, borderRadius: 2, background: "var(--raised)" }} />
          </div>
        ))}
      </div>
    );
  }

  if (!data) {
    return shell(
      <div role="alert" style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", borderRadius: 12, border: "1px solid rgba(229,72,77,.4)", fontSize: 14 }}>
        <Icon name="alert" style={{ color: "var(--err)" }} />
        <span style={{ flex: 1 }}>{error || "No evaluation data available."}</span>
        <span className="rr-mono" style={{ fontSize: 11, color: "var(--muted)" }}>python -m backend.evaluation.retrieval_eval</span>
      </div>
    );
  }

  const { summary, questions } = data;
  const history = data.history ?? [];
  const previous = history.length >= 2 ? history[history.length - 2].summary : undefined;

  const precision = summary.precision_at_k?.[k];
  const recall = summary.recall_at_k?.[k];
  const f1k = summary.f1_at_k_by_k?.[k] ?? (k === "6" ? summary.f1_at_k : f1(precision, recall));
  const prevF1 = previous ? previous.f1_at_k_by_k?.[k] ?? (k === "6" ? previous.f1_at_k : f1(previous.precision_at_k?.[k], previous.recall_at_k?.[k])) : undefined;

  const types = ["all", ...Array.from(new Set(questions.map((q) => q.query_type || "normal")))];
  const shown = questions.filter((q) => typeFilter === "all" || (q.query_type || "normal") === typeFilter);

  return shell(
    <>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span className="rr-lbl" style={{ fontSize: 10 }}>Cutoff</span>
        <div className="rr-seg" role="group" aria-label="Metric cutoff k">
          {K_OPTIONS.filter((option) => summary.precision_at_k?.[option] !== undefined).map((option) => (
            <button key={option} type="button" aria-pressed={k === option} onClick={() => setK(option)}>
              @{option}
            </button>
          ))}
        </div>
        <span style={{ flex: 1 }} />
        <span className="rr-lbl" style={{ fontSize: 10 }}>
          {data.run_id ? `run ${data.run_id}` : ""}
          {data.created_at ? ` · ${new Date(data.created_at).toLocaleString()}` : ""}
        </span>
      </div>

      <section aria-label="Headline metrics" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12 }}>
        <Stat label={`Precision@${k}`} value={pct(precision)} delta={<Delta now={precision} before={previous?.precision_at_k?.[k]} />} />
        <Stat label={`Recall@${k}`} value={pct(recall)} delta={<Delta now={recall} before={previous?.recall_at_k?.[k]} />} />
        <Stat label={`F1@${k}`} value={pct(f1k)} delta={<Delta now={f1k} before={prevF1} />} primary />
        <Stat label="MRR" value={pct(summary.mrr)} delta={<Delta now={summary.mrr} before={previous?.mrr} />} />
        <Stat label="Hit@1" value={pct(summary.hit_at_1)} sub={`Hit@3 ${pct(summary.hit_at_3)} · Hit@5 ${pct(summary.hit_at_5)}`} delta={<Delta now={summary.hit_at_1} before={previous?.hit_at_1} />} />
        <Stat label="Refusal accuracy" value={pct(summary.refusal_accuracy)} sub={`${summary.unsupported_questions ?? 0} unsupported questions`} />
        <Stat label="Avg retrieval" value={`${Math.round(summary.avg_retrieval_latency_ms)} ms`} sub={summary.avg_context_chunks ? `${summary.avg_context_chunks} chunks per answer` : undefined} delta={<Delta now={summary.avg_retrieval_latency_ms} before={previous?.avg_retrieval_latency_ms} invert />} />
        <Stat label="Questions" value={String(summary.total_questions)} sub={`source ${pct(summary.source_accuracy, 0)} · section ${pct(summary.section_accuracy, 0)}`} />
      </section>

      {history.length > 0 && (
        <section className="rr-src" style={{ padding: 0, overflow: "hidden" }} aria-label="Run history">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", padding: "12px 16px", borderBottom: "1px solid var(--line)" }}>
            <span className="rr-h" style={{ fontSize: 14 }}>Run history</span>
            <span className="rr-lbl" style={{ fontSize: 10 }}>last {Math.min(history.length, 6)} of {history.length}</span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-mono)", fontSize: 12 }}>
              <thead>
                <tr className="rr-lbl" style={{ fontSize: 10 }}>
                  {["Run", `P@${k}`, `R@${k}`, "MRR", "Hit@1", "Latency"].map((head) => (
                    <th key={head} style={{ textAlign: head === "Run" ? "left" : "right", padding: "8px 16px", fontWeight: 500 }}>{head}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.slice(-6).reverse().map((run, index) => (
                  <tr key={run.run_id} style={{ borderTop: "1px solid var(--line)", color: index === 0 ? "var(--text)" : "var(--text-2)" }}>
                    <td style={{ padding: "8px 16px", whiteSpace: "nowrap" }}>
                      {index === 0 && <span className="rr-dot" style={{ background: "var(--blue)", marginRight: 8 }} />}
                      {new Date(run.created_at).toLocaleString()}
                    </td>
                    <td style={{ padding: "8px 16px", textAlign: "right" }}>{pct(run.summary.precision_at_k?.[k])}</td>
                    <td style={{ padding: "8px 16px", textAlign: "right" }}>{pct(run.summary.recall_at_k?.[k])}</td>
                    <td style={{ padding: "8px 16px", textAlign: "right" }}>{pct(run.summary.mrr)}</td>
                    <td style={{ padding: "8px 16px", textAlign: "right" }}>{pct(run.summary.hit_at_1)}</td>
                    <td style={{ padding: "8px 16px", textAlign: "right" }}>{Math.round(run.summary.avg_retrieval_latency_ms)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="rr-src" style={{ padding: 0, overflow: "hidden" }} aria-label="Question results">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, padding: "12px 16px", borderBottom: "1px solid var(--line)", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <span className="rr-h" style={{ fontSize: 14 }}>Questions</span>
            <span className="rr-lbl" style={{ fontSize: 10 }}>{shown.length} shown</span>
          </div>
          <div className="rr-seg" role="group" aria-label="Filter by question type" style={{ flexWrap: "wrap" }}>
            {types.map((type) => (
              <button key={type} type="button" aria-pressed={typeFilter === type} onClick={() => setTypeFilter(type)}>
                {type}
              </button>
            ))}
          </div>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr className="rr-lbl" style={{ fontSize: 10 }}>
                {[
                  ["#", "left"],
                  ["Question", "left"],
                  ["Type", "left"],
                  [`P@${k}`, "left"],
                  [`R@${k}`, "left"],
                  ["First hit", "right"],
                  ["Outcome", "left"],
                  ["Latency", "right"],
                ].map(([head, align]) => (
                  <th key={head} style={{ textAlign: align as "left" | "right", padding: "10px 16px", fontWeight: 500, whiteSpace: "nowrap" }}>{head}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((item) => {
                const unsupported = item.refused !== null && item.refused !== undefined;
                const outcome = unsupported
                  ? item.refused ? ["Refused", "ok"] : ["Answered", "err"]
                  : item.first_relevant_rank ? ["Found", "ok"] : ["Missed", "err"];
                return (
                  <tr key={item.id} className="rr-eval-row" style={{ borderTop: "1px solid var(--line)" }}>
                    <td className="rr-mono" style={{ padding: "10px 16px", color: "var(--muted)", fontSize: 11 }}>{item.id}</td>
                    <td style={{ padding: "10px 16px", color: "var(--text)", minWidth: 280, maxWidth: 460 }}>{item.question}</td>
                    <td style={{ padding: "10px 16px" }}><span className="rr-tag">{item.query_type || "normal"}</span></td>
                    <td style={{ padding: "10px 16px" }}>{unsupported ? <span className="rr-mono" style={{ color: "var(--muted)" }}>—</span> : <Meter value={item.precision_at_k?.[k]} />}</td>
                    <td style={{ padding: "10px 16px" }}>{unsupported ? <span className="rr-mono" style={{ color: "var(--muted)" }}>—</span> : <Meter value={item.recall_at_k?.[k]} />}</td>
                    <td className="rr-mono" style={{ padding: "10px 16px", textAlign: "right", color: item.first_relevant_rank === 1 ? "var(--text)" : "var(--text-2)" }}>
                      {item.first_relevant_rank ? `#${item.first_relevant_rank}` : "—"}
                    </td>
                    <td style={{ padding: "10px 16px" }}>
                      <span className="rr-lbl" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 6, color: "var(--text-2)" }}>
                        <span className={`rr-dot ${outcome[1]}`} />
                        {outcome[0]}
                      </span>
                    </td>
                    <td className="rr-mono" style={{ padding: "10px 16px", textAlign: "right", color: "var(--muted)" }}>
                      {typeof item.retrieval_latency_ms === "number" ? `${Math.round(item.retrieval_latency_ms)} ms` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
