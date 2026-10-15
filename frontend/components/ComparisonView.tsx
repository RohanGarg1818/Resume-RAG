"use client";

import { useCallback, useRef, useState } from "react";

import Icon from "@/components/Icon";
import { compareCandidates } from "@/lib/comparison";
import { apiFetch } from "@/lib/config";
import type {
  CandidateAnalysis,
  CandidateScore,
  ComparisonResponse,
  RequirementComparison,
  RequirementMatch,
} from "@/types/comparison";

type Tab = "input" | "results";
type Step = "session" | "upload" | "analyze";

const STEPS: { id: Step; label: string }[] = [
  { id: "session", label: "Session" },
  { id: "upload", label: "Index resume" },
  { id: "analyze", label: "Compare" },
];

const SCORE_ROWS: { key: keyof CandidateScore; label: string }[] = [
  { key: "required_skills", label: "Required skills" },
  { key: "experience", label: "Experience" },
  { key: "projects", label: "Projects" },
  { key: "preferred_skills", label: "Preferred skills" },
  { key: "education", label: "Education" },
];

export default function ComparisonView() {
  const [tab, setTab] = useState<Tab>("input");
  const [jobRequirements, setJobRequirements] = useState("");
  const [candidateBFile, setCandidateBFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState<Step | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [expandedReq, setExpandedReq] = useState<string | null>(null);
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setCandidateBFile(file);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCandidateBFile(file);
    }
    e.target.value = "";
  }, []);

  const handleCompare = useCallback(async () => {
    if (!jobRequirements.trim()) {
      setError("Please enter job requirements.");
      return;
    }
    if (!candidateBFile) {
      setError("Please upload the other candidate's resume.");
      return;
    }

    setError(null);
    setLoading(true);

    try {
      // Step 1: Start a session
      setStep("session");
      const sessionRes = await apiFetch("/api/session/start", { method: "POST", credentials: "include" });
      if (!sessionRes.ok) throw new Error("Failed to start session.");

      // Step 2: Upload the other candidate's resume
      setStep("upload");
      const formData = new FormData();
      formData.append("file", candidateBFile);
      const uploadRes = await apiFetch("/api/session/upload", { method: "POST", body: formData, credentials: "include" });
      if (!uploadRes.ok) {
        const data = await uploadRes.json();
        throw new Error(data.detail || "Failed to upload resume.");
      }

      // Step 3: Run comparison
      setStep("analyze");
      const comparison = await compareCandidates({ job_requirements: jobRequirements });

      setResult(comparison);
      setTab("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Comparison failed.");
    } finally {
      setLoading(false);
      setStep(null);
    }
  }, [jobRequirements, candidateBFile]);

  const toggleReq = (req: string) => {
    setExpandedSection(null);
    setExpandedReq((prev) => (prev === req ? null : req));
  };

  const toggleCandidateSection = (section: string) => {
    setExpandedReq(null);
    setExpandedSection((prev) => (prev === section ? null : section));
  };

  const requirementLines = jobRequirements.split("\n").filter((line) => line.trim()).length;

  // ----------------------------------------------------------------
  // Input
  // ----------------------------------------------------------------

  if (tab === "input") {
    const stepIndex = step ? STEPS.findIndex((s) => s.id === step) : -1;
    return (
      <div className="rr-scroll" style={{ height: "100%", padding: "28px clamp(16px, 4vw, 32px) 56px" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
          <div>
            <div className="rr-lbl" style={{ color: "var(--blue-hi)" }}>Candidate comparison</div>
            <h1 className="rr-h" style={{ margin: "6px 0 0", fontSize: 28, letterSpacing: "-0.03em" }}>Compare two candidates</h1>
            <p style={{ margin: "6px 0 0", fontSize: 14, lineHeight: 1.6, color: "var(--muted)", maxWidth: "70ch" }}>
              Paste the job requirements and add the other candidate&apos;s resume. Both are scored against each requirement using only what their resumes state.
            </p>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 20, alignItems: "stretch" }}>
            {/* Requirements */}
            <div style={{ flex: "999 1 520px", minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <label htmlFor="rr-requirements" className="rr-lbl">Job requirements</label>
                <span className="rr-lbl" style={{ fontSize: 10 }}>{requirementLines} lines · {jobRequirements.length} chars</span>
              </div>
              <div className="rr-well rr-composer" style={{ borderRadius: 16, padding: "14px 16px", flex: 1, display: "flex" }}>
                <textarea
                  id="rr-requirements"
                  value={jobRequirements}
                  onChange={(e) => setJobRequirements(e.target.value)}
                  placeholder={"Senior Backend Engineer\n- 3+ years with Python or Go\n- REST APIs and PostgreSQL\n- Docker; Redis preferred"}
                  rows={12}
                  style={{ flex: 1, resize: "vertical", border: 0, outline: "none", background: "transparent", width: "100%", fontSize: 14, lineHeight: 1.6, color: "var(--text)", fontFamily: "inherit", minHeight: 240 }}
                />
              </div>
            </div>

            {/* Candidates */}
            <div style={{ flex: "1 1 320px", minWidth: 0, display: "flex", flexDirection: "column", gap: 8 }}>
              <span className="rr-lbl">Candidates</span>

              <div className="rr-src" style={{ padding: 16, display: "flex", gap: 12, alignItems: "flex-start" }}>
                <span style={{ flex: "none", width: 24, height: 24, borderRadius: 6, background: "var(--blue)", color: "#fff", display: "inline-flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--font-mono)", fontSize: 11 }}>A</span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 14, color: "var(--text)" }}>Rohan Garg</div>
                  <div className="rr-lbl" style={{ fontSize: 10, marginTop: 4, textTransform: "none", letterSpacing: "0.02em" }}>Permanent profile · resume + LinkedIn</div>
                </div>
              </div>

              <div
                role="button"
                tabIndex={0}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); } }}
                className={`rr-focus ${candidateBFile ? "rr-src" : "rr-dash"}`}
                aria-label="Add the other candidate's resume"
                style={{
                  flex: 1,
                  minHeight: 140,
                  borderRadius: 12,
                  padding: 16,
                  cursor: "pointer",
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                  transition: "border-color 200ms var(--ease), background 200ms var(--ease)",
                  borderColor: dragOver ? "var(--blue-line)" : undefined,
                  background: dragOver ? "rgba(43,107,255,0.04)" : undefined,
                }}
              >
                <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                  <span style={{ flex: "none", width: 24, height: 24, borderRadius: 6, border: "1px solid var(--line-2)", color: "var(--muted)", display: "inline-flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--font-mono)", fontSize: 11 }}>B</span>
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontSize: 14, color: "var(--text)" }}>Other candidate</div>
                    <div className="rr-lbl" style={{ fontSize: 10, marginTop: 4, textTransform: "none", letterSpacing: "0.02em" }}>pdf · docx · md · csv</div>
                  </div>
                </div>
                {candidateBFile ? (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", borderRadius: 8, background: "#080B10", boxShadow: "var(--neu-in)" }}>
                    <Icon name="file" style={{ color: "var(--blue-hi)" }} />
                    <span className="rr-mono" style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text)" }}>{candidateBFile.name}</span>
                    <span className="rr-mono" style={{ fontSize: 11, color: "var(--muted)" }}>{(candidateBFile.size / 1024).toFixed(0)} KB</span>
                    <button
                      type="button"
                      className="rr-btn rr-btn-ghost rr-icon-btn"
                      style={{ width: 24, height: 24, borderRadius: 6 }}
                      aria-label="Remove file"
                      onClick={(e) => { e.stopPropagation(); setCandidateBFile(null); }}
                    >
                      <Icon name="x" size={14} />
                    </button>
                  </div>
                ) : (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--muted)", fontSize: 13 }}>
                    <Icon name="upload" />
                    Drop a resume here or click to browse
                  </div>
                )}
                <input ref={fileInputRef} type="file" hidden accept=".pdf,.docx,.md,.markdown,.csv" onChange={handleFileSelect} />
              </div>
            </div>
          </div>

          {error && (
            <div role="alert" style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 12, border: "1px solid rgba(229,72,77,.4)", fontSize: 13 }}>
              <Icon name="alert" style={{ color: "var(--err)" }} />
              <span style={{ flex: 1 }}>{error}</span>
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={handleCompare}
              disabled={loading || !jobRequirements.trim() || !candidateBFile}
              className="rr-btn rr-btn-primary"
              style={{ height: 40, padding: "0 18px", fontSize: 14 }}
            >
              <Icon name="compare" />
              {loading ? "Comparing…" : "Compare candidates"}
            </button>
            {loading && (
              <ol aria-label="Progress" style={{ display: "flex", gap: 6, listStyle: "none", margin: 0, padding: 0, flexWrap: "wrap" }}>
                {STEPS.map((s, index) => {
                  const state = index < stepIndex ? "done" : index === stepIndex ? "active" : "pending";
                  return (
                    <li key={s.id} className="rr-lbl" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 6, height: 26, padding: "0 10px", borderRadius: 8, border: `1px solid ${state === "active" ? "var(--blue-line)" : "var(--line)"}`, color: state === "pending" ? "var(--muted)" : state === "active" ? "var(--blue-hi)" : "var(--text-2)" }}>
                      {state === "done" ? <Icon name="check" size={12} /> : state === "active" ? <span className="rr-dot live" /> : <span className="rr-dot" />}
                      {s.label}
                    </li>
                  );
                })}
              </ol>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ----------------------------------------------------------------
  // Results
  // ----------------------------------------------------------------

  if (!result) return null;

  const nameA = result.candidate_a.name;
  const nameB = result.candidate_b.name;
  const reqBetterA = result.requirement_comparisons.filter((r) => r.better_match === nameA).length;
  const reqBetterB = result.requirement_comparisons.filter((r) => r.better_match === nameB).length;
  const reqTied = result.requirement_comparisons.filter((r) => r.better_match === "tie").length;
  const winnerA = !result.is_tie && result.recommendation === nameA;
  const winnerB = !result.is_tie && result.recommendation === nameB;

  return (
    <div className="rr-scroll" style={{ height: "100%", padding: "28px clamp(16px, 4vw, 32px) 56px" }}>
      <div style={{ maxWidth: 1080, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
        <div>
          <button type="button" onClick={() => setTab("input")} className="rr-btn rr-btn-ghost" style={{ height: 28, fontSize: 12, marginLeft: -8 }}>
            <Icon name="chevronRight" size={14} style={{ transform: "rotate(180deg)" }} />
            New comparison
          </button>
        </div>

        {/* Recommendation */}
        <section className="rr-src" style={{ padding: 0, overflow: "hidden" }} aria-label="Recommendation">
          <div style={{ padding: "20px 24px", display: "flex", flexWrap: "wrap", gap: 24, alignItems: "flex-end", justifyContent: "space-between" }}>
            <div style={{ minWidth: 0 }}>
              <div className="rr-lbl" style={{ color: "var(--blue-hi)" }}>Recommendation</div>
              <h2 className="rr-h" style={{ margin: "8px 0 0", fontSize: 30, letterSpacing: "-0.03em" }}>
                {result.is_tie ? "Both are suitable" : result.recommendation}
                {!result.is_tie && <span style={{ fontSize: 16, color: "var(--muted)", marginLeft: 12, letterSpacing: "-0.01em" }}>is the stronger fit</span>}
              </h2>
              <div className="rr-lbl" style={{ fontSize: 10, marginTop: 10, display: "flex", gap: 10, flexWrap: "wrap" }}>
                <span>{reqBetterA} favour {nameA}</span>
                <span style={{ color: "var(--faint)" }} aria-hidden="true">·</span>
                <span>{reqTied} tied</span>
                <span style={{ color: "var(--faint)" }} aria-hidden="true">·</span>
                <span>{reqBetterB} favour {nameB}</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 12 }}>
              <ScoreTile label={nameA} score={result.score_a.overall} winner={winnerA} />
              <ScoreTile label={nameB} score={result.score_b.overall} winner={winnerB} />
            </div>
          </div>

          {/* Score breakdown */}
          <div style={{ borderTop: "1px solid var(--line)", padding: "16px 24px", display: "grid", gridTemplateColumns: "minmax(120px, 160px) minmax(0, 1fr)", gap: "10px 20px", alignItems: "center" }}>
            <span className="rr-lbl" style={{ fontSize: 10 }}>Breakdown</span>
            <span className="rr-lbl" style={{ fontSize: 10, display: "flex", gap: 16 }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 10, height: 4, borderRadius: 2, background: "var(--blue)" }} />{nameA}</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}><span style={{ width: 10, height: 4, borderRadius: 2, background: "#7A8596" }} />{nameB}</span>
            </span>
            {SCORE_ROWS.map((row) => (
              <ScoreRow key={row.key} label={row.label} a={result.score_a[row.key]} b={result.score_b[row.key]} />
            ))}
          </div>
        </section>

        {result.reasoning && (
          <section className="rr-src" style={{ padding: "18px 24px" }} aria-label="Reasoning">
            <div className="rr-lbl" style={{ fontSize: 10 }}>
              Why {result.is_tie ? "both are suitable" : `${result.recommendation} is recommended`}
            </div>
            <p style={{ margin: "10px 0 0", fontSize: 15, lineHeight: 1.65, color: "var(--text)", maxWidth: "68ch" }}>{result.reasoning}</p>
          </section>
        )}

        {/* Requirement comparison */}
        <section className="rr-src" style={{ padding: 0, overflowX: "auto" }} aria-label="Requirement comparison">
          <div style={{ minWidth: 640 }}>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 72px 72px 140px 24px", gap: 12, padding: "10px 20px", borderBottom: "1px solid var(--line)" }} className="rr-lbl">
            <span style={{ fontSize: 10 }}>Requirement</span>
            <span style={{ fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={nameA}>{nameA}</span>
            <span style={{ fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={nameB}>{nameB}</span>
            <span style={{ fontSize: 10 }}>Better match</span>
            <span />
          </div>
          {result.requirement_comparisons.map((rc) => (
            <RequirementRow
              key={rc.requirement}
              rc={rc}
              candidateAName={nameA}
              candidateBName={nameB}
              expanded={expandedReq === rc.requirement}
              onToggle={() => toggleReq(rc.requirement)}
            />
          ))}
          {result.requirement_comparisons.length === 0 && (
            <p style={{ margin: 0, padding: "14px 20px", fontSize: 13, color: "var(--muted)" }}>No requirements could be extracted from the text.</p>
          )}
          </div>
        </section>

        {(result.missing_requirements.length > 0 || result.ambiguous_requirements.length > 0) && (
          <section className="rr-src" style={{ padding: "16px 20px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 20 }} aria-label="Evidence notes">
            {result.missing_requirements.length > 0 && (
              <NoteList tone="warn" title="No evidence in either resume" items={result.missing_requirements} />
            )}
            {result.ambiguous_requirements.length > 0 && (
              <NoteList tone="muted" title="Low-confidence evidence" items={result.ambiguous_requirements} />
            )}
          </section>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 16 }}>
          <CandidateCard analysis={result.candidate_a} primary expanded={expandedSection === "a"} onToggle={() => toggleCandidateSection("a")} />
          <CandidateCard analysis={result.candidate_b} expanded={expandedSection === "b"} onToggle={() => toggleCandidateSection("b")} />
        </div>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------

function ScoreTile({ label, score, winner }: { label: string; score: number; winner: boolean }) {
  return (
    <div
      className={winner ? "rr-neu" : undefined}
      style={{
        minWidth: 132,
        padding: "12px 16px",
        borderRadius: 12,
        border: winner ? "1px solid var(--blue-line)" : "1px solid var(--line)",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span className="rr-lbl" style={{ fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 160 }} title={label}>
        {label}
      </span>
      <span className="rr-mono" style={{ fontSize: 28, letterSpacing: "-0.02em", color: winner ? "var(--text)" : "var(--text-2)" }}>
        {Math.round(score * 100)}
        <span style={{ fontSize: 13, color: "var(--muted)" }}>%</span>
      </span>
    </div>
  );
}

function ScoreRow({ label, a, b }: { label: string; a: number; b: number }) {
  const bar = (value: number, color: string) => (
    <span style={{ position: "relative", flex: 1, height: 6, borderRadius: 3, background: "#06080B", boxShadow: "inset 1px 1px 2px rgba(0,0,0,.8)" }}>
      <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${Math.max(0, Math.min(1, value)) * 100}%`, borderRadius: 3, background: color, transition: "width 300ms var(--ease)" }} />
    </span>
  );
  return (
    <>
      <span style={{ fontSize: 13, color: "var(--text-2)" }}>{label}</span>
      <span style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {bar(a, "var(--blue)")}
          <span className="rr-mono" style={{ width: 40, textAlign: "right", color: "var(--text)" }}>{Math.round(a * 100)}%</span>
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {bar(b, "#7A8596")}
          <span className="rr-mono" style={{ width: 40, textAlign: "right", color: "var(--text-2)" }}>{Math.round(b * 100)}%</span>
        </span>
      </span>
    </>
  );
}

function MetTag({ met }: { met: boolean }) {
  return (
    <span className="rr-lbl" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 6, color: met ? "var(--text)" : "var(--muted)" }}>
      <span className={`rr-dot ${met ? "ok" : ""}`} />
      {met ? "Met" : "Not met"}
    </span>
  );
}

function RequirementRow({
  rc,
  candidateAName,
  candidateBName,
  expanded,
  onToggle,
}: {
  rc: RequirementComparison;
  candidateAName: string;
  candidateBName: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const tie = rc.better_match === "tie";
  const favoursA = rc.better_match === candidateAName;

  return (
    <div style={{ borderBottom: "1px solid var(--line)" }}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="rr-row rr-focus"
        style={{ borderRadius: 0, display: "grid", gridTemplateColumns: "minmax(0, 1fr) 72px 72px 140px 24px", gap: 12, padding: "12px 20px", minHeight: 44, alignItems: "center", cursor: "pointer", color: "var(--text)" }}
      >
        <span style={{ fontSize: 13, lineHeight: 1.5 }}>{rc.requirement}</span>
        <MetTag met={rc.candidate_a.met} />
        <MetTag met={rc.candidate_b.met} />
        <span
          className="rr-tag"
          style={{ justifySelf: "start", maxWidth: "100%", ...(tie ? {} : favoursA ? { color: "var(--blue-hi)", borderColor: "var(--blue-line)", background: "var(--blue-wash)" } : { color: "var(--text)" }) }}
          title={tie ? "Tie" : rc.better_match}
        >
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", maxWidth: 110 }}>{tie ? "Tie" : rc.better_match}</span>
        </span>
        <Icon name="chevronRight" size={14} style={{ color: "var(--muted)", transform: `rotate(${expanded ? 90 : 0}deg)`, transition: "transform 200ms" }} />
      </button>
      {expanded && (
        <div className="rr-fade-in" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12, padding: "4px 20px 16px" }}>
          <EvidenceBlock label={candidateAName} match={rc.candidate_a} />
          <EvidenceBlock label={candidateBName} match={rc.candidate_b} />
        </div>
      )}
    </div>
  );
}

function EvidenceBlock({ label, match }: { label: string; match: RequirementMatch }) {
  return (
    <div className="rr-well" style={{ borderRadius: 10, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
        <span className="rr-mono" style={{ color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
        <span className="rr-lbl" style={{ fontSize: 10 }}>confidence {match.confidence}</span>
      </div>
      {match.evidence ? (
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "var(--text-2)" }}>{match.evidence}</p>
      ) : (
        <p style={{ margin: 0, fontSize: 13, color: "var(--muted)" }}>No evidence in this resume.</p>
      )}
    </div>
  );
}

function NoteList({ tone, title, items }: { tone: "warn" | "muted"; title: string; items: string[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <span className="rr-lbl" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 6 }}>
        <span className={`rr-dot ${tone === "warn" ? "warn" : ""}`} />
        {title}
      </span>
      <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 4, fontSize: 13, color: "var(--text-2)" }}>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

function ListBlock({ title, dot, items, limit }: { title: string; dot?: "ok" | "err"; items: string[]; limit?: number }) {
  if (items.length === 0) return null;
  const shown = typeof limit === "number" ? items.slice(0, limit) : items;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span className="rr-lbl" style={{ fontSize: 10, display: "inline-flex", alignItems: "center", gap: 6 }}>
        {dot && <span className={`rr-dot ${dot}`} />}
        {title}
      </span>
      <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 4 }}>
        {shown.map((item) => (
          <li key={item} style={{ fontSize: 13, lineHeight: 1.5, color: "var(--text-2)" }}>{item}</li>
        ))}
      </ul>
      {typeof limit === "number" && items.length > limit && (
        <span className="rr-lbl" style={{ fontSize: 10 }}>+{items.length - limit} more</span>
      )}
    </div>
  );
}

function CandidateCard({
  analysis,
  expanded,
  onToggle,
  primary = false,
}: {
  analysis: CandidateAnalysis;
  expanded: boolean;
  onToggle: () => void;
  primary?: boolean;
}) {
  return (
    <section className="rr-src" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }} aria-label={analysis.name}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "14px 18px", borderBottom: "1px solid var(--line)" }}>
        <span style={{ flex: "none", width: 10, height: 10, borderRadius: 3, background: primary ? "var(--blue)" : "#7A8596" }} />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: 14, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{analysis.name}</div>
          <div className="rr-lbl" style={{ fontSize: 10, marginTop: 2 }}>fit {Math.round(analysis.score.overall * 100)}%</div>
        </div>
        <button type="button" onClick={onToggle} className="rr-btn rr-btn-ghost" aria-expanded={expanded} style={{ height: 28, fontSize: 12 }}>
          {expanded ? "Collapse" : "Details"}
        </button>
      </div>

      <div style={{ padding: "14px 18px", display: "flex", flexDirection: "column", gap: 14 }}>
        <ListBlock title="Strengths" dot="ok" items={analysis.strengths} limit={expanded ? undefined : 3} />
        <ListBlock title="Gaps" dot="err" items={analysis.gaps} limit={expanded ? undefined : 3} />
        {expanded && <ListBlock title="Unique to this candidate" items={analysis.unique_skills} />}
        {expanded && <ListBlock title="Shared strengths" items={analysis.shared_strengths} />}
      </div>

      {expanded && analysis.evidence.length > 0 && (
        <div style={{ borderTop: "1px solid var(--line)", padding: "14px 18px", display: "flex", flexDirection: "column", gap: 8 }}>
          <span className="rr-lbl" style={{ fontSize: 10 }}>Retrieved evidence</span>
          {analysis.evidence.slice(0, 8).map((e, i) => (
            <div key={i} className="rr-well" style={{ borderRadius: 10, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="rr-lbl" style={{ fontSize: 10, display: "flex", justifyContent: "space-between", gap: 8 }}>
                <span>{e.section || "—"}</span>
                <span>sim {e.relevance.toFixed(2)}</span>
              </span>
              <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.55, color: "var(--text-2)", display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{e.text}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
