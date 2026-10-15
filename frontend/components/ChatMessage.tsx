"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import Icon from "@/components/Icon";
import {
  citedLabels,
  highlightTerms,
  linkCitations,
  sourceIndexFor,
  sourceLocation,
  sourceTitle,
} from "@/lib/citations";
import type { ChatMessage as ChatMessageType, Source } from "@/types/chat";

// Line-anchored: "Sources: ...", "**Sources:**", "## References", "Sources"
// on its own line. Everything from that line to the end is removed.
// ("Evidence:" is kept: evaluation answers legitimately use it as a heading.)
export const SOURCES_SECTION_RE =
  /(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?(?:Sources?|References)(?:\*\*|__)?[ \t]*(?::|(?:\*\*|__)?[ \t]*:|(?=\n|$))[\s\S]*$/i;

/** Answer body without a trailing Sources/References section. */
export function stripSourcesSection(text: string): string {
  return text.replace(SOURCES_SECTION_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Plain answer text for copying and speech: no citation markers. */
export function cleanResponse(text: string): string {
  return stripSourcesSection(text)
    .replace(/\s*\[S\d+\]/gi, "")
    .replace(/【S\d+】/gi, "")
    .trim();
}

function cleanForSpeech(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/_(.*?)_/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/\|/g, " ")
    .replace(/<br\s*\/?>/gi, " ")
    .replace(/\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function Highlighted({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length) return <>{text}</>;
  const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return (
    <>
      {text.split(pattern).map((part, index) =>
        index % 2 === 1 ? <mark key={index} className="rr-hl">{part}</mark> : <span key={index}>{part}</span>
      )}
    </>
  );
}

/** Numbered citation chip with a glass hover card showing the passage. */
function CitationChip({
  number,
  source,
  active,
  terms,
  onSelect,
}: {
  number: string;
  source: Source | undefined;
  active: boolean;
  terms: string[];
  onSelect: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [card, setCard] = useState<{ left: number; top: number; below: boolean } | null>(null);

  const open = () => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || !source) return;
    const width = 400;
    const left = Math.max(12, Math.min(rect.left - 16, window.innerWidth - width - 12));
    const below = rect.top < 240;
    setCard({ left, top: below ? rect.bottom + 10 : rect.top - 10, below });
  };

  if (!source) {
    return (
      <span className="rr-chip is-missing" title="This citation does not match a retrieved source">
        {number}
      </span>
    );
  }

  const excerpt = (source.excerpt ?? "").slice(0, 420);

  return (
    <span style={{ display: "inline" }} onMouseEnter={open} onMouseLeave={() => setCard(null)}>
      <button
        ref={ref}
        type="button"
        className="rr-chip"
        aria-pressed={active}
        aria-label={`Source ${number}: ${sourceTitle(source)}`}
        onClick={onSelect}
        onFocus={open}
        onBlur={() => setCard(null)}
      >
        {number}
      </button>
      {card && (
        <span
          role="tooltip"
          className="rr-glass rr-fade-in"
          style={{
            position: "fixed",
            zIndex: 60,
            left: card.left,
            top: card.top,
            transform: card.below ? "none" : "translateY(-100%)",
            width: 400,
            maxWidth: "calc(100vw - 24px)",
            padding: "14px 16px 16px",
            borderRadius: 16,
            display: "block",
            pointerEvents: "none",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
              <span className="rr-chip" style={{ animation: "none", margin: 0 }}>{number}</span>
              <span className="rr-mono" style={{ color: "var(--text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {sourceTitle(source)}
              </span>
            </span>
            <span className="rr-lbl" style={{ fontSize: 10, flex: "none" }}>{sourceLocation(source)}</span>
          </span>
          {excerpt && (
            <span style={{ display: "block", marginTop: 10, fontSize: 13, lineHeight: 1.6, color: "var(--text-2)" }}>
              “<Highlighted text={excerpt} terms={terms} />{excerpt.length < (source.excerpt ?? "").length ? "…" : ""}”
            </span>
          )}
          <span className="rr-lbl" style={{ display: "flex", justifyContent: "space-between", marginTop: 12, fontSize: 10 }}>
            <span>{typeof source.score === "number" ? `similarity ${source.score.toFixed(2)}` : "similarity —"}</span>
            <span style={{ color: "var(--blue-hi)" }}>Click to inspect</span>
          </span>
        </span>
      )}
    </span>
  );
}

interface ChatMessageProps {
  message: ChatMessageType;
  /** Position of the question (Q 01, Q 02 …) for user entries. */
  turn?: number;
  /** Reveal the answer progressively with a caret (newest answer only). */
  animate?: boolean;
  activeSource?: number | null;
  onSelectSource?: (index: number) => void;
  onInspect?: () => void;
}

function useReveal(text: string, enabled: boolean) {
  const words = useMemo(() => text.split(/(\s+)/), [text]);
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    // Long answers finish in roughly three seconds; reduced motion shows all at once.
    const step = reduce ? words.length : Math.max(2, Math.ceil(words.length / 140));
    const timer = window.setInterval(() => {
      setCount((current) => {
        if (current >= words.length) {
          window.clearInterval(timer);
          return current;
        }
        return Math.min(words.length, current + step);
      });
    }, 22);
    return () => window.clearInterval(timer);
  }, [enabled, words]);

  if (!enabled) return { shown: text, done: true };
  return { shown: words.slice(0, count).join(""), done: count >= words.length };
}

export default function ChatMessage({
  message,
  turn,
  animate = false,
  activeSource = null,
  onSelectSource,
  onInspect,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const sources = useMemo(() => message.sources ?? [], [message.sources]);
  const body = useMemo(() => stripSourcesSection(message.content), [message.content]);
  const { shown, done } = useReveal(body, animate && !isUser);
  const linked = useMemo(() => linkCitations(shown, sources), [shown, sources]);
  const terms = useMemo(() => highlightTerms(message.question), [message.question]);
  const cited = useMemo(() => citedLabels(body), [body]);

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);

  if (isUser) {
    return (
      <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
        <span className="rr-lbl" style={{ flex: "none", width: 52, color: "var(--blue-hi)" }}>
          Q {String(turn ?? 0).padStart(2, "0")}
        </span>
        <div style={{ fontSize: 16, lineHeight: 1.5, color: "var(--text)", letterSpacing: "-0.01em", whiteSpace: "pre-wrap", minWidth: 0, overflowWrap: "anywhere" }}>
          {message.content}
        </div>
      </div>
    );
  }

  function toggleSpeech() {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const speech = window.speechSynthesis;
    if (isSpeaking) {
      speech.cancel();
      setIsSpeaking(false);
      return;
    }
    const speechText = cleanForSpeech(cleanResponse(message.content));
    if (!speechText) return;
    if (speech.speaking || speech.pending) speech.cancel();
    const utterance = new SpeechSynthesisUtterance(speechText);
    utterance.lang = "en-IN";
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = (event) => {
      if (event.error !== "canceled") console.error("Speech synthesis error:", event.error);
      setIsSpeaking(false);
    };
    speech.speak(utterance);
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(cleanResponse(message.content));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      // Clipboard can be unavailable on insecure origins.
    }
  }

  const retrieval = message.retrieval;
  const meta: string[] = [];
  if (retrieval) {
    const refusedBeforeModel = retrieval.confidence === "low" || retrieval.confidence === "none";
    if (retrieval.chunks > 0 && refusedBeforeModel) {
      meta.push(`Retrieved ${retrieval.chunks} chunk${retrieval.chunks === 1 ? "" : "s"} in ${Math.round(retrieval.latency_ms)}ms`);
      meta.push("low confidence · not sent to model");
    } else if (retrieval.chunks > 0) {
      meta.push(`Retrieved ${retrieval.chunks} chunk${retrieval.chunks === 1 ? "" : "s"} in ${Math.round(retrieval.latency_ms)}ms`);
      meta.push(`${cited.length} cited`);
      if (retrieval.llm_latency_ms > 0) meta.push(`gen ${(retrieval.llm_latency_ms / 1000).toFixed(1)}s`);
      if (retrieval.route) meta.push(retrieval.route.replace(/_/g, " "));
    } else {
      meta.push("No retrieval · direct reply");
    }
  }

  return (
    <div style={{ display: "flex", gap: 12 }}>
      <div style={{ flex: "none", width: 52, paddingTop: 4 }}>
        <div
          role="img"
          aria-label="Assistant"
          style={{
            width: 16, height: 16, borderRadius: 4,
            background: message.error ? "transparent" : "var(--blue)",
            border: message.error ? "1px solid var(--err)" : "none",
            boxShadow: message.error ? "none" : "inset 0 1px 0 rgba(255,255,255,.25)",
          }}
        />
      </div>
      <div style={{ minWidth: 0, flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="rr-prose" style={message.error ? { color: "var(--err)" } : undefined}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => {
                if (href?.startsWith("#cite-")) {
                  const label = href.slice("#cite-".length);
                  const index = sourceIndexFor(label, sources);
                  return (
                    <CitationChip
                      number={index >= 0 ? String(index + 1) : "?"}
                      source={index >= 0 ? sources[index] : undefined}
                      active={index >= 0 && activeSource === index}
                      terms={terms}
                      onSelect={() => index >= 0 && onSelectSource?.(index)}
                    />
                  );
                }
                return (
                  <a href={href} target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                );
              },
              table: ({ children }) => (
                <div className="rr-table-wrap">
                  <table>{children}</table>
                </div>
              ),
            }}
          >
            {linked}
          </ReactMarkdown>
          {!done && <span className="rr-caret" aria-hidden="true" />}
        </div>

        {done && (
          <div className="rr-fade-in" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {meta.length > 0 && (
              <div className="rr-lbl" style={{ fontSize: 10, display: "flex", flexWrap: "wrap", gap: "4px 10px" }}>
                {meta.map((item, index) => (
                  <span key={item} style={{ display: "inline-flex", gap: 10, color: index === 1 && cited.length ? "var(--blue-hi)" : undefined }}>
                    {index > 0 && <span style={{ color: "var(--faint)" }} aria-hidden="true">·</span>}
                    {item}
                  </span>
                ))}
              </div>
            )}
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              <button type="button" className="rr-btn rr-btn-ghost" style={{ height: 28, fontSize: 12 }} onClick={copy}>
                <Icon name={copied ? "check" : "copy"} size={14} />
                {copied ? "Copied" : "Copy"}
              </button>
              <button type="button" className="rr-btn rr-btn-ghost" style={{ height: 28, fontSize: 12 }} onClick={toggleSpeech}>
                <Icon name={isSpeaking ? "stop" : "volume"} size={14} />
                {isSpeaking ? "Stop" : "Listen"}
              </button>
              {onInspect && (sources.length > 0 || message.trace) && (
                <button type="button" className="rr-btn rr-btn-ghost" style={{ height: 28, fontSize: 12 }} onClick={onInspect}>
                  <Icon name="scan" size={14} />
                  Inspect retrieval
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
