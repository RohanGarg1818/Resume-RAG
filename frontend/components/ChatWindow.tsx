"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import ChatInput, { type ResponseMode } from "@/components/ChatInput";
import ChatMessageComponent from "@/components/ChatMessage";
import DynamicDataUpload, { type SessionFile } from "@/components/DynamicDataUpload";
import Icon from "@/components/Icon";
import IndexStatusPanel from "@/components/IndexStatusPanel";
import { GaugeDefs } from "@/components/RelevanceGauge";
import SourcesPanel, { type PanelTab } from "@/components/SourcesPanel";
import TopBar from "@/components/TopBar";
import { fetchIndexStatus, sendMessage } from "@/lib/api";
import { apiFetch } from "@/lib/config";
import { startRagSession } from "@/lib/session";
import type { ChatMessage, IndexStatus } from "@/types/chat";

const exampleQuestions = [
  "Which project used Redis, and what was it used for?",
  "What did Rohan work on at IEEE?",
  "How many problems has Rohan solved on LeetCode?",
];

let messageCounter = 0;
const nextId = () => `m${Date.now().toString(36)}${(messageCounter += 1)}`;

export default function ChatWindow() {
  const [voiceMode, setVoiceMode] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ResponseMode>("standard");

  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [sessionFiles, setSessionFiles] = useState<SessionFile[]>([]);
  const [indexing, setIndexing] = useState<string | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [animateId, setAnimateId] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const [tab, setTab] = useState<PanelTab>("sources");

  const scrollRef = useRef<HTMLDivElement>(null);

  // Prevent duplicate session creation in development/Strict Mode.
  const sessionStarted = useRef(false);

  // ---------------------------------------------------------------
  // Start a RAG session and read index facts when the workspace mounts
  // ---------------------------------------------------------------

  useEffect(() => {
    if (sessionStarted.current) {
      return;
    }

    sessionStarted.current = true;

    startRagSession().catch((sessionError) => {
      console.error("Failed to start RAG session:", sessionError);
      setError(
        `Unable to start the RAG session. ${
          sessionError instanceof Error ? sessionError.message : ""
        }`.trim()
      );
    });

    fetchIndexStatus().then(setStatus);
  }, []);

  // ---------------------------------------------------------------
  // End the RAG session when the page is refreshed/closed
  // ---------------------------------------------------------------

  useEffect(() => {
    const handlePageExit = () => {
      apiFetch("/api/session/end", {
        method: "POST",
        credentials: "include",
        keepalive: true,
      }).catch(() => {
        // The backend TTL cleanup remains the fallback.
      });
    };

    window.addEventListener("pagehide", handlePageExit);

    return () => {
      window.removeEventListener("pagehide", handlePageExit);
    };
  }, []);

  // Live retrieval timer shown while a question is in flight.
  useEffect(() => {
    if (!loading) return;
    const started = performance.now();
    const timer = window.setInterval(() => setElapsed(Math.round(performance.now() - started)), 50);
    return () => window.clearInterval(timer);
  }, [loading]);

  // Keep the newest entry in view.
  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, loading]);

  // ---------------------------------------------------------------
  // Send message
  // ---------------------------------------------------------------

  const handleMessage = useCallback(
    async (message: string, fromVoice = false) => {
      setError(null);
      setVoiceMode(fromVoice);
      setElapsed(0);

      setMessages((previous) => [
        ...previous,
        { id: nextId(), role: "user", content: message },
      ]);

      setLoading(true);

      try {
        const response = await sendMessage({
          message,
          conversation_id: conversationId,
          response_mode: mode,
          // The trace powers the inspector's Trace tab; it never calls the LLM again.
          debug: true,
        });

        setConversationId(response.conversation_id);

        const id = nextId();
        setMessages((previous) => [
          ...previous,
          {
            id,
            role: "assistant",
            content: response.answer,
            sources: response.sources,
            retrieval: response.retrieval ?? null,
            trace: response.debug ?? null,
            question: message,
          },
        ]);
        setSelectedId(id);
        setAnimateId(fromVoice ? null : id);
        setActiveSource(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      } finally {
        setLoading(false);
      }
    },
    [conversationId, mode]
  );

  // ---------------------------------------------------------------
  // Voice response
  // ---------------------------------------------------------------

  useEffect(() => {
    if (!loading && voiceMode && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];

      if (lastMessage.role === "assistant") {
        const utterance = new SpeechSynthesisUtterance(
          lastMessage.content.replace(/\s*\[S\d+\]/g, "")
        );

        utterance.lang = "en-IN";
        utterance.rate = 1;
        utterance.pitch = 1;

        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utterance);

        // Reset after speaking starts; the effect re-runs on this change.
        window.setTimeout(() => setVoiceMode(false), 0);
      }
    }
  }, [loading, messages, voiceMode]);

  const selected = useMemo(
    () => messages.find((item) => item.id === selectedId && item.role === "assistant") ?? null,
    [messages, selectedId]
  );

  const firstQuestion = messages.find((item) => item.role === "user")?.content;
  const turns = messages.filter((item) => item.role === "user").length;
  const sessionChunks = sessionFiles.reduce((sum, file) => sum + (file.chunks_added ?? 0), 0);

  const newThread = () => {
    setMessages([]);
    setConversationId(null);
    setSelectedId(null);
    setActiveSource(null);
    setError(null);
  };

  const turnById = new Map<string, number>();
  messages.reduce((count, item) => {
    const next = item.role === "user" ? count + 1 : count;
    turnById.set(item.id, next);
    return next;
  }, 0);

  return (
    <div style={{ position: "relative", minHeight: "100dvh", display: "flex", flexDirection: "column", overflow: "hidden", background: "var(--bg)" }}>
      <GaugeDefs />
      <div className="rr-grid-band rr-bluegrid" />
      <TopBar current="chat" status={status} />

      <div className="rr-shell" style={{ position: "relative", zIndex: 1, display: "flex", alignItems: "stretch" }}>
        {/* ========== LEFT RAIL ========== */}
        <aside className="rr-pane" aria-label="Library" style={{ width: 240, flex: "none", height: "100%", display: "flex", flexDirection: "column", background: "var(--surface)" }}>
          <div className="rr-scroll" style={{ flex: 1, minHeight: 0, padding: "16px 12px 8px", display: "flex", flexDirection: "column", gap: 20 }}>
            <DynamicDataUpload
              onUploaded={() => fetchIndexStatus().then(setStatus)}
              onFilesChange={setSessionFiles}
              onIndexingChange={setIndexing}
              onErrorChange={setIngestError}
            />

            <nav aria-label="Collections" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <div className="rr-lbl" style={{ padding: "0 8px 6px" }}>Collections</div>
              <div className="rr-row on" style={{ cursor: "default" }}>
                <span className="rr-dot" style={{ background: "var(--blue)" }} />
                <span style={{ flex: 1 }}>Rohan Garg · profile</span>
                <span className="rr-mono" style={{ fontSize: 11 }}>{status ? status.chunks : "—"}</span>
              </div>
              {Object.entries(status?.by_source ?? {}).map(([source, count]) => (
                <div key={source} className="rr-row" style={{ cursor: "default", paddingLeft: 22 }}>
                  <Icon name={source === "resume" ? "file" : "user"} />
                  <span className="rr-mono" style={{ flex: 1 }}>{source}</span>
                  <span className="rr-mono" style={{ fontSize: 10 }}>{count}</span>
                </div>
              ))}
              {sessionFiles.length > 0 && (
                <div className="rr-row" style={{ cursor: "default" }}>
                  <span className="rr-dot" style={{ background: "var(--faint)" }} />
                  <span style={{ flex: 1 }}>This session</span>
                  <span className="rr-mono" style={{ fontSize: 11 }}>{sessionChunks}</span>
                </div>
              )}
            </nav>
          </div>

          <IndexStatusPanel
            status={status}
            sessionChunks={sessionChunks}
            sessionFiles={sessionFiles.length}
            indexing={indexing}
            error={ingestError}
          />
        </aside>

        <div className="rr-resize rr-hide-sm" aria-hidden="true" />

        {/* ========== CENTER: CONVERSATION ========== */}
        <main className="rr-pane rr-main rr-dotgrid" style={{ flex: "999 1 560px", minWidth: 0, height: "100%", display: "flex", flexDirection: "column" }}>
          <div style={{ flex: "none", minHeight: 44, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "0 28px", borderBottom: "1px solid var(--line)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
              <span className="rr-h" style={{ fontSize: 14, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {firstQuestion ?? "New thread"}
              </span>
              <span className="rr-lbl" style={{ fontSize: 10, flex: "none" }}>
                {turns} {turns === 1 ? "turn" : "turns"}
              </span>
            </div>
            {messages.length > 0 && (
              <button type="button" className="rr-btn rr-btn-ghost" style={{ height: 28, fontSize: 12 }} onClick={newThread} disabled={loading}>
                <Icon name="plus" size={14} />
                New thread
              </button>
            )}
          </div>

          <div ref={scrollRef} className="rr-scroll" style={{ flex: 1, minHeight: 0, padding: "28px 28px 24px" }}>
            {messages.length === 0 ? (
              <div style={{ maxWidth: 640, margin: "0 auto", minHeight: "100%", display: "flex", flexDirection: "column", justifyContent: "center", gap: 32 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <div className="rr-lbl" style={{ color: "var(--blue-hi)" }}>Rohan Garg · profile · ready</div>
                  <h1 className="rr-h" style={{ margin: 0, fontSize: 40, lineHeight: 1.1, letterSpacing: "-0.03em" }}>
                    Ask anything about Rohan&apos;s profile.
                  </h1>
                  <p style={{ margin: 0, fontSize: 15, lineHeight: 1.65, color: "var(--muted)", maxWidth: "54ch" }}>
                    Answers come only from his resume, LinkedIn posts and any documents you add to this session. Each claim links to the passage it came from.
                  </p>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div className="rr-lbl" style={{ fontSize: 10 }}>Try</div>
                  {exampleQuestions.map((question, index) => (
                    <button
                      key={question}
                      type="button"
                      onClick={() => handleMessage(question)}
                      disabled={loading}
                      className="rr-btn rr-src"
                      style={{ height: "auto", padding: "14px 16px", justifyContent: "flex-start", gap: 14, fontSize: 14, fontWeight: 400, color: "var(--text)", whiteSpace: "normal", textAlign: "left" }}
                    >
                      <span className="rr-kbd">{index + 1}</span>
                      <span style={{ flex: 1 }}>{question}</span>
                      <Icon name="arrowRight" style={{ color: "var(--muted)" }} />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ maxWidth: 760, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
                {messages.map((message) => {
                  const isSelected = message.id === selectedId;
                  return (
                    <div
                      key={message.id}
                      onClick={message.role === "assistant" && !isSelected ? () => { setSelectedId(message.id); setActiveSource(null); } : undefined}
                      style={{
                        opacity: message.role === "assistant" && !isSelected && selectedId ? 0.78 : 1,
                        transition: "opacity 200ms var(--ease)",
                      }}
                    >
                      <ChatMessageComponent
                        message={message}
                        turn={turnById.get(message.id)}
                        animate={message.id === animateId}
                        activeSource={isSelected ? activeSource : null}
                        onSelectSource={(index) => {
                          setSelectedId(message.id);
                          setActiveSource(index);
                          setTab("sources");
                        }}
                        onInspect={() => {
                          setSelectedId(message.id);
                          setTab("trace");
                        }}
                      />
                    </div>
                  );
                })}

                {loading && (
                  <div style={{ display: "flex", gap: 12 }} aria-live="polite">
                    <div style={{ flex: "none", width: 52, paddingTop: 4 }}>
                      <div style={{ width: 16, height: 16, borderRadius: 4, background: "var(--blue)" }} />
                    </div>
                    <div className="rr-lbl" style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 8, paddingTop: 6 }}>
                      <span className="rr-dot live" />
                      Retrieving and answering · {elapsed} ms
                      <span className="rr-caret" aria-hidden="true" />
                    </div>
                  </div>
                )}

                {error && (
                  <div role="alert" style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 12, border: "1px solid rgba(229,72,77,.4)", fontSize: 13, color: "var(--text)" }}>
                    <Icon name="alert" style={{ color: "var(--err)" }} />
                    <span style={{ flex: 1 }}>{error}</span>
                  </div>
                )}
              </div>
            )}
            {messages.length === 0 && error && (
              <div role="alert" style={{ maxWidth: 640, margin: "16px auto 0", display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", borderRadius: 12, border: "1px solid rgba(229,72,77,.4)", fontSize: 13 }}>
                <Icon name="alert" style={{ color: "var(--err)" }} />
                <span style={{ flex: 1 }}>{error}</span>
              </div>
            )}
          </div>

          <div style={{ flex: "none", padding: "12px 28px 20px" }}>
            <div style={{ maxWidth: 760, margin: "0 auto" }}>
              <ChatInput
                onSubmit={handleMessage}
                disabled={loading}
                autoSubmitVoice={true}
                mode={mode}
                onModeChange={setMode}
                placeholder={messages.length ? "Ask a follow-up…" : undefined}
              />
            </div>
          </div>
        </main>

        <div className="rr-resize rr-hide-sm" aria-hidden="true" />

        {/* ========== RIGHT: SOURCES INSPECTOR ========== */}
        <SourcesPanel
          key={selected?.id ?? "none"}
          message={selected}
          tab={tab}
          onTab={setTab}
          activeSource={activeSource}
          onSelectSource={setActiveSource}
        />
      </div>
    </div>
  );
}
