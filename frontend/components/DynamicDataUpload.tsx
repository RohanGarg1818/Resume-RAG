"use client";

import { useEffect, useRef, useState } from "react";

import Icon from "@/components/Icon";
import { apiFetch } from "@/lib/config";

export type SessionFile = { filename: string; chunks_added?: number };

type Props = {
  onUploaded?: () => void;
  /** Reports the session's indexed files whenever they change. */
  onFilesChange?: (files: SessionFile[]) => void;
  /** Name of the file being indexed, or null when idle. */
  onIndexingChange?: (filename: string | null) => void;
  /** Last ingestion error, or null. */
  onErrorChange?: (message: string | null) => void;
};

const ACCEPT = ".pdf,.docx,.md,.markdown,.csv";

/**
 * Session ingestion for the left rail: a drop zone, the Ingest button and the
 * list of documents indexed for this browser session. Uploads go to
 * /api/session/upload and stay isolated from the permanent profile index.
 */
export default function DynamicDataUpload({
  onUploaded,
  onFilesChange,
  onIndexingChange,
  onErrorChange,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const [uploading, setUploading] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<SessionFile[]>([]);

  useEffect(() => {
    apiFetch("/api/session/status", { credentials: "include" })
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (Array.isArray(data?.files)) {
          setUploadedFiles(data.files);
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => { onFilesChange?.(uploadedFiles); }, [uploadedFiles, onFilesChange]);
  useEffect(() => { onIndexingChange?.(uploading); }, [uploading, onIndexingChange]);
  useEffect(() => { onErrorChange?.(message || null); }, [message, onErrorChange]);

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    setUploading(file.name);
    setMessage("");

    try {
      const response = await apiFetch("/api/session/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setUploadedFiles((prev) => {
        const next = prev.filter((item) => item.filename !== data.filename);
        return [...next, { filename: data.filename, chunks_added: data.chunks_added }];
      });

      onUploaded?.();
    } catch (error) {
      setMessage(`${file.name}: ${error instanceof Error ? error.message : "Upload failed"}`);
    } finally {
      setUploading(null);
    }
  };

  const removeFile = async (filename: string) => {
    try {
      const response = await apiFetch(
        `/api/session/files/${encodeURIComponent(filename)}`,
        { method: "DELETE", credentials: "include" }
      );
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Could not remove file");
      }
      setUploadedFiles((files) => files.filter((file) => file.filename !== filename));
      setMessage("");
      onUploaded?.();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not remove file");
    }
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      uploadFile(file);
    }
    // Allow selecting the same file again later.
    event.target.value = "";
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file && !uploading) {
      uploadFile(file);
    }
  };

  return (
    <section aria-label="Session documents" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <button
        type="button"
        className="rr-btn rr-btn-neu"
        onClick={() => inputRef.current?.click()}
        disabled={Boolean(uploading)}
        style={{ justifyContent: "space-between", height: 36 }}
      >
        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Icon name="plus" style={{ color: "var(--blue-hi)" }} />
          {uploading ? "Indexing…" : "Ingest"}
        </span>
        <span className="rr-lbl" style={{ fontSize: 10 }}>pdf · docx · md · csv</span>
      </button>

      <div
        onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className="rr-dash"
        style={{
          borderRadius: 12,
          padding: "10px 12px",
          textAlign: "center",
          fontSize: 12,
          color: "var(--muted)",
          transition: "border-color 200ms var(--ease), background 200ms var(--ease)",
          borderColor: dragOver ? "var(--blue-line)" : undefined,
          background: dragOver ? "rgba(43,107,255,0.04)" : "transparent",
        }}
      >
        Drop a file to add it to this session
      </div>

      <input ref={inputRef} type="file" hidden accept={ACCEPT} onChange={handleChange} />

      <div className="rr-lbl" style={{ padding: "8px 8px 2px", display: "flex", justifyContent: "space-between" }}>
        <span>Session uploads</span>
        <span>{uploadedFiles.length}</span>
      </div>

      {uploading && (
        <div className="rr-row on" style={{ cursor: "default" }}>
          <span className="rr-dot live" style={{ margin: "0 5px" }} />
          <span className="rr-mono" style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--text)" }}>{uploading}</span>
          <span className="rr-lbl" style={{ fontSize: 10, color: "var(--blue-hi)" }}>embed</span>
        </div>
      )}

      {uploadedFiles.length === 0 && !uploading && (
        <p style={{ margin: 0, padding: "0 8px", fontSize: 12, lineHeight: 1.5, color: "var(--muted)" }}>
          Uploads are searchable alongside the profile until you close this tab.
        </p>
      )}

      {uploadedFiles.map((file) => (
        <div key={file.filename} className="rr-row" style={{ cursor: "default" }}>
          <Icon name="file" />
          <span className="rr-mono" title={file.filename} style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {file.filename}
          </span>
          {file.chunks_added ? <span className="rr-mono" style={{ fontSize: 10 }}>{file.chunks_added}</span> : null}
          <button
            type="button"
            onClick={() => removeFile(file.filename)}
            className="rr-btn rr-btn-ghost rr-icon-btn"
            style={{ width: 24, height: 24, borderRadius: 6 }}
            aria-label={`Remove ${file.filename}`}
          >
            <Icon name="x" size={14} />
          </button>
        </div>
      ))}

      {message && (
        <p role="alert" className="rr-mono" style={{ margin: 0, padding: "6px 8px", fontSize: 11, lineHeight: 1.5, color: "var(--err)", border: "1px solid rgba(229,72,77,.35)", borderRadius: 8 }}>
          {message}
        </p>
      )}
    </section>
  );
}
