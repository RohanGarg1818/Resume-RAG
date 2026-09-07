
"use client";

import { useRef, useState } from "react";

type Props = {
  onUploaded?: () => void;
};

export default function DynamicDataUpload({
  onUploaded,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [showFiles, setShowFiles] = useState(false);

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setMessage("");

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/session/upload`,
        {
          method: "POST",
          body: formData,
          credentials: "include",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      // Add uploaded filename to the dropdown
      setUploadedFiles((prev) => {
        if (prev.includes(data.filename)) {
          return prev;
        }

        return [...prev, data.filename];
      });

      // No success message
      setMessage("");

      onUploaded?.();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Upload failed"
      );
    } finally {
      setUploading(false);
    }
  };

  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    uploadFile(file);

    // Allow selecting the same file again later.
    event.target.value = "";
  };

  const isError =
    message.toLowerCase().includes("failed") ||
    message.toLowerCase().includes("error");

  return (
    <div className="relative flex items-center gap-2">
      {/* Uploaded files dropdown */}
      {uploadedFiles.length > 0 && (
        <div className="relative">
          <button
            type="button"
            onClick={() => setShowFiles((prev) => !prev)}
            className="flex items-center gap-2 rounded-xl border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-300 transition hover:bg-zinc-900"
          >
            <span>
              {uploadedFiles.length}{" "}
              {uploadedFiles.length === 1 ? "file" : "files"}
            </span>

            <span className="text-xs text-zinc-500">
              {showFiles ? "▲" : "▼"}
            </span>
          </button>

          {showFiles && (
            <div className="absolute right-0 top-full z-50 mt-2 w-72 rounded-xl border border-zinc-800 bg-zinc-950 p-2 shadow-xl">
              <p className="px-2 py-1.5 text-xs font-medium text-zinc-500">
                Input Data
              </p>

              <div className="max-h-48 overflow-y-auto">
                {uploadedFiles.map((filename) => (
                  <div
                    key={filename}
                    className="truncate rounded-lg px-2 py-2 text-sm text-zinc-300 hover:bg-zinc-900"
                    title={filename}
                  >
                    {filename}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Add Data button */}
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        className="rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2 text-sm font-medium text-zinc-300 transition hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {uploading ? "Processing..." : "+ Add Data"}
      </button>

      <input
        ref={inputRef}
        type="file"
        hidden
        accept=".pdf,.docx,.md,.markdown,.csv"
        onChange={handleChange}
      />

      {/* Error message only */}
      {message && (
        <p
          className={`absolute right-0 top-full z-50 mt-2 w-max max-w-[260px] rounded-lg border px-3 py-2 text-xs ${
            isError
              ? "border-red-900/50 bg-red-950/30 text-red-300"
              : "border-zinc-800 bg-zinc-950 text-zinc-400"
          }`}
        >
          {message}
        </p>
      )}
    </div>
  );
}
