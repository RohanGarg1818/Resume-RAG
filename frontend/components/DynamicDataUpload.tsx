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

const uploadFile = async (file: File) => {
const formData = new FormData();

formData.append("file", file);

setUploading(true);
setMessage("");

try {
  const response = await fetch(
    "http://localhost:8000/api/session/upload",
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

  setMessage(
    `${data.filename} added successfully`
  );

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
<div className="relative flex flex-col items-end">
<button
type="button"
onClick={() => inputRef.current?.click()}
disabled={uploading}
className="rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2 text-sm font-medium text-zinc-300 transition hover hover hover disabled disabled"
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

  {message && (
    <p
      className={`absolute right-0 top-full mt-2 w-max max-w-[260px] rounded-lg border px-3 py-2 text-xs ${
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