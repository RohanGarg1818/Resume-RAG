
"use client";

import { useState } from "react";

import type { ChatMessage as ChatMessageType } from "@/types/chat";
import SourceCard from "@/components/SourceCard";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessageProps {
  message: ChatMessageType;
}

function cleanForSpeech(text: string): string {
  return text
    // Remove fenced code blocks
    .replace(/```[\s\S]*?```/g, "")

    // Remove headings
    .replace(/^#{1,6}\s+/gm, "")

    // Remove bold
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")

    // Remove italic
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/_(.*?)_/g, "$1")

    // Remove inline code
    .replace(/`([^`]+)`/g, "$1")

    // Convert markdown links to their visible text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")

    // Remove unordered list markers
    .replace(/^\s*[-*+]\s+/gm, "")

    // Remove numbered list markers
    .replace(/^\s*\d+\.\s+/gm, "")

    // Remove blockquote markers
    .replace(/^\s*>\s?/gm, "")

    // Replace table pipes with spaces
    .replace(/\|/g, " ")

    // Remove HTML line breaks
    .replace(/<br\s*\/?>/gi, " ")

    // Remove newlines
    .replace(/\n+/g, " ")

    // Remove extra spaces
    .replace(/\s+/g, " ")

    .trim();
}

export default function ChatMessage({
  message,
}: ChatMessageProps) {
  const isUser = message.role === "user";

  const [isSpeaking, setIsSpeaking] = useState(false);

  function toggleSpeech() {
    if (typeof window === "undefined") {
      return;
    }

    if (!("speechSynthesis" in window)) {
      console.error(
        "Speech synthesis is not supported in this browser."
      );
      return;
    }

    const speech = window.speechSynthesis;

    // If this message is currently speaking,
    // stop it.
    if (isSpeaking) {
      speech.cancel();
      setIsSpeaking(false);
      return;
    }

    const speechText = cleanForSpeech(message.content);

    if (!speechText) {
      console.warn("No text available for speech.");
      return;
    }

    // Stop speech from any other message.
    if (speech.speaking || speech.pending) {
      speech.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(
      speechText
    );

    utterance.lang = "en-IN";
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = (event) => {
      // "canceled" is expected when speech is
      // manually stopped or another speech starts.
      if (event.error !== "canceled") {
        console.error(
          "Speech synthesis error:",
          event.error
        );
      }

      setIsSpeaking(false);
    };

    speech.speak(utterance);
  }

  return (
    <div
      className={`flex ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-[80%] ${
          isUser
            ? "rounded-2xl rounded-br-md bg-white px-4 py-3 text-zinc-900"
            : "rounded-2xl rounded-bl-md bg-zinc-900 px-4 py-3 text-zinc-100"
        }`}
      >
        {/* Message */}

        <div className="text-sm leading-6">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              p: ({ children }) => (
                <p className="mb-3 last:mb-0">
                  {children}
                </p>
              ),

              strong: ({ children }) => (
                <strong className="font-semibold text-white">
                  {children}
                </strong>
              ),

              em: ({ children }) => (
                <em className="italic">
                  {children}
                </em>
              ),

              ul: ({ children }) => (
                <ul className="my-3 list-disc space-y-2 pl-5">
                  {children}
                </ul>
              ),

              ol: ({ children }) => (
                <ol className="my-3 list-decimal space-y-2 pl-5">
                  {children}
                </ol>
              ),

              li: ({ children }) => (
                <li className="pl-1">
                  {children}
                </li>
              ),

              h1: ({ children }) => (
                <h1 className="mb-3 mt-4 text-lg font-semibold text-white">
                  {children}
                </h1>
              ),

              h2: ({ children }) => (
                <h2 className="mb-3 mt-4 text-base font-semibold text-white">
                  {children}
                </h2>
              ),

              h3: ({ children }) => (
                <h3 className="mb-2 mt-4 font-semibold text-white">
                  {children}
                </h3>
              ),

              code: ({ children }) => (
                <code className="rounded bg-zinc-950 px-1.5 py-0.5 text-xs text-zinc-300">
                  {children}
                </code>
              ),

              blockquote: ({ children }) => (
                <blockquote className="my-3 border-l-2 border-zinc-700 pl-4 text-zinc-400">
                  {children}
                </blockquote>
              ),

              table: ({ children }) => (
                <div className="my-4 w-full overflow-x-auto rounded-lg border border-zinc-800">
                  <table className="w-full min-w-[500px] border-collapse text-left text-xs">
                    {children}
                  </table>
                </div>
              ),

              thead: ({ children }) => (
                <thead className="bg-zinc-950">
                  {children}
                </thead>
              ),

              th: ({ children }) => (
                <th className="border-b border-zinc-800 px-3 py-2 font-semibold text-zinc-200">
                  {children}
                </th>
              ),

              td: ({ children }) => (
                <td className="border-b border-zinc-800 px-3 py-2 align-top text-zinc-400">
                  {children}
                </td>
              ),

              a: ({ children, href }) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-zinc-200 underline underline-offset-2 hover:text-white"
                >
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3">
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Evidence
            </p>
            <div className="grid gap-2 sm:grid-cols-2">
              {message.sources.map((source, index) => (
                <SourceCard
                  key={`${source.citation_id ?? source.source}-${index}`}
                  source={source}
                />
              ))}
            </div>
          </div>
        )}

        {/* Voice */}

        {!isUser && (
          <div className="mt-3 flex items-center gap-2">
            <button
              type="button"
              onClick={toggleSpeech}
              className="rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-xs text-zinc-400 transition hover:border-zinc-700 hover:text-white"
            >
              {isSpeaking ? "■ Stop" : "🔊 Listen"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
