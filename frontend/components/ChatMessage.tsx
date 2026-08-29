"use client";

import { useEffect, useState } from "react";
import type { ChatMessage as ChatMessageType } from "@/types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import SourceCard from "./SourceCard";

interface ChatMessageProps {
  message: ChatMessageType;
}

function cleanForSpeech(text: string): string {
  return text
    // Headings
    .replace(/#{1,6}\s?/g, "")
    // Bold
    .replace(/\*\*(.*?)\*\*/g, "$1")
    // Italic
    .replace(/\*(.*?)\*/g, "$1")
    // Inline code
    .replace(/`([^`]+)`/g, "$1")
    // Markdown links
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    // List markers
    .replace(/^\s*[-*+]\s+/gm, "")
    // Table separators / pipes
    .replace(/\|/g, " ")
    // HTML line breaks
    .replace(/<br\s*\/?>/gi, " ")
    // Multiple newlines
    .replace(/\n+/g, " ")
    // Multiple spaces
    .replace(/\s+/g, " ")
    .trim();
}

export default function ChatMessage({
  message,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const [isSpeaking, setIsSpeaking] = useState(false);

  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  function toggleSpeech() {
    if (!("speechSynthesis" in window)) {
      return;
    }

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    window.speechSynthesis.cancel();

    const speechText = cleanForSpeech(message.content);

    if (!speechText) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(
      speechText
    );

    utterance.lang = "en-IN";
    utterance.rate = 1;
    utterance.pitch = 1;

    utterance.onstart = () => {
      setIsSpeaking(true);
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = () => {
      setIsSpeaking(false);
    };

    window.speechSynthesis.speak(utterance);
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

        
        {/* Sources 
        {!isUser &&
          message.sources &&
          message.sources.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                Sources
              </p>

              {message.sources.map((source, index) => (
                <SourceCard
                  key={`${source.section}-${source.page}-${index}`}
                  source={source}
                />
              ))}
            </div>
          )}

        */}
      </div>
    </div>
  );
}