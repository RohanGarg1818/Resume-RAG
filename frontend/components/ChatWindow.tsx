"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { sendMessage } from "@/lib/api";
import type { ChatMessage } from "@/types/chat";

import ChatInput from "./ChatInput";
import ChatMessageComponent from "./ChatMessage";

const suggestedQuestions = [
  "What projects has he built?",
  "What are his technical skills?",
  "What hackathons has he participated in?",
  "What is his academic qualification?",
];

export default function ChatWindow() {
  const [voiceMode, setVoiceMode] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<
    string | null
  >(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleMessage=useCallback(async(
    message: string,
    fromVoice = false
    )=>{
    setError(null);
    setVoiceMode(fromVoice);

    setMessages((previous) => [
        ...previous,
        {
        role: "user",
        content: message,
        },
    ]);

    setLoading(true);

    try {
        const response = await sendMessage({
        message,
        conversation_id: conversationId,
        });

        setConversationId(response.conversation_id);

        setMessages((previous) => [
        ...previous,
        {
            role: "assistant",
            content: response.answer,
            sources: response.sources,
        },
        ]);
    } catch (err) {
        const errorMessage =
        err instanceof Error
            ? err.message
            : "Something went wrong.";

        setError(errorMessage);
    } finally {
        setLoading(false);
    }
    },[conversationId]);
    useEffect(() => {
        if (!loading && voiceMode && messages.length > 0) {
            const lastMessage =
            messages[messages.length - 1];

            if (lastMessage.role === "assistant") {
            const utterance =
                new SpeechSynthesisUtterance(
                lastMessage.content
                );

            utterance.lang = "en-IN";
            utterance.rate = 1;
            utterance.pitch = 1;

            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);

            setVoiceMode(false);
            }
        }
        }, [loading, messages, voiceMode]);

  return (
    <div className="flex h-[100dvh] flex-col bg-black text-white">
      {/* Header */}
      <header className="border-b border-zinc-900 px-6 py-5">
        <div className="mx-auto max-w-4xl">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500">
            Resume RAG
          </p>

          <h1 className="mt-1 text-xl font-semibold">
            Ask about Rohan
          </h1>

          <p className="mt-1 text-sm text-zinc-500">
            Directly ask questions about experience, projects, skills and
            achievements.
          </p>
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-8">
        <div className="mx-auto flex max-w-4xl flex-col gap-5">
          {messages.length === 0 ? (
            <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-zinc-800 bg-zinc-950 text-xl">
                RG
              </div>

              <h2 className="text-2xl font-semibold">
                Ask anything about the profile
              </h2>

              <p className="mt-2 max-w-md text-sm leading-6 text-zinc-500">
                The answers are grounded in the candidate&apos;s
                resume and linkedIn profile.
              </p>

              <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
                {suggestedQuestions.map((question) => (
                  <button
                    key={question}
                    onClick={() => handleMessage(question)}
                    className="rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 text-left text-sm text-zinc-400 transition hover:border-zinc-700 hover:text-white"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <ChatMessageComponent
                key={`${message.role}-${index}`}
                message={message}
              />
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-md bg-zinc-900 px-4 py-3 text-sm text-zinc-500">
                Thinking...
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>
      </main>

      {/* Input */}
      <div className="border-t border-zinc-900 px-4 py-4">
        <div className="mx-auto max-w-4xl">
           <ChatInput
            onSubmit={handleMessage}
            disabled={loading}
            autoSubmitVoice={true}
            />
        </div>
      </div>
    </div>
  );
}