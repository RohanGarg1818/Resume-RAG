"use client";


import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import Icon from "@/components/Icon";

export type ResponseMode = "standard" | "recruiter";

interface ChatInputProps {
  onSubmit: (
    message: string,
    fromVoice: boolean
  ) => Promise<void>;
  disabled?: boolean;
  autoSubmitVoice?: boolean;
  mode?: ResponseMode;
  onModeChange?: (mode: ResponseMode) => void;
  placeholder?: string;
}

interface SpeechRecognitionEventLike {
  results: {
    [index: number]: {
      [index: number]: {
        transcript: string;
      };
    };
  };
}

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: (
    event: SpeechRecognitionEventLike
  ) => void;
  onend: () => void;
  onerror: () => void;
}

type SpeechRecognitionConstructor =
  new () => SpeechRecognitionLike;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

export default function ChatInput({
  onSubmit,
  disabled = false,
  autoSubmitVoice = false,
  mode = "standard",
  onModeChange,
  placeholder = "Ask about Rohan's experience, projects or skills…",
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] =
    useState(false);

  const recognitionRef =
    useRef<SpeechRecognitionLike | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // "/" focuses the query box from anywhere outside a text field.
  useEffect(() => {
    function onKey(event: globalThis.KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        textareaRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const messageRef = useRef("");

  useEffect(() => {
    messageRef.current = message;
  }, [message]);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      return;
    }

    setSpeechSupported(true);

    const recognition = new SpeechRecognition();

    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const transcript =
        event.results[0][0].transcript.trim();

      setMessage((previous) => {
        const next = previous.trim()
          ? `${previous} ${transcript}`
          : transcript;

        messageRef.current = next;

        return next;
      });
    };

    recognition.onend = async () => {
      setIsListening(false);

      if (!autoSubmitVoice) {
        return;
      }

      const finalMessage =
        messageRef.current.trim();

      if (!finalMessage || disabled) {
        return;
      }

      setMessage("");
      messageRef.current = "";

      await onSubmit(finalMessage,true);
    };

    recognition.onerror = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
      recognitionRef.current = null;
    };
  }, [autoSubmitVoice, disabled, onSubmit]);

  function toggleListening() {
    if (!recognitionRef.current || disabled) {
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    setIsListening(true);
    recognitionRef.current.start();
  }

  async function submitCurrent() {
    const trimmed = message.trim();

    if (!trimmed || disabled) {
      return;
    }

    setMessage("");
    messageRef.current = "";

    await onSubmit(trimmed, false);
  }

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();
    await submitCurrent();
  }

  // Enter or Cmd/Ctrl+Enter sends; Shift+Enter adds a new line.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitCurrent();
    }
  }

  const modes: { value: ResponseMode; label: string }[] = [
    { value: "standard", label: "Standard" },
    { value: "recruiter", label: "Recruiter" },
  ];

  return (
    <form
      onSubmit={handleSubmit}
      className="rr-well rr-composer"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        padding: "14px 14px 10px 16px",
        borderRadius: 16,
        transition: "box-shadow 200ms var(--ease)",
      }}
    >
      <label htmlFor="rr-query" className="sr-only">
        Question
      </label>
      <textarea
        id="rr-query"
        ref={textareaRef}
        rows={2}
        value={message}
        onChange={(event) => {
          setMessage(event.target.value);
          messageRef.current = event.target.value;
        }}
        onKeyDown={handleKeyDown}
        placeholder={isListening ? "Listening…" : placeholder}
        disabled={disabled}
        style={{
          resize: "none",
          border: 0,
          outline: "none",
          background: "transparent",
          width: "100%",
          fontSize: 15,
          lineHeight: 1.55,
          color: "var(--text)",
          fontFamily: "inherit",
        }}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        {onModeChange && (
          <div className="rr-seg" role="group" aria-label="Answer mode">
            {modes.map((item) => (
              <button
                key={item.value}
                type="button"
                aria-pressed={mode === item.value}
                onClick={() => onModeChange(item.value)}
                title={item.value === "recruiter" ? "Focus answers on verified role fit and gaps" : "Direct grounded answers"}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}

        <span style={{ flex: 1 }} />

        <span className="rr-lbl rr-hide-sm" style={{ fontSize: 10, display: "inline-flex", gap: 4, alignItems: "center" }}>
          <span className="rr-kbd">/</span> focus
        </span>

        {speechSupported && (
          <button
            type="button"
            onClick={toggleListening}
            disabled={disabled}
            aria-pressed={isListening}
            aria-label={isListening ? "Stop listening" : "Start voice input"}
            className="rr-btn rr-btn-neu rr-icon-btn"
            style={isListening ? { color: "var(--blue-hi)", boxShadow: "var(--neu-in), 0 0 0 1px var(--blue-line)" } : undefined}
          >
            <Icon name={isListening ? "stop" : "mic"} />
          </button>
        )}

        <button
          type="submit"
          disabled={disabled || !message.trim()}
          className="rr-btn rr-btn-primary"
          aria-label="Send question"
        >
          {disabled ? "Answering" : "Ask"}
          <span
            className="rr-kbd"
            style={{ background: "rgba(255,255,255,.14)", borderColor: "rgba(255,255,255,.2)", color: "#fff", boxShadow: "none", height: 18 }}
          >
            ↵
          </span>
        </button>
      </div>
    </form>
  );
}
