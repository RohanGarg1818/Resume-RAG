"use client";


import {
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

interface ChatInputProps {
  onSubmit: (
    message: string,
    fromVoice: boolean
  ) => Promise<void>;
  disabled?: boolean;
  autoSubmitVoice?: boolean;
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
}: ChatInputProps) {
  const [message, setMessage] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [speechSupported, setSpeechSupported] =
    useState(false);

  const recognitionRef =
    useRef<SpeechRecognitionLike | null>(null);

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

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const trimmed = message.trim();

    if (!trimmed || disabled) {
      return;
    }

    setMessage("");
    messageRef.current = "";

    await onSubmit(trimmed,false);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-950 p-2"
    >
      
      <input
        value={message}
        onChange={(event) => {
          setMessage(event.target.value);
          messageRef.current = event.target.value;
        }}
        placeholder={
          isListening
            ? "Listening..."
            : "Ask about the candidate..."
        }
        disabled={disabled}
        className="flex-1 bg-transparent px-3 py-3 text-sm text-white outline-none placeholder:text-zinc-600"
      />

      {speechSupported && (
        <button
          type="button"
          onClick={toggleListening}
          disabled={disabled}
          aria-label={
            isListening
              ? "Stop listening"
              : "Start voice input"
              
          }
          className={`flex h-11 w-11 items-center justify-center rounded-xl text-lg transition ${
            isListening
              ? "bg-white text-black"
              : "bg-zinc-900 text-zinc-300 hover:bg-zinc-800 hover:text-white"
          } disabled:cursor-not-allowed disabled:opacity-40`}
        >
          {isListening ? "■" : "🎤"}
        </button>
      )}

      <button
        type="submit"
        disabled={
          disabled || !message.trim()
        }
        className="rounded-xl bg-white px-4 py-3 text-sm font-medium text-black transition hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {disabled ? "..." : "Ask"}
      </button>
    </form>
  );
}