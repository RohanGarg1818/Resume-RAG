export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
}

export interface Source {
  source: string;
  section?: string | null;
  page?: number | null;
  project?: string | null;
  company?: string | null;
  role?: string | null;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  conversation_id: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}