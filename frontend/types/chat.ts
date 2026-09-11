export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  response_mode?: "standard" | "recruiter" | "comparison";
  debug?: boolean;
}

export interface Source {
  source: string;
  citation_id?: string | null;
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
