export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  response_mode?: "standard" | "recruiter" | "comparison";
  debug?: boolean;
}

export interface Source {
  source: string;
  citation_id?: string | null;
  /** Every context label ([S#]) merged into this source. */
  citation_ids?: string[];
  section?: string | null;
  page?: number | null;
  project?: string | null;
  company?: string | null;
  role?: string | null;
  candidate?: string | null;
  /** Display relevance 0-1 (cross-encoder probability, else cosine). */
  score?: number | null;
  excerpt?: string | null;
}

export interface RetrievalSummary {
  chunks: number;
  latency_ms: number;
  llm_latency_ms: number;
  confidence?: string | null;
  route?: string | null;
}

export interface TraceChunk {
  rank: number;
  metadata?: Record<string, unknown>;
  channels?: string[];
  semantic_score?: number;
  bm25_score?: number;
  hybrid_score?: number;
  reranker_score?: number | null;
  reasons?: string[];
  text_preview?: string;
}

export interface RetrievalDiagnostics {
  query_type?: string;
  topics?: string[];
  entities?: string[];
  debug?: {
    intent?: string;
    expanded_terms?: string[];
    routed_sections?: string[];
    candidates?: TraceChunk[];
    final_context?: TraceChunk[];
  } | null;
  [key: string]: unknown;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  conversation_id: string;
  retrieval?: RetrievalSummary | null;
  debug?: RetrievalDiagnostics | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  retrieval?: RetrievalSummary | null;
  trace?: RetrievalDiagnostics | null;
  /** The user question an assistant message answers (for highlighting). */
  question?: string;
  error?: boolean;
}

export interface IndexStatus {
  chunks: number;
  by_source: Record<string, number>;
  embedding_model: string;
  reranker_model: string | null;
  llm_model: string;
  llm_location: "local" | "cloud";
}
