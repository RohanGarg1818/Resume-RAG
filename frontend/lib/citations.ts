import type { Source } from "@/types/chat";

/** Inline citation marker emitted by the model, e.g. [S3]. */
export const CITATION_RE = /\[S(\d+)\]/g;

/** Cited labels ("S3") in order of first appearance. */
export function citedLabels(answer: string): string[] {
  const seen: string[] = [];
  for (const match of answer.matchAll(CITATION_RE)) {
    const label = `S${match[1]}`;
    if (!seen.includes(label)) seen.push(label);
  }
  return seen;
}

/** Index of the source that owns a context label, or -1. */
export function sourceIndexFor(label: string, sources: Source[]): number {
  return sources.findIndex(
    (source) => source.citation_id === label || (source.citation_ids ?? []).includes(label)
  );
}

/**
 * Turn [S#] markers into markdown links (#cite-S#) that the renderer draws as
 * numbered chips. Adjacent duplicates ([S1][S1]) collapse to one chip.
 */
export function linkCitations(answer: string, sources: Source[]): string {
  return answer
    .replace(/(\[S\d+\])(\s*\1)+/g, "$1")
    .replace(CITATION_RE, (_match, n: string) => {
      const label = `S${n}`;
      const index = sourceIndexFor(label, sources);
      return `[${index >= 0 ? index + 1 : "?"}](#cite-${label})`;
    });
}

export function sourceTitle(source: Source): string {
  return source.project || source.company || source.section || source.source || "Source";
}

export function sourceLocation(source: Source): string {
  const parts = [
    source.source,
    source.project || source.company ? source.section : null,
    source.page ? `p.${source.page}` : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

const STOP = new Set(
  "a an the and or of in on at to for with from by is are was were be been what which who whom how does did do his her their he she it this that rohan rohan's about tell me any".split(" ")
);

/** Words from the question worth highlighting inside a retrieved passage. */
export function highlightTerms(question: string | undefined): string[] {
  if (!question) return [];
  return Array.from(
    new Set(
      question
        .toLowerCase()
        .split(/[^a-z0-9+#.]+/)
        .map((word) => word.replace(/\.$/, ""))
        .filter((word) => word.length > 2 && !STOP.has(word))
    )
  );
}
