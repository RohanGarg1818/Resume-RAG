import type { Source } from "@/types/chat";

interface SourceCardProps {
  source: Source;
}

export default function SourceCard({
  source,
}: SourceCardProps) {
  const title =
    source.project ||
    source.company ||
    source.section ||
    "Resume";

  const details = [
    source.section,
    source.role,
    source.page ? `Page ${source.page}` : null,
  ].filter(Boolean);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <div className="text-sm font-medium text-zinc-200">
        {title}
      </div>

      {details.length > 0 && (
        <div className="mt-1 text-xs text-zinc-500">
          {details.join(" · ")}
        </div>
      )}
    </div>
  );
}