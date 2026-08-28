import type { ReactNode } from "react";

/**
 * Simple markdown renderer for analysis results.
 * Handles **bold**, bullet points, and line breaks.
 * No external dependencies — pure React.
 */
export function MarkdownText({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  return (
    <div className={className}>
      {text.split("\n").map((line, i) => (
        <Line key={i} line={line} />
      ))}
    </div>
  );
}

function Line({ line }: { line: string }) {
  // Empty line → spacing
  if (line.trim() === "") {
    return <div className="h-2" />;
  }

  // Header line (starts with ** and ends with **)
  if (line.startsWith("**") && line.includes("**:")) {
    const match = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
    if (match) {
      return (
        <div className="mt-2 mb-1 font-bold text-xs uppercase tracking-wider text-[#1A1A2E]/70">
          <InlineMarkdown text={match[1]} />
          {match[2] ? (
            <span className="ml-1 font-normal normal-case tracking-normal text-[#1A1A2E]/50">
              <InlineMarkdown text={match[2]} />
            </span>
          ) : null}
        </div>
      );
    }
  }

  // Bullet point (starts with • or -)
  if (line.trim().startsWith("•") || line.trim().startsWith("-")) {
    const content = line.trim().replace(/^[•\-]\s*/, "");
    return (
      <div className="flex gap-2 py-0.5 text-sm">
        <span className="text-[#EF476F] font-bold shrink-0">•</span>
        <span className="leading-relaxed">
          <InlineMarkdown text={content} />
        </span>
      </div>
    );
  }

  // Indented bullet (starts with space + •)
  if (line.match(/^\s+•/) || line.match(/^\s+-/)) {
    const content = line.trim().replace(/^[•\-]\s*/, "");
    return (
      <div className="flex gap-2 py-0.5 pl-4 text-sm">
        <span className="text-[#1A1A2E]/30 font-bold shrink-0">–</span>
        <span className="leading-relaxed text-[#1A1A2E]/80">
          <InlineMarkdown text={content} />
        </span>
      </div>
    );
  }

  // Regular line with inline markdown
  return (
    <div className="text-sm leading-relaxed">
      <InlineMarkdown text={line} />
    </div>
  );
}

/**
 * Handle inline **bold** and `code` formatting.
 */
function InlineMarkdown({ text }: { text: string }) {
  // Split by ** for bold and ` for code
  const parts: ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Check for **bold**
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    // Check for `code`
    const codeMatch = remaining.match(/`(.+?)`/);

    let nextMatch: { index: number; length: number; type: "bold" | "code" } | null = null;

    if (boldMatch && boldMatch.index !== undefined) {
      nextMatch = { index: boldMatch.index, length: boldMatch[0].length, type: "bold" };
    }
    if (codeMatch && codeMatch.index !== undefined) {
      if (!nextMatch || codeMatch.index < nextMatch.index) {
        nextMatch = { index: codeMatch.index, length: codeMatch[0].length, type: "code" };
      }
    }

    if (!nextMatch) {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }

    // Text before match
    if (nextMatch.index > 0) {
      parts.push(<span key={key++}>{remaining.slice(0, nextMatch.index)}</span>);
    }

    if (nextMatch.type === "bold") {
      parts.push(
        <strong key={key++} className="font-bold">
          {boldMatch![1]}
        </strong>,
      );
    } else {
      parts.push(
        <code
          key={key++}
          className="bg-[#1A1A2E]/5 px-1 py-0.5 text-[0.85em] font-mono"
        >
          {codeMatch![1]}
        </code>,
      );
    }

    remaining = remaining.slice(nextMatch.index + nextMatch.length);
  }

  return <>{parts}</>;
}
