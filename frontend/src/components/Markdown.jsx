/**
 * The small Markdown subset the assistant's answers use: paragraphs,
 * bullet and numbered lists, short headings, **bold** and `code`.
 *
 * Builds React elements only; model text is never inserted as HTML, so
 * nothing in an answer can run as markup.
 */
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`)/g;

export function renderInline(text) {
  return text.split(INLINE).filter(Boolean).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={i}>{renderInline(part.slice(2, -2))}</strong>;  // `code` inside bold
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>;
    return part;
  });
}

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
const HEADING = /^#{1,4}\s+/;

/** Splits text into blocks: [{type: "p"|"ul"|"ol"|"h", lines}] */
export function parseBlocks(text) {
  const blocks = [];
  for (const raw of text.replace(/\r/g, "").split("\n")) {
    const line = raw.trimEnd();
    const last = blocks[blocks.length - 1];
    if (!line.trim()) {
      blocks.push(null); // a blank line ends the current block
      continue;
    }
    const type = HEADING.test(line) ? "h" : BULLET.test(line) ? "ul" : NUMBERED.test(line) ? "ol" : "p";
    const content = line.replace(HEADING, "").replace(BULLET, "").replace(NUMBERED, "");
    if (last && last.type === type && type !== "h") last.lines.push(content);
    else blocks.push({ type, lines: [content] });
  }
  return blocks.filter(Boolean);
}

export function Markdown({ text }) {
  return (
    <div className="md">
      {parseBlocks(text).map((block, i) => {
        if (block.type === "ul" || block.type === "ol") {
          const List = block.type;
          return <List key={i}>{block.lines.map((l, j) => <li key={j}>{renderInline(l)}</li>)}</List>;
        }
        if (block.type === "h") return <p key={i} className="md-h">{renderInline(block.lines[0])}</p>;
        return <p key={i}>{renderInline(block.lines.join(" "))}</p>;
      })}
    </div>
  );
}
