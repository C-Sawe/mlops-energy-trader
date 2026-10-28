/**
 * Pipeline assistant panel (backend: src/serving/assistant.py).
 *
 * A wide side panel where the supervisor can ask about the pipeline. An
 * answer streams in as it is produced: each record the assistant reads
 * appears as a step, its chart or table appears as soon as the data
 * arrives, then the explanation, then clickable follow-up questions.
 * Charts come from the records, not from the model (AssistantVisuals.jsx).
 *
 * Offered only when the backend says the assistant is configured.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchAssistantStatus, streamAssistant } from "../api.js";
import { Visual } from "./AssistantVisuals.jsx";
import { Markdown } from "./Markdown.jsx";

export const STARTERS = [
  "What is the pipeline doing right now?",
  "Why did the model last retrain?",
  "How has the model performed against buy-and-hold?",
  "Trace the latest decision to the model and data behind it",
  "Which models have been promoted, and why?",
  "Show me XOM's recent prices",
];

const FAILURES = {
  throttled: "You've asked a lot of questions in a short time. Wait a few minutes and try again.",
  not_configured: "The assistant isn't configured on this server.",
  unreachable: "Couldn't reach the server. Check your connection and try again.",
  interrupted: "The answer was interrupted. Try again.",
  unauthorized: "Your session has ended. Sign in again to keep asking.",
  invalid: "That question couldn't be sent. Try rephrasing it.",
  failed: "Something went wrong. Try again.",
};

let nextId = 1;
const newId = () => `m${nextId++}`;

/** The conversation as the API expects it: text turns only, ending with the question. */
export function toHistory(messages, question) {
  const turns = messages
    .map((m) => ({ role: m.role, content: m.role === "user" ? m.text : answerText(m.parts), error: m.error }))
    .filter((m) => m.content && !m.error)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));
  return [...turns, { role: "user", content: question }].slice(-20);
}

function SparkIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
      <path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9L12 2zm7 11l.9 2.6 2.6.9-2.6.9L19 20l-.9-2.6-2.6-.9 2.6-.9L19 13z" />
    </svg>
  );
}

function Steps({ steps }) {
  if (!steps.length) return null;
  return (
    <ul className="as-steps" aria-label="What the assistant read">
      {steps.map((s) => (
        <li key={s.id} className={`as-step ${s.status}`}>
          <span className="as-step-icon" aria-hidden="true">{s.status === "running" ? "" : s.status === "ok" ? "✓" : "!"}</span>
          {s.label}
          {s.status === "failed" && <span className="sr-only"> (failed)</span>}
        </li>
      ))}
    </ul>
  );
}

/**
 * An answer's parts in the order they arrived: opening words, the records
 * read (steps), their charts, then the explanation. Consecutive text deltas
 * extend one text part; consecutive steps share one row.
 */
export function addPart(parts, event) {
  const last = parts[parts.length - 1];
  if (event.type === "text") {
    if (last?.kind === "text") return [...parts.slice(0, -1), { ...last, text: last.text + event.delta }];
    return [...parts, { kind: "text", key: `t${parts.length}`, text: event.delta }];
  }
  if (event.type === "step") {
    const step = { id: event.id, label: event.label, status: "running" };
    if (last?.kind === "steps") return [...parts.slice(0, -1), { ...last, steps: [...last.steps, step] }];
    return [...parts, { kind: "steps", key: `s${parts.length}`, steps: [step] }];
  }
  if (event.type === "step_done") {
    const status = event.ok ? "ok" : "failed";
    return parts.map((p) => (p.kind === "steps"
      ? { ...p, steps: p.steps.map((s) => (s.id === event.id ? { ...s, status } : s)) }
      : p));
  }
  if (event.type === "visual") return [...parts, { kind: "visual", key: `v${event.id}`, visual: event.visual }];
  if (event.type === "notice") return [...parts, { kind: "notice", key: `n${parts.length}`, text: event.message }];
  return parts;
}

const answerText = (parts) => parts.filter((p) => p.kind === "text").map((p) => p.text.trim()).join("\n\n");

function AssistantMessage({ message, onAsk, onRetry, busy }) {
  const { parts } = message;
  const last = parts[parts.length - 1];
  const waiting = message.streaming && (!last || last.kind === "visual" || last.kind === "notice"
    || (last.kind === "steps" && last.steps.every((s) => s.status !== "running")));
  return (
    <div className="as-msg assistant" aria-busy={message.streaming}>
      {parts.map((p, i) => {
        if (p.kind === "steps") return <Steps key={p.key} steps={p.steps} />;
        if (p.kind === "visual") return <Visual key={p.key} visual={p.visual} onAsk={busy ? undefined : onAsk} />;
        if (p.kind === "notice") return <p key={p.key} className="as-notice">{p.text}</p>;
        return (
          <div key={p.key} className={`as-text ${message.streaming && i === parts.length - 1 ? "streaming" : ""}`}>
            <Markdown text={p.text.trim()} />
          </div>
        );
      })}
      {waiting && <div className="as-thinking"><span /><span /><span /> Thinking</div>}
      {message.error && (
        <div className="as-error" role="alert">
          {message.error}
          {onRetry && <button className="text-btn" onClick={onRetry}>Try again</button>}
        </div>
      )}
      {!message.streaming && message.followUps.length > 0 && (
        <div className="as-chips" aria-label="Suggested follow-up questions">
          {message.followUps.map((q) => (
            <button key={q} className="as-chip" disabled={busy} onClick={() => onAsk(q)}>{q}</button>
          ))}
        </div>
      )}
    </div>
  );
}

export function AssistantPanel({ open, onClose, model }) {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef(null);
  const inputRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => { if (open) inputRef.current?.focus(); }, [open]);
  useEffect(() => { endRef.current?.scrollIntoView?.({ block: "end" }); }, [messages]);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const update = (id, change) =>
    setMessages((all) => all.map((m) => (m.id === id ? { ...m, ...change(m) } : m)));

  // `base` is the conversation to continue: normally the current one, but a
  // retry passes it with the failed exchange already removed.
  const ask = useCallback(async (raw, base = messages) => {
    const question = raw.trim();
    if (!question || busy) return;
    const history = toHistory(base, question);
    const answerId = newId();
    setDraft("");
    setBusy(true);
    setMessages((all) => [
      ...all,
      { id: newId(), role: "user", text: question },
      { id: answerId, role: "assistant", parts: [], followUps: [], error: null, streaming: true, question },
    ]);
    const controller = new AbortController();
    abortRef.current = controller;

    const onEvent = (e) => {
      if (e.type === "follow_ups") update(answerId, () => ({ followUps: e.questions ?? [] }));
      else if (e.type === "error") update(answerId, () => ({ error: e.message }));
      else update(answerId, (m) => ({ parts: addPart(m.parts, e) }));
    };

    const result = await streamAssistant(history, onEvent, controller.signal);
    update(answerId, (m) => ({
      streaming: false,
      error: result.ok || result.reason === "cancelled"
        ? m.error
        : result.detail ?? FAILURES[result.reason] ?? FAILURES.failed,
      parts: result.reason === "cancelled" ? addPart(m.parts, { type: "text", delta: "\n\n(Stopped.)" }) : m.parts,
    }));
    abortRef.current = null;
    setBusy(false);
  }, [busy, messages]);

  const stop = () => abortRef.current?.abort();
  const reset = () => { stop(); setMessages([]); setDraft(""); inputRef.current?.focus(); };
  const retry = (message) => {
    // Drop the failed answer and the question before it, then ask again.
    const base = messages.slice(0, messages.findIndex((m) => m.id === message.id) - 1);
    setMessages(base);
    ask(message.question, base);
  };

  const onSubmit = (e) => { e.preventDefault(); ask(draft); };
  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask(draft);
    }
  };

  if (!open) return null;
  return (
    <>
      <div className="as-scrim" onClick={onClose} aria-hidden="true" />
      <section className="as-panel" role="dialog" aria-modal="true" aria-labelledby="as-title">
        <header className="as-head">
          <div>
            <h2 id="as-title"><SparkIcon /> Pipeline assistant</h2>
            <p>Answers from this system's own records{model ? ` · ${model}` : ""}</p>
          </div>
          <div className="as-head-actions">
            {messages.length > 0 && <button className="text-btn" onClick={reset}>New chat</button>}
            <button className="icon-btn" onClick={onClose} aria-label="Close the assistant">✕</button>
          </div>
        </header>

        <div className="as-body" aria-live="polite">
          {messages.length === 0 ? (
            <div className="as-welcome">
              <p className="as-welcome-title">Ask about the pipeline</p>
              <p>
                It reads the pipeline's records (status, activity log, decisions, models, performance and prices)
                and shows you the data as charts while it explains. It can't change anything.
              </p>
              <div className="as-chips">
                {STARTERS.map((q) => <button key={q} className="as-chip" onClick={() => ask(q)}>{q}</button>)}
              </div>
            </div>
          ) : (
            messages.map((m) =>
              m.role === "user"
                ? <div key={m.id} className="as-msg user"><p>{m.text}</p></div>
                : <AssistantMessage key={m.id} message={m} onAsk={ask} busy={busy}
                                    onRetry={m.error && !m.streaming ? () => retry(m) : null} />
            )
          )}
          <div ref={endRef} />
        </div>

        <form className="as-compose" onSubmit={onSubmit}>
          <label htmlFor="as-input" className="sr-only">Ask a question about the pipeline</label>
          <textarea id="as-input" ref={inputRef} rows={1} value={draft} maxLength={4000}
                    placeholder="Ask about retrains, decisions, models or performance…"
                    onChange={(e) => setDraft(e.target.value)} onKeyDown={onKeyDown} />
          {busy
            ? <button type="button" className="as-send stop" onClick={stop}>Stop</button>
            : <button type="submit" className="as-send" disabled={!draft.trim()}>Ask</button>}
        </form>
        <p className="as-disclaimer">
          AI-written explanations of the stored records; charts show the records themselves. Not financial advice.
          Questions and the records read are sent to Groq's API.
        </p>
      </section>
    </>
  );
}

/** The launcher button plus panel. Renders nothing unless the backend offers the assistant. */
export default function Assistant() {
  const [status, setStatus] = useState(null);
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    let alive = true;
    fetchAssistantStatus().then((r) => { if (alive && r.ok) setStatus(r.data); });
    return () => { alive = false; };
  }, []);

  if (!status?.enabled) return null;
  return (
    <>
      {!open && (
        <button className="as-launch" onClick={() => setOpen(true)} aria-haspopup="dialog">
          <SparkIcon /> <span className="as-launch-label">Ask the pipeline</span>
        </button>
      )}
      <AssistantPanel open={open} onClose={close} model={status.model} />
    </>
  );
}
