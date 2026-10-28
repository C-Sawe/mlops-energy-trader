/**
 * The pipeline assistant's frontend: the event-stream reader, the Markdown
 * subset (and that model text can't become markup), how an answer's parts
 * are ordered, the panel's flow, and that charts ask follow-ups when
 * clicked. The backend is mocked; nothing reaches a model.
 */
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, fetchAssistantStatus: vi.fn(), streamAssistant: vi.fn() };
});

import * as api from "../api.js";
import { drainEvents } from "../api.js";
import Assistant, { addPart, toHistory, STARTERS } from "./Assistant.jsx";
import { Visual } from "./AssistantVisuals.jsx";
import { Markdown } from "./Markdown.jsx";

afterEach(() => vi.clearAllMocks());

// --------------------------------------------------------------- stream reading
describe("drainEvents", () => {
  it("parses complete events and keeps a partial one for the next chunk", () => {
    const seen = [];
    const rest = drainEvents('data: {"type":"text","delta":"Hi"}\n\ndata: {"type":"te', (e) => seen.push(e));
    expect(seen).toEqual([{ type: "text", delta: "Hi" }]);
    drainEvents(rest + 'xt","delta":" there"}\n\n', (e) => seen.push(e));
    expect(seen[1]).toEqual({ type: "text", delta: " there" });
  });

  it("skips a malformed event instead of ending the answer", () => {
    const seen = [];
    drainEvents('data: {not json}\n\ndata: {"type":"done"}\n\n', (e) => seen.push(e));
    expect(seen).toEqual([{ type: "done" }]);
  });
});

describe("streamAssistant", () => {
  const realStream = async () => (await vi.importActual("../api.js")).streamAssistant;
  const sse = (chunks) => {
    const encoder = new TextEncoder();
    return new ReadableStream({ start(c) { chunks.forEach((x) => c.enqueue(encoder.encode(x))); c.close(); } });
  };

  it("delivers events across chunk boundaries and sends the session cookie", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200,
      body: sse(['data: {"type":"step","id":"t1"', ',"label":"Reading"}\n\ndata: {"type":"done"}\n\n']) });
    vi.stubGlobal("fetch", fetchMock);
    const seen = [];
    const result = await (await realStream())([{ role: "user", content: "hi" }], (e) => seen.push(e));
    expect(result).toEqual({ ok: true });
    expect(seen).toEqual([{ type: "step", id: "t1", label: "Reading" }, { type: "done" }]);
    expect(fetchMock.mock.calls[0][1].credentials).toBe("include");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ messages: [{ role: "user", content: "hi" }] });
    vi.unstubAllGlobals();
  });

  it("reports throttling and a missing configuration with the server's message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 429, json: async () => ({ detail: "Slow down." }) }));
    expect(await (await realStream())([], () => {})).toEqual({ ok: false, reason: "throttled", detail: "Slow down." });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({}) }));
    expect((await (await realStream())([], () => {})).reason).toBe("not_configured");
    vi.unstubAllGlobals();
  });
});

// --------------------------------------------------------------- markdown
describe("Markdown", () => {
  it("renders bold, code and lists", () => {
    const { container } = render(<Markdown text={"The **gate** held.\n\n- one `run`\n- two"} />);
    expect(container.querySelector("strong").textContent).toBe("gate");
    expect(container.querySelector("code").textContent).toBe("run");
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });

  it("renders code inside bold", () => {
    const { container } = render(<Markdown text={"**trigger `ct_decay`**"} />);
    expect(container.querySelector("strong code").textContent).toBe("ct_decay");
    expect(container.textContent).toBe("trigger ct_decay");
  });

  it("never turns model text into markup", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> **bold**'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">');
  });
});

// --------------------------------------------------------------- answer structure
describe("addPart", () => {
  it("keeps parts in arrival order and updates steps in place", () => {
    let parts = [];
    for (const e of [
      { type: "text", delta: "Let me " }, { type: "text", delta: "check." },
      { type: "step", id: "a", label: "Reading" }, { type: "step", id: "b", label: "Counting" },
      { type: "step_done", id: "a", ok: true }, { type: "visual", id: "a", visual: { type: "status" } },
      { type: "text", delta: "Done." },
    ]) parts = addPart(parts, e);
    expect(parts.map((p) => p.kind)).toEqual(["text", "steps", "visual", "text"]);
    expect(parts[0].text).toBe("Let me check.");
    expect(parts[1].steps.map((s) => s.status)).toEqual(["ok", "running"]);
  });

  it("keeps a rate-limit notice as its own part, out of the answer text", () => {
    let parts = addPart([], { type: "notice", message: "Waiting 7 s for Groq's rate limit" });
    parts = addPart(parts, { type: "text", delta: "Answer" });
    expect(parts.map((p) => [p.kind, p.text])).toEqual([
      ["notice", "Waiting 7 s for Groq's rate limit"], ["text", "Answer"]]);
    expect(toHistory([{ role: "assistant", parts }], "Next?")[0].content).toBe("Answer");
  });
});

describe("toHistory", () => {
  it("sends text turns only, skipping failed answers, ending with the question", () => {
    const messages = [
      { role: "user", text: "First?" },
      { role: "assistant", parts: [{ kind: "text", text: "Intro" }, { kind: "visual" }, { kind: "text", text: "Answer" }] },
      { role: "user", text: "Second?" },
      { role: "assistant", parts: [], error: "Something went wrong." },
    ];
    expect(toHistory(messages, "Third?")).toEqual([
      { role: "user", content: "First?" },
      { role: "assistant", content: "Intro\n\nAnswer" },
      { role: "user", content: "Second?" },
      { role: "user", content: "Third?" },
    ]);
  });
});

// --------------------------------------------------------------- the panel
const MODELS = { type: "models", items: [
  { version_id: "aaaa1111-x", promoted_at: "2026-10-20T00:00:00Z", is_active: true, sharpe_ratio: 1.2 },
  { version_id: "bbbb2222-y", promoted_at: "2026-09-17T00:00:00Z", is_active: false, sharpe_ratio: -0.4 },
] };

async function openPanel() {
  api.fetchAssistantStatus.mockResolvedValue({ ok: true, data: { enabled: true, model: "qwen/qwen3.8-27b" } });
  render(<Assistant />);
  await act(async () => {});
  fireEvent.click(screen.getByRole("button", { name: /Ask the pipeline/ }));
}

describe("Assistant", () => {
  it("is not offered when the server has no assistant", async () => {
    api.fetchAssistantStatus.mockResolvedValue({ ok: true, data: { enabled: false, model: null } });
    render(<Assistant />);
    await act(async () => {});
    expect(screen.queryByRole("button", { name: /Ask the pipeline/ })).toBeNull();
  });

  it("streams an answer: steps, the chart, the text, then follow-up questions", async () => {
    api.streamAssistant.mockImplementation(async (history, onEvent) => {
      for (const e of [
        { type: "text", delta: "Let me check." },
        { type: "step", id: "t1", label: "Reading the model history" },
        { type: "step_done", id: "t1", ok: true },
        { type: "visual", id: "t1", visual: MODELS },
        { type: "text", delta: "The **active** model was promoted on 20 Oct." },
        { type: "follow_ups", questions: ["Why did it retrain?"] },
        { type: "done" },
      ]) onEvent(e);
      return { ok: true };
    });
    await openPanel();
    const dialog = screen.getByRole("dialog", { name: "Pipeline assistant" });
    expect(within(dialog).getByText(/qwen\/qwen3\.8-27b/)).toBeTruthy();

    await act(async () => { fireEvent.click(within(dialog).getByRole("button", { name: STARTERS[4] })); });

    expect(api.streamAssistant.mock.calls[0][0]).toEqual([{ role: "user", content: STARTERS[4] }]);
    expect(within(dialog).getByText("Reading the model history")).toBeTruthy();
    expect(within(dialog).getByText("Model history")).toBeTruthy();
    expect(within(dialog).getByText("active", { selector: "strong" })).toBeTruthy();
    expect(within(dialog).getByRole("button", { name: "Why did it retrain?" })).toBeTruthy();
  });

  it("asks a follow-up when a chart item is clicked, with the conversation so far", async () => {
    api.streamAssistant
      .mockImplementationOnce(async (h, onEvent) => {
        onEvent({ type: "visual", id: "t1", visual: MODELS });
        onEvent({ type: "text", delta: "Two models." });
        return { ok: true };
      })
      .mockResolvedValueOnce({ ok: true });
    await openPanel();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: STARTERS[4] })); });

    await act(async () => { fireEvent.click(screen.getByTitle(/Version aaaa1111/)); });

    const history = api.streamAssistant.mock.calls[1][0];
    expect(history.at(-1)).toEqual({ role: "user", content: "Tell me about model version aaaa1111." });
    expect(history[1]).toEqual({ role: "assistant", content: "Two models." });
  });

  it("explains a failure and offers to try again", async () => {
    api.streamAssistant.mockResolvedValueOnce({ ok: false, reason: "throttled", detail: null })
      .mockResolvedValueOnce({ ok: true });
    await openPanel();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: STARTERS[0] })); });

    expect(screen.getByRole("alert").textContent).toMatch(/asked a lot of questions/);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Try again" }));
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(api.streamAssistant).toHaveBeenCalledTimes(2);
    expect(api.streamAssistant.mock.calls[1][0]).toEqual([{ role: "user", content: STARTERS[0] }]);
  });

  it("sends with Enter and closes with Escape", async () => {
    api.streamAssistant.mockResolvedValue({ ok: true });
    await openPanel();
    const input = screen.getByLabelText("Ask a question about the pipeline");
    fireEvent.change(input, { target: { value: "Is the fail-safe on?" } });
    await act(async () => { fireEvent.keyDown(input, { key: "Enter" }); });
    expect(api.streamAssistant.mock.calls[0][0].at(-1).content).toBe("Is the fail-safe on?");

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

// --------------------------------------------------------------- visuals
describe("visuals", () => {
  it("traces a decision through version, run and data windows", () => {
    render(<Visual visual={{ type: "provenance", decision: {
      ticker: "XOM", discrete_action: "BUY", raw_weight: 0.62, vix_at_decision: 18.4, failsafe_triggered: false,
      decided_at: "2026-10-19T07:23:00Z", version_id: "3aad819d-1", version_is_active: true,
      version_promoted_at: "2026-09-28T00:00:00Z", run_id: "99b19816-2", trigger_reason: "ct_decay",
      sharpe_ratio: 0.02, train_start: "2024-08-29", train_end: "2026-08-28", eval_start: "2026-08-29",
      eval_end: "2026-09-28", hyperparameters: { algorithm: "ppo", seed: 0, device: "cpu" },
    } }} />);
    expect(screen.getByText("3aad819d")).toBeTruthy();
    expect(screen.getByText("Retrain after performance fell below target")).toBeTruthy();
    expect(screen.getByRole("img", { name: /Trained 29 Aug 2024 to 28 Aug 2026, evaluated 29 Aug 2026/ })).toBeTruthy();
    expect(screen.queryByText("device")).toBeNull(); // not useful to show
  });

  it("lets a decision row ask for its provenance", () => {
    const onAsk = vi.fn();
    render(<Visual onAsk={onAsk} visual={{ type: "decisions", items: [
      { decision_id: "d-123", decided_at: "2026-10-19T07:23:00Z", ticker: "NEE", raw_weight: 0.157, action: "HOLD",
        vix: 20, failsafe: false, version_id: "3aad819d-1" },
    ] }} />);
    fireEvent.click(screen.getByTitle("Trace this decision"));
    expect(onAsk).toHaveBeenCalledWith("Trace decision d-123 to the model and data behind it.");
  });

  it("shows the Sharpe and VIX against their limits in words as well as bars", () => {
    render(<Visual visual={{ type: "status", status: {
      status: "SERVING", rolling_sharpe: 0.6, target_sharpe_threshold: 1, current_vix: 40, vix_critical_threshold: 35,
    } }} />);
    expect(screen.getByText(/below target/)).toBeTruthy();
    expect(screen.getByText(/fail-safe active/)).toBeTruthy();
    expect(screen.getByRole("img", { name: "VIX 40.0 against limit 35.0" })).toBeTruthy();
  });
});
