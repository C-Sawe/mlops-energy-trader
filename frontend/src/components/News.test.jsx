/**
 * Market headlines: the dashboard card and the list the assistant reuses.
 * The fetch is mocked; no network.
 */
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../api.js", async (importOriginal) => ({ ...(await importOriginal()), fetchNews: vi.fn() }));

import * as api from "../api.js";
import { NewsCard, NewsList, NEWS_NOTE } from "./News.jsx";
import { Visual } from "./AssistantVisuals.jsx";

const NOW = new Date("2026-10-22T16:00:00Z");
const item = (over = {}) => ({
  id: "1", published_at: "2026-10-22T15:00:00Z", headline: "Exxon and Shell explore chemicals sale",
  summary: "A joint venture may be sold.", source: "benzinga", url: "https://example.com/a",
  tickers: ["XOM", "SHEL"], other_symbols: 0, ...over,
});

afterEach(() => vi.resetAllMocks());

describe("NewsList", () => {
  it("links to the publisher in a new tab, safely, with source, time and tickers", () => {
    render(<NewsList items={[item()]} now={NOW} />);
    const link = screen.getByRole("link", { name: "Exxon and Shell explore chemicals sale" });
    expect(link.getAttribute("href")).toBe("https://example.com/a");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    expect(screen.getByText("benzinga")).toBeTruthy();
    expect(screen.getByText("1 h ago")).toBeTruthy();
    expect(screen.getByText("SHEL")).toBeTruthy();
    expect(screen.queryByText("A joint venture may be sold.")).toBeNull(); // summaries off by default
  });

  it("shows plain text without a link, and how many other companies an article tags", () => {
    render(<NewsList items={[item({ url: null, other_symbols: 12 })]} summaries now={NOW} />);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("+12 other companies")).toBeTruthy();
    expect(screen.getByText("A joint venture may be sold.")).toBeTruthy();
  });
});

async function renderCard(ticker = "XOM") {
  await act(async () => { render(<NewsCard ticker={ticker} />); });
}

describe("NewsCard", () => {
  it("shows the ticker's headlines and says news never drives a decision", async () => {
    api.fetchNews.mockResolvedValue({ ok: true, data: { enabled: true, ticker: "XOM", items: [item()] } });
    await renderCard();
    expect(api.fetchNews).toHaveBeenCalledWith("XOM", 7, 8);
    expect(screen.getByText("Exxon and Shell explore chemicals sale")).toBeTruthy();
    expect(screen.getByText(/the trading agent doesn't read news/)).toBeTruthy();
  });

  it("switches to all five tickers", async () => {
    api.fetchNews.mockResolvedValue({ ok: true, data: { enabled: true, ticker: null, items: [] } });
    await renderCard();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "All five" })); });
    expect(api.fetchNews).toHaveBeenLastCalledWith(null, 7, 8);
    expect(screen.getByText("No headlines in the last 7 days.")).toBeTruthy();
  });

  it("is hidden when the server has no news configured", async () => {
    api.fetchNews.mockResolvedValue({ ok: true, data: { enabled: false, ticker: "XOM", items: [] } });
    await renderCard();
    expect(screen.queryByText("Market news")).toBeNull();
  });

  it("says so when the feed is down, instead of failing", async () => {
    api.fetchNews.mockResolvedValue({ ok: false, reason: "unreachable" });
    await renderCard();
    expect(screen.getByText("News is unavailable right now.")).toBeTruthy();
  });
});

describe("assistant news visual", () => {
  it("lists the headlines, labelled as the news feed rather than the records", () => {
    render(<Visual visual={{ type: "news", ticker: "XOM", days: 7, items: [item()] }} />);
    expect(screen.getByText("Headlines · XOM · last 7 days")).toBeTruthy();
    expect(screen.getByText("From the news feed")).toBeTruthy();
    expect(screen.getByText(NEWS_NOTE)).toBeTruthy();
  });
});
