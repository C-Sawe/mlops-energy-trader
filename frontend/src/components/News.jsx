/**
 * Market headlines (backend: src/dataops/news.py, GET /news).
 *
 * Context for the reader, never an input to a decision: the agent doesn't
 * read news, and the card says so. Headlines are plain text from the
 * server; links open the publisher's page in a new tab.
 */
import { useEffect, useState } from "react";
import { fetchNews, poll } from "../api.js";
import { relativeTime } from "./Activity.jsx";
import { GlassCard, CardHead } from "./Primitives.jsx";

export const NEWS_NOTE =
  "Context only: the trading agent doesn't read news, so no decision shown here is based on these headlines.";

/** Headlines newest first. `summaries` adds each article's short summary. */
export function NewsList({ items, summaries = false, now = new Date() }) {
  return (
    <ul className="news-list">
      {items.map((n) => {
        const at = new Date(n.published_at);
        return (
          <li key={n.id} className="news-item">
            {n.url
              ? <a className="news-head" href={n.url} target="_blank" rel="noopener noreferrer">{n.headline}</a>
              : <span className="news-head">{n.headline}</span>}
            <div className="news-meta">
              {n.source && <span>{n.source}</span>}
              <time dateTime={n.published_at} title={at.toLocaleString()}>{relativeTime(at, now)}</time>
              {n.tickers.map((t) => <span key={t} className="news-tag">{t}</span>)}
              {n.other_symbols > 0 && (
                <span title="Other companies this article is tagged with">
                  +{n.other_symbols} other {n.other_symbols === 1 ? "company" : "companies"}
                </span>
              )}
            </div>
            {summaries && n.summary && <p className="news-sum">{n.summary}</p>}
          </li>
        );
      })}
    </ul>
  );
}

/** The dashboard card. Follows the chart's ticker; renders nothing unless the server offers news. */
export function NewsCard({ ticker }) {
  const [scope, setScope] = useState("ticker"); // "ticker" | "all"
  const [state, setState] = useState({ loading: true });
  const query = scope === "all" ? null : ticker;

  useEffect(() => {
    setState({ loading: true });
    return poll(() => fetchNews(query, 7, 8), 600000, (r) => {
      if (r.ok) setState({ enabled: r.data.enabled, items: r.data.items });
      else if (r.reason !== "unauthorized") setState((s) => ({ ...s, loading: false, failed: true }));
    });
  }, [query]);

  if (state.enabled === false) return null;
  return (
    <GlassCard>
      <CardHead
        title="Market news"
        sub={`Headlines tagging ${query ?? "XOM, CVX, SHEL, BP or NEE"} · last 7 days`}
        right={
          <div className="seg" role="group" aria-label="News scope">
            <button aria-pressed={scope === "ticker"} onClick={() => setScope("ticker")}>{ticker}</button>
            <button aria-pressed={scope === "all"} onClick={() => setScope("all")}>All five</button>
          </div>
        }
      />
      <div className="news-body">
      {state.loading && !state.items
        ? <div className="chart-empty">Loading headlines…</div>
        : state.failed && !state.items
          ? <div className="chart-empty">News is unavailable right now.</div>
          : state.items.length === 0
            ? <div className="chart-empty">No headlines in the last 7 days.</div>
            : <NewsList items={state.items} summaries />}
      <p className="news-note">{NEWS_NOTE} Source: Alpaca's news feed (Benzinga and other publishers).</p>
      </div>
    </GlassCard>
  );
}
