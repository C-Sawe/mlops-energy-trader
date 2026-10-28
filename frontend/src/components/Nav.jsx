/**
 * Top-level navigation, after Apple's guidance: a sidebar on wide screens,
 * a bottom tab bar on narrow ones, and no more than five sections. One
 * <nav> element; CSS moves it between the two positions.
 *
 * The section lives in the URL hash (#pipeline), so the back button,
 * reloads and shared links all land on the same page.
 */
import { useEffect, useState } from "react";

export const SECTIONS = [
  {
    id: "overview",
    label: "Overview",
    sub: "PPO trading agent · continuous training pipeline · XOM · CVX · SHEL · BP · NEE",
  },
  { id: "pipeline", label: "Pipeline", sub: "How the continuous-training loop checks, retrains and promotes the model" },
  { id: "decisions", label: "Decisions", sub: "Every action, traced to the model version and data behind it" },
  { id: "trading", label: "Trading", sub: "Prices, headlines and the Alpaca paper account" },
];

const IDS = SECTIONS.map((s) => s.id);
const fromHash = () => {
  const id = window.location.hash.replace(/^#\/?/, "");
  return IDS.includes(id) ? id : IDS[0];
};

/** [section, go]: the current section, and a function that navigates to one. */
export function useSection() {
  const [section, setSection] = useState(fromHash);
  useEffect(() => {
    const onHash = () => setSection(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (id) => {
    if (id === section) return;
    window.location.hash = id; // hashchange updates state, and history gets an entry
    setSection(id);
  };
  return [section, go];
}

const ICONS = {
  overview: (
    <>
      <rect x="3.5" y="3.5" width="7" height="7" rx="2" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="2" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="2" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="2" />
    </>
  ),
  pipeline: (
    <>
      <path d="M19.5 12a7.5 7.5 0 0 1-12.8 5.3" />
      <path d="M4.5 12a7.5 7.5 0 0 1 12.8-5.3" />
      <path d="M17.5 3.5v3.3h-3.3" />
      <path d="M6.5 20.5v-3.3h3.3" />
    </>
  ),
  decisions: (
    <>
      <path d="M9 6.5h11M9 12h11M9 17.5h11" />
      <circle cx="4.75" cy="6.5" r="1.1" />
      <circle cx="4.75" cy="12" r="1.1" />
      <circle cx="4.75" cy="17.5" r="1.1" />
    </>
  ),
  trading: (
    <>
      <path d="M7 3.5v17M17 3.5v17M12 6v12" />
      <rect x="5" y="7.5" width="4" height="7" rx="1" />
      <rect x="15" y="9" width="4" height="6" rx="1" />
      <rect x="10" y="10" width="4" height="4.5" rx="1" />
    </>
  ),
};

const BUSY = { EVALUATING: "Evaluating", RETRAINING: "Retraining" };

export function SectionNav({ current, onSelect, pipeline }) {
  return (
    <nav className="sidenav" aria-label="Sections">
      <div className="side-brand">
        <span className="side-name">Energy Agent Telemetry</span>
        <span className="side-sub">Continuous training for RL trading</span>
      </div>
      <ul>
        {SECTIONS.map((s) => (
          <li key={s.id}>
            <a
              href={`#${s.id}`}
              className="nav-item"
              aria-current={s.id === current ? "page" : undefined}
              onClick={(e) => {
                e.preventDefault();
                onSelect(s.id);
              }}
            >
              <svg className="nav-icon" viewBox="0 0 24 24" aria-hidden="true">{ICONS[s.id]}</svg>
              <span className="nav-label">{s.label}</span>
              {s.id === "pipeline" && BUSY[pipeline] && (
                <span className="nav-badge" title={BUSY[pipeline]}>
                  <span className="sr-only">({BUSY[pipeline]})</span>
                </span>
              )}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
