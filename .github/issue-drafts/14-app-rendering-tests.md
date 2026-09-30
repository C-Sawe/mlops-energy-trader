title: "Frontend: add rendering tests for App.jsx (polling, mock fallback, decision-detail sheet)"
labels: testing, frontend
---
## Gap
The 47 Vitest tests cover pure logic (adapters, outlier clamping, tone mapping, `api.js` date math). **`App.jsx`'s component has no rendering tests** (CLAUDE.md §7, "Scope, honestly stated"): data fetching, the 15 s poll loop, the switch to mock data when the backend dies (IR-07), and opening/closing the decision-detail sheet.

## Tests to add
- [ ] Backend down → "Example data — not live results" banner visible
- [ ] Backend returns → banner disappears on the next poll (fake timers)
- [ ] Backend dies mid-session → falls back again, no crash
- [ ] Click a decision row → sheet shows the version → run → partition chain (NFR-07); fail-safe row shows its VIX trigger (NFR-11)
- [ ] "Force Check" posts to `/ct/evaluate`
