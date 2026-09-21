#!/usr/bin/env node
/**
 * NFR-09: "Dashboard renders without unbounded memory growth over [X] h."
 *
 * A one-off validation experiment (same status as scripts/finrl_crosscheck.py
 * — not run in CI, not a project dependency). Playwright is not a listed
 * dependency anywhere in this project; install it into a scratch location
 * before running this, the same way finrl_crosscheck.py's docstring asks
 * for an isolated venv rather than polluting the main dependency tree:
 *
 *     mkdir -p /tmp/pw-scratch && cd /tmp/pw-scratch
 *     npm init -y && npm install playwright && npx playwright install chromium
 *     node /path/to/this/script.mjs
 *
 * CLAUDE.md §5 records the full finding this script produces. The short
 * version, because it matters for interpreting the numbers below: Chrome
 * DevTools' `Performance.getMetrics()` "JSEventListeners" and
 * "JSHeapUsedSize" climb continuously between garbage-collection passes on
 * ANY sufficiently active page — that is expected V8 behaviour, not a leak.
 * The only way to tell a real leak from ordinary uncollected garbage is to
 * force (or wait for) a GC pass and see whether the *post-collection*
 * baseline is flat across cycles or drifts upward. This script does both:
 * it samples the raw metric every minute (so a real leak's tell-tale
 * unbounded climb would still show up), and every few minutes forces a GC
 * pass via `HeapProfiler.collectGarbage` and logs the reclaimed baseline
 * separately, so the two effects aren't conflated.
 *
 * Usage:
 *     node scripts/nfr09_memory_check.mjs [duration_minutes]
 */
import { chromium } from "playwright";

const DURATION_MIN = parseInt(process.argv[2] || "30", 10);
const SAMPLE_INTERVAL_MS = 60_000;
const GC_EVERY_N_SAMPLES = 2;
const url = process.env.DASHBOARD_URL || "http://localhost:5173";

const browser = await chromium.launch({ args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1300, height: 900 } });
const client = await page.context().newCDPSession(page);
await client.send("Performance.enable");
await client.send("HeapProfiler.enable");

const errors = [];
page.on("pageerror", (err) => errors.push(String(err)));
page.on("console", (msg) => { if (msg.type() === "error") errors.push(msg.text()); });

await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
console.log(`loaded ${url}; sampling every 60s for ${DURATION_MIN} minutes (raw metric + periodic forced-GC baseline)`);

async function metrics() {
  const m = await client.send("Performance.getMetrics");
  const byName = Object.fromEntries(m.metrics.map((x) => [x.name, x.value]));
  return { listeners: byName.JSEventListeners, heapMB: byName.JSHeapUsedSize / 1e6, nodes: byName.Nodes };
}

for (let i = 1; i <= DURATION_MIN; i++) {
  await page.waitForTimeout(SAMPLE_INTERVAL_MS);
  const raw = await metrics();
  let line = `[t=${i}min] raw: listeners=${raw.listeners} heap=${raw.heapMB.toFixed(2)}MB nodes=${raw.nodes} errors=${errors.length}`;
  if (i % GC_EVERY_N_SAMPLES === 0) {
    await client.send("HeapProfiler.collectGarbage");
    const post = await metrics();
    line += `  | post-GC: listeners=${post.listeners} heap=${post.heapMB.toFixed(2)}MB`;
  }
  console.log(line);
}

console.log(`done. ${errors.length} console/page errors total.`);
if (errors.length) console.log(errors.slice(0, 10));
await browser.close();
