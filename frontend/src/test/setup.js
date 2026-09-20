import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// Without this, successive renders across tests in the same file accumulate
// in the same jsdom document (RTL's auto-cleanup relies on detecting a
// *global* afterEach, which `globals: false` in vite.config.js deliberately
// doesn't provide) — later queries then match leftover elements from
// earlier tests, not just the current render.
afterEach(cleanup);
