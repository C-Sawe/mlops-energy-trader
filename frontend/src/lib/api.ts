// Mirrors src/serving/schemas.py — kept in sync by hand, not generated,
// since this project has no OpenAPI codegen step (FastAPI serves the spec
// at /openapi.json but nothing here consumes it yet).

export type DiscreteAction = "BUY" | "HOLD" | "SELL" | "LIQUIDATE";

export interface TickerDecision {
  ticker: string;
  raw_weight: number | null;
  discrete_action: DiscreteAction;
}

export interface PredictResponse {
  decisions: TickerDecision[];
  failsafe_triggered: boolean;
  vix_at_decision: number | null;
  version_id: string | null;
  decided_at: string;
}

export interface EquityPoint {
  date: string;
  equity_value: number;
  rolling_sharpe_30d: number | null;
  max_drawdown: number | null;
  cumulative_return: number | null;
}

export interface TelemetryResponse {
  points: EquityPoint[];
}

export interface DecisionLogEntry {
  decision_id: string;
  version_id: string;
  decided_at: string;
  ticker: string;
  raw_weight: number | null;
  discrete_action: DiscreteAction;
  vix_at_decision: number | null;
  failsafe_triggered: boolean;
}

export interface DecisionLogResponse {
  items: DecisionLogEntry[];
  page: number;
  page_size: number;
  total: number;
}

export type CTStatus = "SERVING" | "EVALUATING" | "RETRAINING";

export interface CTStatusResponse {
  status: CTStatus;
  active_version_id: string | null;
  rolling_sharpe: number | null;
  target_sharpe_threshold: number;
  last_evaluated_at: string | null;
}

export interface HealthResponse {
  status: string;
  active_version_id: string | null;
}

const BASE = "/api";

// Single-user local operation (CLAUDE.md §12): the bearer token, if the
// deployment set one, is supplied at build/runtime via this env var rather
// than a login flow — there is no user table to log in against.
const TOKEN = import.meta.env.VITE_API_BEARER_TOKEN as string | undefined;

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers as Record<string, string> | undefined),
  };
  if (TOKEN) headers["Authorization"] = `Bearer ${TOKEN}`;

  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  predict: (positions: Record<string, number>, cash_weight: number) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify({ positions, cash_weight }),
    }),

  telemetry: (start?: string, end?: string) => {
    const params = new URLSearchParams();
    if (start) params.set("start", start);
    if (end) params.set("end", end);
    const qs = params.toString();
    return request<TelemetryResponse>(`/telemetry${qs ? `?${qs}` : ""}`);
  },

  decisions: (page = 1, pageSize = 20) =>
    request<DecisionLogResponse>(`/decisions?page=${page}&page_size=${pageSize}`),

  ctStatus: () => request<CTStatusResponse>("/ct-status"),

  triggerEvaluation: () => request<CTStatusResponse>("/ct/evaluate", { method: "POST" }),
};

export { ApiError };
