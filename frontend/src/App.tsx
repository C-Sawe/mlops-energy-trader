import { useState } from "react";
import { Header } from "./components/Header";
import { FailsafeBanner } from "./components/FailsafeBanner";
import { CTStatusPanel } from "./components/CTStatusPanel";
import {
  EquityCurveChart,
  MaxDrawdownChart,
  RollingSharpeChart,
} from "./components/PerformanceCharts";
import { DecisionLog } from "./components/DecisionLog";
import { api, type CTStatusResponse } from "./lib/api";
import { usePolling } from "./lib/usePolling";

export default function App() {
  const { data: telemetry } = usePolling(() => api.telemetry(), 10000);
  const { data: polledStatus } = usePolling(() => api.ctStatus(), 5000);
  const [status, setStatus] = useState<CTStatusResponse | null>(null);

  const effectiveStatus = status ?? polledStatus;
  const points = telemetry?.points ?? [];

  return (
    <div className="min-h-screen bg-canvas-light font-sans text-black dark:bg-canvas-dark dark:text-white">
      <Header />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <FailsafeBanner />

        <div className="grid grid-cols-2 gap-6">
          <CTStatusPanel status={effectiveStatus} onRefreshed={setStatus} />
          <EquityCurveChart points={points} />
          <RollingSharpeChart points={points} />
          <MaxDrawdownChart points={points} />
          <DecisionLog />
        </div>
      </main>
    </div>
  );
}
