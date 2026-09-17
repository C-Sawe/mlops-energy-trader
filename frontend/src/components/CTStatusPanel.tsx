import { useState } from "react";
import { GlassCard } from "./GlassCard";
import { CTStatusPill } from "./StatusPill";
import { AppleButton } from "./AppleButton";
import { api, type CTStatusResponse } from "../lib/api";

function relativeTime(iso: string | null): string {
  if (!iso) return "never";
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  return `${hours}h ago`;
}

export function CTStatusPanel({
  status,
  onRefreshed,
}: {
  status: CTStatusResponse | null;
  onRefreshed: (s: CTStatusResponse) => void;
}) {
  const [triggering, setTriggering] = useState(false);

  const handleTriggerEvaluation = async () => {
    setTriggering(true);
    try {
      const result = await api.triggerEvaluation();
      onRefreshed(result);
    } finally {
      setTriggering(false);
    }
  };

  const belowTarget =
    status?.rolling_sharpe != null && status.rolling_sharpe < status.target_sharpe_threshold;

  return (
    <GlassCard className="col-span-2 flex items-center justify-between gap-6 p-6">
      <div className="flex items-center gap-4">
        <div>
          <div className="text-sm text-black/50 dark:text-white/50">CT Pipeline</div>
          <div className="mt-1">{status && <CTStatusPill status={status.status} />}</div>
        </div>
        <div className="h-10 w-px bg-black/10 dark:bg-white/10" />
        <div>
          <div className="text-sm text-black/50 dark:text-white/50">Rolling Sharpe</div>
          <div
            className={`mt-1 text-xl font-semibold tabular-nums tracking-tight ${
              belowTarget ? "text-red-500" : "text-black dark:text-white"
            }`}
          >
            {status?.rolling_sharpe != null ? status.rolling_sharpe.toFixed(2) : "—"}
            <span className="ml-1 text-sm font-normal text-black/40 dark:text-white/40">
              / target {status?.target_sharpe_threshold.toFixed(2) ?? "—"}
            </span>
          </div>
        </div>
        <div className="h-10 w-px bg-black/10 dark:bg-white/10" />
        <div>
          <div className="text-sm text-black/50 dark:text-white/50">Last evaluated</div>
          <div className="mt-1 text-sm font-medium">{relativeTime(status?.last_evaluated_at ?? null)}</div>
        </div>
      </div>
      <AppleButton
        onClick={handleTriggerEvaluation}
        disabled={triggering || status?.status !== "SERVING"}
        variant="neutral"
      >
        {triggering ? "Checking…" : "Force Check"}
      </AppleButton>
    </GlassCard>
  );
}
