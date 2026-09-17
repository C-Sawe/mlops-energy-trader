import { useState } from "react";
import { motion } from "framer-motion";
import { GlassCard } from "./GlassCard";
import { ActionPill } from "./StatusPill";
import { AppleButton } from "./AppleButton";
import { DecisionDetailSheet } from "./DecisionDetailSheet";
import { api, type DecisionLogEntry } from "../lib/api";
import { usePolling } from "../lib/usePolling";

const PAGE_SIZE = 8;

export function DecisionLog() {
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<DecisionLogEntry | null>(null);
  const { data } = usePolling(() => api.decisions(page, PAGE_SIZE), 5000);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <GlassCard className="col-span-2 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Decision Log</h2>
          <p className="text-sm text-black/50 dark:text-white/50">FR-19 · every autonomous action</p>
        </div>
        <div className="text-sm text-black/40 dark:text-white/40">
          page {page} of {totalPages}
        </div>
      </div>

      <div className="mt-4 overflow-hidden rounded-2xl border border-black/5 dark:border-white/10">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-black/5 text-left text-black/40 dark:border-white/10 dark:text-white/40">
              <th className="px-4 py-2 font-medium">Time</th>
              <th className="px-4 py-2 font-medium">Ticker</th>
              <th className="px-4 py-2 font-medium">Action</th>
              <th className="px-4 py-2 font-medium">Weight</th>
              <th className="px-4 py-2 font-medium">VIX</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((item, i) => (
              <motion.tr
                key={item.decision_id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
                onClick={() => setSelected(item)}
                className={`cursor-pointer border-b border-black/5 last:border-0 hover:bg-black/[0.02] dark:border-white/5 dark:hover:bg-white/[0.03] ${
                  item.failsafe_triggered ? "bg-red-500/5" : ""
                }`}
              >
                <td className="px-4 py-2.5 tabular-nums text-black/60 dark:text-white/60">
                  {new Date(item.decided_at).toLocaleTimeString()}
                </td>
                <td className="px-4 py-2.5 font-semibold">{item.ticker}</td>
                <td className="px-4 py-2.5">
                  <ActionPill action={item.discrete_action} />
                </td>
                <td className="px-4 py-2.5 tabular-nums">
                  {item.raw_weight != null ? item.raw_weight.toFixed(2) : "—"}
                </td>
                <td className="px-4 py-2.5 tabular-nums">
                  {item.vix_at_decision != null ? item.vix_at_decision.toFixed(1) : "—"}
                </td>
              </motion.tr>
            ))}
            {data?.items.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-black/40 dark:text-white/40">
                  No decisions logged yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex justify-end gap-2">
        <AppleButton
          variant="neutral"
          disabled={page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          Previous
        </AppleButton>
        <AppleButton
          variant="neutral"
          disabled={page >= totalPages}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </AppleButton>
      </div>

      <DecisionDetailSheet decision={selected} onClose={() => setSelected(null)} />
    </GlassCard>
  );
}
