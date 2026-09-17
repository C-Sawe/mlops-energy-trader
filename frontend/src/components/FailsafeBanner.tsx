import { AnimatePresence, motion } from "framer-motion";
import { api } from "../lib/api";
import { usePolling } from "../lib/usePolling";

// I5/NFR-11: the fail-safe takes precedence over any model output and is
// logged with its trigger value — this banner is the "and visibly
// surfaced" half of that, so a capital-preservation event never goes
// unnoticed just because nobody happened to open the decision log.
export function FailsafeBanner() {
  const { data } = usePolling(() => api.decisions(1, 5), 5000);
  const active = data?.items.some((d) => d.failsafe_triggered) ?? false;
  const vix = data?.items.find((d) => d.failsafe_triggered)?.vix_at_decision;

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          initial={{ opacity: 0, height: 0, marginBottom: 0 }}
          animate={{ opacity: 1, height: "auto", marginBottom: 16 }}
          exit={{ opacity: 0, height: 0, marginBottom: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="overflow-hidden rounded-2xl border border-red-500/20 bg-red-500/10 px-5 py-3 text-sm font-medium text-red-600 dark:text-red-400"
        >
          Fail-safe active — VIX {vix?.toFixed(1) ?? "critical"} exceeded the threshold; inference
          was bypassed and capital-preservation (LIQUIDATE) was returned instead (FR-12, I5).
        </motion.div>
      )}
    </AnimatePresence>
  );
}
