import { AnimatePresence, motion } from "framer-motion";
import type { DecisionLogEntry } from "../lib/api";
import { ActionPill } from "./StatusPill";

// Apple's fluid bottom sheet: slides up as one physical object, heavier
// (higher damping) than a button because it's a much larger element.
export function DecisionDetailSheet({
  decision,
  onClose,
}: {
  decision: DecisionLogEntry | null;
  onClose: () => void;
}) {
  return (
    <AnimatePresence>
      {decision && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
          />

          <motion.div
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            exit={{ y: "100%" }}
            transition={{ type: "spring", stiffness: 300, damping: 32, mass: 1 }}
            className="fixed bottom-0 left-0 right-0 z-50 rounded-t-[32px] border-t border-white/20 bg-white/90 p-6 pb-10 shadow-2xl backdrop-blur-2xl dark:border-white/10 dark:bg-black/85"
          >
            <div className="mx-auto mb-6 h-1.5 w-12 rounded-full bg-black/15 dark:bg-white/20" />

            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold tracking-tight">{decision.ticker}</h2>
              <ActionPill action={decision.discrete_action} />
            </div>

            <dl className="mt-6 grid grid-cols-2 gap-4 text-sm">
              <div>
                <dt className="text-black/50 dark:text-white/50">Decided at</dt>
                <dd className="mt-0.5 font-medium tabular-nums">
                  {new Date(decision.decided_at).toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-black/50 dark:text-white/50">Raw weight</dt>
                <dd className="mt-0.5 font-medium tabular-nums">
                  {decision.raw_weight != null ? decision.raw_weight.toFixed(3) : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-black/50 dark:text-white/50">VIX at decision</dt>
                <dd className="mt-0.5 font-medium tabular-nums">
                  {decision.vix_at_decision != null ? decision.vix_at_decision.toFixed(1) : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-black/50 dark:text-white/50">Fail-safe</dt>
                <dd
                  className={`mt-0.5 font-medium ${
                    decision.failsafe_triggered ? "text-red-500" : ""
                  }`}
                >
                  {decision.failsafe_triggered ? "Triggered (I5, FR-12)" : "Not triggered"}
                </dd>
              </div>
              <div className="col-span-2">
                <dt className="text-black/50 dark:text-white/50">Model version</dt>
                <dd className="mt-0.5 break-all font-mono text-xs">{decision.version_id}</dd>
              </div>
            </dl>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
