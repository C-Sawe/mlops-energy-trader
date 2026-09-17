import { motion } from "framer-motion";
import type { CTStatus, DiscreteAction } from "../lib/api";

// Purposeful colour, not decorative: green reads as "healthy/going up" for
// BUY and SERVING, red as "stop/destructive" for SELL/LIQUIDATE, blue as
// "informational/in progress" for EVALUATING, amber as "active work
// happening" for RETRAINING — never used interchangeably.
const CT_STYLES: Record<CTStatus, string> = {
  SERVING: "bg-green-500/15 text-green-600 dark:text-green-400",
  EVALUATING: "bg-blue-500/15 text-blue-600 dark:text-blue-400",
  RETRAINING: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
};

const ACTION_STYLES: Record<DiscreteAction, string> = {
  BUY: "bg-green-500/15 text-green-600 dark:text-green-400",
  SELL: "bg-red-500/15 text-red-600 dark:text-red-400",
  HOLD: "bg-gray-500/15 text-gray-600 dark:text-gray-300",
  LIQUIDATE: "bg-red-600/20 text-red-700 dark:text-red-400",
};

function Pill({ label, className }: { label: string; className: string }) {
  return (
    <motion.span
      key={label}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold tracking-tight ${className}`}
    >
      {label}
    </motion.span>
  );
}

export function CTStatusPill({ status }: { status: CTStatus }) {
  return <Pill label={status} className={CT_STYLES[status]} />;
}

export function ActionPill({ action }: { action: DiscreteAction }) {
  return <Pill label={action} className={ACTION_STYLES[action]} />;
}
