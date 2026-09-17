import { motion } from "framer-motion";
import type { ReactNode } from "react";

// Apple's "Liquid Glass" material: translucency + blur for elevation
// instead of a drop shadow doing all the work.
export function GlassCard({
  children,
  className = "",
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30, mass: 1, delay }}
      className={`rounded-3xl border border-white/20 bg-white/70 shadow-sm backdrop-blur-2xl dark:border-white/10 dark:bg-black/60 ${className}`}
    >
      {children}
    </motion.div>
  );
}
