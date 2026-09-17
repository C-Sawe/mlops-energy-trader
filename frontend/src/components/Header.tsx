import { motion } from "framer-motion";

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="sticky top-0 z-30 border-b border-white/20 bg-white/70 px-6 py-4 backdrop-blur-2xl dark:border-white/10 dark:bg-black/60"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-tight">Energy Trading Agent</h1>
          <p className="text-xs text-black/50 dark:text-white/50">
            Continuous Training Dashboard · XOM · CVX · SHEL · BP · NEE
          </p>
        </div>
      </div>
    </motion.header>
  );
}
