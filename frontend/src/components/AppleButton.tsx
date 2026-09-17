import { motion } from "framer-motion";
import type { ReactNode } from "react";

const VARIANTS = {
  primary: "bg-blue-500 text-white",
  destructive: "bg-red-500 text-white",
  neutral: "bg-black/5 text-black dark:bg-white/10 dark:text-white",
};

export function AppleButton({
  children,
  onClick,
  disabled = false,
  variant = "primary",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: keyof typeof VARIANTS;
  className?: string;
}) {
  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      whileTap={disabled ? undefined : { scale: 0.95 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
      style={{ WebkitTapHighlightColor: "transparent" }}
      className={`rounded-2xl px-5 py-2.5 text-sm font-semibold tracking-tight disabled:opacity-40 ${VARIANTS[variant]} ${className}`}
    >
      {children}
    </motion.button>
  );
}
