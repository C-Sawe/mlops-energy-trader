/**
 * Interface primitives carrying the HIG behaviours.
 *
 * Motion is spring-based, never time-eased: useSpring integrates a critically
 * damped spring (stiffness 300, damping 30, mass 1) so the sheet settles fast
 * without the cartoon wobble a bouncy spring gives. Buttons compress under
 * press rather than merely recolouring.
 */
import { forwardRef, useCallback, useEffect, useRef, useState } from "react";

/* ----------------------------------------------------------------- spring */
export function useSpring(target, { stiffness = 300, damping = 30, mass = 1 } = {}) {
  const [value, setValue] = useState(target);
  const state = useRef({ x: target, v: 0, raf: 0 });

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setValue(target); return; }

    const dt = 1 / 60;
    const step = () => {
      const s = state.current;
      // Two substeps per frame keeps the integration stable at high stiffness.
      for (let i = 0; i < 2; i++) {
        const acc = (-stiffness * (s.x - target) - damping * s.v) / mass;
        s.v += acc * dt;
        s.x += s.v * dt;
      }
      setValue(s.x);
      if (Math.abs(s.x - target) > 0.15 || Math.abs(s.v) > 0.15) {
        s.raf = requestAnimationFrame(step);
      } else {
        s.x = target; s.v = 0; setValue(target);
      }
    };
    cancelAnimationFrame(state.current.raf);
    state.current.raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(state.current.raf);
  }, [target, stiffness, damping, mass]);

  return value;
}

/* ------------------------------------------------------------------- card */
export const GlassCard = ({ children, className = "", ...rest }) => (
  <section className={`glass ${className}`} {...rest}>{children}</section>
);

export const CardHead = ({ title, sub, right }) => (
  <div className="card-head">
    <div>
      <h2 className="card-title">{title}</h2>
      {sub && <p className="card-sub">{sub}</p>}
    </div>
    {right}
  </div>
);

/* ------------------------------------------------------------------- pill */
/** Status is encoded in shape and text as well as colour — never colour alone. */
export const Pill = ({ tone = "idle", children }) => (
  <span className={`pill ${tone}`}>
    <span className="dot" />
    {children}
  </span>
);

/* --------------------------------------------------------------- pressable */
/* forwardRef is load-bearing: the sheet moves focus to this button on open.
   A plain function component would drop the ref silently in a production
   build, leaving the dialog open with focus still behind it. */
export const Button = forwardRef(function Button(
  { children, className = "", ...rest }, ref
) {
  return (
    <button ref={ref} className={`btn ${className}`} {...rest}>
      {children}
    </button>
  );
});

/* ------------------------------------------------------------ status strip */
export function StatusStrip({ pipeline, model, lastIngest, vix, failSafeThreshold }) {
  const tripped = vix != null && failSafeThreshold != null && vix > failSafeThreshold;
  const tone = { SERVING: "good", EVALUATING: "idle", RETRAINING: "warn" }[pipeline] || "idle";
  const label = { SERVING: "Serving", EVALUATING: "Evaluating", RETRAINING: "Retraining" }[pipeline]
    || pipeline;

  return (
    <section className="glass strip" aria-label="System status">
      <div className="strip-cell">
        <span className="strip-k">CT pipeline</span>
        <Pill tone={tone}>{label}</Pill>
      </div>
      <div className="strip-cell">
        <span className="strip-k">Active model</span>
        <span className="strip-v num">{model ?? "—"}</span>
      </div>
      <div className="strip-cell">
        <span className="strip-k">Last ingest</span>
        <span className="strip-v num">{lastIngest ?? "—"}</span>
      </div>
      <div className="strip-cell">
        <span className="strip-k">Volatility fail-safe</span>
        <Pill tone={tripped ? "crit" : "idle"}>
          {tripped ? "Triggered" : "Inactive"} · VIX {vix?.toFixed(1) ?? "—"}
        </Pill>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ sheet */
/**
 * iOS-style bottom sheet. Heavier than a button, so it uses the same spring at
 * a lower stiffness — a larger object should feel like it has more mass.
 */
export function Sheet({ open, onClose, title, subtitle, children }) {
  const y = useSpring(open ? 0 : 100, { stiffness: 300, damping: 30, mass: 1 });
  const [mounted, setMounted] = useState(open);
  const closeRef = useRef(null);
  const restoreRef = useRef(null);

  useEffect(() => {
    if (open) {
      restoreRef.current = document.activeElement;
      setMounted(true);
    } else if (y > 99.5) {
      setMounted(false);
      restoreRef.current?.focus?.();
    }
  }, [open, y]);

  useEffect(() => { if (open) closeRef.current?.focus(); }, [open]);

  const onKey = useCallback((e) => { if (e.key === "Escape") onClose(); }, [onClose]);
  useEffect(() => {
    if (!open) return;
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onKey]);

  if (!mounted) return null;

  return (
    <>
      <div
        className={`scrim ${open ? "on" : ""}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        style={{ transform: `translate3d(0, ${y}%, 0)` }}
      >
        <div className="grabber" />
        <h2>{title}</h2>
        {subtitle && <p className="sub">{subtitle}</p>}
        {children}
        <Button ref={closeRef} onClick={onClose}>Done</Button>
      </div>
    </>
  );
}
