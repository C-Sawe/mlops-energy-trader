/**
 * Operator override (FR-23): the human's brake on the autonomous loop.
 *
 * Three switches and a rollback, from `/oversight`. The switches can only
 * restrain the system: none makes it trade, train or promote where it
 * otherwise wouldn't. Rollback does choose which model decides, limited to
 * ones that have served before. Switching one on
 * asks for a reason, because that is what the audit trail records about
 * why a human stepped in. The banner shows on every section while any is
 * on, so nobody reads the dashboard as fully autonomous while it isn't.
 */
import { useState } from "react";
import { GlassCard, CardHead, Pill } from "./Primitives.jsx";

export const CONTROL_INFO = {
  halt_trading: {
    on: "Trading halted",
    does: "Every decision becomes capital preservation (LIQUIDATE), as when VIX trips the fail-safe. The paper account acts on it at its next daily cycle; queued orders are not cancelled.",
  },
  pause_retraining: {
    on: "Retraining paused",
    does: "The loop keeps checking the model and drawing its charts, but never trains or promotes a candidate.",
  },
  pause_paper_trading: {
    on: "Paper trading paused",
    does: "The daily paper cycle records a skip instead of placing orders, including the ones a trading halt would place. Queued orders are not cancelled.",
  },
};

export const activeOverrides = (controls) => (controls ?? []).filter((c) => c.enabled);

const fmtWhen = (iso) =>
  iso
    ? new Date(iso).toLocaleString("en-GB", {
        day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
      })
    : null;

/** Why an action failed, in words the operator can act on. */
export function failureText(result) {
  if (result.reason === "conflict") return result.detail?.detail ?? "Refused by the server.";
  if (result.reason === "invalid") {
    const d = result.detail?.detail;
    return Array.isArray(d) ? d.map((e) => e.msg.replace(/^Value error, /, "")).join("; ") : "The request was refused.";
  }
  if (result.reason === "timeout") return "The backend took too long to answer. Check the state before retrying.";
  return "No answer from the backend, so the change may or may not have been made. Check the state before retrying.";
}

export function OverrideBanner({ controls }) {
  const active = activeOverrides(controls);
  if (!active.length) return null;
  return (
    <section className="ovr-banner" role="status" aria-label="Operator overrides in force">
      <span className="ovr-banner-icon" aria-hidden="true">!</span>
      <div>
        <b>Human override in force.</b>{" "}
        {active.map((c, i) => (
          <span key={c.name}>
            {i > 0 && " · "}
            {CONTROL_INFO[c.name]?.on ?? c.label} by {c.set_by ?? "an operator"}
            {c.set_at && `, ${fmtWhen(c.set_at)}`}
            {c.reason && <> — “{c.reason}”</>}
          </span>
        ))}
      </div>
    </section>
  );
}

/** One switch: state, what it does, and an inline confirm step. */
function ControlRow({ control, busy, disabled, onSet }) {
  const [confirming, setConfirming] = useState(false);
  const [reason, setReason] = useState("");
  const info = CONTROL_INFO[control.name] ?? { on: control.label, does: "" };
  const switchingOn = !control.enabled;
  const id = `ovr-reason-${control.name}`;

  const submit = async (e) => {
    e.preventDefault();
    const ok = await onSet(control.name, switchingOn, reason.trim() || null);
    if (ok) { setConfirming(false); setReason(""); }
  };

  return (
    <div className={`ovr-row ${control.enabled ? "on" : ""}`}>
      <div className="ovr-row-head">
        <div>
          <div className="ovr-name">{control.label}</div>
          <p className="ovr-does">{info.does}</p>
        </div>
        <Pill tone={control.enabled ? "crit" : "idle"}>{control.enabled ? "On" : "Off"}</Pill>
      </div>
      {control.enabled && (
        <p className="ovr-meta">
          Switched on by <b>{control.set_by ?? "an operator"}</b>
          {control.set_at && <> on {fmtWhen(control.set_at)}</>}
          {control.reason && <> — “{control.reason}”</>}
        </p>
      )}
      {!confirming ? (
        <button
          className={`ovr-btn ${switchingOn ? "danger" : ""}`}
          disabled={disabled || busy}
          onClick={() => setConfirming(true)}
        >
          {switchingOn ? `Switch on ${control.label.toLowerCase()}…` : `Lift ${control.label.toLowerCase()}…`}
        </button>
      ) : (
        <form className="ovr-confirm" onSubmit={submit}>
          <label htmlFor={id} className="ovr-label">
            {switchingOn ? "Reason (required, kept in the audit trail)" : "Note (optional)"}
          </label>
          <textarea
            id={id}
            value={reason}
            maxLength={300}
            rows={2}
            autoFocus
            onChange={(e) => setReason(e.target.value)}
            placeholder={switchingOn ? "e.g. VIX feed looks wrong; holding until checked" : ""}
          />
          <div className="ovr-actions">
            <button type="button" className="ovr-btn" onClick={() => setConfirming(false)} disabled={busy}>
              Cancel
            </button>
            <button
              type="submit"
              className={`ovr-btn ${switchingOn ? "danger" : "primary"}`}
              disabled={busy || (switchingOn && !reason.trim())}
            >
              {busy ? "Saving…" : switchingOn ? "Confirm" : "Lift it"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

function RollbackRow({ versions, pipelineStatus, busy, disabled, onRollback }) {
  const choices = versions.filter((v) => !v.is_active);
  const active = versions.find((v) => v.is_active);
  const [target, setTarget] = useState("");
  const [reason, setReason] = useState("");
  const serving = pipelineStatus === "SERVING";
  const name = (v) => v.label ?? v.version_id.slice(0, 8);

  const submit = async (e) => {
    e.preventDefault();
    const ok = await onRollback(target, reason.trim());
    if (ok) { setTarget(""); setReason(""); }
  };

  return (
    <div className="ovr-row wide">
      <div className="ovr-row-head">
        <div>
          <div className="ovr-name">Roll back the model</div>
          <p className="ovr-does">
            Put back a model that has served before, hot-swapped with no restart. Serving now:{" "}
            <b title={active?.version_id}>{active ? name(active) : "none"}</b>. The loop may retrain it
            later if its Sharpe is below target; pause retraining too to keep it.
          </p>
        </div>
      </div>
      {choices.length === 0 ? (
        <p className="ovr-meta">No earlier model to roll back to yet.</p>
      ) : (
        <form className="ovr-confirm ovr-rollback" onSubmit={submit}>
          <div className="ovr-field">
            <label htmlFor="ovr-rollback-version" className="ovr-label">Model</label>
            <select id="ovr-rollback-version" value={target} onChange={(e) => setTarget(e.target.value)}
                    disabled={disabled || busy}>
              <option value="">Choose a model…</option>
              {choices.map((v) => (
                <option key={v.version_id} value={v.version_id}>
                  {name(v)}{v.sharpe_ratio != null ? ` · eval Sharpe ${v.sharpe_ratio.toFixed(2)}` : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="ovr-field">
            <label htmlFor="ovr-rollback-reason" className="ovr-label">Reason (required)</label>
            <textarea id="ovr-rollback-reason" rows={1} maxLength={300} value={reason}
                      onChange={(e) => setReason(e.target.value)} disabled={disabled || busy} />
          </div>
          <button type="submit" className="ovr-btn danger solid"
                  disabled={disabled || busy || !serving || !target || !reason.trim()}>
            {busy ? "Rolling back…" : "Roll back"}
          </button>
          {!serving && (
            <p className="ovr-meta ovr-wait">
              Waiting for the loop to finish {pipelineStatus?.toLowerCase()} before a rollback is allowed.
            </p>
          )}
        </form>
      )}
    </div>
  );
}

/**
 * `oversight` is `/oversight`'s body, or null while it loads or when the
 * page is on example data (`demo`), where the controls are shown but can't
 * be used. `onSet`/`onRollback` resolve to the API result.
 */
export function OversightCard({ oversight, demo = false, onSet, onRollback }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const run = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      const result = await fn();
      if (!result.ok) setError(failureText(result));
      return result.ok;
    } finally {
      setBusy(false);
    }
  };

  return (
    <GlassCard>
      <CardHead
        title="Operator controls"
        sub="Human override of the autonomous loop. The switches can only restrain it; a rollback restores a model that has served before. Every use is recorded with who, when and why."
      />
      <div className="ovr-body">
      {demo && <p className="ovr-meta">Example data: the controls work once the backend answers.</p>}
      {!demo && !oversight && <p className="ovr-meta">Loading…</p>}
      {error && <p className="ovr-error" role="alert">{error}</p>}
      {(oversight || demo) && (
        <div className="ovr-list">
          {(oversight?.controls ?? DEMO_CONTROLS).map((c) => (
            <ControlRow key={c.name} control={c} busy={busy} disabled={demo}
                        onSet={(name, enabled, reason) => run(() => onSet(name, enabled, reason))} />
          ))}
          <RollbackRow versions={oversight?.versions ?? []} pipelineStatus={oversight?.pipeline_status ?? "SERVING"}
                       busy={busy} disabled={demo}
                       onRollback={(id, reason) => run(() => onRollback(id, reason))} />
        </div>
      )}
      </div>
    </GlassCard>
  );
}

const DEMO_CONTROLS = [
  { name: "halt_trading", label: "Trading halt", enabled: false },
  { name: "pause_retraining", label: "Retraining pause", enabled: false },
  { name: "pause_paper_trading", label: "Paper-trading pause", enabled: false },
];
