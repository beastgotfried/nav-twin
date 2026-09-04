import type { TwinState } from "../lib/protocol";
import { fmtDuration } from "../lib/format";

/**
 * Maintenance advisory. The protocol (10-Twin/README.md) carries no
 * maintenance fields of its own, so every row here is derived from the twin
 * state on screen: the banner follows alarm.level, the advisory sentence
 * quotes the leading diagnosis, and the figures come from the alarm and the
 * residuals. Nothing is fabricated; the footer says what the card is.
 */

function worstZ(twin: TwinState): number {
  let m = 0;
  for (const c of twin.cylinders) {
    m = Math.max(m, Math.abs(c.z_EGT), Math.abs(c.z_CHT));
  }
  return Math.max(m, Math.abs(twin.oil.z_p), Math.abs(twin.oil.z_T));
}

const BANNER: Record<string, string> = {
  nominal: "CONTINUE MISSION",
  caution: "MONITOR FLAGGED CHANNEL",
  warning: "GROUNDED PENDING INSPECTION",
};

export function MaintenanceAdvisory({ twin }: { twin: TwinState | null }) {
  const level = twin?.alarm.level ?? "nominal";
  const active = twin?.alarm.active ?? false;
  const top = twin?.diagnosis[0];

  const flaggedFor =
    twin && active && twin.alarm.since_t_s !== null
      ? Math.max(0, twin.t_s - twin.alarm.since_t_s)
      : 0;

  return (
    <section className="card">
      <h2 className="card-title">MAINTENANCE ADVISORY</h2>
      <div className={`adv-banner adv-${level}`}>{BANNER[level]}</div>

      {twin === null ? (
        <p className="empty-hint">awaiting telemetry</p>
      ) : (
        <>
          <p className="adv-text">
            {active && top
              ? `Flagged ${top.label}. Inspect the flagged cylinder before next flight.`
              : "No advisories. All channels are inside the predicted band."}
          </p>
          <div className="adv-rows">
            <div className="kv-row">
              <span className="kv-label">Time flagged</span>
              <span className="kv-value mono">{fmtDuration(flaggedFor)}</span>
            </div>
            <div className="kv-row">
              <span className="kv-label">Worst |z| now</span>
              <span className="kv-value mono">{worstZ(twin).toFixed(2)}</span>
            </div>
            <div className="kv-row">
              <span className="kv-label">Flagged cylinder</span>
              <span className="kv-value mono">
                {active && twin.alarm.cylinder !== null
                  ? `cyl ${twin.alarm.cylinder}`
                  : "none"}
              </span>
            </div>
          </div>
          <p className="adv-foot">advisory only, not a certified figure</p>
        </>
      )}
    </section>
  );
}
