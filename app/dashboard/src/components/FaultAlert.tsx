import type { TwinState } from "../lib/protocol";
import { fmtPct } from "../lib/format";

/**
 * Ranked differential diagnosis, straight from the twin state. Bar colour
 * follows the mockup: the leading hypothesis is red (it is the suspected
 * fault), the runner-up orange, the rest ink.
 */

function barClass(rank: number): string {
  if (rank === 1) return "bar-fill bar-top";
  if (rank === 2) return "bar-fill bar-second";
  return "bar-fill bar-rest";
}

function confClass(rank: number): string {
  if (rank === 1) return "mono conf conf-top";
  if (rank === 2) return "mono conf conf-second";
  return "mono conf conf-rest";
}

export function FaultAlert({ twin }: { twin: TwinState | null }) {
  const level = twin?.alarm.level ?? "nominal";
  const active = twin?.alarm.active ?? false;
  const entries = twin?.diagnosis ?? [];

  const cardClass =
    level === "warning"
      ? "card alert-card alert-warning"
      : level === "caution"
        ? "card alert-card alert-caution"
        : "card alert-card";

  const titleClass = active ? `card-title title-${level}` : "card-title";
  const titleSuffix = active && twin?.alarm.cylinder != null
    ? ` cyl ${twin.alarm.cylinder}`
    : "";

  return (
    <section className={cardClass} aria-live="polite">
      <h2 className={titleClass}>
        FAULT ALERT
        {titleSuffix && <span className="title-suffix">{titleSuffix}</span>}
      </h2>
      <p className="alert-sub">ranked differential diagnosis</p>

      {entries.length === 0 ? (
        <div className="alert-clear">
          <p className="alert-clear-main">no active flags</p>
          <p className="alert-clear-sub">
            all channels inside the predicted band
          </p>
        </div>
      ) : (
        <>
          <div className="diag-list">
            {entries.map((d) => (
              <div key={d.rank} className="diag-entry">
                <div className="diag-row">
                  <span className="diag-label">{d.label}</span>
                  <span className={confClass(d.rank)}>{fmtPct(d.confidence)}</span>
                </div>
                <div className="bar-track">
                  <div
                    className={barClass(d.rank)}
                    style={{ width: `${Math.min(d.confidence, 1) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          {entries[0].evidence.length > 0 && (
            <div className="evidence">
              <div className="evidence-title">evidence</div>
              {entries[0].evidence.map((e, i) => (
                <div key={i} className="evidence-line mono">
                  {e}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
