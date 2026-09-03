import type { TwinState } from "../lib/protocol";
import { fmtDuration, fmtScenario } from "../lib/format";
import type { MissionStats } from "../lib/useTwin";

const NEXT_ACTION: Record<string, string> = {
  nominal: "continue",
  caution: "monitor",
  warning: "inspect",
};

export function MissionReport({
  twin,
  stats,
}: {
  twin: TwinState | null;
  stats: MissionStats;
}) {
  const level = twin?.alarm.level ?? "nominal";
  return (
    <section className="card">
      <h2 className="card-title">MISSION REPORT</h2>
      {twin === null ? (
        <p className="empty-hint">awaiting telemetry</p>
      ) : (
        <div className="kv-rows">
          <div className="kv-row">
            <span className="kv-label">Profile</span>
            <span className="kv-value mono">{fmtScenario(twin.scenario)}</span>
          </div>
          <div className="kv-row">
            <span className="kv-label">Flown</span>
            <span className="kv-value mono">{fmtDuration(twin.t_s)}</span>
          </div>
          <div className="kv-row">
            <span className="kv-label">Anomalies flagged</span>
            <span className="kv-value mono">
              {stats.anomalies}
              {stats.flaggedCylinders.length > 0 &&
                ` (cyl ${stats.flaggedCylinders.join(", ")})`}
            </span>
          </div>
          <div className="kv-row">
            <span className="kv-label">Next action</span>
            <span className={`kv-value kv-action kv-${level}`}>
              {NEXT_ACTION[level]}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
