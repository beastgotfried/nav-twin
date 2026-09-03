import type { AlarmLevel, TwinState } from "../lib/protocol";
import { fmtClock, fmtGrouped, fmtScenario } from "../lib/format";
import type { ConnState } from "../lib/useTwin";

const LEVEL_COLOR: Record<AlarmLevel, string> = {
  nominal: "var(--green)",
  caution: "var(--orange)",
  warning: "var(--red)",
};

export function StatusPill({ level }: { level: AlarmLevel | null }) {
  if (level === null) {
    return <span className="status-pill status-standby">STANDBY</span>;
  }
  return (
    <span className="status-pill" style={{ background: LEVEL_COLOR[level] }}>
      {level.toUpperCase()}
    </span>
  );
}

export function Header({
  twin,
  conn,
  mode = "live",
}: {
  twin: TwinState | null;
  conn: ConnState;
  mode?: "live" | "canned";
}) {
  const nCyl = twin?.cylinders.length ?? 4;
  const meta = twin
    ? [
        "MALE UAV",
        `${nCyl}-cyl`,
        fmtScenario(twin.scenario),
        `${fmtGrouped(twin.inputs.altitude_m)} m`,
        fmtClock(twin.t_s),
      ]
    : ["MALE UAV", `${nCyl}-cyl`, "no mission", "T+00:00"];

  return (
    <header className="card header">
      <div className="brand">
        <h1>SITAARA</h1>
        <p>AERO PISTON ENGINE DIGITAL TWIN</p>
      </div>
      <div className="header-meta mono">
        {meta.map((m, i) => (
          <span key={i}>
            {i > 0 && <span className="meta-dot">·</span>}
            {m}
          </span>
        ))}
      </div>
      <div className="header-right">
        {mode === "canned" && (
          <span
            className="replay-badge mono"
            title="Replaying telemetry precomputed by the real twin; the live system runs this exact pipeline against the FastAPI backend"
          >
            REPLAY
          </span>
        )}
        <span
          className={`conn-dot conn-${conn}`}
          title={`stream ${conn}`}
        />
        <StatusPill level={twin ? twin.alarm.level : null} />
      </div>
    </header>
  );
}
