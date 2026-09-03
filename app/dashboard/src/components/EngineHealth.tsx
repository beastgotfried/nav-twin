import type { ChannelStatus, TwinState } from "../lib/protocol";
import { fmtGrouped, fmtKPa, fmtTempC, fmtZ } from "../lib/format";
import type { TrendChannel } from "./HealthTrend";

/** Residual colouring per app/README.md: caution from |z| in [2, 3),
 * warning from |z| >= 3. */
function zClass(z: number): string {
  const a = Math.abs(z);
  if (a >= 3) return "z-flag";
  if (a >= 2) return "z-caution";
  return "z-ok";
}

function ZLabel({ z }: { z: number }) {
  return (
    <span className={`row-sub mono ${zClass(z)}`}>
      z {fmtZ(z)}
      {Math.abs(z) >= 3 && " FLAG"}
    </span>
  );
}

function Row({
  label,
  value,
  unit,
  sub,
}: {
  label: string;
  value: string;
  unit: string;
  sub: React.ReactNode;
}) {
  return (
    <div className="health-row">
      <span className="row-label">{label}</span>
      <span className="row-value mono">
        {value} <span className="row-unit">{unit}</span>
      </span>
      {sub}
    </div>
  );
}

function CylinderCells({
  title,
  kind,
  twin,
  selected,
  onSelect,
}: {
  title: string;
  kind: "EGT" | "CHT";
  twin: TwinState;
  selected: TrendChannel;
  onSelect: (ch: TrendChannel) => void;
}) {
  return (
    <div className="cyl-group">
      <div className="cyl-group-title">{title}</div>
      <div className="cyl-grid">
        {twin.cylinders.map((c) => {
          const isSel = selected.kind === kind && selected.cyl === c.n;
          const value = kind === "EGT" ? c.EGT_K : c.CHT_K;
          const z = kind === "EGT" ? c.z_EGT : c.z_CHT;
          const status: ChannelStatus = c.status;
          return (
            <button
              key={c.n}
              type="button"
              className={`cyl-cell ${isSel ? "cyl-selected" : ""}`}
              onClick={() => onSelect({ kind, cyl: c.n })}
              title={`show ${title} cylinder ${c.n} in the trend chart`}
            >
              <span className="cyl-n">cyl {c.n}</span>
              <span
                className={`cyl-value mono ${
                  status === "warning"
                    ? "v-flag"
                    : status === "caution"
                      ? "v-caution"
                      : ""
                }`}
              >
                {fmtTempC(value)}
              </span>
              <ZLabel z={z} />
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function EngineHealth({
  twin,
  selected,
  onSelect,
}: {
  twin: TwinState | null;
  selected: TrendChannel;
  onSelect: (ch: TrendChannel) => void;
}) {
  return (
    <section className="card">
      <h2 className="card-title">ENGINE HEALTH</h2>
      {twin === null ? (
        <p className="empty-hint">awaiting telemetry</p>
      ) : (
        <>
          <Row
            label="RPM"
            value={fmtGrouped(twin.inputs.N_rpm)}
            unit="rpm"
            sub={<span className="row-sub sub-ok">measured</span>}
          />
          <CylinderCells
            title="EGT"
            kind="EGT"
            twin={twin}
            selected={selected}
            onSelect={onSelect}
          />
          <CylinderCells
            title="CHT"
            kind="CHT"
            twin={twin}
            selected={selected}
            onSelect={onSelect}
          />
          <Row
            label="Oil pressure"
            value={fmtKPa(twin.oil.p_Pa)}
            unit="kPa"
            sub={<ZLabel z={twin.oil.z_p} />}
          />
          <Row
            label="Oil temp"
            value={fmtTempC(twin.oil.T_K)}
            unit="°C"
            sub={<ZLabel z={twin.oil.z_T} />}
          />
          <Row
            label="Fuel flow"
            value={twin.inputs.fuel_flow_total_kg_h.toFixed(1)}
            unit="kg/h"
            sub={<span className="row-sub">commanded</span>}
          />
        </>
      )}
    </section>
  );
}
