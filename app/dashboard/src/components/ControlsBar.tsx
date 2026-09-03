import { useState } from "react";
import {
  ENGINE_WIDE_FAULTS,
  FAULT_KINDS,
  SENSOR_CHANNELS,
  type FaultKind,
  type SensorChannel,
} from "../lib/protocol";

/**
 * Scenario and fault-injection controls, posting to /api/control exactly as
 * app/README.md specifies. The strip is intentionally quiet: controls
 * are not data, so they stay neutral and let the status colours carry the
 * meaning.
 */

interface ControlProps {
  scenarios: string[];
  paused: boolean;
  control: (body: Record<string, unknown>) => Promise<boolean>;
  setPaused: (p: boolean) => void;
  clearHistory: () => void;
}

export function ControlsBar({
  scenarios,
  paused,
  control,
  setPaused,
  clearHistory,
}: ControlProps) {
  const [scenario, setScenario] = useState(scenarios[1] ?? scenarios[0]);
  const [kind, setKind] = useState<FaultKind>("detonation");
  const [cylinder, setCylinder] = useState(1);
  const [severity, setSeverity] = useState(1.0);
  const [channel, setChannel] = useState<SensorChannel>("CHT_K");
  const [biasK, setBiasK] = useState("10");
  const [rampS, setRampS] = useState("0");

  const engineWide = ENGINE_WIDE_FAULTS.includes(kind);
  const isDrift = kind === "sensor_drift";

  const start = () =>
    control({ action: "start", scenario, fault_events: [] }).then((ok) => {
      if (ok) {
        setPaused(false);
        clearHistory();
      }
    });

  const inject = () => {
    // ramp_s rides inside the fault object, exactly as app/README.md's
    // inject example shows it.
    const ramp = Number(rampS) || 0;
    const fault: Record<string, unknown> = isDrift
      ? { kind, cylinder, sensor_channel: channel, bias_K: Number(biasK), ramp_s: ramp }
      : engineWide
        ? { kind, severity, ramp_s: ramp }
        : { kind, cylinder, severity, ramp_s: ramp };
    return control({ action: "inject", fault });
  };

  return (
    <div className="card controls">
      <div className="ctl-group">
        <label className="ctl-label" htmlFor="scenario">scenario</label>
        <select
          id="scenario"
          className="ctl-input mono"
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
        >
          {scenarios.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <button type="button" className="btn btn-primary" onClick={start}>
          start
        </button>
        <button
          type="button"
          className="btn"
          onClick={() =>
            control({ action: paused ? "resume" : "pause" }).then(
              (ok) => ok && setPaused(!paused),
            )
          }
        >
          {paused ? "resume" : "pause"}
        </button>
        <button
          type="button"
          className="btn btn-quiet"
          onClick={() =>
            control({ action: "reset" }).then((ok) => {
              if (ok) {
                setPaused(false);
                clearHistory();
              }
            })
          }
        >
          reset
        </button>
      </div>

      <div className="ctl-divider" />

      <div className="ctl-group">
        <label className="ctl-label" htmlFor="fault-kind">fault</label>
        <select
          id="fault-kind"
          className="ctl-input mono"
          value={kind}
          onChange={(e) => setKind(e.target.value as FaultKind)}
        >
          {FAULT_KINDS.map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>

        {!engineWide && (
          <>
            <label className="ctl-label" htmlFor="fault-cyl">cyl</label>
            <select
              id="fault-cyl"
              className="ctl-input mono ctl-narrow"
              value={cylinder}
              onChange={(e) => setCylinder(Number(e.target.value))}
            >
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </>
        )}

        {isDrift ? (
          <>
            <label className="ctl-label" htmlFor="drift-channel">channel</label>
            <select
              id="drift-channel"
              className="ctl-input mono ctl-narrow"
              value={channel}
              onChange={(e) => setChannel(e.target.value as SensorChannel)}
            >
              {SENSOR_CHANNELS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <label className="ctl-label" htmlFor="drift-bias">bias K</label>
            <input
              id="drift-bias"
              className="ctl-input mono ctl-num"
              type="number"
              step="1"
              value={biasK}
              onChange={(e) => setBiasK(e.target.value)}
            />
          </>
        ) : (
          <>
            <label className="ctl-label" htmlFor="fault-sev">severity</label>
            <input
              id="fault-sev"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={severity}
              onChange={(e) => setSeverity(Number(e.target.value))}
              className="ctl-range"
            />
            <span className="mono ctl-sev">{severity.toFixed(2)}</span>
          </>
        )}

        <label className="ctl-label" htmlFor="fault-ramp">ramp s</label>
        <input
          id="fault-ramp"
          className="ctl-input mono ctl-num"
          type="number"
          min="0"
          step="5"
          value={rampS}
          onChange={(e) => setRampS(e.target.value)}
        />

        <button type="button" className="btn btn-inject" onClick={inject}>
          inject
        </button>
        <button
          type="button"
          className="btn btn-quiet"
          onClick={() => control({ action: "clear_faults" })}
        >
          clear faults
        </button>
      </div>
    </div>
  );
}
