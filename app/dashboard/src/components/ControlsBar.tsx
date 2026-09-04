import { useCallback, useRef, useState } from "react";
import {
  ENGINE_WIDE_FAULTS,
  FAULT_KINDS,
  SENSOR_CHANNELS,
  type FaultKind,
  type SensorChannel,
} from "../lib/protocol";

/**
 * Scenario and fault-injection controls, posting to /api/control exactly as
 * 10-Twin/README.md specifies. The strip is intentionally quiet: controls
 * are not data, so they stay neutral and let the status colours carry the
 * meaning.
 *
 * Quiet is not the same as silent. Every button here awaits a POST and then
 * did nothing visible with the answer, which made a working "start" look
 * broken: restarting a mission that is already streaming the same scenario
 * changes little on screen beyond the clock, so with no acknowledgement the
 * only honest reading was that the click was lost. Worse, `control` returns
 * false on an error and every caller dropped it, so a genuine failure was
 * indistinguishable from a success. useAction below fixes both: the button
 * says it is working, then says whether it worked.
 */

/** How long the ok / failed mark stays up before the button goes quiet. */
const ACK_MS = 1100;

type ActState = "idle" | "busy" | "ok" | "err";

/**
 * Wrap a control action so the button that fires it can show its own state.
 *
 * Returns props to spread onto the button. Pending clicks are swallowed
 * rather than queued: these actions restart or perturb a running mission, so
 * firing a second one because the first had not answered yet is never what
 * the operator meant.
 */
function useAction(run: () => Promise<boolean>) {
  const [state, setState] = useState<ActState>("idle");
  const timer = useRef<number | undefined>(undefined);
  const busy = state === "busy";

  const onClick = useCallback(() => {
    if (busy) return;
    window.clearTimeout(timer.current);
    setState("busy");
    void run().then(
      (ok) => {
        setState(ok ? "ok" : "err");
        timer.current = window.setTimeout(() => setState("idle"), ACK_MS);
      },
      () => {
        setState("err");
        timer.current = window.setTimeout(() => setState("idle"), ACK_MS);
      },
    );
  }, [busy, run]);

  return {
    onClick,
    "data-act": state,
    "aria-busy": busy,
    "aria-live": "polite" as const,
  };
}

interface ControlProps {
  scenarios: string[];
  onGuided: () => void;
  paused: boolean;
  control: (body: Record<string, unknown>) => Promise<boolean>;
  setPaused: (p: boolean) => void;
  clearHistory: () => void;
}

export function ControlsBar({
  scenarios,
  onGuided,
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

  const start = useAction(() =>
    control({ action: "start", scenario, fault_events: [] }).then((ok) => {
      if (ok) {
        setPaused(false);
        clearHistory();
      }
      return ok;
    }),
  );

  const pauseResume = useAction(() =>
    control({ action: paused ? "resume" : "pause" }).then((ok) => {
      if (ok) setPaused(!paused);
      return ok;
    }),
  );

  const reset = useAction(() =>
    control({ action: "reset" }).then((ok) => {
      if (ok) {
        setPaused(false);
        clearHistory();
      }
      return ok;
    }),
  );

  const clearFaults = useAction(() => control({ action: "clear_faults" }));

  const inject = () => {
    // ramp_s rides inside the fault object, exactly as 10-Twin/README.md's
    // inject example shows it.
    const ramp = Number(rampS) || 0;
    const fault: Record<string, unknown> = isDrift
      ? { kind, cylinder, sensor_channel: channel, bias_K: Number(biasK), ramp_s: ramp }
      : engineWide
        ? { kind, severity, ramp_s: ramp }
        : { kind, cylinder, severity, ramp_s: ramp };
    return control({ action: "inject", fault });
  };

  const injectAction = useAction(inject);

  return (
    <div className="card controls">
      {/* The way back into the guided flight.
       *
       * It used to be reachable only from the welcome overlay, which shows
       * once per page load. Dismissing that overlay therefore threw away the
       * phase timeline, the speed control and the captions with no way to get
       * them back short of a reload, and nothing on screen said they had ever
       * existed. It is the first control in the strip because for a first
       * time viewer it is the right one to press. */}
      <div className="ctl-group">
        <button type="button" className="btn btn-guided" onClick={onGuided}>
          guided flight
        </button>
      </div>

      <div className="ctl-divider" />

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
        <button type="button" className="btn btn-primary" {...start}>
          start
        </button>
        <button type="button" className="btn" {...pauseResume}>
          {paused ? "resume" : "pause"}
        </button>
        <button type="button" className="btn btn-quiet" {...reset}>
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

        <button type="button" className="btn btn-inject" {...injectAction}>
          inject
        </button>
        <button type="button" className="btn btn-quiet" {...clearFaults}>
          clear faults
        </button>
      </div>
    </div>
  );
}
