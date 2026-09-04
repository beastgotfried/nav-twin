import type { TwinState } from "../lib/protocol";
import { fmtPct } from "../lib/format";

/**
 * The learned layer's view, straight from state.ml (twin/ml_bridge.py):
 * the isolation forest's anomaly verdict, the classifier's ranked
 * diagnosis with SHAP evidence, and the particle filter's bounded RUL.
 * Renders nothing on a rules-only checkout (no model artifacts), which
 * keeps the dashboard honest about what is actually deployed.
 */
export function MLPanel({ twin }: { twin: TwinState | null }) {
  const ml = twin?.ml;
  if (!ml || !ml.available) return null;

  const flag = ml.anomaly_flag ?? false;
  const top = ml.diagnosis?.[0];
  const rul = ml.rul;

  return (
    <section className={`card ml-card ${flag ? "alert-caution" : ""}`}>
      <h2 className="card-title">LEARNED LAYER</h2>
      <p className="alert-sub">isolation forest + classifier + particle filter</p>

      <div className="kv-row">
        <span className="kv-label">anomaly detector</span>
        <span className={`kv-value mono ${flag ? "adv-warning-text" : ""}`}>
          {ml.note ? ml.note : flag ? "FLAGGED" : "quiet"}
          {ml.anomaly_score !== undefined &&
            `  (score ${ml.anomaly_score.toFixed(3)})`}
        </span>
      </div>

      {top && (
        <div className="diag-list">
          {ml.diagnosis!.map((d) => (
            <div key={d.rank} className="diag-entry">
              <div className="diag-row">
                <span className="diag-label">{d.label}</span>
                <span className="mono conf conf-top">{fmtPct(d.confidence)}</span>
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill bar-second"
                  style={{ width: `${Math.min(d.confidence, 1) * 100}%` }}
                />
              </div>
            </div>
          ))}
          {top.evidence.length > 0 && (
            <div className="evidence">
              <div className="evidence-title">
                model evidence{top.attribution ? ` (${top.attribution})` : ""}
              </div>
              {top.evidence.map((e, i) => (
                <div key={i} className="evidence-line mono">
                  {e}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {rul && (
        <div className="rul-block">
          <div className="evidence-title">degradation tracker · {rul.subsystem}</div>
          <div className="kv-row">
            <span className="kv-label">severity</span>
            <span className="kv-value mono">
              {rul.severity_median.toFixed(2)} [{rul.severity_p10.toFixed(2)}
              {" .. "}{rul.severity_p90.toFixed(2)}]
            </span>
          </div>
          {rul.projection && (
            <div className="kv-row">
              <span className="kv-label">time to threshold</span>
              <span className="kv-value mono">
                {rul.projection.t_to_failure_hr_median.toFixed(1)} h [
                {rul.projection.t_to_failure_hr_p10.toFixed(1)}
                {" .. "}{rul.projection.t_to_failure_hr_p90.toFixed(1)}]
              </span>
            </div>
          )}
          <p className="adv-foot">{rul.framing}</p>
        </div>
      )}
    </section>
  );
}
