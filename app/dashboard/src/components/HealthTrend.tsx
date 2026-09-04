import { useMemo } from "react";
import type { TrendBuffers, TrendPoint } from "../lib/useTwin";
import { fmtClock, fmtGrouped } from "../lib/format";

export interface TrendChannel {
  kind: "EGT" | "CHT";
  cyl: number;
}

/**
 * Live band chart, styled after 09-Visuals figures: predicted line in blue
 * with a 13% alpha wash for the +/- 2 sigma band (the caution threshold in
 * 10-Twin/README.md), observed line in red, and a red wash where the
 * observed trace has exited the band. Temperatures are drawn in the
 * dashboard's display unit, degrees C.
 */

const W = 560;
const H = 240;
const PAD_L = 44;
const PAD_R = 10;
const PAD_T = 18;
const PAD_B = 24;

const BAND_SIGMAS = 2;

function toC(points: TrendPoint[]): TrendPoint[] {
  return points.map((p) => ({
    t: p.t,
    obs: p.obs - 273.15,
    pred: p.pred - 273.15,
    sigma: p.sigma,
  }));
}

function pathFrom(xs: number[], ys: number[]): string {
  return xs
    .map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`)
    .join(" ");
}

export function HealthTrend({
  history,
  histVersion,
  channel,
  onChannel,
  phases,
}: {
  history: TrendBuffers;
  histVersion: number;
  channel: TrendChannel;
  onChannel: (ch: TrendChannel) => void;
  phases?: { start_s: number; name: string }[];
}) {
  const key = `${channel.kind}_${channel.cyl}`;
  const raw = history[key] ?? [];

  const model = useMemo(() => {
    const pts = toC(raw);
    if (pts.length < 2) return null;

    // Scale to the BAND, not to the observation.
    //
    // Fitting the range to both meant one large excursion set the scale for
    // the whole window: a 147 K detonation deviation squashed the +/-2 sigma
    // band into a hairline, and the band is the entire point of this chart.
    // Act 3 depends on being able to SEE the observation sitting inside it.
    //
    // So the band always gets a guaranteed share of the plot height. The
    // observation is allowed to extend the range beyond that, but only up to
    // a bounded multiple of the band's own width, past which it clips and
    // the red excess fill carries the story instead.
    let bandLo = Infinity;
    let bandHi = -Infinity;
    let obsLo = Infinity;
    let obsHi = -Infinity;
    for (const p of pts) {
      bandLo = Math.min(bandLo, p.pred - BAND_SIGMAS * p.sigma);
      bandHi = Math.max(bandHi, p.pred + BAND_SIGMAS * p.sigma);
      obsLo = Math.min(obsLo, p.obs);
      obsHi = Math.max(obsHi, p.obs);
    }
    const bandSpan = Math.max(bandHi - bandLo, 1e-6);
    // The band keeps at least 1/MAX_RANGE_FACTOR of the vertical space.
    const MAX_RANGE_FACTOR = 3.5;
    const room = bandSpan * MAX_RANGE_FACTOR;
    const mid = (bandLo + bandHi) / 2;
    let lo = Math.max(obsLo, mid - room / 2);
    let hi = Math.min(obsHi, mid + room / 2);
    // Never crop the band itself.
    lo = Math.min(lo, bandLo);
    hi = Math.max(hi, bandHi);
    const span = Math.max(hi - lo, 1e-6);
    lo -= span * 0.06;
    hi += span * 0.06;
    const t0 = pts[0].t;
    const t1 = Math.max(pts[pts.length - 1].t, t0 + 1);

    const x = (t: number) => PAD_L + ((t - t0) / (t1 - t0)) * (W - PAD_L - PAD_R);
    const y = (v: number) => PAD_T + (1 - (v - lo) / (hi - lo)) * (H - PAD_T - PAD_B);

    const xs = pts.map((p) => x(p.t));
    const yObs = pts.map((p) => y(p.obs));
    const yPred = pts.map((p) => y(p.pred));
    const yTop = pts.map((p) => y(p.pred + BAND_SIGMAS * p.sigma));
    const yBot = pts.map((p) => y(p.pred - BAND_SIGMAS * p.sigma));

    const ribbon = (top: number[], bot: number[]) =>
      pathFrom(xs, top) +
      " " +
      xs
        .map((_, i) => pts.length - 1 - i)
        .map((i) => `L${xs[i].toFixed(1)},${bot[i].toFixed(1)}`)
        .join(" ") +
      " Z";

    const band = ribbon(yTop, yBot);

    // The one-sigma core, drawn inside the two-sigma band.
    //
    // Without it the band is a single flat slab covering most of the plot,
    // and a slab says only "somewhere in here", which is not what the twin
    // computed. Two nested ribbons say where the probability actually is, so
    // a trace hugging the centreline and a trace grazing the edge stop
    // looking alike. It costs no new data: sigma is already per point.
    const yTop1 = pts.map((p) => y(p.pred + p.sigma));
    const yBot1 = pts.map((p) => y(p.pred - p.sigma));
    const core = ribbon(yTop1, yBot1);

    // Red wash segments where the observed trace sits outside the band.
    const excess: string[] = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const outHi =
        pts[i].obs > pts[i].pred + BAND_SIGMAS * pts[i].sigma ||
        pts[i + 1].obs > pts[i + 1].pred + BAND_SIGMAS * pts[i + 1].sigma;
      const outLo =
        pts[i].obs < pts[i].pred - BAND_SIGMAS * pts[i].sigma ||
        pts[i + 1].obs < pts[i + 1].pred - BAND_SIGMAS * pts[i + 1].sigma;
      if (!outHi && !outLo) continue;
      const edgeA = outHi ? yTop[i] : yBot[i];
      const edgeB = outHi ? yTop[i + 1] : yBot[i + 1];
      excess.push(
        `M${xs[i].toFixed(1)},${edgeA.toFixed(1)} ` +
          `L${xs[i].toFixed(1)},${yObs[i].toFixed(1)} ` +
          `L${xs[i + 1].toFixed(1)},${yObs[i + 1].toFixed(1)} ` +
          `L${xs[i + 1].toFixed(1)},${edgeB.toFixed(1)} Z`,
      );
    }

    // Four horizontal gridlines with mono tick labels, like the figures.
    const ticks = [0, 1, 2, 3].map((i) => {
      const v = lo + ((i + 0.5) / 3.5) * (hi - lo);
      return { v, y: y(v) };
    });

    const last = pts[pts.length - 1];
    const phaseTicks = (phases ?? [])
      .filter((p) => p.start_s > t0 && p.start_s < t1)
      .map((p) => ({ x: x(p.start_s), name: p.name }));
    return {
      xs,
      yObs,
      yPred,
      band,
      core,
      excess,
      ticks,
      phaseTicks,
      t0,
      t1,
      lastX: xs[xs.length - 1],
      lastY: yObs[yObs.length - 1],
      last,
      n: pts.length,
    };
    // histVersion bumps on every appended point; raw identity is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw, histVersion, phases]);

  return (
    <section className="card trend-card">
      <div className="trend-head">
        <h2 className="card-title">HEALTH TREND</h2>
        <div className="trend-select">
          <span className="seg">
            {(["EGT", "CHT"] as const).map((k) => (
              <button
                key={k}
                type="button"
                className={`seg-btn ${channel.kind === k ? "seg-on" : ""}`}
                onClick={() => onChannel({ kind: k, cyl: channel.cyl })}
              >
                {k}
              </button>
            ))}
          </span>
          <span className="seg">
            {[1, 2, 3, 4].map((n) => (
              <button
                key={n}
                type="button"
                className={`seg-btn mono ${channel.cyl === n ? "seg-on" : ""}`}
                onClick={() => onChannel({ kind: channel.kind, cyl: n })}
              >
                {n}
              </button>
            ))}
          </span>
        </div>
      </div>

      {model === null ? (
        <div className="trend-empty">
          <p className="empty-hint">awaiting telemetry</p>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="trend-svg"
          role="img"
          aria-label={
            `${channel.kind} cylinder ${channel.cyl}: observed ` +
            `${(model.last.obs).toFixed(0)} degrees C against a predicted ` +
            `${(model.last.pred).toFixed(0)} plus or minus ` +
            `${(BAND_SIGMAS * model.last.sigma).toFixed(0)}, ` +
            (model.excess.length > 0
              ? "currently outside the predicted band"
              : "inside the predicted band")
          }
        >
          <defs>
            {/* The band is a distribution, not a slab. Fading it from the
                centre outwards says "most of the probability is near the
                prediction" without drawing a second chart. */}
            <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--blue)" stopOpacity="0.05" />
              <stop offset="50%" stopColor="var(--blue)" stopOpacity="0.14" />
              <stop offset="100%" stopColor="var(--blue)" stopOpacity="0.05" />
            </linearGradient>
            <linearGradient id="excessGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--red)" stopOpacity="0.22" />
              <stop offset="100%" stopColor="var(--red)" stopOpacity="0.05" />
            </linearGradient>
            {/* A soft glow so the observed trace reads as a lit instrument
                line rather than a hairline scratch on a dark ground. */}
            <filter id="obsGlow" x="-20%" y="-40%" width="140%" height="180%">
              <feGaussianBlur stdDeviation="2.2" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {model.ticks.map((tk, i) => (
            <g key={i}>
              <line
                x1={PAD_L}
                x2={W - PAD_R}
                y1={tk.y}
                y2={tk.y}
                className="grid-line"
              />
              <text x={PAD_L - 6} y={tk.y + 3} textAnchor="end" className="tick">
                {fmtGrouped(tk.v)}
              </text>
            </g>
          ))}
          <text x={PAD_L - 6} y={PAD_T - 6} textAnchor="end" className="tick">
            °C
          </text>
          <path d={model.band} className="band-fill" />
            <path d={model.core} className="band-core" />
          {model.excess.map((d, i) => (
            <path key={i} d={d} className="excess-fill" />
          ))}
          {model.phaseTicks.map((pt, i) => (
            <g key={`ph-${i}`}>
              <line
                x1={pt.x}
                x2={pt.x}
                y1={PAD_T}
                y2={H - PAD_B}
                className="phase-tick"
              />
              <text x={pt.x + 3} y={PAD_T + 9} className="phase-tick-label">
                {pt.name}
              </text>
            </g>
          ))}
          <path d={pathFrom(model.xs, model.yPred)} className="line-pred" />
          <path
            d={pathFrom(model.xs, model.yObs)}
            className="line-obs"
            filter="url(#obsGlow)"
          />
          {/* Current reading, called out where the eye already is. The trace
              is never smoothed: the jitter is real sensor noise, and
              flattening it would be drawing a nicer number than we measured. */}
          <g className="trend-now">
            <line
              x1={model.lastX}
              x2={W - PAD_R}
              y1={model.lastY}
              y2={model.lastY}
              className="now-rule"
            />
            <circle cx={model.lastX} cy={model.lastY} r={4.2} className="obs-halo" />
            <circle cx={model.lastX} cy={model.lastY} r={2.6} className="obs-dot" />
            <text
              x={W - PAD_R}
              y={model.lastY - 7}
              textAnchor="end"
              className="now-value"
            >
              {fmtGrouped(model.last.obs)} °C
            </text>
          </g>
          <line
            x1={PAD_L}
            x2={W - PAD_R}
            y1={H - PAD_B}
            y2={H - PAD_B}
            className="axis-line"
          />
          <text x={PAD_L} y={H - 8} textAnchor="start" className="tick">
            {fmtClock(model.t0)}
          </text>
          <text x={W - PAD_R} y={H - 8} textAnchor="end" className="tick">
            {fmtClock(model.t1)}
          </text>
        </svg>
      )}

      <div className="trend-legend">
        <span className="legend-item">
          <span className="swatch swatch-pred" />
          predicted ± {BAND_SIGMAS}σ
        </span>
        <span className="legend-item">
          <span className="swatch swatch-obs" />
          observed
        </span>
        <span className="legend-item legend-channel mono">
          {channel.kind} cyl {channel.cyl}
        </span>
      </div>

      {/* The plain-language half. deck-strategy.md section 2: annotate in
          plain language, label technically. The axes and the sigma notation
          carry the precision for a propulsion reader; this sentence is what
          a non-specialist reads instead, and it changes with the state so it
          is never a caption nobody looks at twice. */}
      {model !== null && (
        <p className="trend-plain">
          {model.excess.length > 0
            ? "The engine is doing something the model cannot account for. Red is the gap between what was predicted and what the sensor read."
            : "The shaded band is what the twin expects this sensor to read right now. Inside it means normal, however the number looks."}
        </p>
      )}
    </section>
  );
}
