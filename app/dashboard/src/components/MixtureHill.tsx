import { useEffect, useMemo, useState } from "react";
import type { TwinState } from "../lib/protocol";

/**
 * The mixture hill.
 *
 * This is the project's load-bearing physics argument. Exhaust gas
 * temperature is NOT monotonic in equivalence ratio: it rises to a peak near
 * phi = 1 and falls away on both sides, so the same EGT reading occurs at two
 * different mixtures. That is why threshold monitoring on EGT is ambiguous by
 * construction.
 *
 * The curve is NOT drawn here. It is computed by physics.cycle's
 * egt_steady_state_K, exported by 10-Twin/export_mixture_curve.py, and
 * shipped as static JSON, the same contract as the canned telemetry. If it
 * ever stops being a hill, the exporter's own assert fails at build time
 * rather than the shape quietly going wrong on a slide.
 *
 * WHAT THE FIRST VERSION GOT WRONG, kept as a record per the project's habit
 * of looking at the render. It plotted the curve correctly and buried the
 * argument anyway:
 *
 *  - The two flanks differ in slope by about 17x (+723 degC per phi lean,
 *    -42 rich, read off the exported curve). On one linear axis the rich
 *    flank is visually flat, so the picture read as "a peak and a plateau"
 *    rather than "two mixtures, one temperature".
 *  - A large gradient slab filled most of the panel and carried nothing.
 *  - The equal-EGT pair, which IS the argument, was a thin dashed line in the
 *    top tenth of the plot with no numbers on it.
 *  - The y axis was labelled "degC" with no values, so nothing on it could be
 *    checked.
 *
 * The redraw keeps the geometry honest (same curve, same linear axis, the
 * kink at stoichiometric left sharp because the min(phi,1) term is real) and
 * spends the panel on the pair instead: one horizontal read at a temperature
 * that has two solutions, both solutions marked and labelled with their own
 * phi, and the distance between them called out. The flat rich flank stops
 * being a rendering problem and becomes the point, because a nearly flat
 * flank is exactly why the second solution is so far away.
 *
 * One honesty constraint drove the rest. The curve's absolute position shifts
 * with intake and ambient temperature, so it is evaluated at a single stated
 * reference condition and the caption says so. Each cylinder is placed on the
 * curve by its own live phi, which answers "which side of the peak is this
 * cylinder on". It does not claim the curve's EGT equals that cylinder's
 * observed EGT: the observed value lives in the trend chart, plotted against
 * its own prediction.
 */

interface Curve {
  reference: { altitude_m: number; T_amb_K: number; T_im_K: number };
  phi: number[];
  EGT_K: number[];
  peak: { phi: number; EGT_K: number };
}

const W = 360;
const H = 232;
const PAD_L = 44;
const PAD_R = 14;
const PAD_T = 18;
const PAD_B = 46;

/**
 * Where to take the equal-EGT read, as a fraction of the way from the rich
 * end of the curve up to the peak.
 *
 * Anchored against the RICH END rather than the full temperature range, and
 * that is forced by the physics rather than chosen for looks. The rich flank
 * is nearly flat (the min(phi,1) term collapses the slope past
 * stoichiometric, 00-STREAM 2.18), so a drop sized against the whole curve
 * overshoots the rich flank entirely and no second solution exists. Sitting
 * close to the rich end guarantees one crossing on each flank and puts the
 * rich crossing far from the peak, which is where the separation the argument
 * needs comes from. The lean flank is steep, so its crossing stays near
 * stoichiometric whatever this value is.
 */
const READ_AT = 0.12;

export function MixtureHill({ twin }: { twin: TwinState | null }) {
  const [curve, setCurve] = useState<Curve | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/canned/mixture_curve.json");
        if (!res.ok || cancelled) return;
        setCurve((await res.json()) as Curve);
      } catch {
        // No curve shipped: the panel stays empty rather than inventing one.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const model = useMemo(() => {
    if (!curve) return null;
    const phis = curve.phi;
    const egtC = curve.EGT_K.map((k) => k - 273.15);

    const x0 = phis[0];
    const x1 = phis[phis.length - 1];
    const lo = Math.min(...egtC);
    const hi = Math.max(...egtC);
    // Headroom above the peak and below the lean end so the curve never
    // touches the frame and the peak label has somewhere to sit.
    const yLo = lo - (hi - lo) * 0.08;
    const yHi = hi + (hi - lo) * 0.14;

    const X = (p: number) =>
      PAD_L + ((p - x0) / (x1 - x0)) * (W - PAD_L - PAD_R);
    const Y = (v: number) =>
      PAD_T + (1 - (v - yLo) / (yHi - yLo)) * (H - PAD_T - PAD_B);

    const path = phis
      .map(
        (p, i) => `${i === 0 ? "M" : "L"}${X(p).toFixed(1)},${Y(egtC[i]).toFixed(1)}`,
      )
      .join(" ");

    /** EGT at a given phi, by linear interpolation along the computed curve. */
    const egtAt = (p: number): number => {
      if (p <= x0) return egtC[0];
      if (p >= x1) return egtC[egtC.length - 1];
      let i = 1;
      while (i < phis.length && phis[i] < p) i++;
      const t = (p - phis[i - 1]) / (phis[i] - phis[i - 1]);
      return egtC[i - 1] + t * (egtC[i] - egtC[i - 1]);
    };

    // The equal-EGT pair, solved against the computed curve so the two marks
    // are genuinely at the same temperature rather than placed by eye.
    const peakC = curve.peak.EGT_K - 273.15;
    const richEndC = egtC[egtC.length - 1];
    const target = richEndC + (peakC - richEndC) * READ_AT;
    let leanPhi: number | null = null;
    let richPhi: number | null = null;
    for (let i = 1; i < phis.length; i++) {
      const a = egtC[i - 1];
      const b = egtC[i];
      if (a < target && b >= target && leanPhi === null) {
        leanPhi = phis[i - 1] + ((target - a) / (b - a)) * (phis[i] - phis[i - 1]);
      }
      if (a >= target && b < target) {
        richPhi = phis[i - 1] + ((target - a) / (b - a)) * (phis[i] - phis[i - 1]);
      }
    }

    // Y gridlines at round temperatures inside the drawn range, so every
    // height on the panel can actually be read off rather than guessed.
    const step = 100;
    const ticks: number[] = [];
    for (let v = Math.ceil(yLo / step) * step; v <= yHi; v += step) ticks.push(v);

    return {
      X, Y, path, egtAt, peakC, target, leanPhi, richPhi, x0, x1, ticks, yLo,
    };
  }, [curve]);

  const flagged = twin?.cylinders.filter((c) => c.status !== "nominal") ?? [];

  return (
    <section className="card mix-card">
      <h2 className="card-title">MIXTURE</h2>
      {model === null || curve === null ? (
        <p className="empty-hint">curve not built</p>
      ) : (
        <>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="mix-svg"
            role="img"
            aria-label={
              `Exhaust gas temperature against equivalence ratio, computed by the ` +
              `engine model. The curve peaks near phi ${curve.peak.phi.toFixed(2)}. ` +
              (model.leanPhi !== null && model.richPhi !== null
                ? `A reading of ${model.target.toFixed(0)} degrees occurs at both phi ` +
                  `${model.leanPhi.toFixed(2)} and phi ${model.richPhi.toFixed(2)}, ` +
                  `so the temperature alone cannot say which mixture the engine is at.`
                : "")
            }
          >
            <defs>
              <linearGradient id="mixGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--blue)" stopOpacity="0.16" />
                <stop offset="100%" stopColor="var(--blue)" stopOpacity="0" />
              </linearGradient>
            </defs>

            {/* Gridlines first, so everything else sits on top of them. */}
            {model.ticks.map((v) => (
              <g key={v}>
                <line
                  x1={PAD_L}
                  x2={W - PAD_R}
                  y1={model.Y(v)}
                  y2={model.Y(v)}
                  className="mix-grid"
                />
                <text x={PAD_L - 7} y={model.Y(v) + 3.2} className="mix-ytick">
                  {v}
                </text>
              </g>
            ))}

            <path
              d={`${model.path} L${(W - PAD_R).toFixed(1)},${model.Y(model.yLo).toFixed(1)} L${PAD_L.toFixed(1)},${model.Y(model.yLo).toFixed(1)} Z`}
              className="mix-area"
            />

            {/* The peak, marked where it is rather than implied by the shape. */}
            <line
              x1={model.X(curve.peak.phi)}
              x2={model.X(curve.peak.phi)}
              y1={model.Y(model.peakC) - 4}
              y2={H - PAD_B}
              className="mix-peak"
            />
            <circle
              cx={model.X(curve.peak.phi)}
              cy={model.Y(model.peakC)}
              r={2.6}
              className="mix-peak-dot"
            />
            <text
              x={model.X(curve.peak.phi) + 7}
              y={model.Y(model.peakC) - 5}
              className="mix-peak-label"
            >
              peak &phi; {curve.peak.phi.toFixed(2)}
            </text>

            <path d={model.path} className="mix-curve" />

            {/* The equal-EGT pair: the whole argument, drawn as one read
                across the curve with both answers labelled. */}
            {model.leanPhi !== null && model.richPhi !== null && (
              <>
                <line
                  x1={PAD_L}
                  x2={W - PAD_R}
                  y1={model.Y(model.target)}
                  y2={model.Y(model.target)}
                  className="mix-tie"
                />
                <text
                  x={PAD_L - 7}
                  y={model.Y(model.target) + 3.2}
                  className="mix-ytick mix-ytick-read"
                >
                  {model.target.toFixed(0)}
                </text>

                {/* Drop lines, so each solution can be read down to its own
                    mixture on the axis. */}
                {[model.leanPhi, model.richPhi].map((p, i) => (
                  <g key={i}>
                    <line
                      x1={model.X(p)}
                      x2={model.X(p)}
                      y1={model.Y(model.target)}
                      y2={H - PAD_B}
                      className="mix-drop"
                    />
                    <circle
                      cx={model.X(p)}
                      cy={model.Y(model.target)}
                      r={4.2}
                      className="mix-amb"
                    />
                    <text
                      x={model.X(p)}
                      y={H - PAD_B + 14}
                      className="mix-amb-label"
                    >
                      &phi; {p.toFixed(2)}
                    </text>
                  </g>
                ))}

                {/* Below the read, not above it. Above, it landed on the
                    peak and on the cylinder cluster, which sit at the same
                    height and the same place for a healthy engine. */}
                <text
                  x={(model.X(model.leanPhi) + model.X(model.richPhi)) / 2}
                  y={model.Y(model.target) + 16}
                  className="mix-note"
                >
                  one reading, two mixtures
                </text>
              </>
            )}

            {/* Live cylinders, placed by their own forward-computed phi. */}
            {twin?.cylinders.map((c) =>
              c.phi === undefined ? null : (
                <g key={c.n}>
                  <circle
                    cx={model.X(c.phi)}
                    cy={model.Y(model.egtAt(c.phi))}
                    r={4.4}
                    className={`mix-cyl ${c.status !== "nominal" ? "mix-cyl-flag" : ""}`}
                  />
                  {/* Numbered only once it has separated from the pack. A
                      healthy engine runs every cylinder at the same mixture,
                      so the dots SHOULD sit on top of each other and four
                      labels in one spot is an unreadable blob. A cylinder
                      drifting off the cluster is the signal, and that is
                      exactly the one worth naming. */}
                  {c.status !== "nominal" && (
                    <text
                      x={model.X(c.phi)}
                      y={model.Y(model.egtAt(c.phi)) - 9}
                      className="mix-cyl-label"
                    >
                      {c.n}
                    </text>
                  )}
                </g>
              ),
            )}

            <line
              x1={PAD_L}
              x2={W - PAD_R}
              y1={H - PAD_B}
              y2={H - PAD_B}
              className="axis-line"
            />
            <text x={PAD_L} y={H - PAD_B + 27} className="mix-xend">
              leaner
            </text>
            <text
              x={W - PAD_R}
              y={H - PAD_B + 27}
              textAnchor="end"
              className="mix-xend"
            >
              richer
            </text>
            <text x={PAD_L - 7} y={PAD_T - 5} textAnchor="end" className="mix-axis-unit">
              &deg;C
            </text>
          </svg>

          <p className="mix-foot">
            EGT against equivalence ratio, computed by the engine model at{" "}
            {curve.reference.altitude_m.toFixed(0)} m. Peak at &phi;{" "}
            {curve.peak.phi.toFixed(2)}.{" "}
            {flagged.length > 0
              ? `Cylinder ${flagged.map((c) => c.n).join(", ")} has left the cluster.`
              : "All four cylinders share a mixture, so their marks overlap."}
          </p>
          <p className="panel-plain">
            A thermocouple reports one number, and the curve gives that number
            two answers: one lean of the peak, one rich of it. Which one an
            engine is actually at changes what a fault means, so Sitaara
            computes &phi; forward from fuel and air and never reads it back
            off the temperature.
          </p>
        </>
      )}
    </section>
  );
}
