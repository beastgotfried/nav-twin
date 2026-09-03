import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { TrendBuffers, TrendPoint } from "../lib/useTwin";
import { fmtClock, fmtGrouped } from "../lib/format";

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  CanvasRenderer,
]);

export interface TrendChannel {
  kind: "EGT" | "CHT";
  cyl: number;
}

/**
 * Live band chart on Apache ECharts, styled after 09-Visuals figures:
 * predicted line in blue with a 13% alpha wash for the +/- 2 sigma band
 * (the caution threshold in app/README.md), observed line in red, and
 * a red wash over the time ranges where the observed trace has exited the
 * band. Temperatures are drawn in the dashboard's display unit, degrees C.
 */

const BAND_SIGMAS = 2;

/* Dark-glass palette, mirrors the CSS tokens in styles.css. */
const C = {
  blue: "#4fb0ff",
  red: "#ff5d74",
  ink: "#e9eff5",
  soft: "#9aa9b8",
  mute: "#647384",
  grid: "rgba(255,255,255,0.07)",
  blueWash: "rgba(79,176,255,0.13)",
  redWash: "rgba(255,93,116,0.13)",
  mono: "'Geist Mono', ui-monospace, monospace",
};

function toC(points: TrendPoint[]): TrendPoint[] {
  return points.map((p) => ({
    t: p.t,
    obs: p.obs - 273.15,
    pred: p.pred - 273.15,
    sigma: p.sigma,
  }));
}

/** Contiguous [tStart, tEnd] runs where the observed trace exits the band. */
function excessRuns(pts: TrendPoint[]): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const out =
      p.obs > p.pred + BAND_SIGMAS * p.sigma ||
      p.obs < p.pred - BAND_SIGMAS * p.sigma;
    if (out && start === null) start = p.t;
    if (!out && start !== null) {
      runs.push([start, pts[i - 1].t]);
      start = null;
    }
  }
  if (start !== null) runs.push([start, pts[pts.length - 1].t]);
  return runs;
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
  const chartEl = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.EChartsType | null>(null);

  const option = useMemo(() => {
    const pts = toC(raw);
    if (pts.length < 2) return null;

    const t0 = pts[0].t;
    const t1 = Math.max(pts[pts.length - 1].t, t0 + 1);
    const last = pts[pts.length - 1];

    const bandBase = pts.map((p) => [p.t, p.pred - BAND_SIGMAS * p.sigma]);
    const bandSpan = pts.map((p) => [p.t, 2 * BAND_SIGMAS * p.sigma]);
    const pred = pts.map((p) => [p.t, p.pred]);
    const obs = pts.map((p) => [p.t, p.obs]);

    const phaseTicks = (phases ?? [])
      .filter((p) => p.start_s > t0 && p.start_s < t1)
      .map((p) => ({
        xAxis: p.start_s,
        name: p.name,
        label: { formatter: p.name },
      }));

    return {
      // Live telemetry: animation is off so 1 Hz appends never replay
      // entry transitions (that replay is what read as flicker).
      animation: false,
      grid: { left: 52, right: 16, top: 22, bottom: 28 },
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "line" as const,
          lineStyle: { color: C.mute, width: 1, type: "dashed" as const },
        },
        backgroundColor: "rgba(13, 19, 27, 0.92)",
        borderColor: "rgba(255,255,255,0.1)",
        textStyle: { color: C.ink, fontFamily: C.mono, fontSize: 11 },
        formatter: (params: unknown) => {
          const list = (params as { seriesName: string; value: number[] }[]);
          const o = list.find((p) => p.seriesName === "observed")?.value;
          const pr = list.find((p) => p.seriesName === "predicted")?.value;
          if (!o || !pr) return "";
          const s = pts.find((p) => p.t === o[0])?.sigma ?? 0;
          const lo = pr[1] - BAND_SIGMAS * s;
          const hi = pr[1] + BAND_SIGMAS * s;
          return [
            fmtClock(o[0]),
            `obs  ${o[1].toFixed(1)} °C`,
            `pred ${pr[1].toFixed(1)} °C`,
            `band ${lo.toFixed(1)} .. ${hi.toFixed(1)}`,
          ].join("<br/>");
        },
      },
      xAxis: {
        type: "value" as const,
        min: t0,
        max: t1,
        splitNumber: 6,
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.18)" } },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: {
          color: C.soft,
          fontFamily: C.mono,
          fontSize: 10,
          hideOverlap: true,
          formatter: (v: number) => fmtClock(v),
        },
      },
      yAxis: {
        type: "value" as const,
        scale: true,
        name: "°C",
        nameTextStyle: {
          color: C.soft,
          fontFamily: C.mono,
          fontSize: 10,
          align: "left" as const,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: C.grid, width: 0.7 } },
        axisLabel: {
          color: C.soft,
          fontFamily: C.mono,
          fontSize: 10,
          formatter: (v: number) => fmtGrouped(v),
        },
      },
      series: [
        {
          name: "band-base",
          type: "line" as const,
          data: bandBase,
          stack: "band",
          showSymbol: false,
          silent: true,
          lineStyle: { opacity: 0 },
          emphasis: { disabled: true },
          tooltip: { show: false },
        },
        {
          name: "band",
          type: "line" as const,
          data: bandSpan,
          stack: "band",
          showSymbol: false,
          silent: true,
          lineStyle: { opacity: 0 },
          areaStyle: { color: C.blueWash },
          emphasis: { disabled: true },
          tooltip: { show: false },
        },
        {
          name: "predicted",
          type: "line" as const,
          data: pred,
          showSymbol: false,
          lineStyle: {
            color: C.blue,
            width: 1.75,
            join: "round" as const,
            cap: "round" as const,
          },
          markLine: {
            silent: true,
            symbol: "none",
            data: phaseTicks,
            lineStyle: {
              color: C.soft,
              width: 0.7,
              type: "dashed" as const,
              opacity: 0.55,
            },
            label: {
              color: C.soft,
              fontSize: 8.5,
              fontFamily: "'Geist Variable', system-ui, sans-serif",
              position: "insideStartTop" as const,
            },
          },
        },
        {
          name: "observed",
          type: "line" as const,
          data: obs,
          showSymbol: false,
          lineStyle: {
            color: C.red,
            width: 1.75,
            join: "round" as const,
            cap: "round" as const,
          },
          markPoint: {
            symbol: "circle",
            symbolSize: 6.5,
            itemStyle: {
              color: C.red,
              borderColor: "rgba(255,255,255,0.65)",
              borderWidth: 1,
            },
            label: { show: false },
            data: [{ coord: [last.t, last.obs] }],
          },
          markArea: {
            silent: true,
            itemStyle: { color: C.redWash },
            data: excessRuns(pts).map(([a, b]) => [
              { xAxis: a },
              { xAxis: Math.max(b, a + (t1 - t0) * 0.004) },
            ]),
          },
        },
      ],
    };
    // histVersion bumps on every appended point; raw identity is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw, histVersion, phases]);

  /*
   * Init once on mount and never unmount the chart node. ECharts manages
   * DOM inside the container; unmounting that container (when history is
   * cleared on start/reset) makes React 19's deletion pass trip over nodes
   * it no longer owns, crashing the whole app with a removeChild
   * NotFoundError. The empty state is an overlay sibling instead.
   */
  useEffect(() => {
    const el = chartEl.current;
    if (!el) return;
    const chart = echarts.init(el, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    const ro = new ResizeObserver(() => {
      if (!chart.isDisposed()) chart.resize();
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Merge, never replace: replaceMerge re-creates the series on every
    // tick, which restarts rendering and shows up as flicker.
    const chart = chartRef.current;
    if (!chart || chart.isDisposed()) return;
    try {
      if (option) {
        chart.setOption(option);
      } else {
        chart.clear();
      }
    } catch (err) {
      console.error("trend chart update failed", err);
    }
  }, [option]);

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

      <div className="trend-body">
        <div
          ref={chartEl}
          className="trend-chart"
          style={{ visibility: option === null ? "hidden" : "visible" }}
          role="img"
          aria-label={`${channel.kind} cylinder ${channel.cyl} temperature trend, predicted versus observed`}
        />
        {option === null && (
          <div className="trend-empty">
            <p className="empty-hint">awaiting telemetry</p>
          </div>
        )}
      </div>

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
    </section>
  );
}
