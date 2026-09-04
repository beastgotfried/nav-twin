/**
 * Canned-mode twin hook for the static hosted build.
 *
 * The hosted site (Vercel) has no FastAPI backend: it replays twin states
 * precomputed by 10-Twin/export_canned.py from the real mission + twin
 * pipeline, shipped as static JSON under /canned/. The interface mirrors
 * useTwin so the components render identically; the header badge marks the
 * page as a replay, and this file's data is never edited after export.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { TwinState } from "./protocol";
import { FALLBACK_SCENARIOS } from "./protocol";
import type { ConnState, MissionStats, TrendBuffers } from "./useTwin";

interface CannedEntry {
  name: string;
  scenario: string;
  faults: {
    kind: string;
    cylinder: number | null;
    severity: number;
    sensor_channel: string | null;
    bias_K: number;
  }[];
}

interface CannedFile {
  name: string;
  profile: string;
  frames: TwinState[];
  phases?: { start_s: number; name: string; caption: string }[];
}

export interface DemoPhase {
  start_s: number;
  name: string;
  caption: string;
}

const MAX_POINTS = 2000;
const TICK_MS = 1000;

function appendPoint(buffers: TrendBuffers, key: string, p: { t: number; obs: number; pred: number; sigma: number }) {
  const arr = buffers[key] ?? (buffers[key] = []);
  arr.push(p);
  if (arr.length > MAX_POINTS) arr.splice(0, arr.length - MAX_POINTS);
}

/** Map an inject request to the canned entry that reproduces it. */
function matchFault(entries: CannedEntry[], scenario: string, fault: Record<string, unknown>): CannedEntry | undefined {
  const kind = String(fault.kind ?? "");
  const severity = Number(fault.severity ?? 1);
  return entries.find((e) => {
    if (e.faults.length !== 1) return false;
    const f = e.faults[0];
    if (f.kind !== kind) return false;
    if (kind === "injector_restriction") {
      // Two canned cases: partial on both cruise points, blockage on rich.
      if (severity >= 0.5) return e.name === "cruise_rich_blockage";
      return e.scenario === scenario && f.severity < 0.5;
    }
    // detonation, misfire, sensor_drift are canned on endurance; a fault
    // requested on any scenario replays the endurance case (same physics).
    return true;
  });
}

export function useCanned(enabled: boolean) {
  const [conn, setConn] = useState<ConnState>("connecting");
  const [scenarios, setScenarios] = useState<string[]>(FALLBACK_SCENARIOS);
  const [twin, setTwin] = useState<TwinState | null>(null);
  const [paused, setPaused] = useState(false);
  const [stats, setStats] = useState<MissionStats>({ anomalies: 0, flaggedCylinders: [] });
  const [histVersion, setHistVersion] = useState(0);

  const manifestRef = useRef<CannedEntry[]>([]);
  const fileRef = useRef<CannedFile | null>(null);
  const idxRef = useRef(0);
  const timerRef = useRef<number | undefined>(undefined);
  const speedRef = useRef(1);
  const buffersRef = useRef<TrendBuffers>({});
  const prevAlarmRef = useRef(false);
  const scenarioRef = useRef("endurance");
  const pausedRef = useRef(false);
  const [phases, setPhases] = useState<DemoPhase[]>([]);

  const clearHistory = useCallback(() => {
    buffersRef.current = {};
    prevAlarmRef.current = false;
    setStats({ anomalies: 0, flaggedCylinders: [] });
    setHistVersion((v) => v + 1);
  }, []);

  const emitFrame = useCallback((frame: TwinState) => {
    for (const c of frame.cylinders) {
      appendPoint(buffersRef.current, `EGT_${c.n}`, {
        t: frame.t_s, obs: c.EGT_K, pred: c.EGT_pred_K, sigma: c.sigma_EGT_K,
      });
      appendPoint(buffersRef.current, `CHT_${c.n}`, {
        t: frame.t_s, obs: c.CHT_K, pred: c.CHT_pred_K, sigma: c.sigma_CHT_K,
      });
    }
    const rising = frame.alarm.active && !prevAlarmRef.current;
    prevAlarmRef.current = frame.alarm.active;
    if (rising) {
      setStats((st) => ({
        anomalies: st.anomalies + 1,
        flaggedCylinders:
          frame.alarm.cylinder !== null && !st.flaggedCylinders.includes(frame.alarm.cylinder)
            ? [...st.flaggedCylinders, frame.alarm.cylinder]
            : st.flaggedCylinders,
      }));
    }
    setTwin(frame);
    setHistVersion((v) => v + 1);
  }, []);

  const startTicker = useCallback(() => {
    window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      const file = fileRef.current;
      if (!file || pausedRef.current) return;
      if (idxRef.current >= file.frames.length) {
        // A guided flight ends at its last frame and holds there; an
        // ambient scenario replay loops.
        if (file.phases && file.phases.length > 0) {
          setPaused(true);
          return;
        }
        idxRef.current = 0;
        buffersRef.current = {};
      }
      emitFrame(file.frames[idxRef.current]);
      idxRef.current += 1;
    }, TICK_MS / speedRef.current);
  }, [emitFrame]);

  const load = useCallback(async (entryName: string, seekT = 0) => {
    const res = await fetch(`/canned/${entryName}.json`);
    if (!res.ok) return false;
    const file = (await res.json()) as CannedFile;
    fileRef.current = file;
    setPhases(file.phases ?? []);
    const idx = file.frames.findIndex((f) => f.t_s >= seekT);
    idxRef.current = idx >= 0 ? idx : 0;
    clearHistory();
    return true;
  }, [clearHistory]);

  /** Jump to a mission time inside the currently loaded replay. */
  const seek = useCallback((t: number) => {
    const file = fileRef.current;
    if (!file) return;
    const idx = file.frames.findIndex((f) => f.t_s >= t);
    idxRef.current = idx >= 0 ? idx : 0;
    clearHistory();
  }, [clearHistory]);

  /** Playback rate multiplier (1 = real time). Rebuilds the ticker. */
  const setSpeed = useCallback((mult: number) => {
    speedRef.current = Math.max(0.25, mult);
    startTicker();
  }, [startTicker]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void (async () => {
      const res = await fetch("/canned/index.json");
      if (!res.ok || cancelled) return;
      const manifest = (await res.json()) as { entries: CannedEntry[] };
      manifestRef.current = manifest.entries;
      setScenarios([...new Set(manifest.entries.map((e) => e.scenario))]);
      await load("endurance");
      if (!cancelled) {
        setConn("live");
        startTicker();
      }
    })();
    return () => {
      cancelled = true;
      window.clearInterval(timerRef.current);
    };
  }, [enabled, load, startTicker]);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  const control = useCallback(async (body: Record<string, unknown>) => {
    const action = String(body.action ?? "");
    if (action === "start") {
      const scenario = String(body.scenario ?? "endurance");
      scenarioRef.current = scenario;
      const entry = manifestRef.current.find(
        (e) => e.scenario === scenario && e.faults.length === 0,
      );
      return entry ? load(entry.name) : false;
    }
    if (action === "pause") { setPaused(true); return true; }
    if (action === "resume") { setPaused(false); return true; }
    if (action === "reset") {
      const healthy = manifestRef.current.find(
        (e) => e.scenario === scenarioRef.current && e.faults.length === 0,
      );
      return healthy ? load(healthy.name) : false;
    }
    if (action === "inject") {
      const fault = (body.fault ?? {}) as Record<string, unknown>;
      const t = twin?.t_s ?? 0;
      const entry = matchFault(manifestRef.current, scenarioRef.current, fault);
      return entry ? load(entry.name, t) : false;
    }
    if (action === "clear_faults") {
      const t = twin?.t_s ?? 0;
      const healthy = manifestRef.current.find(
        (e) => e.scenario === scenarioRef.current && e.faults.length === 0,
      );
      return healthy ? load(healthy.name, t) : false;
    }
    return false;
  }, [load, twin]);

  return {
    conn,
    scenarios,
    twin,
    paused,
    setPaused,
    stats,
    history: buffersRef.current,
    histVersion,
    clearHistory,
    control,
    seek,
    setSpeed,
    phases,
    mode: "canned" as const,
  };
}
