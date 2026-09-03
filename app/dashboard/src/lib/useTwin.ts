/**
 * Twin connection hook. Opens WS /ws (same origin in production, proxied to
 * :8734 by Vite in dev), keeps the latest twin state, accumulates the chart
 * history client-side (app/README.md: state messages are accumulated by
 * the client; GET /api/history backfills since the last reset), and posts
 * control actions.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  HistoryResponse,
  ServerMessage,
  TwinState,
} from "./protocol";
import { FALLBACK_SCENARIOS } from "./protocol";

export type ConnState = "connecting" | "live" | "closed";

export interface TrendPoint {
  t: number;
  obs: number;
  pred: number;
  sigma: number;
}

/** key: "EGT_1" .. "CHT_4" */
export type TrendBuffers = Record<string, TrendPoint[]>;

export interface MissionStats {
  anomalies: number;
  flaggedCylinders: number[];
}

const MAX_POINTS = 2000;

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

function appendPoint(buffers: TrendBuffers, key: string, p: TrendPoint) {
  const arr = buffers[key] ?? (buffers[key] = []);
  arr.push(p);
  if (arr.length > MAX_POINTS) arr.splice(0, arr.length - MAX_POINTS);
}

export function useTwin() {
  const [conn, setConn] = useState<ConnState>("connecting");
  const [scenarios, setScenarios] = useState<string[]>(FALLBACK_SCENARIOS);
  const [twin, setTwin] = useState<TwinState | null>(null);
  const [paused, setPaused] = useState(false);
  const [stats, setStats] = useState<MissionStats>({
    anomalies: 0,
    flaggedCylinders: [],
  });
  const [histVersion, setHistVersion] = useState(0);
  const buffersRef = useRef<TrendBuffers>({});
  const lastTRef = useRef<number | null>(null);
  const prevAlarmRef = useRef(false);

  const clearHistory = useCallback(() => {
    buffersRef.current = {};
    lastTRef.current = null;
    prevAlarmRef.current = false;
    setStats({ anomalies: 0, flaggedCylinders: [] });
    setHistVersion((v) => v + 1);
  }, []);

  const backfill = useCallback(async () => {
    try {
      const res = await fetch("/api/history");
      if (!res.ok) return;
      const h = (await res.json()) as HistoryResponse;
      const bufs: TrendBuffers = {};
      h.t_s.forEach((t, i) => {
        for (const [cyl, ch] of Object.entries(h.cylinders)) {
          if (ch.EGT_K[i] != null) {
            appendPoint(bufs, `EGT_${cyl}`, {
              t,
              obs: ch.EGT_K[i],
              pred: ch.EGT_pred_K[i],
              sigma: ch.sigma_EGT_K[i],
            });
          }
          if (ch.CHT_K[i] != null) {
            appendPoint(bufs, `CHT_${cyl}`, {
              t,
              obs: ch.CHT_K[i],
              pred: ch.CHT_pred_K[i],
              sigma: ch.sigma_CHT_K[i],
            });
          }
        }
      });
      buffersRef.current = bufs;
      setHistVersion((v) => v + 1);
    } catch {
      // No backend yet; the chart simply starts empty.
    }
  }, []);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let stopped = false;

    const connect = () => {
      setConn("connecting");
      socket = new WebSocket(wsUrl());
      socket.onopen = () => setConn("live");
      socket.onclose = () => {
        setConn("closed");
        if (!stopped) retry = window.setTimeout(connect, 1000);
      };
      socket.onerror = () => socket?.close();
      socket.onmessage = (ev) => {
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data as string) as ServerMessage;
        } catch {
          return;
        }
        if (msg.type === "hello") {
          setScenarios(msg.scenarios);
          clearHistory();
          void backfill();
          return;
        }
        if (msg.type !== "state") return;
        const s = msg.state;

        // A clock that moves backwards means the mission was reset or a new
        // one started: drop everything accumulated from the previous run.
        if (lastTRef.current !== null && s.t_s < lastTRef.current - 0.5) {
          buffersRef.current = {};
          prevAlarmRef.current = false;
          setStats({ anomalies: 0, flaggedCylinders: [] });
        }
        lastTRef.current = s.t_s;

        for (const c of s.cylinders) {
          appendPoint(buffersRef.current, `EGT_${c.n}`, {
            t: s.t_s,
            obs: c.EGT_K,
            pred: c.EGT_pred_K,
            sigma: c.sigma_EGT_K,
          });
          appendPoint(buffersRef.current, `CHT_${c.n}`, {
            t: s.t_s,
            obs: c.CHT_K,
            pred: c.CHT_pred_K,
            sigma: c.sigma_CHT_K,
          });
        }

        const rising = s.alarm.active && !prevAlarmRef.current;
        prevAlarmRef.current = s.alarm.active;
        if (rising) {
          setStats((st) => ({
            anomalies: st.anomalies + 1,
            flaggedCylinders:
              s.alarm.cylinder !== null &&
              !st.flaggedCylinders.includes(s.alarm.cylinder)
                ? [...st.flaggedCylinders, s.alarm.cylinder]
                : st.flaggedCylinders,
          }));
        }

        setTwin(s);
        setHistVersion((v) => v + 1);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (retry !== undefined) window.clearTimeout(retry);
      socket?.close();
    };
  }, [backfill, clearHistory]);

  const control = useCallback(async (body: Record<string, unknown>) => {
    try {
      const res = await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

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
  };
}
