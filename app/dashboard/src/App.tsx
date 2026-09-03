import { useEffect, useRef, useState } from "react";
import { useTwin } from "./lib/useTwin";
import { useCanned } from "./lib/useCanned";
import { useGuidedDemo } from "./lib/useGuidedDemo";
import { focusCylinder } from "./lib/demoScript";
import { Header } from "./components/Header";
import { ControlsBar } from "./components/ControlsBar";
import { EngineHealth } from "./components/EngineHealth";
import { FaultAlert } from "./components/FaultAlert";
import { MaintenanceAdvisory } from "./components/MaintenanceAdvisory";
import { HealthTrend, type TrendChannel } from "./components/HealthTrend";
import { MissionReport } from "./components/MissionReport";
import { DemoBar } from "./components/DemoBar";
import { WelcomeOverlay } from "./components/WelcomeOverlay";

export default function App() {
  const live = useTwin();
  // No backend on the static hosted build: after a few seconds of failed
  // websocket retries, fall back to replaying precomputed twin states. The
  // conn state oscillates (closed -> connecting -> closed) on each retry,
  // so the timer is only cancelled by a genuine "live" connection, never
  // by a state change.
  const [cannedActive, setCannedActive] = useState(false);
  const fallbackTimer = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (cannedActive) return;
    if (live.conn === "live") {
      window.clearTimeout(fallbackTimer.current);
      fallbackTimer.current = undefined;
      return;
    }
    if (fallbackTimer.current === undefined) {
      fallbackTimer.current = window.setTimeout(
        () => setCannedActive(true),
        5000,
      );
    }
  }, [live.conn, cannedActive]);
  const canned = useCanned(cannedActive);

  const data = cannedActive ? canned : live;
  const mode = cannedActive ? ("canned" as const) : ("live" as const);
  const {
    conn,
    scenarios,
    twin,
    paused,
    setPaused,
    stats,
    history,
    histVersion,
    clearHistory,
    control,
  } = data;

  const [channel, setChannelState] = useState<TrendChannel>(() => {
    // Deep-link: the trend channel lives in the URL so a link captures
    // exactly what the viewer was watching (Web Interface Guidelines:
    // stateful UI belongs in query params).
    const p = new URLSearchParams(window.location.search);
    const kind = p.get("kind") === "CHT" ? "CHT" : "EGT";
    const cyl = [1, 2, 3, 4].includes(Number(p.get("cyl")))
      ? Number(p.get("cyl"))
      : 1;
    return { kind, cyl };
  });
  const setChannel = (ch: TrendChannel) => {
    setChannelState(ch);
    const p = new URLSearchParams(window.location.search);
    p.set("kind", ch.kind);
    p.set("cyl", String(ch.cyl));
    window.history.replaceState(null, "", `?${p.toString()}`);
  };
  const [welcomed, setWelcomed] = useState(false);

  const guidedApi = {
    mode,
    twin,
    control,
    seek: cannedActive ? canned.seek : undefined,
    setSpeed: cannedActive ? canned.setSpeed : undefined,
    phases: cannedActive ? canned.phases : undefined,
    setPaused,
  };
  const guided = useGuidedDemo(guidedApi);

  // Guide the viewer's eye: during the guided flight the trend chart
  // follows the cylinder the current phase is about; outside it, any
  // active alarm pulls the chart to the flagged cylinder.
  useEffect(() => {
    if (!guided.guided) return;
    const focus = focusCylinder(guided.phaseIdx);
    if (focus !== null && focus !== channel.cyl) {
      setChannel({ kind: channel.kind, cyl: focus });
      return;
    }
    if (focus === null && twin?.alarm.cylinder && twin.alarm.cylinder !== channel.cyl) {
      setChannel({ kind: channel.kind, cyl: twin.alarm.cylinder });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guided.guided, guided.phaseIdx, twin?.alarm.cylinder]);

  const startGuided = async () => {
    setWelcomed(true);
    setChannel({ kind: "EGT", cyl: 1 });
    await guided.start();
  };

  // Deep links: ?guided=1 starts the guided flight immediately (skips the
  // welcome overlay), &t=300 seeks to a mission time. Used for sharing a
  // link to a specific moment and for screenshot verification.
  useEffect(() => {
    if (welcomed || conn !== "live") return;
    const params = new URLSearchParams(window.location.search);
    if (params.get("guided") !== "1") return;
    void startGuided().then(() => {
      const t = Number(params.get("t") ?? "0");
      if (t > 0) guided.seekTo(t);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conn, welcomed]);

  return (
    <div className="app">
      {!welcomed && conn === "live" && (
        <WelcomeOverlay
          mode={mode}
          onGuided={startGuided}
          onExplore={() => setWelcomed(true)}
        />
      )}
      <Header twin={twin} conn={conn} mode={mode} />
      {guided.guided ? (
        <DemoBar
          phases={guided.phases}
          phaseIdx={guided.phaseIdx}
          t={twin?.t_s ?? 0}
          paused={paused}
          speed={guided.speed}
          canSeek={guided.canSeek}
          onPlayPause={() => setPaused(!paused)}
          onSeek={guided.seekTo}
          onSpeed={guided.setSpeed}
          onExit={guided.exit}
        />
      ) : (
        <ControlsBar
          scenarios={scenarios}
          paused={paused}
          control={control}
          setPaused={setPaused}
          clearHistory={clearHistory}
        />
      )}
      <main className="grid">
        <EngineHealth twin={twin} selected={channel} onSelect={setChannel} />
        <FaultAlert twin={twin} />
        <MaintenanceAdvisory twin={twin} />
        <div className="col-stack">
          <HealthTrend
            history={history}
            histVersion={histVersion}
            channel={channel}
            onChannel={setChannel}
            phases={guided.guided ? guided.phases : undefined}
          />
          <MissionReport twin={twin} stats={stats} />
        </div>
      </main>
    </div>
  );
}
