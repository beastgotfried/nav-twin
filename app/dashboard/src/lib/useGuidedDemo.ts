/**
 * Guided demo orchestration. Holds the "watching the guided flight" state
 * and translates it into the same control calls the manual controls use,
 * so the dashboard renders one data path whether the viewer is driving or
 * the script is. Speed and seek are canned-mode features (client-side
 * replay); in live mode the server paces the mission and seek is hidden.
 */

import { useCallback, useMemo, useState } from "react";
import type { TwinState } from "./protocol";
import { FULL_MISSION_PHASES, phaseAt } from "./demoScript";
import type { DemoPhase } from "./useCanned";

export interface GuidedApi {
  mode: "live" | "canned";
  twin: TwinState | null;
  control: (body: Record<string, unknown>) => Promise<boolean>;
  seek?: (t: number) => void;
  setSpeed?: (mult: number) => void;
  phases?: DemoPhase[];
  setPaused?: (p: boolean) => void;
}

export function useGuidedDemo(api: GuidedApi) {
  const [guided, setGuided] = useState(false);
  const [speed, setSpeedState] = useState(4);

  const phases = useMemo(
    () => (api.phases && api.phases.length > 0 ? api.phases : FULL_MISSION_PHASES),
    [api.phases],
  );

  const t = api.twin?.t_s ?? 0;
  const phaseIdx = guided ? phaseAt(phases, t) : -1;

  const start = useCallback(async () => {
    setGuided(true);
    api.setSpeed?.(speed);
    await api.control({ action: "start", scenario: "full_mission", speed });
    api.setPaused?.(false);
  }, [api, speed]);

  const exit = useCallback(async () => {
    setGuided(false);
    await api.control({ action: "start", scenario: "endurance" });
  }, [api]);

  const seekTo = useCallback(
    (t_s: number) => {
      api.seek?.(t_s);
    },
    [api],
  );

  const setSpeed = useCallback(
    (mult: number) => {
      setSpeedState(mult);
      api.setSpeed?.(mult);
    },
    [api],
  );

  return {
    guided,
    phases,
    phaseIdx,
    speed,
    start,
    exit,
    seekTo,
    setSpeed,
    canSeek: api.mode === "canned",
  };
}
