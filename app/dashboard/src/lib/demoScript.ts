/**
 * The guided flight's narrative, shared by the DemoBar and the fallback
 * paths. The canned full_mission.json carries the same table (exported from
 * mission.py); this local copy covers the live-server mode, where the
 * backend streams states without phase metadata.
 */

import type { DemoPhase } from "./useCanned";

export const FULL_MISSION_PHASES: DemoPhase[] = [
  {
    start_s: 0,
    name: "Takeoff and climb",
    caption:
      "The twin is predicting every gauge. The shaded band is its own " +
      "computed uncertainty, not a tuned threshold.",
  },
  {
    start_s: 90,
    name: "Cruise, and a small fault",
    caption:
      "An 8 percent injector restriction begins on cylinder 3 at t=150. " +
      "Watch the system stay quiet: the deviation sits inside its own " +
      "uncertainty band.",
  },
  {
    start_s: 210,
    name: "Same fault, leaner mixture",
    caption:
      "The mixture leans and the cylinder slides past the heat-release " +
      "peak. The residual steps outside the band and the alarm fires; the " +
      "diagnosis settles on the injector as the slow head-temperature " +
      "channel confirms it.",
  },
  {
    start_s: 330,
    name: "Escalation to full blockage",
    caption:
      "The same parameter worsens. The exhaust temperature sign flips on " +
      "its own and the diagnosis becomes a dead cylinder. One knob, two " +
      "signatures.",
  },
  {
    start_s: 450,
    name: "Detonation",
    caption:
      "Cylinder 3 recovers as its fault clears. Then a different fault on " +
      "cylinder 1: head temperature climbs while exhaust temperature " +
      "falls. Opposite directions from one cause; no single channel sees it.",
  },
  {
    start_s: 620,
    name: "Recovery and descent",
    caption:
      "Faults cleared. Watch the flagged channels settle back into the " +
      "band as the engine cools, then the twin returns to nominal for the " +
      "descent home.",
  },
];

export const FULL_MISSION_END_S = 720;

/** The phase a mission time belongs to (last phase whose start <= t). */
export function phaseAt(phases: DemoPhase[], t: number): number {
  let idx = 0;
  for (let i = 0; i < phases.length; i++) {
    if (phases[i].start_s <= t) idx = i;
  }
  return idx;
}

/** Cylinder the viewer should watch during a phase, if one matters. */
export function focusCylinder(phaseIdx: number): number | null {
  if (phaseIdx >= 2 && phaseIdx <= 3) return 3;
  if (phaseIdx === 4) return 1;
  return null;
}
