/**
 * The guided flight's narrative, shared by the DemoBar and the fallback
 * paths. The canned full_mission.json carries the same table (exported from
 * mission.py); this local copy covers the live-server mode, where the
 * backend streams states without phase metadata. It mirrors
 * simulator/mission.py FULL_MISSION_PHASES exactly and contains no data,
 * only phase names and captions; every number on screen still comes from
 * the twin.
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
    start_s: 630,
    name: "A lying sensor",
    caption:
      "Cylinder 2's exhaust probe starts lying at t=630, instantly, with " +
      "no engine change. Physics says a real combustion change must reach " +
      "the head within about a minute; when it never does, the system " +
      "calls it what it is: sensor drift, not an engine fault.",
  },
  {
    start_s: 750,
    name: "A dead cylinder",
    caption:
      "Ignition fails on cylinder 2. Watch it run COLD, not hot: no " +
      "combustion means no heat. Both its temperatures collapse while " +
      "the other three stay in band.",
  },
  {
    start_s: 870,
    name: "Bearings wearing out",
    caption:
      "Oil temperature rising and pressure falling together, gradually. " +
      "The particle filter tracks the wear and projects a bounded time " +
      "to the failure threshold. The bounds are the honest part: a " +
      "spread, not a date.",
  },
  {
    start_s: 990,
    name: "Cooling degrades slowly",
    caption:
      "A baffle crack on cylinder 4 worsens over a minute and a half. " +
      "Exhaust stays flat, head temperature climbs, and the RUL tracker " +
      "follows the severity rising with its uncertainty bounds.",
  },
  {
    start_s: 1110,
    name: "Climb, and a fading turbo",
    caption:
      "The turbo loses efficiency as we climb. Every temperature channel " +
      "stays in band, because physics says they should. But the twin " +
      "watches more than temperatures: the gap between commanded and " +
      "achieved manifold pressure grows with altitude, and the diagnosis " +
      "names the fading turbo while the gauges read normal.",
  },
  {
    start_s: 1260,
    name: "Recovery and descent",
    caption:
      "Faults cleared. Watch the flagged channels settle back into the " +
      "band as the engine cools, then the twin returns to nominal for " +
      "the descent home.",
  },
];

export const FULL_MISSION_END_S = 1390;

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
  if (phaseIdx >= 1 && phaseIdx <= 3) return 3; // injector phases
  if (phaseIdx === 4) return 1; // detonation
  if (phaseIdx === 5 || phaseIdx === 6) return 2; // drift, misfire
  if (phaseIdx === 8) return 4; // cooling degradation
  return null; // oil (bearing) and MAP (turbo) phases: no single cylinder
}
