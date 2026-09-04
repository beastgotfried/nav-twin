/**
 * Cylinder head temperature to colour.
 *
 * Ported unchanged in substance from 03-Design/layout-study/engine-state.html,
 * which is where the ramp was designed and where the reasoning lives. Two
 * things about it are load bearing and must not be "simplified":
 *
 * 1. The anchors are measured, not chosen. The guided full_mission spans
 *    116.1 to 235.7 degC mean-of-four and 260 degC is the published redline
 *    for this engine class, the same figure fig_residual_band.py uses.
 *
 * 2. The pale stop at 148 degC is a fix for a real defect, not a style
 *    preference. Interpolating saturated blue straight to saturated gold has
 *    to pass through desaturated colour, and the first ramp produced #999ec6
 *    at 140 degC: a muddy lavender that reads as a rendering fault rather
 *    than a temperature. Figma's own Temperature slider runs blue to neutral
 *    to warm, so a pale middle is correct; it just has to be a colour someone
 *    picked rather than one the interpolator invented.
 *
 * Interpolation is in linear light. Mixing sRGB directly pulls a blue to gold
 * ramp through a muddy grey exactly in the middle, and the middle of this
 * ramp is normal cruise, the state an operator sees most.
 */

export interface TintStop {
  /** degrees C */
  C: number;
  hex: string;
  label: string;
}

/**
 * At rest. No mission, no telemetry, nothing to report: the aircraft is drawn
 * in plain ink, the same colour as the strongest text on screen.
 *
 * White is the correct resting state because it is the ABSENCE of the signal
 * rather than a low value of it. Every colour this file produces means "the
 * engine is at some temperature"; white means the question is not currently
 * being asked. The study used a dim grey here, which read as a very cold
 * engine instead of as no engine.
 */
export const OFF_COLOUR = "#e8ebed";

export const STOPS: readonly TintStop[] = [
  { C: 20, hex: "#0b5fd0", label: "cold" },
  { C: 100, hex: "#0099ff", label: "cool" },
  { C: 148, hex: "#cfe0ec", label: "warming" },
  { C: 190, hex: "#d9a441", label: "running" },
  { C: 225, hex: "#dc762d", label: "hot" },
  { C: 260, hex: "#fb2c55", label: "redline" },
];

const hex2rgb = (h: string): number[] =>
  [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

const rgb2hex = (a: number[]): string =>
  "#" +
  a
    .map((v) =>
      Math.round(Math.max(0, Math.min(255, v)))
        .toString(16)
        .padStart(2, "0"),
    )
    .join("");

const toLin = (v: number): number => {
  const u = v / 255;
  return u <= 0.04045 ? u / 12.92 : ((u + 0.055) / 1.055) ** 2.4;
};

const toSrgb = (v: number): number =>
  255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055);

/** Colour for a cylinder head temperature in degrees C. null means off. */
export function tintFor(C: number | null): string {
  if (C === null || Number.isNaN(C)) return OFF_COLOUR;
  if (C <= STOPS[0].C) return STOPS[0].hex;
  if (C >= STOPS[STOPS.length - 1].C) return STOPS[STOPS.length - 1].hex;
  for (let i = 1; i < STOPS.length; i++) {
    if (C <= STOPS[i].C) {
      const a = STOPS[i - 1];
      const b = STOPS[i];
      const t = (C - a.C) / (b.C - a.C);
      const A = hex2rgb(a.hex).map(toLin);
      const B = hex2rgb(b.hex).map(toLin);
      return rgb2hex(A.map((v, k) => toSrgb(v + (B[k] - v) * t)));
    }
  }
  return OFF_COLOUR;
}

/** The word for a temperature, for the readout and the screen reader. */
export function stateName(C: number | null): string {
  if (C === null || Number.isNaN(C)) return "off";
  const s = [...STOPS].reverse().find((s) => C >= s.C);
  return s ? s.label : "cold";
}

/**
 * How hot, 0 to 1, across the running-to-redline span only. Drives the bloom,
 * which must stay absent on a normal engine: a glow that is always present is
 * decoration, one that only arrives past "running" is information.
 */
export function heatFraction(C: number | null): number {
  if (C === null || Number.isNaN(C)) return 0;
  return Math.max(0, Math.min(1, (C - 190) / (260 - 190)));
}

/**
 * The colour the aircraft is actually drawn in, given both what the engine is
 * doing and what the twin thinks of it.
 *
 * Temperature sets the colour; an alarm overrides it. That ordering is the
 * point rather than a shortcut. Temperature is a continuous fact and answers
 * "what is the engine doing"; an alarm is a discrete judgement and answers
 * "is anything wrong". When the second question has an answer it is strictly
 * the more urgent one, so it takes the surface.
 *
 * The alarm colours are the locked accents and mean here exactly what they
 * mean in every other panel and in every matplotlib figure in the repo, so a
 * red aircraft and a red row cannot disagree about severity.
 */
export function engineTint(
  cht_C: number | null,
  alarm: { active: boolean; level: string } | null,
): string {
  if (alarm?.active) {
    if (alarm.level === "warning") return "#fb2c55";
    if (alarm.level === "caution") return "#dc762d";
  }
  return tintFor(cht_C);
}
