/**
 * Drift the contour field at a rate set by engine speed.
 *
 * WHY THIS IS NOT A CSS ANIMATION
 * -------------------------------
 * It was one: a linear `atmo-drift` keyframe of one tile, parked and unparked
 * by a class. That is the right shape for motion at a fixed rate, and the
 * wrong one the moment the rate has to follow a number. Changing
 * `animation-duration` on a running animation recomputes the current position
 * from the elapsed time against the new duration, so the field jumps to a
 * different phase of the loop every time the rate changes. Against a
 * continuously varying RPM that is a lurch several times a second.
 *
 * Accumulating the offset per frame has no such discontinuity: the position
 * is whatever it was plus this frame's travel, so the rate can change however
 * it likes and the field never moves anywhere it was not already going.
 *
 * WHY RPM
 * -------
 * The field is the ground; the ground goes past faster when the engine is
 * working harder. That is the whole mapping, and it means the motion is
 * reporting a measured quantity rather than decorating the page. It is the
 * same discipline the rest of the screen is held to: this project's claim is
 * that everything on screen is computed, so a background that moved at a rate
 * nobody could point at would be exactly the theatre the design study warns
 * about (03-Design/operator-interface.md rule 2).
 *
 * The anchors are measured, not guessed. Across the canned missions N_rpm
 * spans 3791 to 5512 on `full_mission` and sits near 5000 for `endurance`,
 * so the ramp is pinned to that range: idle-ish at the bottom of it, full
 * travel at the top. A mission that starts on the ground and climbs to
 * takeoff power therefore visibly accelerates the field, which is the point.
 *
 * Writes a CSS custom property rather than setting `transform` directly, so
 * the value lands on `.atmosphere` and the `::before` that actually carries
 * the tile picks it up. Nothing here goes through React state: this runs
 * every frame, and a re-render per frame to move a background would cost more
 * than everything else on the page put together.
 */

import { useEffect, useRef } from "react";

/**
 * One tile, in px. MUST match `background-size` on `.atmosphere::before`, or
 * the wrap lands mid-pattern and the field jumps once per cycle.
 */
const TILE_PX = 620;

/** Bottom and top of the measured RPM range across the canned missions. */
const RPM_LO = 3800;
const RPM_HI = 5500;

/**
 * Travel in px/s at each end of that range. Slow enough at the bottom to read
 * as "running" rather than as a screensaver, and still unhurried at the top:
 * at 30 px/s a tile takes about 20 s to pass, which registers peripherally
 * without ever competing with a trace that is actually changing.
 */
const PX_PER_S_LO = 4;
const PX_PER_S_HI = 30;

/**
 * Rate of approach toward the target speed, per second. RPM itself steps
 * around between frames and the raw value would make the field twitch, so the
 * speed is eased. Slower than the CHT easing in EngineHero because this is
 * further from the eye and has less business being precise.
 */
const EASE_PER_S = 2.2;

/** Positive is down the screen, which is the direction the ground travels. */
function pxPerSecFor(rpm: number | null): number {
  if (rpm === null || Number.isNaN(rpm)) return 0;
  const f = (rpm - RPM_LO) / (RPM_HI - RPM_LO);
  const clamped = Math.max(0, Math.min(1, f));
  return PX_PER_S_LO + clamped * (PX_PER_S_HI - PX_PER_S_LO);
}

export function useFieldDrift(rpm: number | null, streaming: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  // Read inside the frame loop rather than captured, so a change to either
  // does not tear down and restart the loop (which would drop the frame
  // clock and stutter).
  const target = useRef(0);
  target.current = streaming ? pxPerSecFor(rpm) : 0;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduced.matches) return;

    let raf = 0;
    let last = 0;
    let y = 0;
    let speed = 0;

    const frame = (now: number) => {
      // First frame has no interval to integrate over. Clamped because a
      // backgrounded tab resumes with a gap of seconds, and integrating that
      // in one step would teleport the field.
      const dt = last === 0 ? 0 : Math.min((now - last) / 1000, 0.1);
      last = now;

      speed += (target.current - speed) * Math.min(1, EASE_PER_S * dt);
      y = (y + speed * dt) % TILE_PX;
      el.style.setProperty("--drift-y", `${y.toFixed(2)}px`);

      raf = window.requestAnimationFrame(frame);
    };

    raf = window.requestAnimationFrame(frame);
    return () => window.cancelAnimationFrame(raf);
  }, []);

  return ref;
}
