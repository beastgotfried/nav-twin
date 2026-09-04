import { useEffect, useState } from "react";
import type { TwinState } from "../lib/protocol";
import { engineTint, stateName, heatFraction } from "../lib/thermalTint";

/**
 * The hero: the airframe as a neutral outline with the engine glowing inside
 * it, tinted by the live mean cylinder head temperature.
 *
 * This is mode A of 03-Design/layout-study/engine-state.html, graduated into
 * the dashboard exactly as that study's README says it should be ("when this
 * graduates into the dashboard they move to 10-Twin/dashboard/public/").
 * Nothing about the mechanism is new here; what is new is that the number
 * driving it arrives over the websocket instead of out of a scrub bar.
 *
 * Why the airframe stays neutral and only the engine takes colour: the engine
 * is the thing we monitor. Colouring the whole airframe would say the
 * aircraft is hot, and this project models an engine, not an aircraft
 * (CLAUDE.md section 8). The outline is context, the glow is state.
 *
 * Why it sits behind the panels rather than in one: it answers "what is the
 * engine doing", which is a different and slower question than "is anything
 * wrong". 03-Design/operator-interface.md rule 2 spends motion only on
 * things that are wrong, and this earns its place by staying peripheral: a
 * tint that moves over minutes, never a thing that grabs.
 *
 * It carries no numbers of its own. Everything quantitative is in the panels
 * in front of it, and a background that competed with them would be theatre.
 */

/** Milliseconds between eases. Matches the study's 60 ms tick. */
const EASE_MS = 60;
/** Fraction of the remaining gap closed per tick. */
const EASE_ALPHA = 0.25;
/** Below this we treat the two as settled and stop scheduling work. */
const SETTLED_K = 0.05;

function meanCHT_C(twin: TwinState | null): number | null {
  if (!twin || twin.cylinders.length === 0) return null;
  const sum = twin.cylinders.reduce((a, c) => a + c.CHT_K, 0);
  return sum / twin.cylinders.length - 273.15;
}

export function EngineHero({ twin }: { twin: TwinState | null }) {
  const target = meanCHT_C(twin);

  // The shown temperature chases the target rather than cutting to it. A
  // cylinder head has a thermal time constant of tens of seconds, so easing
  // is the more truthful rendering, not merely the prettier one.
  const [shown, setShown] = useState<number | null>(null);

  useEffect(() => {
    if (target === null) {
      setShown(null);
      return;
    }
    // First frame of a mission: adopt the value instead of easing up from
    // cold, which would show a warm-up that did not happen.
    setShown((s) => (s === null ? target : s));

    const tick = () => {
      setShown((s) => {
        if (s === null) return target;
        if (Math.abs(target - s) < SETTLED_K) return target;
        return s + (target - s) * EASE_ALPHA;
      });
    };
    const id = window.setInterval(tick, EASE_MS);
    return () => window.clearInterval(id);
  }, [target]);

  const alarm = twin?.alarm ?? null;
  const tint = engineTint(shown, alarm);
  const label = alarm?.active ? alarm.level : stateName(shown);
  // The bloom follows temperature only. An alarm already has the whole
  // airframe's colour; adding a glow to it would be shouting the same thing
  // twice, and the glow is the one part of this that is genuinely decorative
  // if it is not tied to heat.
  const heat = heatFraction(shown);

  return (
    <div
      className="hero"
      aria-hidden="true"
      style={
        {
          "--tint": tint,
          // The bloom only exists past "running", and only ever gently. Alpha
          // is appended as hex so the colour and its glow can never drift
          // apart.
          "--glow":
            heat > 0.02
              ? `${tint}${Math.round(heat * 90)
                  .toString(16)
                  .padStart(2, "0")}`
              : "transparent",
        } as React.CSSProperties
      }
      data-state={label}
    >
      {/* A darkened pool under the aircraft. The contour field runs edge to
          edge, and line art on top of line work is hard to read; sinking the
          ground behind the airframe separates the two without hiding either. */}
      <div className="hero-veil" />
      <div className="hero-layer hero-engine" />
      <div className="hero-layer hero-outline" />
    </div>
  );
}
