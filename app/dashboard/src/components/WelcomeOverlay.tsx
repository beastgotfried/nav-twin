/**
 * First-visit welcome overlay. One screen that says what this is and gives
 * the visitor exactly two ways in: watch the guided flight (the story), or
 * explore the dashboard directly. Every claim on it is verified behaviour
 * of the pipeline; the fine print says the replay is precomputed by the
 * real twin, matching the header's REPLAY badge.
 *
 * Built on the native <dialog> in modal mode, which is where the focus trap,
 * the Escape key, the inert background and the modal semantics come from.
 * The hand-rolled div version had none of those: keyboard users tabbed
 * straight through to the dashboard behind it.
 */

import { useEffect, useRef } from "react";

export function WelcomeOverlay({
  mode,
  onGuided,
  onExplore,
}: {
  mode: "live" | "canned";
  onGuided: () => void;
  onExplore: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el && !el.open) el.showModal();
  }, []);

  return (
    <dialog
      ref={ref}
      className="welcome-backdrop"
      aria-labelledby="welcome-title"
      // Escape would close the dialog without choosing a way in, leaving a
      // dashboard nobody asked for. Dismissing it IS "explore on my own".
      onCancel={(e) => {
        e.preventDefault();
        onExplore();
      }}
    >
      <div className="welcome-card">
        <p className="welcome-kicker mono">AERO PISTON ENGINE DIGITAL TWIN</p>
        <h1 className="welcome-title" id="welcome-title">SITAARA</h1>
        <p className="welcome-lede">
          A physics-based digital twin flying a MALE UAV engine mission. It
          predicts what every sensor should read, measures the gap against
          what they actually read, and explains, in ranked evidence, what is
          going wrong before any redline is crossed.
        </p>
        <div className="welcome-actions">
          <button type="button" className="welcome-primary" onClick={onGuided}>
            WATCH THE GUIDED FLIGHT
            <span className="welcome-sub">
              takeoff to landing, faults included, about 3 minutes at 4x
            </span>
          </button>
          <button type="button" className="welcome-secondary" onClick={onExplore}>
            EXPLORE ON MY OWN
            <span className="welcome-sub">
              scenarios and fault injection, hands on
            </span>
          </button>
        </div>
        <p className="welcome-fine">
          {mode === "canned"
            ? "This site replays telemetry precomputed by the real twin, the same numbers the live system produces."
            : "Live: the twin is running on this machine, computing predictions as you watch."}
        </p>
      </div>
    </dialog>
  );
}
