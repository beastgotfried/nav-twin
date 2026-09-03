/**
 * The guided-flight bar: transport controls, a segmented phase timeline
 * that doubles as a seek bar, and the current phase caption. This is the
 * element that turns the dashboard from an instrument panel into a story.
 * Rendered only while the guided demo is active.
 */

import type { DemoPhase } from "../lib/useCanned";
import { fmtClock } from "../lib/format";
import { FULL_MISSION_END_S } from "../lib/demoScript";

export function DemoBar({
  phases,
  phaseIdx,
  t,
  paused,
  speed,
  canSeek,
  onPlayPause,
  onSeek,
  onSpeed,
  onExit,
}: {
  phases: DemoPhase[];
  phaseIdx: number;
  t: number;
  paused: boolean;
  speed: number;
  canSeek: boolean;
  onPlayPause: () => void;
  onSeek: (t_s: number) => void;
  onSpeed: (mult: number) => void;
  onExit: () => void;
}) {
  const phase = phaseIdx >= 0 ? phases[phaseIdx] : null;
  const end = phases.length > 0 ? phases[phases.length - 1].start_s : FULL_MISSION_END_S;
  const total = Math.max(end, FULL_MISSION_END_S);

  return (
    <section className="card demo-bar">
      <div className="demo-row">
        <div className="demo-transport">
          <button
            type="button"
            className="demo-play"
            onClick={onPlayPause}
            aria-label={paused ? "Resume the guided flight" : "Pause the guided flight"}
            title={paused ? "Resume the flight" : "Pause the flight"}
          >
            {paused ? "▶" : "❚❚"}
          </button>
          <span className="demo-clock mono">{fmtClock(t)}</span>
          <span className="seg">
            {[1, 4, 16].map((m) => (
              <button
                key={m}
                type="button"
                className={`seg-btn mono ${speed === m ? "seg-on" : ""}`}
                onClick={() => onSpeed(m)}
              >
                {m}x
              </button>
            ))}
          </span>
        </div>

        <div className="demo-timeline" role="list">
          {phases.map((p, i) => {
            const next = i + 1 < phases.length ? phases[i + 1].start_s : total;
            const width = ((next - p.start_s) / total) * 100;
            const done = t >= next;
            const current = i === phaseIdx;
            const fill = done ? 100 : current ? ((t - p.start_s) / (next - p.start_s)) * 100 : 0;
            return (
              <button
                key={p.start_s}
                type="button"
                role="listitem"
                className={`demo-seg ${current ? "demo-seg-current" : ""} ${done ? "demo-seg-done" : ""}`}
                style={{ width: `${width}%` }}
                onClick={() => canSeek && onSeek(p.start_s)}
                disabled={!canSeek}
                title={`${p.name} (from ${fmtClock(p.start_s)})`}
              >
                <span className="demo-seg-fill" style={{ width: `${fill}%` }} />
                <span className="demo-seg-label">{i + 1}</span>
              </button>
            );
          })}
        </div>

        <button type="button" className="demo-exit" onClick={onExit}>
          EXIT FLIGHT
        </button>
      </div>

      {phase && (
        <div className="demo-caption">
          <span className="demo-caption-name">
            Phase {phaseIdx + 1}: {phase.name}
          </span>
          <span className="demo-caption-text">{phase.caption}</span>
        </div>
      )}
    </section>
  );
}
