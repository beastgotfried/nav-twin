"""twin/replay.py -- mission replay (problem statement Section E).

Re-runs a logged mission through the twin after the fact, cylinder by
cylinder, and returns the states plus a compact flags timeline: when an
alarm went active, on which channel, and what the top diagnosis was. This
is the same code path as the live stream; replay is just a different frame
source, which is what architecture.md Layer 5 means by "the same engine run
without a live comparison target", inverted: the same twin, fed from a log.
"""

from . import Twin


def replay_frames(frames, twin: Twin = None, calibrate_s: float = 30.0):
    """frames: iterable of telemetry frames (mission.frames_from_csv output).
    Returns {"states": [...], "flags": [...]}. The twin calibrates its
    baseline on the first calibrate_s seconds, which must be healthy
    (Handbook 6.6: fit delta on known-healthy data only, then freeze)."""
    twin = twin or Twin(calibrate_s=calibrate_s)
    states, flags = [], []
    alarm_was = False
    for fr in frames:
        st = twin.step(fr)
        states.append(st)
        alarm_on = st["alarm"]["active"]
        if alarm_on and not alarm_was:
            top = st["diagnosis"][0]["label"] if st["diagnosis"] else \
                "unexplained residual excursion"
            flags.append({
                "t_s": st["t_s"], "event": "alarm_on",
                "channel": st["alarm"]["channel"],
                "cylinder": st["alarm"]["cylinder"],
                "top_diagnosis": top,
            })
        elif not alarm_on and alarm_was:
            flags.append({"t_s": st["t_s"], "event": "alarm_off"})
        alarm_was = alarm_on
    return {"states": states, "flags": flags}
