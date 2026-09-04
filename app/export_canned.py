"""export_canned.py -- precompute demo telemetry for the static hosted build.

Vercel cannot run the live twin (serverless, no websockets, no long-lived
mission loop), so the hosted dashboard replays REAL twin states that this
script computes ahead of time. Every number in the canned files comes out of
the same mission.py -> twin pipeline as the live system; nothing is drawn,
mocked or hand-edited. The hosted UI labels itself as a replay.

Output: dashboard/public/canned/*.json plus an index.json manifest the
frontend uses to map (scenario, injected fault) -> replay file. These files
are build artefacts: regenerate, do not edit.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulator"))

from mission import run_mission, FaultEvent
from physics.faults import FaultSpec
from twin import Twin

OUT = Path(__file__).resolve().parent / "dashboard" / "public" / "canned"


def compact(state):
    """Round floats to shrink the JSON; keep the exact state shape."""
    def r(x):
        return round(x, 2) if isinstance(x, float) else x
    cyls = [{k: r(v) if isinstance(v, float) else v for k, v in c.items()}
            for c in state["cylinders"]]
    return {
        "t_s": r(state["t_s"]),
        "scenario": state["scenario"],
        "inputs": {k: r(v) for k, v in state["inputs"].items()},
        "cylinders": cyls,
        "oil": {k: r(v) for k, v in state["oil"].items()},
        "alarm": state["alarm"],
        "diagnosis": state["diagnosis"],
        "calibrated": state["calibrated"],
        "ml": state.get("ml"),
    }


def build(name, profile, faults=(), seconds=150, seed=5):
    events = [FaultEvent(60.0, f) for f in faults]
    tw = Twin()
    frames = [compact(tw.step(fr))
              for fr in run_mission(profile, fault_events=events, seed=seed)
              if fr["t_s"] <= seconds]
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"name": name, "profile": profile,
                                "frames": frames}, separators=(",", ":")))
    print(f"  {name}: {len(frames)} frames, {path.stat().st_size//1024} KiB")
    return {"name": name, "scenario": profile,
            "faults": ([{"kind": f.kind, "cylinder": f.cylinder,
                         "severity": f.severity,
                         "sensor_channel": f.sensor_channel,
                         "bias_K": f.bias_K} for f in faults] or [])}


def _build_full_mission():
    """full_mission uses its shared schedule (mission.full_mission_faults),
    not the per-entry fault list, so it gets its own builder."""
    from mission import run_mission, full_mission_faults, FULL_MISSION_PHASES
    tw = Twin()
    frames = [compact(tw.step(fr))
              for fr in run_mission("full_mission",
                                    fault_events=full_mission_faults(),
                                    seed=5)]
    path = OUT / "full_mission.json"
    path.write_text(json.dumps({"name": "full_mission",
                                "profile": "full_mission",
                                "phases": FULL_MISSION_PHASES,
                                "frames": frames}, separators=(",", ":")))
    print(f"  full_mission: {len(frames)} frames, "
          f"{path.stat().st_size//1024} KiB")
    return {"name": "full_mission", "scenario": "full_mission",
            "faults": [], "guided": True}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    print("building canned replay data (real twin, this takes a few minutes)")
    entries.append(build("endurance", "endurance", seconds=300, seed=3))
    entries.append(build("rapid_throttle", "rapid_throttle", seconds=240, seed=7))
    entries.append(build("high_altitude", "high_altitude", seconds=300, seed=3))
    entries.append(build("hot_weather", "hot_weather", seconds=240, seed=3))
    entries.append(build("cruise_rich", "cruise_rich", seconds=150))
    entries.append(build("cruise_lean", "cruise_lean", seconds=150))
    entries.append(build("cruise_rich_injector", "cruise_rich",
                         [FaultSpec("injector_restriction", cylinder=3,
                                    severity=0.10)]))
    entries.append(build("cruise_lean_injector", "cruise_lean",
                         [FaultSpec("injector_restriction", cylinder=3,
                                    severity=0.10)]))
    entries.append(build("cruise_rich_blockage", "cruise_rich",
                         [FaultSpec("injector_restriction", cylinder=3,
                                    severity=1.0)]))
    entries.append(build("detonation", "endurance",
                         [FaultSpec("detonation", cylinder=1, severity=1.0)],
                         seconds=180))
    entries.append(build("misfire", "endurance",
                         [FaultSpec("misfire", cylinder=1, severity=1.0)]))
    entries.append(build("sensor_drift", "endurance",
                         [FaultSpec("sensor_drift", cylinder=2,
                                    sensor_channel="EGT_K", bias_K=80.0)],
                         seconds=180))
    # The guided flight: one stitched takeoff-to-landing mission with its
    # own shared fault schedule and narrative phases (mission.py).
    entries.append(_build_full_mission())
    (OUT / "index.json").write_text(json.dumps({"entries": entries},
                                               separators=(",", ":")))
    print(f"manifest: {len(entries)} entries -> {OUT}")
