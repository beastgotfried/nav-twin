"""demo.py -- the scripted demo runner for 04-Deliverables/demo-script.md.

Drives the live dashboard through the demo acts via the server control API,
printing the narration cues in the terminal while the presenter talks. The
dashboard (http://127.0.0.1:8734/) shows everything this script does.

Usage:
  1. ../.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8734
  2. open http://127.0.0.1:8734/ in the demo browser
  3. ../.venv/bin/python demo.py            # full run, real-time pacing
     ../.venv/bin/python demo.py --speed 8  # rehearsal pacing

Covers Acts 1-7 and 9 of the demo script. Act 8 (RUL what-if) is out of MVP
scope, see the MVP plan's stretch list.

The act 3/4 pair runs at the two cruise points verify_twin.py was built
around (cruise_rich phi=1.15, cruise_lean phi=0.85): the operating points
where the model itself puts an 8% leaning inside the band on one side and
outside it on the other. We did not tune the demo until it worked; we
measured where the physics makes it true.
"""

import argparse
import sys
import time
import urllib.request
import json

BASE = "http://127.0.0.1:8734"


def ctl(body: dict) -> dict:
    req = urllib.request.Request(
        BASE + "/api/control", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def say(act: str, text: str):
    print(f"\n{'=' * 74}\n{act}\n{'-' * 74}\n{text}\n", flush=True)


def wait(sim_seconds: float, speed: float, note: str = ""):
    """Wall-clock wait for a number of simulated seconds at the run speed."""
    if note:
        print(f"   ... {note}", flush=True)
    time.sleep(sim_seconds / speed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=8.0,
                    help="simulated seconds per wall second (default 8)")
    args = ap.parse_args()
    speed = args.speed
    cal = 35.0   # calibration window (twin calibrates at 30 sim-s) + margin

    say("SETUP", "Server on :8734, dashboard open, mission about to start.\n"
        "This script prints what to say; the dashboard shows what it says.")

    # Act 1: normal operation
    ctl({"action": "start", "scenario": "endurance", "speed": speed})
    say("ACT 1 - normal operation (30 s)",
        "All gauges green. Point at the HEALTH TREND chart: the blue line is\n"
        "the twin predicting, the shaded band is its computed uncertainty,\n"
        "the red line is the observed sensor. 'The twin is predicting, not\n"
        "just displaying.' Wait for the baseline to freeze (~30 s sim).")
    wait(cal, speed, "baseline calibration in progress (healthy data only)")

    # Act 2: the band breathes
    say("ACT 2 - the band breathes (30 s)",
        "Switching to rapid throttle transitions. Watch the band WIDEN and\n"
        "NARROW as the engine moves through the envelope. 'The system knows\n"
        "where it is confident. That is not a threshold we tuned, it is\n"
        "computed from the physics.'")
    ctl({"action": "start", "scenario": "rapid_throttle", "speed": speed})
    wait(cal + 30.0, speed, "watch the band on the throttle steps")

    # Act 3: the fault that is correctly ignored
    say("ACT 3 - the fault that is correctly ignored (20 s)",
        "Cruise at phi=1.15, deep rich. Injecting an 8% injector restriction\n"
        "on cylinder 3. The deviation is real, and it sits INSIDE the band,\n"
        "so the system stays quiet. 'A fixed threshold would either miss\n"
        "this or false-alarm constantly. We know this deviation is within\n"
        "what our own model uncertainty can explain.'")
    ctl({"action": "start", "scenario": "cruise_rich", "speed": speed})
    wait(cal, speed, "recalibrating at the rich cruise point")
    ctl({"action": "inject",
         "fault": {"kind": "injector_restriction", "cylinder": 3,
                   "severity": 0.10}})
    wait(25.0, speed, "no alarm: the band is doing its job")

    # Act 4: the same fault fires
    say("ACT 4 - the same fault fires (20 s)",
        "Same fault, same severity, lean cruise at phi=0.85. The leaning\n"
        "drives cylinder 3 past the mixture kink onto the steep flank, the\n"
        "residual leaves the band, and the alarm fires. 'Calibrated\n"
        "uncertainty, adaptive thresholding, honest limits, in twenty\n"
        "seconds. No fixed-threshold system can do this pair.'")
    ctl({"action": "start", "scenario": "cruise_lean", "speed": speed})
    wait(cal, speed, "recalibrating at the lean cruise point")
    ctl({"action": "inject",
         "fault": {"kind": "injector_restriction", "cylinder": 3,
                   "severity": 0.10}})
    wait(20.0, speed, "alarm fires; FAULT ALERT ranks injector restriction")

    # Act 5: differential diagnosis
    say("ACT 5 - differential diagnosis (30 s)",
        "Open the FAULT ALERT card: ranked candidates with evidence, z-scores\n"
        "per channel, not an anomaly score. 'A diagnosis a maintenance\n"
        "engineer can act on.'")
    wait(15.0, speed, "let the ranking settle")

    # Act 6: severity inversion
    say("ACT 6 - severity inversion (20 s)",
        "Same cylinder, same parameter, severity to full blockage. The EGT\n"
        "signature FLIPS from rising to falling and the diagnosis updates to\n"
        "a dead cylinder. 'We never coded two faults. One parameter, one\n"
        "severity knob. The physics inverted the signature by itself because\n"
        "the mixture crossed the heat-release peak.'")
    ctl({"action": "clear_faults"})
    ctl({"action": "inject",
         "fault": {"kind": "injector_restriction", "cylinder": 3,
                   "severity": 1.0}})
    wait(25.0, speed, "watch EGT sign flip and diagnosis follow")

    # Act 7: detonation
    say("ACT 7 - detonation (15 s)",
        "Detonation on cylinder 1: head temperature climbs while exhaust\n"
        "temperature FALLS, opposite directions from one cause. 'No single\n"
        "channel identifies this. The joint pattern does.'")
    ctl({"action": "clear_faults"})
    ctl({"action": "inject",
         "fault": {"kind": "detonation", "cylinder": 1, "severity": 1.0}})
    wait(25.0, speed, "CHT up, EGT down, detonation ranked first")

    # Act 9: replay
    say("ACT 9 - mission replay (15 s)",
        "Replaying a logged endurance mission with a misfire injected at\n"
        "t=60 s. The same twin, fed from the log: the alarm and the\n"
        "diagnosis arrive on cue, cylinder by cylinder. 'Three flights\n"
        "earlier than any redline.'")
    ctl({"action": "replay",
         "path": "app/logs/misfire_endurance.csv",
         "seek_s": 40, "speed": speed})
    wait(40.0, speed, "replay: misfire flags at t=60+ s")

    say("DONE",
        "Hand over to the honesty slide:\n"
        "- signature directions are research-backed, magnitudes need the rig\n"
        "- RUL is a bounded relative index, not a certified time to failure\n"
        "- physics+ML hybrid is published practice; our claim is the\n"
        "  combination for the MALE UAV case.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"cannot reach the server at {BASE}: {e}\n"
              "start it first: ../.venv/bin/python -m uvicorn server:app "
              "--host 127.0.0.1 --port 8734", file=sys.stderr)
        raise SystemExit(1)
