"""verify_twin.py -- the demo acts as headless assertions.

This is the MVP's go/no-go gate: the moments demo-script.md calls "the
single strongest twenty seconds available to us", run end to end through
mission.py (the engine), the nominal physics, the residual sigma band, and
the diagnosis rules, with no UI involved.

What is asserted, mapped to the demo script:

  Acts 3/4  the same fault is correctly ignored inside the band and fires
            outside it. The operating points below were chosen by probing
            the real model, not by assertion: at phi=1.15 the band is narrow
            AND an 8% leaning moves EGT toward peak only ~+4 K (z ~ +1.4),
            while at phi=0.85 the same 8% cut drives the cylinder past the
            kink onto the steep lean flank (z ~ -3.6), outside even the
            wider band there.
  Act 6     severity inversion: one parameter, and the EGT sign flips as
            severity crosses the mixture peak (fault-signatures.md 3).
  Act 7     detonation: opposite-sign pair, detonation ranked first.
  Drift     a biased sensor moves exactly one channel, no corroboration,
            and sensor drift ranks first.
  Misfire   both channels collapse on the right cylinder only.
  Health    a clean endurance mission never alarms.

Run AFTER verify_mission.py and verify_sigma_table.py; it assumes both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulator"))

from mission import run_mission, MissionPoint, FaultEvent
from physics.faults import FaultSpec
from twin import Twin, replay_frames

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
failures = 0


def check(ok, label, hint=""):
    global failures
    print(f"  {PASS if ok else FAIL} {label}" + (f"  ({hint})" if hint else ""))
    if not ok:
        failures += 1


def run(phi, fault=None, dur=150, seed=5, calibrate_s=30.0):
    pts = [MissionPoint(t_s=float(t), N_rpm=5000.0, MAP_Pa=120_000.0,
                        altitude_m=3000.0, phi=phi) for t in range(0, dur + 1)]
    ev = [FaultEvent(60.0, fault)] if fault else []
    tw = Twin(calibrate_s=calibrate_s)
    states = [tw.step(fr) for fr in run_mission(pts, fault_events=ev, seed=seed)]
    return states


def acts_3_4():
    inside = run(1.15, FaultSpec("injector_restriction", cylinder=3,
                                 severity=0.10))
    after = inside[-1]["cylinders"][2]
    check(not any(s["alarm"]["active"] for s in inside[70:]),
          "Act 3: small injector fault inside the band is correctly ignored",
          f"phi=1.15, z_EGT(3) settled at {after['z_EGT']:+.2f}, |z|<3 throughout")

    outside = run(0.85, FaultSpec("injector_restriction", cylinder=3,
                                  severity=0.10))
    after = outside[-1]["cylinders"][2]
    fired = [s["t_s"] for s in outside if s["alarm"]["active"]]
    check(bool(fired) and fired[0] <= 60.0 + 60.0,
          "Act 4: the same fault fires once the operating point puts it "
          "outside the band",
          f"phi=0.85, z_EGT(3)={after['z_EGT']:+.2f}, alarm at "
          f"t={fired[0]:.0f}s" if fired else "never fired")
    top = outside[-1]["diagnosis"][0]["label"] if outside[-1]["diagnosis"] else ""
    check("injector" in top,
          "Act 4: diagnosis names the injector, not a generic alarm",
          f"top: {top}")


def act_6():
    partial = run(1.15, FaultSpec("injector_restriction", cylinder=3,
                                  severity=0.10))
    blocked = run(1.15, FaultSpec("injector_restriction", cylinder=3,
                                  severity=1.0))
    ze_partial = partial[-1]["cylinders"][2]["z_EGT"]
    ze_blocked = blocked[-1]["cylinders"][2]["z_EGT"]
    check(ze_partial > 0 > ze_blocked,
          "Act 6: one severity knob flips the EGT sign across the peak",
          f"partial z_EGT={ze_partial:+.2f}, blocked z_EGT={ze_blocked:+.2f}")
    top = blocked[-1]["diagnosis"][0]["label"]
    check("misfire" in top or "dead cylinder" in top,
          "Act 6: full blockage reads as a dead cylinder", f"top: {top}")


def act_7():
    states = run(1.05, FaultSpec("detonation", cylinder=1, severity=1.0),
                 dur=180)
    c1 = states[-1]["cylinders"][0]
    check(c1["z_CHT"] > 3.0 > c1["z_EGT"] and c1["z_EGT"] < 0,
          "Act 7: detonation splits the channels (CHT up, EGT down)",
          f"z_CHT(1)={c1['z_CHT']:+.1f}, z_EGT(1)={c1['z_EGT']:+.1f}")
    top = states[-1]["diagnosis"][0]["label"]
    check("detonation" in top, "Act 7: detonation ranked first", f"top: {top}")


def drift_and_misfire():
    drift = run(1.05, FaultSpec("sensor_drift", cylinder=2,
                                sensor_channel="EGT_K", bias_K=80.0), dur=210)
    out_channels = set()
    for s in drift[70:]:
        for c in s["cylinders"]:
            if abs(c["z_EGT"]) >= 3.0:
                out_channels.add(("EGT_K", c["n"]))
            if abs(c["z_CHT"]) >= 3.0:
                out_channels.add(("CHT_K", c["n"]))
        if abs(s["oil"]["z_p"]) >= 3.0:
            out_channels.add(("p_oil_Pa", None))
        if abs(s["oil"]["z_T"]) >= 3.0:
            out_channels.add(("T_oil_K", None))
    check(out_channels == {("EGT_K", 2)},
          "Drift: exactly one channel ever leaves the band",
          f"channels: {sorted(out_channels, key=str)}")
    top = drift[-1]["diagnosis"][0]
    check("sensor drift" in top["label"],
          "Drift: sensor drift ranked first (lag-ratio discriminator)",
          f"top: {top['label']} ({top['confidence']})")

    mis = run(1.05, FaultSpec("misfire", cylinder=1, severity=1.0), dur=150)
    c1, c2 = mis[-1]["cylinders"][0], mis[-1]["cylinders"][1]
    check(c1["z_EGT"] < -50 and c1["z_CHT"] < -10,
          "Misfire: the dead cylinder runs cold on both channels",
          f"z_EGT(1)={c1['z_EGT']:+.1f}, z_CHT(1)={c1['z_CHT']:+.1f}")
    check(abs(c2["z_EGT"]) < 3.0 and abs(c2["z_CHT"]) < 3.0,
          "Misfire: the other cylinders stay inside the band",
          f"cyl2 z_EGT={c2['z_EGT']:+.2f}")
    top = mis[-1]["diagnosis"][0]["label"]
    check("misfire" in top, "Misfire: ranked first", f"top: {top}")


def healthy():
    states = run(0.92, None, dur=300, seed=3)
    alarms = [s for s in states if s["alarm"]["active"]]
    check(not alarms, "Healthy endurance mission: zero alarms in 300 s",
          f"{len(alarms)} alarmed steps")
    maxz = max(max(abs(c["z_EGT"]), abs(c["z_CHT"]))
               for s in states[40:] for c in s["cylinders"])
    check(maxz < 3.0, "Healthy: every z stays inside the band after calibration",
          f"max |z|={maxz:.2f}")


def replay_path():
    import io
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulator"))
    from mission import log_mission
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.csv")
        ev = [FaultEvent(60.0, FaultSpec("misfire", cylinder=1, severity=1.0))]
        log_mission(path, "endurance", fault_events=ev, seed=11)
        from mission import frames_from_csv
        res = replay_frames(frames_from_csv(path), Twin())
    ons = [f for f in res["flags"] if f["event"] == "alarm_on"]
    check(bool(ons), "Replay: the injected misfire alarms on the logged mission",
          f"first flag at t={ons[0]['t_s']:.0f}s" if ons else "never fired")
    # The diagnosis at alarm onset can legitimately be "unexplained": EGT
    # collapses within seconds while CHT is still falling (tau_CHT). The
    # researched signature completes over the next seconds; assert THAT.
    tops = [s["diagnosis"][0]["label"] for s in res["states"]
            if s["alarm"]["active"] and s["diagnosis"]]
    check(any(t.startswith("misfire") for t in tops),
          "Replay: misfire tops the diagnosis once the signature completes",
          f"tops seen: {sorted(set(tops))[:3]}")


if __name__ == "__main__":
    print("=" * 78)
    print("verify_twin.py -- the demo acts, headless")
    print("=" * 78)
    print("\n-- Acts 3/4: inside the band ignored, outside it fires --")
    acts_3_4()
    print("\n-- Act 6: severity inversion --")
    act_6()
    print("\n-- Act 7: detonation opposite-sign pair --")
    act_7()
    print("\n-- Sensor drift and misfire --")
    drift_and_misfire()
    print("\n-- Healthy mission --")
    healthy()
    print("\n-- Replay path --")
    replay_path()
    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {failures} CHECK(S) FAILED")
        raise SystemExit(1)
    print("RESULT: ALL CHECKS PASSED")
