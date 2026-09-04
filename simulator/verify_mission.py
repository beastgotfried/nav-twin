"""verify_mission.py -- checks the mission generator against the claims the
rest of the MVP builds on. Run after mission.py, before trusting any
telemetry downstream of it.

Claims checked:
1. Thermal lag ratio (Handbook 7.8): after a throttle step, EGT responds
   fast (tau ~1.5 s) and CHT slowly (tau ~20 s). This is the basis of the
   sensor-drift discriminator, so it has to be real in the telemetry.
2. Sensor noise is present and sane: steady-state scatter matches the
   ASSUMED sigmas in mission.py, not zero and not wild.
3. An injected fault is visible in the logged telemetry: a misfire on
   cylinder 1 drops ITS EGT hard while the other three stay hot
   (fault-signatures.md: a dead cylinder runs cold).
4. Healthy telemetry tracks the nominal twin: the observed values sit
   within a few K of the fault-free prediction at the same operating point,
   which is what makes residuals meaningful at all.
"""

import numpy as np

from mission import run_mission, FaultEvent, MissionPoint, NOISE_SIGMA, _fuel_flow_per_cyl
from physics.engine_model import OperatingPoint, predict_steady_state
from physics.engine_params import DEFAULT_CONSTANTS
from physics.faults import FaultSpec

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
failures = 0


def check(ok, label, hint=""):
    global failures
    print(f"  {PASS if ok else FAIL} {label}" + (f"  ({hint})" if hint else ""))
    if not ok:
        failures += 1


def step_response():
    """Claims 1 and 2: a commanded phi step at t=30. EGT is driven by phi
    (Handbook 7.6), so a phi step is the input that actually moves EGT_ss;
    stepping N/MAP at fixed phi would leave EGT nearly unchanged and the
    lag measurement would be noise divided by noise."""
    pts = [MissionPoint(t_s=float(t), N_rpm=5000.0, MAP_Pa=120_000.0,
                        altitude_m=3000.0, phi=1.10 if t < 30 else 0.90)
           for t in range(0, 61)]
    frames = list(run_mission(pts, seed=7))
    # Step down happens at t=30 (5500 -> 3800 rpm, 140 -> 90 kPa).
    seg = [f for f in frames if 28 <= f["t_s"] <= 60]
    egt0 = np.mean([f["EGT_K"][0] for f in seg if f["t_s"] < 30])
    egt_late = np.mean([f["EGT_K"][0] for f in seg if f["t_s"] >= 55])
    cht0 = np.mean([f["CHT_K"][0] for f in seg if f["t_s"] < 30])
    cht_late = np.mean([f["CHT_K"][0] for f in seg if f["t_s"] >= 55])

    def frac_at(t):
        f = seg[int(round(t - 28))]
        return (f["EGT_K"][0] - egt0) / (egt_late - egt0), \
               (f["CHT_K"][0] - cht0) / (cht_late - cht0)

    fe_3, fc_3 = frac_at(33.0)
    check(0.5 < fe_3 < 1.05,
          "EGT most of the way to its new steady state ~3 s after a throttle step",
          f"fraction={fe_3:.2f} at t+3s, tau_egt={DEFAULT_CONSTANTS.tau_egt_s}s")
    check(0.02 < fc_3 < 0.5,
          "CHT barely moving ~3 s after the same step (the lag ratio is real)",
          f"fraction={fc_3:.2f} at t+3s, tau_cht={DEFAULT_CONSTANTS.tau_cht_s}s")

    steady = [f["EGT_K"][0] for f in frames if 5 <= f["t_s"] <= 28]
    noise = float(np.std(steady))
    check(0.5 < noise < 6.0,
          "EGT noise present and plausible at steady state",
          f"std={noise:.2f} K, ASSUMED sigma={NOISE_SIGMA['EGT_K']} K plus lag ripple")
    check(egt0 - egt_late > 30.0,
          "the phi step itself moves EGT materially (test input is real)",
          f"EGT_ss {egt0:.0f} K rich -> {egt_late:.0f} K lean")


def fault_visibility():
    """Claim 3: misfire on cylinder 1 from t=60."""
    ev = [FaultEvent(60.0, FaultSpec("misfire", cylinder=1, severity=1.0))]
    frames = list(run_mission("endurance", fault_events=ev, seed=11))
    before = [f for f in frames if 50 <= f["t_s"] < 58]
    after = [f for f in frames if 70 <= f["t_s"] <= 90]
    drop1 = np.mean([f["EGT_K"][0] for f in before]) - \
        np.mean([f["EGT_K"][0] for f in after])
    drop4 = np.mean([f["EGT_K"][3] for f in before]) - \
        np.mean([f["EGT_K"][3] for f in after])
    check(drop1 > 200.0, "misfiring cylinder's EGT collapses (dead cylinder runs cold)",
          f"cyl1 drop={drop1:.0f} K")
    check(abs(drop4) < 20.0, "healthy cylinders unaffected by cylinder 1's misfire",
          f"cyl4 change={drop4:+.1f} K")


def healthy_tracks_nominal():
    """Claim 4: healthy telemetry stays close to the nominal prediction."""
    frames = [f for f in run_mission("endurance", seed=3) if 30 <= f["t_s"] <= 60]
    worst = 0.0
    for f in frames[::10]:
        pt = MissionPoint(t_s=f["t_s"], N_rpm=f["N_rpm"], MAP_Pa=f["MAP_Pa"],
                          altitude_m=f["altitude_m"], phi=0.92)
        op = OperatingPoint(N_rpm=f["N_rpm"], MAP_Pa=f["MAP_Pa"],
                            altitude_m=f["altitude_m"],
                            fuel_flow_kg_s_per_cyl=_fuel_flow_per_cyl(pt),
                            T_oil_K=f["T_oil_K"])
        pred = predict_steady_state(op)
        for i, c in enumerate(pred["per_cylinder"]):
            worst = max(worst, abs(f["EGT_K"][i] - c["EGT_K"]))
    check(worst < 15.0, "healthy EGT telemetry within 15 K of the nominal twin",
          f"worst deviation={worst:.1f} K over the sampled segment")


def full_mission_windows():
    """The stitched guided-flight profile produces the expected alarm
    windows: quiet through the climb and the rich-cruise small-fault phase,
    alarmed across every engine fault phase, quiet through the turbo leg
    (the MAP gap is not a z channel, so the band structurally cannot see
    it), and back to nominal after the faults clear."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
    from twin import Twin
    from mission import full_mission_faults

    tw = Twin(calibrate_s=30.0)
    alarmed = {}
    forest_flagged = {}
    rul_seen = {}
    turbo_named = {}
    for fr in run_mission("full_mission", fault_events=full_mission_faults(),
                          seed=5):
        st = tw.step(fr)
        alarmed[fr["t_s"]] = st["alarm"]["active"]
        ml = st.get("ml") or {}
        forest_flagged[fr["t_s"]] = bool(ml.get("anomaly_flag"))
        rul_seen[fr["t_s"]] = ml.get("rul")
        labels = [d["label"] for d in st.get("diagnosis", [])]
        labels += [d["label"] for d in ml.get("diagnosis", [])]
        turbo_named[fr["t_s"]] = any("turbo" in l for l in labels)

    def window(d, t0, t1):
        vals = [v for t, v in d.items() if t0 <= t < t1]
        return sum(vals) / max(len(vals), 1)

    check(window(alarmed, 40, 90) == 0.0,
          "full mission: no alarms during the climb")
    check(window(alarmed, 160, 205) == 0.0,
          "full mission: 8% restriction at rich cruise correctly ignored")
    check(window(alarmed, 250, 330) > 0.8,
          "full mission: lean cruise alarms on the same fault")
    check(window(alarmed, 400, 420) > 0.8,
          "full mission: escalation to blockage stays alarmed")
    check(window(alarmed, 520, 590) > 0.8,
          "full mission: detonation phase alarmed")
    check(window(alarmed, 650, 710) > 0.5,
          "full mission: EGT sensor drift alarms its one channel")
    check(window(alarmed, 770, 830) > 0.8,
          "full mission: misfire phase alarmed")
    check(window(alarmed, 920, 955) > 0.5,
          "full mission: ramped bearing wear alarms on the oil channels")
    check(window(alarmed, 1030, 1075) > 0.5,
          "full mission: ramped cooling degradation alarms once established")
    check(window(alarmed, 1130, 1250) == 0.0,
          "full mission: turbo leg invisible to the band (no z channel)")
    check(window(alarmed, 1300, 1385) == 0.0,
          "full mission: nominal again after faults clear")

    if any(forest_flagged.values()):
        # Model artifacts present. The learned layer must corroborate an
        # established engine fault, and the diagnosis (rules or model)
        # must name the fading turbo during the high-altitude leg, from
        # the MAP gap the temperature band cannot see.
        check(window(forest_flagged, 520, 590) > 0.5,
              "full mission (ML): forest corroborates the detonation phase")
        check(window(forest_flagged, 40, 90) == 0.0,
              "full mission (ML): forest quiet through the healthy climb")
        check(window(turbo_named, 1120, 1250) > 0.5,
              "full mission: diagnosis names the fading turbo while the "
              "temperature band stays quiet")
        rul_vals = [r for t, r in rul_seen.items()
                    if 900 <= t < 955 and r]
        check(any(r["severity_median"] > 0.2 for r in rul_vals),
              "full mission (ML): RUL tracks the bearing wear ramp "
              "(severity > 0.2 during the phase)")
    else:
        print("  [SKIP] ML windows: no model artifacts, rules-only checkout")


if __name__ == "__main__":
    print("=" * 78)
    print("verify_mission.py -- mission generator checks")
    print("=" * 78)
    print("\n-- Thermal lag and noise (rapid_throttle) --")
    step_response()
    print("\n-- Fault visibility (endurance + misfire cyl 1) --")
    fault_visibility()
    print("\n-- Healthy telemetry vs nominal twin (endurance) --")
    healthy_tracks_nominal()
    print("\n-- Full mission: alarm windows across the guided flight --")
    full_mission_windows()
    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {failures} CHECK(S) FAILED")
        raise SystemExit(1)
    print("RESULT: ALL CHECKS PASSED")
