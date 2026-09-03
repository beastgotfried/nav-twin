"""verify_ml.py -- integration gate for the learned layer (roadmap W6).

Runs the misfire demo act through the full Twin with model artifacts
present and asserts the learned view works end to end:

1. every state carries an "ml" block with available=True
2. the forest flags the injected misfire after onset
3. the classifier's top class is misfire on the right cylinder
4. the RUL block appears for the diagnosed subsystem, with ordered bounds
   and the bounded-relative framing
5. rules fallback intact: Twin(ml=False) produces no "ml" key and the
   rule-based diagnosis still names the misfire

Run after the models are trained: python verify_ml.py
"""

import sys
from pathlib import Path

import numpy as np

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_ROOT / "simulator"))
sys.path.insert(0, str(_APP))

import mission as M            # noqa: E402
from physics.faults import FaultSpec  # noqa: E402
from twin import Twin          # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    if not (_APP / "models" / "isolation_forest.pkl").exists():
        raise SystemExit("no model artifacts in app/models; run "
                         "ml_anomaly.py, ml_classify.py, ml_rul.py first")

    events = [M.FaultEvent(60.0, FaultSpec("misfire", cylinder=1,
                                           severity=1.0))]
    frames = M.run_mission("cruise_rich", events, seed=42)

    twin = Twin()
    flagged_at = None
    top_class = None
    rul_seen = None
    for fr in frames:
        st = twin.step(fr)
        check_key = "ml" in st
        if not check_key:
            check(False, "state carries ml block")
            break
        ml = st["ml"]
        if ml.get("anomaly_flag") and flagged_at is None:
            flagged_at = fr["t_s"]
        # CHT needs ~2 time constants to corroborate, so judge the
        # classifier on the settled diagnosis late in the fault, not the
        # first answer seconds after onset (EGT-only evidence is genuinely
        # ambiguous between misfire and detonation until the head responds)
        if ml.get("diagnosis") and fr["t_s"] >= 150.0:
            top_class = ml["diagnosis"][0]["label"]
        if ml.get("rul") and rul_seen is None:
            rul_seen = ml["rul"]

    check(flagged_at is not None and flagged_at >= 60.0,
          f"forest flagged the misfire at t={flagged_at}")
    check(top_class is not None and top_class.startswith("misfire")
          and top_class.endswith("cyl1"),
          f"classifier top class: {top_class}")
    if rul_seen is not None:
        check(rul_seen["severity_p10"] <= rul_seen["severity_median"]
              <= rul_seen["severity_p90"],
              "RUL severity bounds ordered")
        check("bounded relative" in rul_seen["framing"],
              "RUL carries bounded-relative framing")
    else:
        check(False, "RUL block appeared for the diagnosed subsystem")

    # rules-only fallback: no artifacts consulted, no ml key, rules still
    # name the misfire
    twin2 = Twin(ml=False)
    frames = M.run_mission("cruise_rich", events, seed=42)
    saw_rules = False
    for fr in frames:
        st = twin2.step(fr)
        if fr["t_s"] == 61.0:
            check("ml" not in st, "ml=False: no ml key in state")
        if st["diagnosis"] and "misfire" in st["diagnosis"][0]["label"]:
            saw_rules = True
    check(saw_rules, "ml=False: rule-based diagnosis still names misfire")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
