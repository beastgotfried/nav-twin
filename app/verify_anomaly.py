"""verify_anomaly.py -- the gate for the learned anomaly layer (roadmap W3).

Deployment semantics first, because the gates follow from them: the forest
ships as a UNION with the band (alert = band OR forest). The band reacts to
instantaneous z excursions and stays the fast path for library faults; the
forest's windowed features trade a few seconds of latency for coverage of
faults the band structurally cannot see (e.g. turbo degradation, which
shows up in the commanded-vs-achieved MAP gap, a channel that has no z).

The gates, all required, therefore check what the forest is actually FOR:

1. CALIBRATION HONESTY: measured false-alarm rate on held-out healthy
   missions must not exceed 1.5x the target the threshold was chosen for.
2. ADDED COVERAGE: the forest must alert on at least one held-out fault
   mission that the band never alerts on. If it adds nothing, it does not
   deploy.
3. LATENCY SANITY: forest-alone median onset-to-alert time must be within
   the feature window plus slack (65 s). Reported, and bounded so a broken
   feature pipeline cannot pass silently.

Band-alone and forest-alone numbers are printed for the record.

Run after ml_anomaly.py: python verify_anomaly.py --corpus ../corpus
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_APP))

from ml_anomaly import (alerts_from_scores, band_alerts, z_block,  # noqa: E402
                        load_healthy_features, TARGET_FPR)
from ml_features import features_from_corpus_mission, WINDOW_S  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def first_alert_t(alerts, t):
    idx = np.nonzero(alerts)[0]
    return float(t[idx[0]]) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    args = ap.parse_args()

    art = joblib.load(_APP / "models" / "isolation_forest.pkl")
    pipe, threshold = art["model"], art["threshold"]
    persistence = art["persistence"]

    import csv
    with open(Path(args.corpus) / "index.csv") as fh:
        all_rows = list(csv.DictReader(fh))

    # gate 1: healthy FPR on the recorded val split
    val_rows = [r for r in all_rows
                if r["label"] == "healthy"
                and int(r["seed"]) in set(art["val_seeds"])]
    Xva, _ = load_healthy_features(args.corpus, val_rows)
    scores = pipe.score_samples(Xva)
    fpr = float((scores < threshold).mean())
    check(fpr <= TARGET_FPR * 1.5,
          f"healthy FPR {fpr:.4f} within 1.5x target {TARGET_FPR}")

    # gates 2+3: held-out fault missions
    fault_rows = [r for r in all_rows if r["label"] == "fault"]
    lat_forest, lat_band = [], []
    added = 0
    per_kind = {}
    for r in fault_rows:
        d = np.load(Path(args.corpus) / r["file"])
        t, F, idx = features_from_corpus_mission(d)
        onset = float(r["onset_s"])
        a_f = alerts_from_scores(pipe.score_samples(F), threshold,
                                 persistence)
        a_b = band_alerts(z_block(d)[idx], persistence)
        tf, tb = first_alert_t(a_f, t), first_alert_t(a_b, t)
        if tf is not None and tf >= onset:
            lat_forest.append(tf - onset)
        if tb is not None and tb >= onset:
            lat_band.append(tb - onset)
        if tf is not None and tb is None:
            added += 1
        k = r["kind"]
        per_kind.setdefault(k, [0, 0])
        per_kind[k][0] += 1 if tf is not None else 0
        per_kind[k][1] += 1 if tb is not None else 0

    print(f"fault missions: {len(fault_rows)}")
    for k, (f, b) in sorted(per_kind.items()):
        print(f"  {k:24s} forest {f:3d}  band {b:3d}")

    check(added > 0, f"added coverage: forest caught {added} missions the "
                     "band never alerted on")
    med_f = float(np.median(lat_forest)) if lat_forest else float("inf")
    med_b = float(np.median(lat_band)) if lat_band else float("inf")
    print(f"median latency: forest {med_f:.1f}s, band {med_b:.1f}s "
          f"(deployed as union: effective latency is min of the two)")
    check(med_f <= WINDOW_S + 5.0 + persistence,
          f"forest latency {med_f:.1f}s within window+slack "
          f"({WINDOW_S + 5.0 + persistence:.0f}s)")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES -- model does not deploy, "
              "band stays primary")
        sys.exit(1)
    print("RESULT: ALL GATES PASSED -- forest deploys alongside the band")


if __name__ == "__main__":
    main()
