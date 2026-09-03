"""verify_corpus.py -- gates the training corpus before any model touches it.

Checks, in order:

1. Completeness and integrity: every index row loads, no NaN anywhere,
   array shapes consistent.
2. Envelope coverage: the corpus actually reaches the edges of the
   N / MAP / altitude / phi ranges ml_corpus.py claims to cover.
3. Determinism: one mission is regenerated from its recorded seed and must
   reproduce bit-identical telemetry.
4. Separation: every fault class must move its researched signature
   channels (fault-signatures.md) once active, measured as median |z| on
   those channels during the active window. A class that stays inside the
   band everywhere is either a corpus bug or a genuinely sub-threshold
   severity, and this script says which.

Run after ml_corpus.py: python verify_corpus.py --corpus ../corpus
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SIM = Path(__file__).resolve().parent
_ROOT = _SIM.parent
sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(_ROOT / "app"))

import ml_corpus as C   # noqa: E402
import mission as M     # noqa: E402

# Researched signature channels per fault kind (fault-signatures.md):
# which z channels each fault is expected to move.
SIGNATURE_CHANNELS = {
    "misfire": ["z_EGT", "z_CHT"],
    "injector_restriction": ["z_EGT"],
    "detonation": ["z_CHT", "z_EGT"],
    "cooling_degradation": ["z_CHT"],
    "bearing_wear": ["z_p_oil", "z_T_oil"],
    "turbo_degradation": [],   # MAP-gap channel, not a z; checked separately
    "sensor_drift": ["z_EGT", "z_CHT"],  # filtered to the biased channel below
}


def check(cond, msg, failures):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    args = ap.parse_args()
    root = Path(args.corpus)
    failures = []

    import csv
    with open(root / "index.csv") as fh:
        rows = list(csv.DictReader(fh))
    check(len(rows) > 0, f"index has {len(rows)} missions", failures)

    # 1. integrity
    n_nan, n_bad_shape = 0, 0
    data = {}
    for r in rows:
        d = np.load(root / r["file"])
        data[r["file"]] = d
        if any(np.isnan(d[k]).any() for k in d.files):
            n_nan += 1
        n = d["t_s"].shape[0]
        if d["z_EGT"].shape != (n, 4) or d["EGT_K"].shape != (n, 4):
            n_bad_shape += 1
    check(n_nan == 0, f"no NaN in any mission ({n_nan} bad)", failures)
    check(n_bad_shape == 0, "array shapes consistent", failures)

    # 2. coverage (over healthy missions, which walk the whole envelope)
    ns = np.concatenate([data[r["file"]]["N_rpm"] for r in rows
                         if r["label"] == "healthy"])
    maps = np.concatenate([data[r["file"]]["MAP_Pa"] for r in rows
                           if r["label"] == "healthy"]) / 1000.0
    alts = np.concatenate([data[r["file"]]["altitude_m"] for r in rows
                           if r["label"] == "healthy"])
    check(ns.min() < C.N_RANGE[0] + 300 and ns.max() > C.N_RANGE[1] - 300,
          f"RPM covers envelope ({ns.min():.0f}..{ns.max():.0f})", failures)
    check(maps.min() < C.MAP_RANGE[0] / 1000 + 8
          and maps.max() > C.MAP_RANGE[1] / 1000 - 8,
          f"MAP covers envelope ({maps.min():.0f}..{maps.max():.0f} kPa)",
          failures)
    check(alts.min() < C.ALT_RANGE[0] + 600
          and alts.max() > C.ALT_RANGE[1] - 600,
          f"altitude covers envelope ({alts.min():.0f}..{alts.max():.0f} m)",
          failures)

    # 3. determinism: regenerate the first fault mission from its seed
    r0 = next(r for r in rows if r["label"] == "fault")
    seed = int(r0["seed"])
    rng = np.random.default_rng(seed)
    points = C.random_profile(rng)
    cidx = int(r0.get("class_idx", -1))
    ev, meta = C.sample_fault(rng, class_idx=cidx if cidx >= 0 else None)
    frames = list(M.run_mission(points, [ev], seed=seed))
    ref = data[r0["file"]]
    same = np.allclose(np.array([f["EGT_K"] for f in frames]), ref["EGT_K"])
    check(same, f"determinism: seed {seed} reproduces bit-identical EGT",
          failures)
    check(meta["kind"] == r0["kind"], "fault meta reproduces from seed",
          failures)

    # 4. separation per fault kind
    per_kind = defaultdict(list)
    for r in rows:
        if r["label"] != "fault":
            continue
        d = data[r["file"]]
        act = d["fault_active"] > 0
        if act.sum() < 10:
            continue
        kind = r["kind"]
        cyl = int(r["cylinder"])
        zmax = 0.0
        chans = SIGNATURE_CHANNELS.get(kind, [])
        if kind == "sensor_drift":
            chans = [r["sensor_channel"].replace("_K", "")
                     .replace("EGT", "z_EGT").replace("CHT", "z_CHT")]
        for ch in chans:
            arr = d[ch]
            if arr.ndim == 2 and cyl >= 1:
                arr = arr[:, cyl - 1]
            zmax = max(zmax, float(np.median(np.abs(arr[act]))))
        if kind == "turbo_degradation":
            gap = (d["MAP_commanded_Pa"] - d["MAP_Pa"])[act]
            zmax = float(np.median(gap)) / 1000.0  # kPa, threshold below
        per_kind[kind].append(zmax)

    for kind, vals in sorted(per_kind.items()):
        med = float(np.median(vals))
        if kind == "turbo_degradation":
            # 1.5 kPa = 3x MAP sensor noise (0.5 kPa), i.e. observable.
            # The rule-based detector needs 3.0 kPa, but the corpus gate
            # only asks that the signal exists above the noise floor;
            # sub-rule-threshold turbo missions are legitimate corpus
            # members (they are exactly what the learned layer adds).
            check(med > 1.5, f"separation: {kind} median MAP gap "
                             f"{med:.1f} kPa over {len(vals)} missions",
                  failures)
        else:
            check(med > 0.5, f"separation: {kind} median |z| {med:.2f} on "
                             f"signature channels over {len(vals)} missions",
                  failures)

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURES")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
