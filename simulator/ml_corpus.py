"""ml_corpus.py -- training corpus generator for the ML layer (roadmap W1).

Sweeps the UNMODIFIED mission generator over the operating envelope to
produce two corpus families:

- healthy: randomised missions with no faults, for the unsupervised
  anomaly detector and the baseline/delta machinery
- fault: randomised missions with exactly one injected fault each, labelled,
  for the supervised classifier and the RUL trajectories (ramped onsets
  double as progressive degradation)

Per mission this stores BOTH the raw telemetry frames and the twin states
(z residuals), so feature extraction downstream is pure numpy windowing and
never re-runs the physics.

Deliberate choices:

- Fault onset is never before t=45 s, because the twin calibrates its frozen
  baseline on the opening seconds (twin/__init__.py, DEFAULT_CALIBRATE_S).
  Onset inside the calibration window would contaminate the baseline.
- One fault per mission. Multi-fault interaction is a later research
  question, not training data v1.
- Profiles are random walks over the envelope, not the named demo
  scenarios, so the model cannot memorise the demo.
- Everything is seeded. verify_corpus.py regenerates one mission from its
  recorded seed and requires bit-identical arrays.

Run: python ml_corpus.py --healthy 600 --out ../corpus --workers 14
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_SIM = Path(__file__).resolve().parent
_ROOT = _SIM.parent
sys.path.insert(0, str(_SIM))
sys.path.insert(0, str(_ROOT / "app"))

import mission as M                      # noqa: E402
from physics.faults import FaultSpec     # noqa: E402
from twin import Twin                    # noqa: E402

# Envelope the corpus walks over. Matches the ranges the sigma tables were
# built on, so the band lookup never clamps during corpus generation.
N_RANGE = (3500.0, 5800.0)
MAP_RANGE = (85_000.0, 145_000.0)
ALT_RANGE = (300.0, 7600.0)
PHI_RANGE = (0.80, 1.20)

DURATION_S = 300.0
DT_S = 1.0
MIN_ONSET_S = 45.0          # after the twin's 30 s calibration window

CYLINDER_FAULTS = ("misfire", "injector_restriction", "detonation",
                   "cooling_degradation")
ENGINE_FAULTS = ("bearing_wear", "turbo_degradation")
SEVERITY_GRID = (0.10, 0.30, 0.60, 1.00)
DRIFT_BIASES_K = (-25.0, 25.0)

# The explicit class list. Fault missions cycle through it round-robin
# (severity, onset and ramp still random within the class), so every class
# is guaranteed training coverage. Pure random sampling left some classes
# empty on small corpora, which verify_ml.py exposed when the classifier
# could not name a class it had never seen.
CLASS_LIST = (
    [(k, c, "") for k in CYLINDER_FAULTS for c in (1, 2, 3, 4)]
    + [(k, 0, "") for k in ENGINE_FAULTS]
    + [("sensor_drift", c, ch)
       for ch in ("EGT_K", "CHT_K") for c in (1, 2, 3, 4)]
)


def random_profile(rng, duration_s=DURATION_S, dt_s=DT_S):
    """Piecewise-ramped random walk over the envelope: 4 to 6 random
    operating points, linear ramps between them. Returns MissionPoints."""
    n_knots = int(rng.integers(4, 7))
    knots = []
    for i in range(n_knots):
        t = duration_s * i / (n_knots - 1)
        knots.append((t,
                      rng.uniform(*N_RANGE),
                      rng.uniform(*MAP_RANGE),
                      rng.uniform(*ALT_RANGE),
                      rng.uniform(*PHI_RANGE)))
    pts = []
    t = 0.0
    while t <= duration_s:
        # find bracketing knots
        j = 0
        while j < n_knots - 2 and knots[j + 1][0] < t:
            j += 1
        t0, n0, m0, a0, p0 = knots[j]
        t1, n1, m1, a1, p1 = knots[j + 1]
        f = 0.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
        pts.append(M.MissionPoint(
            t_s=t,
            N_rpm=n0 + f * (n1 - n0),
            MAP_Pa=m0 + f * (m1 - m0),
            altitude_m=a0 + f * (a1 - a0),
            phi=p0 + f * (p1 - p0),
        ))
        t += dt_s
    return pts


def sample_fault(rng, class_idx=None):
    """One labelled fault spec. class_idx indexes CLASS_LIST for stratified
    coverage; None draws uniformly at random. Returns (FaultEvent, meta)."""
    if class_idx is None:
        kind = rng.choice(CYLINDER_FAULTS + ENGINE_FAULTS + ("sensor_drift",))
        cylinder = 0 if kind in ENGINE_FAULTS else int(rng.integers(1, 5))
        channel = str(rng.choice(["EGT_K", "CHT_K"])) \
            if kind == "sensor_drift" else ""
    else:
        kind, cylinder, channel = CLASS_LIST[class_idx % len(CLASS_LIST)]
    severity = float(rng.choice(SEVERITY_GRID))
    onset_s = float(rng.uniform(MIN_ONSET_S, DURATION_S - 90.0))
    ramp_s = float(rng.choice([0.0, 30.0, 60.0]))
    bias = 0.0
    if kind == "sensor_drift":
        bias = float(rng.choice(DRIFT_BIASES_K))
        # drift magnitude is physical (K), not a 0-1 severity
        severity = abs(bias)
    spec = FaultSpec(kind, cylinder=cylinder or None, severity=severity,
                     sensor_channel=channel or None, bias_K=bias)
    ev = M.FaultEvent(onset_s, spec, ramp_s=ramp_s)
    meta = {"kind": kind, "cylinder": cylinder or 0, "severity": severity,
            "onset_s": onset_s, "ramp_s": ramp_s,
            "sensor_channel": channel, "bias_K": bias}
    return ev, meta


def run_one(task):
    """One mission end to end: profile -> telemetry frames -> twin states.
    Returns a dict of arrays plus meta. Runs in worker processes."""
    seed, label, out_dir = task["seed"], task["label"], task["out_dir"]
    rng = np.random.default_rng(seed)
    points = random_profile(rng)
    events = []
    meta = {"kind": "healthy", "cylinder": 0, "severity": 0.0,
            "onset_s": -1.0, "ramp_s": 0.0, "sensor_channel": "",
            "bias_K": 0.0}
    if label == "fault":
        ev, meta = sample_fault(rng, class_idx=task.get("class_idx"))
        events = [ev]

    frames = list(M.run_mission(points, events, seed=seed))
    twin = Twin()
    states = [twin.step(fr) for fr in frames]

    n = len(frames)
    def col(key):
        return np.array([f[key] for f in frames], dtype=float)
    def cyl_col(key):
        return np.array([f[key] for f in frames], dtype=float)  # (n, 4)
    def state_cyl(key):
        return np.array([[c[key] for c in s["cylinders"]] for s in states],
                        dtype=float)

    t = col("t_s")
    active = np.zeros(n)
    sev_t = np.zeros(n)
    if meta["onset_s"] >= 0:
        on = meta["onset_s"]
        active = (t >= on).astype(float)
        if meta["ramp_s"] > 0:
            sev_t = meta["severity"] * np.clip((t - on) / meta["ramp_s"], 0, 1)
        else:
            sev_t = meta["severity"] * active

    name = f"{label}_{seed:08d}.npz"
    np.savez_compressed(
        Path(out_dir) / label / name,
        t_s=t,
        N_rpm=col("N_rpm"), MAP_Pa=col("MAP_Pa"),
        MAP_commanded_Pa=col("MAP_commanded_Pa"), altitude_m=col("altitude_m"),
        fuel_flow=cyl_col("fuel_flow_kg_s_per_cyl"),
        EGT_K=cyl_col("EGT_K"), CHT_K=cyl_col("CHT_K"),
        p_oil_Pa=col("p_oil_Pa"), T_oil_K=col("T_oil_K"),
        z_EGT=state_cyl("z_EGT"), z_CHT=state_cyl("z_CHT"),
        z_p_oil=np.array([s["oil"]["z_p"] for s in states]),
        z_T_oil=np.array([s["oil"]["z_T"] for s in states]),
        fault_active=active, severity_t=sev_t,
    )
    return {"file": f"{label}/{name}", "label": label, "seed": seed,
            "class_idx": task.get("class_idx", -1), **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy", type=int, default=600)
    ap.add_argument("--fault", type=int, default=320)
    ap.add_argument("--out", default=str(_ROOT / "corpus"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "healthy").mkdir(parents=True, exist_ok=True)
    (out / "fault").mkdir(parents=True, exist_ok=True)

    tasks = ([{"seed": args.seed * 1_000_000 + i, "label": "healthy",
               "out_dir": str(out)} for i in range(args.healthy)]
             + [{"seed": args.seed * 1_000_000 + 500_000 + i,
                 "label": "fault", "out_dir": str(out), "class_idx": i}
                for i in range(args.fault)])

    t0 = time.time()
    rows = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(run_one, tasks)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    dt = time.time() - t0
                    print(f"{i+1}/{len(tasks)} missions, "
                          f"{dt:.0f}s elapsed, eta "
                          f"{dt/(i+1)*(len(tasks)-i-1):.0f}s", flush=True)
    else:
        for i, tk in enumerate(tasks):
            rows.append(run_one(tk))
            if (i + 1) % 10 == 0:
                print(f"{i+1}/{len(tasks)}", flush=True)

    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} missions in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
