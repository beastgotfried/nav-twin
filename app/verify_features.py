"""verify_features.py -- gates the feature extractor (roadmap W2).

Three properties, each checked directly:

1. NO LOOK-AHEAD. Features at time t must be bit-identical when the future
   is corrupted. If this fails, every downstream metric is lying, because
   the model would train on information it cannot have at runtime.
2. DETERMINISM. Same input, same features.
3. WINDOW CORRECTNESS. Constant input gives zero slope and mean equal to
   the constant; a step input moves the mean gradually, not instantly.

Run: python verify_features.py   (needs no corpus, self-contained)
"""

import numpy as np

from ml_features import FeatureBuilder, FEATURE_NAMES, N_FEATURES

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def run_stream(rows):
    fb = FeatureBuilder()
    out = {}
    for t, z, ctx in rows:
        fb.push(t, z, ctx)
        f = fb.features()
        if f is not None:
            out[t] = f
    return out


def make_rows(n=120, seed=7):
    rng = np.random.default_rng(seed)
    return [(float(t), rng.normal(0, 1, 10), rng.uniform(0, 1, 4))
            for t in range(n)]


# 1. no look-ahead: corrupt everything after t=80, features up to t=80
# must be identical to the uncorrupted run.
rows = make_rows()
clean = run_stream(rows)
corrupted_rows = [(t, (np.full(10, 999.0) if t > 80 else z), ctx)
                  for t, z, ctx in rows]
corrupt = run_stream(corrupted_rows)
same = all(np.array_equal(clean[t], corrupt[t]) for t in clean if t <= 80)
check(same, "no look-ahead: features at t<=80 unchanged when future corrupted")
check(any(t > 80 for t in corrupt)
      and not np.array_equal(clean[119], corrupt[119]),
      "sanity: corrupted future DID change later features (test is real)")

# 2. determinism
again = run_stream(make_rows())
check(all(np.array_equal(clean[t], again[t]) for t in clean),
      "determinism: identical input gives identical features")

# 3. window correctness
fb = FeatureBuilder()
for t in range(120):
    fb.push(float(t), np.full(10, 2.5), np.zeros(4))
f = fb.features()
check(f is not None and f.shape == (N_FEATURES,),
      f"feature shape is {N_FEATURES}")
means = f[10:20]
slopes = f[20:30]
check(np.allclose(means, 2.5), "constant input: window mean == constant")
check(np.allclose(slopes, 0.0, atol=1e-12),
      "constant input: window slope == 0")

# step at t=60: one window later (t=119) the mean should sit strictly
# between the old and new values, and settle to the new value only once
# the pre-step data has aged out (t=180).
fb = FeatureBuilder()
f_at_119 = None
for t in range(181):
    fb.push(float(t), np.full(10, 0.0 if t < 60 else 4.0), np.zeros(4))
    if t == 119:
        f_at_119 = fb.features()
f = fb.features()
check(0.0 < f_at_119[10] < 4.0,
      f"step input: window mean moves gradually "
      f"(mean one window after step={f_at_119[10]:.2f})")
check(abs(f[10] - 4.0) < 1e-9, "mean settles to new value once window aged out")
check(abs(f[0] - 4.0) < 1e-12, "current z reflects the step immediately")

print()
if failures:
    print(f"RESULT: {len(failures)} FAILURES")
    raise SystemExit(1)
print(f"RESULT: ALL CHECKS PASSED ({N_FEATURES} features: "
      f"{', '.join(FEATURE_NAMES[:3])}, ...)")
