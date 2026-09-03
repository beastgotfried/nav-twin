"""ml_anomaly.py -- learned anomaly layer (roadmap W3).

Isolation Forest trained on healthy-corpus features only. It learns the
shape of normal, so it can flag faults OUTSIDE the injection library, the
property the band alone does not have (ml-layer.md section 1).

It does not replace the band. verify_anomaly.py gates it against the
band-only baseline on held-out data: false-alarm rate no worse, detection
latency no worse. If the gate fails the model is not deployed and the band
stays primary; the roadmap says so explicitly.

Model artifact: app/models/isolation_forest.pkl (joblib dict: sklearn
Pipeline(scaler + forest), threshold, feature names, training metadata).
Threshold is chosen on held-out healthy data at the target false-alarm
rate, never on training data.
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_APP))

from ml_features import features_from_corpus_mission, FEATURE_NAMES  # noqa: E402

MODELS_DIR = _APP / "models"
TARGET_FPR = 0.01        # 1% of healthy timesteps flagged, calibration target
PERSISTENCE = 3          # consecutive flagged steps before an alert,
                         # same as ALARM_ON_STEPS in twin/anomaly.py


def load_healthy_features(corpus_root, files):
    """features and per-row mission id for a list of index rows."""
    X, mid = [], []
    for k, r in enumerate(files):
        d = np.load(Path(corpus_root) / r["file"])
        _, F, _ = features_from_corpus_mission(d)
        X.append(F)
        mid.append(np.full(len(F), k))
    return np.vstack(X), np.concatenate(mid)


def alerts_from_scores(scores, threshold, persistence=PERSISTENCE):
    """Persistence-filtered binary alerts, mirroring twin/anomaly.py."""
    raw = scores < threshold
    out = np.zeros_like(raw)
    run = 0
    for i, r in enumerate(raw):
        run = run + 1 if r else 0
        if run >= persistence:
            out[i] = True
    return out


def band_alerts(z_block, persistence=PERSISTENCE):
    """The rule baseline on the same timesteps: any |z| >= 3 with the same
    persistence. z_block: (n, 10) raw z values straight from the corpus."""
    raw = (np.abs(z_block) >= 3.0).any(axis=1)
    out = np.zeros_like(raw)
    run = 0
    for i, r in enumerate(raw):
        run = run + 1 if r else 0
        if run >= persistence:
            out[i] = True
    return out


def z_block(d):
    return np.hstack([d["z_EGT"], d["z_CHT"],
                      d["z_p_oil"][:, None], d["z_T_oil"][:, None]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    import csv
    with open(Path(args.corpus) / "index.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] == "healthy"]
    if len(rows) < 10:
        raise SystemExit(f"need at least 10 healthy missions, found "
                         f"{len(rows)}; is the corpus finished?")

    # split BY MISSION, never by timestep (correlated rows would leak)
    order = np.random.default_rng(1).permutation(len(rows))
    n_val = max(1, int(len(rows) * args.val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    print(f"healthy missions: {len(train_rows)} train / {len(val_rows)} val")

    t0 = time.time()
    Xtr, _ = load_healthy_features(args.corpus, train_rows)
    Xva, _ = load_healthy_features(args.corpus, val_rows)
    print(f"features: train {Xtr.shape}, val {Xva.shape} "
          f"({time.time()-t0:.0f}s)")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("iforest", IsolationForest(
                         n_estimators=200, max_samples=0.25,
                         random_state=0, n_jobs=-1))])
    t0 = time.time()
    pipe.fit(Xtr)
    print(f"fit in {time.time()-t0:.0f}s")

    # threshold on HELD-OUT healthy data at the target FPR
    val_scores = pipe.score_samples(Xva)
    threshold = float(np.quantile(val_scores, TARGET_FPR))

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": pipe, "threshold": threshold,
                 "feature_names": FEATURE_NAMES,
                 "persistence": PERSISTENCE,
                 "trained_on": f"{len(train_rows)} healthy missions",
                 "train_seeds": [int(r["seed"]) for r in train_rows],
                 "val_seeds": [int(r["seed"]) for r in val_rows],
                 "sklearn_fpr_target": TARGET_FPR},
                MODELS_DIR / "isolation_forest.pkl")
    print(f"saved {MODELS_DIR / 'isolation_forest.pkl'} "
          f"(threshold {threshold:.4f})")


if __name__ == "__main__":
    main()
