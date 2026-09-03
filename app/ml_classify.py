"""ml_classify.py -- supervised fault classifier (roadmap W4).

HistGradientBoosting over the 36 windowed features, one row per
fault-active timestep, classes = fault kind x cylinder (sensor drift also
carries its channel). Fires only after an anomaly, so there is deliberately
no "healthy" class: the two-stage design of ml-layer.md section 2.

Split BY MISSION, never by timestep: consecutive rows of one flight are
heavily correlated and would leak between train and test.

Attribution: SHAP TreeExplainer when it supports the fitted estimator;
falls back to permutation importance on a validation subsample and says so
in the artifact. The evidence strings the dashboard shows come from this
attribution, which is why the fallback exists rather than a hard failure.

Artifact: app/models/classifier.pkl (joblib dict: model, classes,
attribution metadata, per-class expected signature features, split seeds).
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_APP))

from ml_features import features_from_corpus_mission, FEATURE_NAMES  # noqa: E402

MODELS_DIR = _APP / "models"
WARMUP_S = 5.0       # rows within onset+5s are excluded (transient onset)
EVERY_NTH = 2        # subsample timesteps to decorrelate and halve size

# Researched signature channels per fault kind (fault-signatures.md),
# expressed over FEATURE_NAMES. Used by verify_classify.py for the
# signature-consistency gate and by the integration layer for evidence.
def expected_features(kind, cylinder=0, channel=""):
    cyl = max(int(cylinder), 1)
    egt = [f"z_egt_{cyl}", f"zmean_z_egt_{cyl}", f"zslope_z_egt_{cyl}"]
    cht = [f"z_cht_{cyl}", f"zmean_z_cht_{cyl}", f"zslope_z_cht_{cyl}"]
    oil = ["z_p_oil", "z_t_oil", "zmean_z_p_oil", "zmean_z_t_oil",
           "zslope_z_p_oil", "zslope_z_t_oil"]
    table = {
        "misfire": egt + cht,
        "injector_restriction": egt + cht,
        "detonation": cht + egt,
        "cooling_degradation": cht,
        "bearing_wear": oil,
        "turbo_degradation": ["ctx_map_gap"],
        "sensor_drift": egt if channel == "EGT_K" else cht,
    }
    return table.get(kind, [])


def class_name(r):
    kind = r["kind"]
    if kind == "sensor_drift":
        return f"sensor_drift_{r['sensor_channel']}_cyl{r['cylinder']}"
    if int(r["cylinder"]) >= 1:
        return f"{kind}_cyl{r['cylinder']}"
    return kind


def load_fault_dataset(corpus_root, rows):
    """Features + labels + mission ids for the fault corpus."""
    X, y, mid = [], [], []
    for k, r in enumerate(rows):
        d = np.load(Path(corpus_root) / r["file"])
        onset = float(r["onset_s"])
        t, F, idx = features_from_corpus_mission(d)
        keep = (t > onset + WARMUP_S) & (d["fault_active"][idx] > 0)
        keep &= (np.arange(len(t)) % EVERY_NTH == 0)
        if keep.sum() == 0:
            continue
        X.append(F[keep])
        y += [class_name(r)] * int(keep.sum())
        mid.append(np.full(int(keep.sum()), r["seed"]))
    return np.vstack(X), np.array(y), np.concatenate(mid)


def split_by_mission(y, mid, seed=0):
    """70/15/15 train/val/test, whole missions in one split only,
    class-stratified at the mission level."""
    rng = np.random.default_rng(seed)
    missions = np.unique(mid)
    mission_class = {m: y[mid == m][0] for m in missions}
    splits = {"train": set(), "val": set(), "test": set()}
    for cls in np.unique(y):
        ms = np.array([m for m in missions if mission_class[m] == cls])
        rng.shuffle(ms)
        n = len(ms)
        if n >= 5:
            n_tr = int(round(0.7 * n))
            n_va = int(round(0.15 * n))
        elif n >= 3:
            n_tr, n_va = n - 2, 1     # small class: 1 val, 1 test
        elif n == 2:
            n_tr, n_va = 1, 0         # tiny class: 1 train, 1 test
        else:
            n_tr, n_va = n, 0         # single mission: train only
        splits["train"] |= set(ms[:n_tr])
        splits["val"] |= set(ms[n_tr:n_tr + n_va])
        splits["test"] |= set(ms[n_tr + n_va:])
    return {k: np.isin(mid, list(v)) for k, v in splits.items()}


def compute_attribution(model, X_val, y_val, classes, X_fallback=None):
    """Mean |importance| per feature per class. SHAP TreeExplainer if it
    accepts the estimator, else permutation importance (same vector shape
    per class via one-per-class binary relevance), honestly labelled.

    If the validation split is empty or tiny (small corpora put every
    mission of a class into train/test), attribution falls back to a
    training subsample; SHAP on an empty set silently produced an all-NaN
    matrix, which the signature-consistency gate then misread."""
    if len(X_val) < 50 and X_fallback is not None and len(X_fallback) > 0:
        print("note: validation split too small for attribution "
              f"({len(X_val)} rows); using a training subsample")
        X_val = X_fallback[:2000]
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_val[:2000])
        # shap returns (n, f) for binary, list[(n, f)] or (n, f, k) for multi
        if isinstance(sv, list):
            arr = np.stack([np.abs(s).mean(axis=0) for s in sv])
        else:
            arr = np.abs(sv)
            if arr.ndim == 3:      # (n, features, classes)
                arr = arr.mean(axis=0).T
            else:                  # (n, features); binary
                arr = arr.mean(axis=0)[None, :]
        if arr.shape[0] == len(classes):
            if np.isnan(arr).any():
                raise ValueError("NaN in SHAP attribution")
            return arr, "shap.TreeExplainer"
        raise ValueError(f"unexpected shap shape {arr.shape}")
    except Exception as e:  # fall back, and say so in the artifact
        print(f"note: SHAP unavailable for this estimator ({e}); "
              "using permutation importance fallback")
        from sklearn.inspection import permutation_importance
        arr = np.zeros((len(classes), X_val.shape[1]))
        for ci, cls in enumerate(classes):
            y_bin = (y_val == cls).astype(int)
            if y_bin.sum() < 10 or y_bin.sum() > len(y_bin) - 10:
                continue
            r = permutation_importance(model, X_val[:4000],
                                       y_bin[:4000], n_repeats=3,
                                       random_state=0, scoring="f1")
            arr[ci] = r.importances_mean
        return arr, "permutation_importance (SHAP fallback)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    args = ap.parse_args()

    import csv
    with open(Path(args.corpus) / "index.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] == "fault"]
    if len(rows) < 30:
        raise SystemExit(f"need at least 30 fault missions, found "
                         f"{len(rows)}; is the corpus finished?")

    t0 = time.time()
    X, y, mid = load_fault_dataset(args.corpus, rows)
    print(f"fault dataset: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes ({time.time()-t0:.0f}s)")

    splits = split_by_mission(y, mid)
    Xtr, ytr = X[splits["train"]], y[splits["train"]]
    Xva, yva = X[splits["val"]], y[splits["val"]]
    print(f"split by mission: {len(ytr)} train / {len(yva)} val / "
          f"{splits['test'].sum()} test rows")

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=400, early_stopping=True, validation_fraction=0.1,
        random_state=0)
    model.fit(Xtr, ytr)
    print(f"fit in {time.time()-t0:.0f}s")

    classes = list(model.classes_)
    attr, attr_method = compute_attribution(model, Xva, yva, classes,
                                            X_fallback=Xtr)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "classes": classes,
        "feature_names": FEATURE_NAMES,
        "attribution": attr,
        "attribution_method": attr_method,
        "expected": {c: expected_features(
            c.split("_cyl")[0].replace("sensor_drift_EGT_K", "sensor_drift")
            .replace("sensor_drift_CHT_K", "sensor_drift"),
            int(c.rsplit("_cyl", 1)[1]) if "_cyl" in c else 0,
            "EGT_K" if "EGT_K" in c else ("CHT_K" if "CHT_K" in c else ""))
            for c in classes},
        "test_seeds": sorted({str(s) for s in mid[splits["test"]]}),
        "train_rows": len(ytr),
    }, MODELS_DIR / "classifier.pkl")
    print(f"saved {MODELS_DIR / 'classifier.pkl'} "
          f"(attribution: {attr_method})")


if __name__ == "__main__":
    main()
