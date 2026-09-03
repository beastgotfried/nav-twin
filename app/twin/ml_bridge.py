"""twin/ml_bridge.py -- the learned layer, bolted onto the twin (roadmap W6).

Design contract:

- The rule-based pipeline (residual -> band -> diagnose) is UNCHANGED. This
  module adds a parallel learned view under the state's "ml" key. If the
  model artifacts are absent, available=False and the twin behaves exactly
  as before, which is what keeps verify_twin.py green on a fresh clone.
- The forest runs every step. The classifier fires only when the forest's
  persisted flag is active (the two-stage design, ml-layer.md section 2).
- RUL tracking follows a model diagnosis: one particle filter per
  (kind, cylinder) subsystem, updated while that diagnosis leads. Bounded
  relative index only, per ml_rul.py's framing.
- Evidence strings come from the artifact's per-class attribution (mean
  |SHAP| or its labelled fallback) combined with the current feature
  values. This is population-level attribution, said plainly; per-instance
  SHAP at 1 Hz is a later refinement.

Nothing here is imported, and no model is loaded, unless the artifacts
exist, so the pure-Python twin core keeps its no-heavy-deps property on a
fresh checkout.
"""

import sys
from pathlib import Path

import numpy as np

_APP = str(Path(__file__).resolve().parent.parent)
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from ml_features import (FeatureBuilder, z_vector_from_state,  # noqa: E402
                         ctx_vector_from_state, FEATURE_NAMES)

_MODELS = None


def _load_models():
    """Lazy global load. Returns dict of artifacts or None."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS or None
    from pathlib import Path
    md = Path(__file__).resolve().parent.parent / "models"
    try:
        import joblib
        arts = {}
        for name in ("isolation_forest", "classifier", "rul"):
            p = md / f"{name}.pkl"
            if p.exists():
                arts[name] = joblib.load(p)
        _MODELS = arts if "isolation_forest" in arts else {}
    except Exception:
        _MODELS = {}
    return _MODELS or None


class MLLayer:
    """Stateful per-twin learned view. One per Twin instance."""

    def __init__(self):
        arts = _load_models()
        self.available = arts is not None
        self.arts = arts or {}
        self.fb = FeatureBuilder()
        self._flag_run = 0
        self._filters = {}      # (kind, cylinder) -> ParticleFilter
        self._lead = None       # current leading (kind, cylinder)

    def reset(self):
        self.fb.reset()
        self._flag_run = 0
        self._filters = {}
        self._lead = None

    def step(self, state: dict) -> dict:
        self.fb.push_state(state)
        if not self.available:
            return {"available": False}

        # The learned layer respects the same calibration boundary as the
        # rules: before the frozen baseline exists, z values are
        # uncalibrated startup transients and must not be scored
        # (verify_ml.py caught the forest alarming at t=6 s otherwise).
        if not state.get("calibrated", False):
            return {"available": True, "note": "calibrating"}

        out = {"available": True}
        feats = self.fb.features()
        if feats is None:
            out["note"] = "warming up, window not full"
            return out

        # --- forest ---
        fa = self.arts["isolation_forest"]
        score = float(fa["model"].score_samples(feats[None, :])[0])
        raw = score < fa["threshold"]
        self._flag_run = self._flag_run + 1 if raw else 0
        flag = self._flag_run >= fa["persistence"]
        out["anomaly_score"] = score
        out["anomaly_flag"] = bool(flag)

        # --- classifier, only when the forest flags ---
        if flag and "classifier" in self.arts:
            ca = self.arts["classifier"]
            proba = ca["model"].predict_proba(feats[None, :])[0]
            order = np.argsort(-proba)[:3]
            diag = []
            for rank, ci in enumerate(order):
                cls = ca["classes"][ci]
                attr = ca["attribution"][ci]
                top = sorted(((float(attr[j]), FEATURE_NAMES[j], int(j))
                              for j in np.argsort(-attr)[:5]),
                             reverse=True)
                ev = [f"{name} = {feats[j]:+.2f}" for _, name, j in top]
                diag.append({"rank": rank + 1, "label": cls,
                             "confidence": round(float(proba[ci]), 2),
                             "evidence": ev, "source": "model",
                             "attribution": ca["attribution_method"]})
            out["diagnosis"] = diag
            top = diag[0]
            kind_cyl = self._parse_class(top["label"])
            if top["confidence"] >= 0.4 and kind_cyl is not None:
                self._lead = kind_cyl
        else:
            self._lead = None

        # --- RUL tracking on the leading diagnosis ---
        if self._lead and "rul" in self.arts:
            out["rul"] = self._rul_step(self._lead, feats)
        return out

    def _parse_class(self, label):
        kind = label.split("_cyl")[0]
        kind = kind.replace("sensor_drift_EGT_K", "sensor_drift") \
                   .replace("sensor_drift_CHT_K", "sensor_drift")
        if kind == "sensor_drift":
            return None          # not physical degradation, by design
        cyl = int(label.rsplit("_cyl", 1)[1]) if "_cyl" in label else 0
        if kind not in self.arts["rul"]["observation"]:
            return None
        return (kind, cyl)

    def _rul_step(self, lead, feats):
        from ml_rul import ParticleFilter, health_index
        kind, cyl = lead
        p = self.arts["rul"]["observation"][kind]
        if lead not in self._filters:
            self._filters[lead] = ParticleFilter(p["g"], b=p.get("b", 0.0), sigma_log=p.get("sigma_log"),
                                                 seed=42)
        pf = self._filters[lead]
        est = pf.update(health_index(feats, kind, cyl))
        proj = pf.project()
        out = {"subsystem": f"{kind}" + (f" cyl {cyl}" if cyl else ""),
               "severity_median": est["median"],
               "severity_p10": est["p10"], "severity_p90": est["p90"],
               "framing": self.arts["rul"]["framing"]}
        if proj:
            out["projection"] = proj
        return out
