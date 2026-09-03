"""ml_features.py -- feature extraction for the ML layer (roadmap W2).

One FeatureBuilder class, two call sites:

- offline: verify/training code feeds it rows from a corpus mission
  (simulator/ml_corpus.py output)
- online: twin/Twin feeds it live twin states

Same code both ways, so the model never sees a train/serve skew. This is
the single most common way ML layers silently break in production, and the
reason the builder owns the windowing rather than the corpus script.

Feature vector (36 dims), all derived from the twin's own outputs:

  0-9    current z per channel (z_EGT x4, z_CHT x4, z_p_oil, z_T_oil)
  10-19  60 s windowed mean of each z
  20-29  60 s windowed slope of each z (per second)
  30-31  per-cylinder spread of current z_EGT and z_CHT (max - min)
  32-35  context: N_rpm, MAP_Pa, altitude_m, commanded-achieved MAP gap,
         each scaled to O(1)

The 60 s window matches LAG_CONFIRM_S in twin/diagnose.py: roughly two CHT
time constants, the horizon over which a real combustion change must reach
the head (fault-signatures.md section 7). Features at time t use only data
up to and including t; verify_features.py proves the no-look-ahead property.
"""

from collections import deque

import numpy as np

WINDOW_S = 60.0

Z_NAMES = [f"z_egt_{i}" for i in range(1, 5)] + \
          [f"z_cht_{i}" for i in range(1, 5)] + ["z_p_oil", "z_t_oil"]
CTX_NAMES = ["ctx_n", "ctx_map", "ctx_alt", "ctx_map_gap"]
FEATURE_NAMES = (Z_NAMES + [f"zmean_{n}" for n in Z_NAMES]
                 + [f"zslope_{n}" for n in Z_NAMES]
                 + ["spread_z_egt", "spread_z_cht"] + CTX_NAMES)
N_FEATURES = len(FEATURE_NAMES)  # 36


def z_vector_from_state(state: dict) -> np.ndarray:
    """Twin state dict -> the 10-dim z vector, in Z_NAMES order."""
    return np.array([c["z_EGT"] for c in state["cylinders"]]
                    + [c["z_CHT"] for c in state["cylinders"]]
                    + [state["oil"]["z_p"], state["oil"]["z_T"]], dtype=float)


def ctx_vector_from_state(state: dict) -> np.ndarray:
    """Twin state dict -> the 4-dim scaled context vector."""
    i = state["inputs"]
    return np.array([i["N_rpm"] / 6000.0, i["MAP_Pa"] / 1e5,
                     i["altitude_m"] / 8000.0,
                     (i.get("MAP_commanded_Pa", i["MAP_Pa"]) - i["MAP_Pa"])
                     / 1e4], dtype=float)


class FeatureBuilder:
    """Rolling 60 s window over (z, ctx). push() one timestep, features()
    reads the current vector. Not valid until the window has enough points
    to estimate a slope (min 5); features() returns None before that."""

    def __init__(self, window_s: float = WINDOW_S, min_points: int = 5):
        self.window_s = window_s
        self.min_points = min_points
        self._buf = deque()   # (t_s, z(10), ctx(4))

    def reset(self):
        self._buf.clear()

    def push(self, t_s: float, z: np.ndarray, ctx: np.ndarray):
        self._buf.append((float(t_s), np.asarray(z, float),
                          np.asarray(ctx, float)))
        while self._buf and self._buf[0][0] < t_s - self.window_s:
            self._buf.popleft()

    def push_state(self, state: dict):
        self.push(state["t_s"], z_vector_from_state(state),
                  ctx_vector_from_state(state))

    def features(self):
        if len(self._buf) < self.min_points:
            return None
        ts = np.array([b[0] for b in self._buf])
        zs = np.array([b[1] for b in self._buf])     # (w, 10)
        ctx = self._buf[-1][2]
        z_now = zs[-1]
        z_mean = zs.mean(axis=0)
        t_rel = ts - ts[0]
        if np.ptp(t_rel) < 1e-9:
            slopes = np.zeros(zs.shape[1])
        else:
            # least-squares slope per channel, vectorised
            a = np.vstack([t_rel, np.ones_like(t_rel)]).T
            slopes = np.linalg.lstsq(a, zs, rcond=None)[0][0]
        spread_egt = float(z_now[:4].max() - z_now[:4].min())
        spread_cht = float(z_now[4:8].max() - z_now[4:8].min())
        return np.concatenate([z_now, z_mean, slopes,
                               [spread_egt, spread_cht], ctx])


def features_from_corpus_mission(d: dict):
    """Corpus npz dict -> (t_s array, feature matrix (n, 36) with leading
    None rows dropped, row index mapping). Used by all training scripts."""
    fb = FeatureBuilder()
    t_all = d["t_s"]
    n = len(t_all)
    z = np.hstack([d["z_EGT"], d["z_CHT"],
                   d["z_p_oil"][:, None], d["z_T_oil"][:, None]])
    ctx = np.column_stack([
        d["N_rpm"] / 6000.0, d["MAP_Pa"] / 1e5, d["altitude_m"] / 8000.0,
        (d["MAP_commanded_Pa"] - d["MAP_Pa"]) / 1e4])
    feats, t_out, idx = [], [], []
    for i in range(n):
        fb.push(t_all[i], z[i], ctx[i])
        f = fb.features()
        if f is not None:
            feats.append(f)
            t_out.append(t_all[i])
            idx.append(i)
    return np.array(t_out), np.array(feats), np.array(idx)
