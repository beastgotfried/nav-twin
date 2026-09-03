"""ml_rul.py -- bounded degradation tracking and RUL projection (roadmap W5).

Per ml-layer.md section 4: health indices derived from residual trends,
tracked by a particle filter under an exponential degradation model. The
particle spread IS the uncertainty; forward projection gives a spread of
crossing times, reported as bounds.

NON-NEGOTIABLE FRAMING (00-STREAM section 7): the output is a bounded
relative degradation index with confidence bounds, never a certified
time-to-failure. The growth-rate prior used for projection is ASSUMED, not
measured; the artifact says so in its own fields.

Subsystems and their observation signals (health indices), all computed
from the 60 s windowed z means already in the feature pipeline:

  per-cylinder combustion (misfire, injector, detonation, cooling):
      HI = max(|zmean_z_egt_c|, |zmean_z_cht_c|)
  bearing:  HI = max(|zmean_z_p_oil|, |zmean_z_t_oil|)
  turbo:    HI = ctx_map_gap (commanded-achieved MAP, in 10 kPa units)

Sensor drift is excluded by design: it is not physical degradation.

Model: hidden severity x >= 0, dynamics x *= exp(beta*dt) (x=0 stays 0,
so the filter seeds x from a small positive prior), observation
HI = g*x + eps. g and eps are FITTED from corpus ramped trajectories
(least squares through the origin per fault kind); beta is an ASSUMED
prior range, sampled per particle during projection.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_APP))

from ml_features import FEATURE_NAMES  # noqa: E402

MODELS_DIR = _APP / "models"

# ASSUMED growth-rate prior for projection (per hour). Not measured; the
# particle spread over this prior is what makes the RUL bound honest.
BETA_PRIOR_PER_HR = (np.log(2) / 100.0, np.log(2) / 10.0)  # doubling 10-100 h
FAILURE_X = 1.0            # severity scale: 1.0 is the corpus's worst case
N_PARTICLES = 500

RUL_KINDS = ("misfire", "injector_restriction", "detonation",
             "cooling_degradation", "bearing_wear", "turbo_degradation")


def health_index(feat_row, kind, cylinder=0):
    """HI from one 36-dim feature row. Feature indices resolved by name so
    this survives feature-list edits."""
    def idx(name):
        return FEATURE_NAMES.index(name)
    if kind in ("misfire", "injector_restriction", "detonation",
                "cooling_degradation"):
        c = max(int(cylinder), 1)
        return max(abs(feat_row[idx(f"zmean_z_egt_{c}")]),
                   abs(feat_row[idx(f"zmean_z_cht_{c}")]))
    if kind == "bearing_wear":
        return max(abs(feat_row[idx("zmean_z_p_oil")]),
                   abs(feat_row[idx("zmean_z_t_oil")]))
    if kind == "turbo_degradation":
        return float(feat_row[idx("ctx_map_gap")])
    raise ValueError(f"no health index for {kind!r}")


class ParticleFilter:
    """Severity tracker for one subsystem. See module docstring for the
    model. All outputs are distributions, never points.

    Heteroscedastic likelihood in LOG space: HI noise is multiplicative
    (proportional scatter across operating points), so the observation
    model is log(HI) = log(b + g*x) + eps with constant sigma_log. A linear
    Gaussian with severity-scaled sigma was tried first and fails
    structurally: with honest fault-end scatter (+-60% of the mean), a
    severity-0.5 state sits within 1.7 sigma of a healthy observation, so
    teleported high particles survive, resampling drift fills the cloud
    upward, and the median floats to ~0.5 on a healthy engine
    (verify_rul.py gate 2 caught it). In log space the same comparison is
    12 sigma apart, and the filter localises correctly at both ends."""

    def __init__(self, g, sigma_obs=None, b=0.0, sigma0=None, sigma1=None,
                 sigma_log=None, n=N_PARTICLES, seed=0):
        self.g = float(g)
        self.b = float(b)
        if sigma_log is None:
            # rough conversion for legacy artifacts: relative scatter
            sigma_log = max(0.1, (sigma_obs or 1.0) / max(abs(g), 1e-6))
        self.sigma_log = float(max(sigma_log, 0.05))
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.uniform(1e-4, 0.02, n)

    def _loglik(self, hi):
        hi_c = max(float(hi), 0.02)
        mu = self.b + self.g * self.x
        return -0.5 * ((np.log(hi_c) - np.log(np.maximum(mu, 1e-9)))
                       / self.sigma_log) ** 2

    def update(self, hi, dt_s=1.0):
        beta = self.rng.uniform(*BETA_PRIOR_PER_HR, len(self.x)) / 3600.0
        self.x *= np.exp(beta * dt_s)
        self.x *= self.rng.lognormal(0.0, 0.05, len(self.x))  # process noise
        # Innovation-adaptive rejuvenation in the same log space.
        med = float(np.median(self.x))
        innov = abs(np.log(max(hi, 0.02))
                    - np.log(max(self.b + self.g * med, 1e-9))) \
            / self.sigma_log
        p_jump = 0.02 if innov < 2.0 else 0.30
        jump = self.rng.random(len(self.x)) < p_jump
        self.x[jump] = self.rng.uniform(0.0, 1.05, int(jump.sum()))
        # log-likelihood, shifted by its max before exp so a fully
        # inconsistent cloud still resamples discriminatively
        logw = self._loglik(hi)
        logw -= logw.max()
        w = np.exp(logw) + 1e-12
        w /= w.sum()
        ess = 1.0 / (w ** 2).sum()
        if ess < len(self.x) / 2:
            cum = np.cumsum(w)
            u0 = self.rng.uniform(0, 1 / len(self.x))
            u = u0 + np.arange(len(self.x)) / len(self.x)
            self.x = self.x[np.searchsorted(cum, u)]
        return self.estimate()

    def estimate(self):
        return {"median": float(np.median(self.x)),
                "p10": float(np.percentile(self.x, 10)),
                "p90": float(np.percentile(self.x, 90))}

    def project(self):
        """Forward-project each particle to FAILURE_X under the beta prior.
        Returns crossing-time bounds in hours, or None if the state is too
        healthy to project. Bounded relative index, not a certified time."""
        med = np.median(self.x)
        if med < 0.05:
            return None
        beta = self.rng.uniform(*BETA_PRIOR_PER_HR, len(self.x))
        with np.errstate(divide="ignore"):
            t_hr = np.log(FAILURE_X / np.maximum(self.x, 1e-9)) / beta
        return {"t_to_failure_hr_p10": float(np.percentile(t_hr, 10)),
                "t_to_failure_hr_median": float(np.median(t_hr)),
                "t_to_failure_hr_p90": float(np.percentile(t_hr, 90)),
                "framing": "bounded relative degradation index, assumed "
                           "growth prior, not a certified time-to-failure"}


def fit_observation_model(corpus_root, rows):
    """Fit the observation model per fault kind: HI = b + g*x + eps.

    b is the healthy HI floor (mean over healthy missions): a real health
    index is not zero on a healthy engine, and without modelling the floor
    the filter has no likelihood gradient below it and floats upward,
    crying degradation on healthy data (caught by verify_rul.py gate 2).
    g is least squares through the origin on HI - b against the known
    severity_t; sigma_obs is the residual std over both populations.
    """
    from ml_features import features_from_corpus_mission

    # healthy floor per subsystem family, from healthy missions
    healthy_rows = [r for r in rows if r["label"] == "healthy"]
    # healthy floor AND its scatter per subsystem family, from healthy
    # missions.
    floor_hi = {k: [] for k in ("cyl", "bearing_wear", "turbo_degradation")}
    for r in healthy_rows[:40]:
        d = np.load(Path(corpus_root) / r["file"])
        t, F, idx = features_from_corpus_mission(d)
        floor_hi["cyl"].append(np.array([
            health_index(f, "misfire", 1) for f in F[::5]]))
        floor_hi["bearing_wear"].append(np.array([
            health_index(f, "bearing_wear") for f in F[::5]]))
        floor_hi["turbo_degradation"].append(np.array([
            health_index(f, "turbo_degradation") for f in F[::5]]))
    floors = {k: float(np.concatenate(v).mean()) if v else 0.0
              for k, v in floor_hi.items()}

    params = {}
    for kind in RUL_KINDS:
        fam = "cyl" if kind in ("misfire", "injector_restriction",
                                "detonation", "cooling_degradation") else kind
        b = floors[fam]
        kind_rows = [r for r in rows if r["kind"] == kind]
        sev_all, hi_all = [], []
        for r in kind_rows[:60]:   # enough for a stable fit, keeps it fast
            d = np.load(Path(corpus_root) / r["file"])
            t, F, idx = features_from_corpus_mission(d)
            sev = d["severity_t"][idx]
            his = np.array([health_index(f, kind, int(r["cylinder"]))
                            for f in F])
            # only rows where the 60 s HI window is saturated with the
            # current severity: past onset + ramp + half the window,
            # otherwise the fit confuses ramp-up lag with small severity
            settled = t > (float(r["onset_s"]) + float(r["ramp_s"])
                           + 30.0)
            mask = (sev > 0.02) & settled
            if mask.sum() > 5:
                sev_all.append(sev[mask])
                hi_all.append(his[mask])
        if not sev_all:
            continue
        s = np.concatenate(sev_all)
        h = np.concatenate(hi_all) - b
        g = float((s * h).sum() / (s * s).sum())
        resid = h - g * s
        # multiplicative scatter: sigma_log is the std of the log-ratio
        # residual, the space the filter's likelihood lives in
        log_resid = np.log(np.maximum(h + b, 0.02)) - \
            np.log(np.maximum(b + g * s, 1e-9))
        params[kind] = {"g": g, "b": b,
                        "sigma_log": float(log_resid.std() + 1e-3),
                        "sigma_obs": float(resid.std() + 1e-6),
                        "n_points": int(len(s))}
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "corpus"))
    args = ap.parse_args()

    import csv
    with open(Path(args.corpus) / "index.csv") as fh:
        rows = list(csv.DictReader(fh))   # fit needs healthy rows too,
                                          # for the HI floor and sigma0

    params = fit_observation_model(args.corpus, rows)
    for k, p in sorted(params.items()):
        print(f"{k:24s} g={p['g']:.3f}  sigma_obs={p['sigma_obs']:.3f} "
              f"(n={p['n_points']})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"observation": params,
                 "beta_prior_per_hr": BETA_PRIOR_PER_HR,
                 "failure_x": FAILURE_X,
                 "framing": "bounded relative degradation index with "
                            "confidence bounds; growth prior ASSUMED, not "
                            "measured; not a certified time-to-failure"},
                MODELS_DIR / "rul.pkl")
    print(f"saved {MODELS_DIR / 'rul.pkl'}")


if __name__ == "__main__":
    main()
