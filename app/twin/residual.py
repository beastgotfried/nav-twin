"""twin/residual.py -- the residual engine (architecture.md Layer 3).

Per telemetry frame: run the NOMINAL physics twin (no faults), look up the
residual sigma band (input-uncertainty-only table plus sensor noise in
quadrature, see build_sigma_residual_table.py), and produce the normalised
residual z per channel per cylinder (Handbook 6.2).

Two design points that are easy to get wrong:

1. The twin applies the SAME first-order thermal lags (Handbook 7.8) to its
   own predictions that the real engine applies to physics. Steady-state
   predictions compared against lagging observations would produce large
   spurious z during every transient. The lag is part of the physics, so it
   is part of the prediction.

2. The frozen baseline correction delta (Handbook 6.6/6.7) is fitted here as
   one constant offset per channel per cylinder over an initial
   known-healthy window, then frozen forever. That is the MVP simplification
   of the Gaussian-process delta(x): our "engine" and our twin share the
   same physics code, so a constant captures the entire systematic part.
   The freeze is the load-bearing property: after freezing, no new
   systematic deviation can be explained away as model error.
"""

import numpy as np

from physics.engine_model import OperatingPoint, predict_steady_state
from physics.engine_params import DEFAULT_CONSTANTS
from physics.sigma_lookup import (sigma_egt_residual, sigma_cht_residual,
                                  SIGMA_P_OIL_PA, SIGMA_T_OIL_K)
from physics.thermal import first_order_lag_step

# Sensor noise on the OBSERVED side, added in quadrature with the propagated
# input uncertainty. Mirrors mission.NOISE_SIGMA (same ASSUMED values; the
# twin knows the sensor spec, that is legitimate).
_NOISE = {"EGT_K": 2.0, "CHT_K": 1.0, "p_oil_Pa": 5000.0, "T_oil_K": 1.0}


class ResidualEngine:
    """Stateful: carries the lagged predictions and the frozen baseline."""

    def __init__(self, n_cyl: int = 4):
        self.n_cyl = n_cyl
        self._lag_pred = None          # {"EGT_K": [...], "CHT_K": [...], "T_oil_K": float}
        self._baseline = []            # healthy-window (obs - pred) rows
        self.delta = None              # frozen baseline offsets (dict of arrays)
        self.t_last_s = None

    def reset(self):
        self.__init__(self.n_cyl)

    @property
    def calibrated(self) -> bool:
        return self.delta is not None

    def freeze_baseline(self):
        """Fit delta on the accumulated healthy window, then freeze
        (Handbook 6.6: fit on known-healthy data ONLY, then stop learning)."""
        if not self._baseline:
            return
        arr = np.array(self._baseline)   # rows: [EGT x4, CHT x4, p_oil, T_oil]
        d = np.median(arr, axis=0)
        self.delta = {
            "EGT_K": d[: self.n_cyl],
            "CHT_K": d[self.n_cyl: 2 * self.n_cyl],
            "p_oil_Pa": float(d[2 * self.n_cyl]),
            "T_oil_K": float(d[2 * self.n_cyl + 1]),
        }
        self._baseline = []

    def step(self, frame: dict) -> dict:
        """One telemetry frame in, one twin-state dict out (app README)."""
        dt = 1.0 if self.t_last_s is None else max(frame["t_s"] - self.t_last_s, 1e-6)
        self.t_last_s = frame["t_s"]

        op = OperatingPoint(
            N_rpm=frame["N_rpm"], MAP_Pa=frame["MAP_Pa"],
            altitude_m=frame["altitude_m"],
            fuel_flow_kg_s_per_cyl=list(frame["fuel_flow_kg_s_per_cyl"]),
            T_oil_K=frame["T_oil_K"])
        pred = predict_steady_state(op)
        t_amb = pred["atmosphere"]["T_amb_K"]
        tau = DEFAULT_CONSTANTS

        egt_ss = [c["EGT_K"] for c in pred["per_cylinder"]]
        cht_ss = [c["CHT_K"] for c in pred["per_cylinder"]]
        t_oil_ss = pred["T_oil_ss_K"]

        if self._lag_pred is None:
            self._lag_pred = {"EGT_K": egt_ss, "CHT_K": cht_ss, "T_oil_K": t_oil_ss}
        else:
            lp = self._lag_pred
            lp["EGT_K"] = [first_order_lag_step(v, s, tau.tau_egt_s, dt)
                           for v, s in zip(lp["EGT_K"], egt_ss)]
            lp["CHT_K"] = [first_order_lag_step(v, s, tau.tau_cht_s, dt)
                           for v, s in zip(lp["CHT_K"], cht_ss)]
            lp["T_oil_K"] = first_order_lag_step(lp["T_oil_K"], t_oil_ss,
                                                 tau.tau_oil_s, dt)

        lp = self._lag_pred
        row = (list(np.array(frame["EGT_K"]) - np.array(lp["EGT_K"]))
               + list(np.array(frame["CHT_K"]) - np.array(lp["CHT_K"]))
               + [frame["p_oil_Pa"] - pred["p_oil_Pa"],
                  frame["T_oil_K"] - lp["T_oil_K"]])
        if not self.calibrated:
            self._baseline.append(row)

        d = self.delta or {"EGT_K": [0.0] * self.n_cyl,
                           "CHT_K": [0.0] * self.n_cyl,
                           "p_oil_Pa": 0.0, "T_oil_K": 0.0}

        cylinders = []
        for i, c in enumerate(pred["per_cylinder"]):
            sig_e = float(np.hypot(sigma_egt_residual(c["phi"], t_amb),
                                   _NOISE["EGT_K"]))
            sig_c = float(np.hypot(
                sigma_cht_residual(frame["N_rpm"], frame["MAP_Pa"], c["phi"], t_amb),
                _NOISE["CHT_K"]))
            z_e = (frame["EGT_K"][i] - lp["EGT_K"][i] - d["EGT_K"][i]) / sig_e
            z_c = (frame["CHT_K"][i] - lp["CHT_K"][i] - d["CHT_K"][i]) / sig_c
            cylinders.append({
                "n": i + 1,
                "EGT_K": frame["EGT_K"][i], "CHT_K": frame["CHT_K"][i],
                "EGT_pred_K": lp["EGT_K"][i] + d["EGT_K"][i],
                "CHT_pred_K": lp["CHT_K"][i] + d["CHT_K"][i],
                "sigma_EGT_K": sig_e, "sigma_CHT_K": sig_c,
                "z_EGT": z_e, "z_CHT": z_c,
                # Forward-computed equivalence ratio, from fuel and air.
                # The mixture panel places cylinders on the hill with it.
                # Never inverted from EGT (that map is two-valued).
                "phi": c["phi"],
                "status": _status(max(abs(z_e), abs(z_c))),
            })

        z_p = ((frame["p_oil_Pa"] - pred["p_oil_Pa"] - d["p_oil_Pa"])
               / float(np.hypot(SIGMA_P_OIL_PA, _NOISE["p_oil_Pa"])))
        z_t = ((frame["T_oil_K"] - lp["T_oil_K"] - d["T_oil_K"])
               / float(np.hypot(SIGMA_T_OIL_K, _NOISE["T_oil_K"])))

        return {
            "t_s": frame["t_s"],
            "scenario": frame["scenario"],
            "inputs": {
                "N_rpm": frame["N_rpm"], "MAP_Pa": frame["MAP_Pa"],
                "MAP_commanded_Pa": frame.get("MAP_commanded_Pa", frame["MAP_Pa"]),
                "altitude_m": frame["altitude_m"],
                "fuel_flow_total_kg_h":
                    float(sum(frame["fuel_flow_kg_s_per_cyl"]) * 3600.0),
            },
            "cylinders": cylinders,
            "oil": {"p_Pa": frame["p_oil_Pa"], "T_K": frame["T_oil_K"],
                    "p_pred_Pa": pred["p_oil_Pa"] + d["p_oil_Pa"],
                    "T_pred_K": lp["T_oil_K"] + d["T_oil_K"],
                    "z_p": z_p, "z_T": z_t},
            "calibrated": self.calibrated,
        }


def _status(z_abs_max: float) -> str:
    if z_abs_max >= 3.0:
        return "warning"
    if z_abs_max >= 2.0:
        return "caution"
    return "nominal"
