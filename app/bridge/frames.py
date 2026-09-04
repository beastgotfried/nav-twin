"""bridge/frames.py -- the frame contract and its validation.

One function matters: normalize(payload) -> internal frame dict. Every
adapter (mavlink, native, anything future) funnels through here, so the
twin sees exactly one shape and every unit conversion lives in exactly
one place.

Internal frame (twin/residual.py consumes this):
    t_s: float                    seconds since mission/engine start
    scenario: str                 free-form source tag, shown in the UI
    N_rpm, MAP_Pa, altitude_m:    operating point
    fuel_flow_kg_s_per_cyl: [4]   per-cylinder fuel flow
    EGT_K, CHT_K: [4]             per-cylinder temperatures
    p_oil_Pa, T_oil_K:            oil subsystem
"""

import math

FRAME_FIELDS = ("t_s", "scenario", "N_rpm", "MAP_Pa", "altitude_m",
                "fuel_flow_kg_s_per_cyl", "EGT_K", "CHT_K",
                "p_oil_Pa", "T_oil_K")

_N_CYL = 4


class FrameError(ValueError):
    """Raised for any payload that cannot become a valid frame. Carries a
    plain-language reason; the API surfaces it verbatim."""


def _num(v, name, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FrameError(f"{name} must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise FrameError(f"{name} must be finite, got {v!r}")
    if lo is not None and not lo <= v <= hi:
        raise FrameError(f"{name} {v} outside sane range [{lo}, {hi}]")
    return v


def _cyl_array(v, name, lo, hi):
    if not isinstance(v, (list, tuple)) or len(v) != _N_CYL:
        raise FrameError(f"{name} must be a list of {_N_CYL} values")
    return [_num(x, f"{name}[{i}]", lo, hi) for i, x in enumerate(v)]


def normalize(d: dict) -> dict:
    """Validate a fully-formed internal frame. Units are SI as listed in
    the module docstring; sane-range checks catch unit mistakes (Kelvin
    vs Celsius is the classic) rather than feeding them to the twin."""
    if not isinstance(d, dict):
        raise FrameError("frame must be an object")
    out = {
        "t_s": _num(d.get("t_s"), "t_s", 0.0, 1e7),
        "scenario": str(d.get("scenario", "external")),
        "N_rpm": _num(d.get("N_rpm"), "N_rpm", 0.0, 20000.0),
        "MAP_Pa": _num(d.get("MAP_Pa"), "MAP_Pa", 5000.0, 300000.0),
        "altitude_m": _num(d.get("altitude_m"), "altitude_m",
                           -500.0, 20000.0),
        "fuel_flow_kg_s_per_cyl": _cyl_array(
            d.get("fuel_flow_kg_s_per_cyl"), "fuel_flow_kg_s_per_cyl",
            0.0, 0.05),
        "EGT_K": _cyl_array(d.get("EGT_K"), "EGT_K", 250.0, 1500.0),
        "CHT_K": _cyl_array(d.get("CHT_K"), "CHT_K", 250.0, 700.0),
        "p_oil_Pa": _num(d.get("p_oil_Pa"), "p_oil_Pa", 0.0, 2e6),
        "T_oil_K": _num(d.get("T_oil_K"), "T_oil_K", 250.0, 500.0),
    }
    if "MAP_commanded_Pa" in d and d["MAP_commanded_Pa"] is not None:
        out["MAP_commanded_Pa"] = _num(d["MAP_commanded_Pa"],
                                       "MAP_commanded_Pa", 5000.0, 300000.0)
    return out
