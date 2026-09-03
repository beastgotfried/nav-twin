"""Handbook Part 7.9 -- Step 8: oil subsystem.

Reproduces the bearing-wear signature from 02-Research/fault-signatures.md
5: as bearing clearance opens and oil warms, pressure falls from BOTH the
viscosity-drop and the clearance-cubed leak term at once, which is why
worn bearings show good pressure cold and poor pressure hot.

Simplification, documented: uses a simple exponential (Andrade-type)
viscosity-temperature law rather than the full Vogel equation named in
Handbook 7.9. ASSUMED/EMPIRICAL constants throughout -- this subsystem
has the least first-principles grounding of the model and is the first
candidate for Phase 2 rig calibration.
"""

import math
from .engine_params import CalibrationConstants, DEFAULT_CONSTANTS


def oil_viscosity_Pa_s(T_oil_K: float, constants: CalibrationConstants = DEFAULT_CONSTANTS) -> float:
    """mu(T_oil). ASSUMED simplified exponential decay (see module docstring)."""
    dT = T_oil_K - constants.oil_mu_ref_T_K
    return constants.oil_mu_ref_pa_s * math.exp(-constants.oil_mu_decay_per_K * dT)


def oil_pump_pressure_Pa(N_rpm: float, constants: CalibrationConstants = DEFAULT_CONSTANTS) -> float:
    """p_pump(N). ASSUMED linear pump curve, clipped at the relief valve setting."""
    raw = constants.oil_pump_gain_pa_per_rpm * N_rpm
    return min(raw, constants.oil_pump_max_pa)


def oil_pressure_Pa(
    N_rpm: float,
    T_oil_K: float,
    bearing_clearance_m: float,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
) -> float:
    """Handbook 7.9: p_oil = p_pump(N) - k_leak*(c_bearing)^3/mu(T_oil).
    The cubed clearance term is DERIVED (lubrication theory: flow through
    a bearing gap scales with the cube of the gap). k_leak is ASSUMED,
    tuned so nominal clearance gives a plausible hot oil pressure."""
    mu = oil_viscosity_Pa_s(T_oil_K, constants)
    leak_term = constants.oil_leak_coeff * (bearing_clearance_m ** 3) / mu
    p = oil_pump_pressure_Pa(N_rpm, constants) - leak_term
    return max(p, 0.0)


def friction_heat_W(
    N_rpm: float,
    bearing_clearance_m: float,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
) -> float:
    """Q_fric(N, c_bearing). ASSUMED: friction heat rises with speed and
    with clearance growth beyond nominal (surrogate for increased
    metal-to-metal contact as a worn bearing loses its oil film)."""
    speed_factor = (N_rpm / 5000.0) ** 2
    wear_ratio = bearing_clearance_m / constants.bearing_clearance_nominal_m
    wear_factor = 1.0 + max(wear_ratio - 1.0, 0.0) * 3.0
    return constants.friction_heat_nominal_W * speed_factor * wear_factor


def oil_temp_steady_state_K(
    N_rpm: float,
    bearing_clearance_m: float,
    T_amb_K: float,
    Q_blowby_W: float = 0.0,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
) -> float:
    """Handbook 7.9: steady-state target for the oil temperature lag ODE.
    T_oil_ss = T_amb + (Q_fric + Q_blowby) / (m_oil * cooling_rate).
    ASSUMED: models oil-to-air cooling as a fixed effective heat-loss
    coefficient rather than a modelled oil cooler."""
    Q_fric = friction_heat_W(N_rpm, bearing_clearance_m, constants)
    cooling_coeff_W_per_K = 8.0  # ASSUMED effective oil cooler conductance
    return T_amb_K + (Q_fric + Q_blowby_W) / cooling_coeff_W_per_K
