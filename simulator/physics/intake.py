"""Handbook Part 7.3 -- Step 2: air mass flow."""

from .constants import R_AIR
from .engine_params import EngineGeometry, CalibrationConstants, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS


def air_mass_flow_kg_s(
    N_rpm: float,
    MAP_Pa: float,
    T_im_K: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    eta_v: float | None = None,
) -> float:
    """Total engine air mass flow rate. DERIVED formula; eta_v is ASSUMED
    (a full eta_v(N, MAP) map is a later refinement -- Handbook 7.3 notes
    this explicitly). Reproduces Handbook 7.3 worked example.
    """
    if eta_v is None:
        eta_v = constants.eta_v_nominal
    rho_im = MAP_Pa / (R_AIR * T_im_K)
    N_over_120 = N_rpm / 120.0  # four-stroke: one induction per 2 revs
    m_air = eta_v * geometry.total_displacement_m3 * N_over_120 * rho_im
    return m_air


def air_mass_per_cycle_per_cyl_kg(
    N_rpm: float,
    MAP_Pa: float,
    T_im_K: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    eta_v: float | None = None,
) -> float:
    """Air mass trapped in ONE cylinder on ONE induction stroke.
    Used by combustion.py to get per-cycle charge mass for Wiebe heat release."""
    if eta_v is None:
        eta_v = constants.eta_v_nominal
    rho_im = MAP_Pa / (R_AIR * T_im_K)
    return eta_v * geometry.displacement_per_cyl_m3 * rho_im
