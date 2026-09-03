"""Handbook Part 7.4 (Step 3, equivalence ratio) and 3.4/3.6 (Step 4, heat
release and flame speed). This is the load-bearing module: the mixture hill
(EGT non-monotonic in phi) falls out of `heat_release_per_kg_charge`, and the
CHT/EGT peak offset falls out of `laminar_flame_speed` peaking richer than
heat release does. Neither is curve-fitted to produce that shape; see
Handbook 3.4-3.6 for the derivation."""

import math
from .constants import Q_LHV, FA_STOICH
from .engine_params import CalibrationConstants, DEFAULT_CONSTANTS


def equivalence_ratio(m_fuel_kg_s: float, m_air_kg_s: float) -> float:
    """phi. Handbook 3.3/7.4. FORWARD ONLY -- never invert this from a
    temperature reading (Handbook 3.5: the EGT(phi) map is two-valued)."""
    if m_air_kg_s <= 0:
        return float("inf")
    fa_actual = m_fuel_kg_s / m_air_kg_s
    return fa_actual / FA_STOICH


def combustion_efficiency(phi: float) -> float:
    """Handbook 3.4. DERIVED: oxygen-limited combustion efficiency."""
    if phi <= 1.0:
        return 1.0
    return 1.0 / phi


def heat_release_per_kg_charge(phi: float, combustion_efficiency_multiplier: float = 1.0) -> float:
    """q(phi), J per kg of charge (air + fuel). Handbook 3.4 / 7.5.

    DERIVED structure (min(phi,1) oxygen-limiting term); Q_LHV and
    FA_STOICH are published fuel properties, not fitted. This is the
    function whose peak at phi ~ 1 produces the exhaust-temperature hill
    -- verify this shape numerically before trusting anything downstream.

    combustion_efficiency_multiplier is a SEPARATE factor from the
    phi-driven oxygen-limiting term above: it represents fault-induced
    combustion loss (e.g. a plug not firing) independent of mixture
    strength. 1.0 = nominal. 0.0 = complete misfire, no heat release
    regardless of phi. This is the fault-injection hook for Table 2's
    "misfire: combustion efficiency of that cylinder set to zero"
    -- see faults.py.
    """
    numerator = Q_LHV * FA_STOICH * min(phi, 1.0) * combustion_efficiency_multiplier
    denominator = 1.0 + phi * FA_STOICH
    return numerator / denominator


def laminar_flame_speed(
    phi: float,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    S_L_max: float = 0.4,
) -> float:
    """S_L(phi), m/s. Handbook 3.6. ASSUMED Gaussian approximation centred
    at phi=1.1 (real flame speed is not exactly Gaussian, but this
    reproduces the qualitative peak-shift that drives the CHT/EGT offset).
    S_L_max is ASSUMED (typical SI gasoline peak laminar flame speed order
    of magnitude)."""
    b = constants.flame_speed_width_b
    return S_L_max * math.exp(-b * (phi - 1.1) ** 2)


def burn_duration_deg(
    phi: float,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    S_L_max: float = 0.4,
) -> float:
    """delta_theta(phi), crank degrees. Handbook 3.6: faster flame -> shorter
    burn -> combustion completes nearer TDC -> higher peak pressure ->
    more heat into the head. This is the mechanism behind the CHT/EGT
    peak offset, not the EGT peak itself."""
    S_L = laminar_flame_speed(phi, constants, S_L_max)
    S_L = max(S_L, 1e-6)  # avoid divide-by-zero at extreme mixtures
    return constants.burn_duration_deg_at_peak * S_L_max / S_L


def wiebe_burn_fraction(theta_deg: float, theta0_deg: float, delta_theta_deg: float,
                         constants: CalibrationConstants = DEFAULT_CONSTANTS) -> float:
    """x_b(theta). Handbook 3.6 / 7.6. EMPIRICAL (standard Wiebe function,
    a and m are textbook-standard values, not fitted to our engine)."""
    if theta_deg < theta0_deg:
        return 0.0
    a, m = constants.wiebe_a, constants.wiebe_m
    x = (theta_deg - theta0_deg) / delta_theta_deg
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-a * x ** (m + 1))


def wiebe_burn_rate_per_deg(theta_deg: float, theta0_deg: float, delta_theta_deg: float,
                             constants: CalibrationConstants = DEFAULT_CONSTANTS) -> float:
    """dx_b/dtheta, analytic derivative of the Wiebe function. Used to drive
    the crank-angle-resolved heat release rate in cycle.py."""
    if theta_deg < theta0_deg or delta_theta_deg <= 0:
        return 0.0
    a, m = constants.wiebe_a, constants.wiebe_m
    x = (theta_deg - theta0_deg) / delta_theta_deg
    if x <= 0:
        return 0.0
    return (a * (m + 1) / delta_theta_deg) * (x ** m) * math.exp(-a * x ** (m + 1))
