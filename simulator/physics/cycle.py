"""Handbook Part 7.6/7.7 -- Step 5 (EGT, algebraic air-standard route) and
Step 6 (CHT, crank-angle-resolved Woschni/Wiebe route).

EGT uses the simple lumped air-standard cycle from Handbook 7.6. As the
handbook itself warns, this overpredicts raw combustion temperature (T3)
by roughly 1000-1500 K versus real peak cylinder temperature, because it
ignores temperature-dependent specific heat, dissociation and in-cylinder
heat loss during combustion. k_egt and the residual/discrepancy-term
architecture exist specifically to absorb this -- see Handbook 6.6 and
7.6. This is flagged, not hidden.

CHT requires knowing peak cylinder pressure, which the algebraic route
cannot give, so it needs an actual crank-angle-resolved pressure trace:
compression, Wiebe-phased combustion, expansion, with Woschni wall heat
transfer integrated along the way (Handbook 3.6/7.7).
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from .constants import R_AIR, GAMMA_HOT
from .engine_params import EngineGeometry, CalibrationConstants, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from .combustion import heat_release_per_kg_charge, wiebe_burn_rate_per_deg, burn_duration_deg


# ---------------------------------------------------------------------------
# Step 5: EGT, algebraic air-standard route (Handbook 7.6)
# ---------------------------------------------------------------------------

def egt_steady_state_K(
    phi: float,
    T_im_K: float,
    T_amb_K: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    gamma: float = GAMMA_HOT,
) -> dict:
    """T2 (compression) -> T3 (combustion) -> T4 (expansion) -> EGT_ss.
    Returns intermediate temperatures too, so the sanity check can report
    the raw T3 overprediction the handbook warns about."""
    r = geometry.compression_ratio
    c_v = R_AIR / (gamma - 1.0)

    T2 = T_im_K * r ** (gamma - 1.0)
    q = heat_release_per_kg_charge(phi, constants.combustion_efficiency_multiplier)
    T3 = T2 + q / c_v
    T4 = T3 * r ** (-(gamma - 1.0))
    egt_ss = T_amb_K + constants.k_egt * (T4 - T_amb_K)

    return {"T2_K": T2, "T3_K": T3, "T4_K": T4, "EGT_ss_K": egt_ss, "q_J_per_kg": q}


# ---------------------------------------------------------------------------
# Step 6: CHT, crank-angle-resolved cycle with Woschni + Wiebe (Handbook 7.7)
# ---------------------------------------------------------------------------

def cylinder_volume_m3(theta_rad: float, geometry: EngineGeometry) -> float:
    """Slider-crank volume relation. theta=0 at TDC. DERIVED kinematics."""
    a = geometry.crank_radius_m
    Rr = geometry.rod_to_crank_ratio
    B = geometry.bore_m
    piston_area = math.pi / 4.0 * B ** 2
    s = a * (Rr + 1.0 - math.cos(theta_rad) - math.sqrt(Rr ** 2 - math.sin(theta_rad) ** 2))
    return geometry.clearance_volume_m3 + piston_area * s


def cylinder_volume_derivative_m3_per_rad(theta_rad: float, geometry: EngineGeometry) -> float:
    """dV/dtheta, theta in radians. DERIVED kinematics (analytic derivative
    of the slider-crank volume relation)."""
    a = geometry.crank_radius_m
    Rr = geometry.rod_to_crank_ratio
    B = geometry.bore_m
    piston_area = math.pi / 4.0 * B ** 2
    sin_t, cos_t = math.sin(theta_rad), math.cos(theta_rad)
    ds_dtheta = a * (sin_t + (sin_t * cos_t) / math.sqrt(max(Rr ** 2 - sin_t ** 2, 1e-9)))
    return piston_area * ds_dtheta


def _woschni_C1(theta_deg: float, theta0_deg: float) -> float:
    """Heywood's published split of the Woschni C1 constant (compression
    vs. combustion/expansion, swirl-free). DERIVED (published values),
    not fitted. The full Woschni correlation also has a C2 term
    involving a motored-pressure reference trace during combustion; we
    omit it (documented simplification -- see module docstring and
    02-Research/sources/openwam.md for the Phase 2 higher-fidelity path)."""
    return 2.28 if theta_deg < theta0_deg else 6.18


def simulate_cylinder_cycle(
    N_rpm: float,
    MAP_Pa: float,
    T_im_K: float,
    phi: float,
    m_charge_per_cycle_kg: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    n_points: int = 720,
) -> dict:
    """Integrate cylinder pressure from BDC (start of compression, theta=-180)
    through TDC (theta=0) to BDC (theta=+180, end of expansion), using a
    single-zone closed-system energy balance with Wiebe combustion and
    Woschni wall heat loss.

    Returns p_max_Pa, T_max_K, and total wall heat loss per cycle (J),
    which feeds the CHT lumped-thermal-resistance calculation.

    Simplifications, all documented: single effective gamma for the whole
    cycle (Handbook 7.6 makes the same simplification for EGT); Woschni
    C2 (pressure-rise) term omitted; ideal-gas R_gas held at R_AIR
    throughout despite changing gas composition after combustion.
    """
    theta0_deg = -constants.ignition_timing_deg_btdc  # e.g. -20 = 20 BTDC
    delta_theta_deg = burn_duration_deg(phi, constants)
    q = heat_release_per_kg_charge(phi, constants.combustion_efficiency_multiplier)
    Q_total = q * m_charge_per_cycle_kg  # J, whole-cycle heat release

    omega = 2.0 * math.pi * N_rpm / 60.0  # rad/s
    gamma = GAMMA_HOT
    R_gas = R_AIR
    mean_piston_speed = 2.0 * geometry.stroke_m * N_rpm / 60.0

    piston_area = geometry.piston_area_m2
    head_area = piston_area  # ASSUMED: flat head approximation, head area == bore area

    theta_start = math.radians(-180.0)
    theta_end = math.radians(180.0)

    # initial condition at BDC (start of compression): treat MAP/T_im as
    # trapped conditions at IVC (a simplification -- ignores intake ram/loss
    # dynamics between MAP and in-cylinder pressure at IVC).
    V0 = cylinder_volume_m3(theta_start, geometry)
    p0 = MAP_Pa
    T0 = T_im_K

    def rhs(theta_rad, y):
        p = y[0]
        theta_deg = math.degrees(theta_rad)
        V = cylinder_volume_m3(theta_rad, geometry)
        dV_dtheta = cylinder_volume_derivative_m3_per_rad(theta_rad, geometry)
        T = max(p * V / (m_charge_per_cycle_kg * R_gas), 200.0)  # guard against non-physical T

        # combustion heat release rate, per radian
        dxb_ddeg = wiebe_burn_rate_per_deg(theta_deg, theta0_deg, delta_theta_deg, constants)
        dQcomb_dtheta = Q_total * dxb_ddeg * (180.0 / math.pi)

        # Woschni wall heat transfer
        C1 = _woschni_C1(theta_deg, theta0_deg)
        w = C1 * mean_piston_speed
        w = max(w, 0.1)
        p_kPa = max(p / 1000.0, 1e-3)
        h_g = constants.woschni_C * geometry.bore_m ** (-0.2) * p_kPa ** 0.8 * T ** (-0.53) * w ** 0.8

        s = max((V - geometry.clearance_volume_m3) / piston_area, 0.0)
        wall_area = piston_area + head_area + math.pi * geometry.bore_m * s
        T_wall = 450.0  # K, ASSUMED representative wall/head metal temperature
        dQwall_dt = h_g * wall_area * (T - T_wall)  # W
        dQwall_dtheta = dQwall_dt / omega  # J/rad

        dp_dtheta = (-gamma * p / V) * dV_dtheta + (gamma - 1.0) / V * (dQcomb_dtheta - dQwall_dtheta)
        return [dp_dtheta]

    theta_eval = np.linspace(theta_start, theta_end, n_points)
    sol = solve_ivp(rhs, [theta_start, theta_end], [p0], t_eval=theta_eval,
                     method="RK45", rtol=1e-6, atol=1.0, max_step=math.radians(1.0))

    if not sol.success:
        raise RuntimeError(f"cylinder cycle integration failed: {sol.message}")

    p_trace = sol.y[0]
    theta_deg_trace = np.degrees(theta_eval)
    V_trace = np.array([cylinder_volume_m3(t, geometry) for t in theta_eval])
    T_trace = np.clip(p_trace * V_trace / (m_charge_per_cycle_kg * R_gas), 200.0, None)

    # integrate wall heat loss over the whole cycle for the CHT calculation
    Q_wall_total = 0.0
    for i in range(len(theta_eval) - 1):
        theta_deg = theta_deg_trace[i]
        p = p_trace[i]
        T = T_trace[i]
        C1 = _woschni_C1(theta_deg, theta0_deg)
        w = max(C1 * mean_piston_speed, 0.1)
        p_kPa = max(p / 1000.0, 1e-3)
        h_g = constants.woschni_C * geometry.bore_m ** (-0.2) * p_kPa ** 0.8 * T ** (-0.53) * w ** 0.8
        s = max((V_trace[i] - geometry.clearance_volume_m3) / piston_area, 0.0)
        wall_area = piston_area + head_area + math.pi * geometry.bore_m * s
        dQwall_dt = h_g * wall_area * (T - 450.0)
        dt = (theta_eval[i + 1] - theta_eval[i]) / omega
        Q_wall_total += dQwall_dt * dt

    i_max = int(np.argmax(p_trace))
    return {
        "p_max_Pa": float(p_trace[i_max]),
        "T_max_K": float(T_trace[i_max]),
        "theta_at_pmax_deg": float(theta_deg_trace[i_max]),
        "Q_wall_per_cycle_J": float(Q_wall_total),
        "theta_deg": theta_deg_trace,
        "p_trace_Pa": p_trace,
        "T_trace_K": T_trace,
    }


def cht_steady_state_K(
    Q_wall_per_cycle_J: float,
    N_rpm: float,
    T_cool_K: float,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
) -> float:
    """Handbook 7.7: CHT_ss = T_cool + Q_wall_rate * R_th.
    Converts per-cycle wall heat loss into an average heat transfer rate
    (W) using cycle frequency, then applies the lumped thermal resistance."""
    cycles_per_sec = N_rpm / 120.0  # one power cycle per 2 revs
    Q_wall_rate_W = Q_wall_per_cycle_J * cycles_per_sec
    return T_cool_K + Q_wall_rate_W * constants.R_th
