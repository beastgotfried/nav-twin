"""Orchestrates atmosphere -> intake -> combustion -> cycle -> thermal -> oil
into one steady-state prediction for a given operating point. This is the
"physics twin" referred to throughout the handbook and vault: given
commanded/measured inputs, predict what every sensor SHOULD read.

Deliberately does not include fault injection or Monte Carlo uncertainty
here -- those are separate modules, built and verified against this one.
"""

from dataclasses import dataclass, field

from .atmosphere import isa_atmosphere
from .intake import air_mass_flow_kg_s, air_mass_per_cycle_per_cyl_kg
from .combustion import equivalence_ratio
from .cycle import egt_steady_state_K, simulate_cylinder_cycle, cht_steady_state_K
from .constants import R_AIR, GAMMA_HOT
from .oil import oil_pressure_Pa, oil_temp_steady_state_K
from .engine_params import EngineGeometry, CalibrationConstants, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS


@dataclass
class OperatingPoint:
    N_rpm: float
    MAP_Pa: float
    altitude_m: float
    fuel_flow_kg_s_per_cyl: list  # length == geometry.num_cylinders
    T_oil_K: float = 353.15          # 80 C, used for oil pressure at this instant
    bearing_clearance_m: float = None  # None -> nominal from constants


def predict_steady_state(
    op: OperatingPoint,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    run_full_cycle_for_cht: bool = True,
    per_cylinder_constants: dict = None,
) -> dict:
    """Predict what every monitored channel SHOULD read at steady state for
    this operating point. Per-cylinder EGT and CHT; scalar oil pressure and
    temperature (oil subsystem is not modelled per-cylinder).

    Set run_full_cycle_for_cht=False to skip the expensive crank-angle
    integration (useful for quick sweeps that only need EGT, e.g. the
    mixture-hill figure).

    per_cylinder_constants: optional {cylinder_number (1-indexed):
    CalibrationConstants} to override the shared `constants` for specific
    cylinders. This is how faults.py injects a single-cylinder fault
    (misfire, detonation, cooling degradation) without touching the
    other three cylinders, which stay on the nominal shared constants.
    """
    per_cylinder_constants = per_cylinder_constants or {}
    atm = isa_atmosphere(op.altitude_m)
    T_im_K = atm["T_amb_K"]  # ASSUMED: no intercooler modelled; intake temp == ambient temp

    total_air_kg_s = air_mass_flow_kg_s(op.N_rpm, op.MAP_Pa, T_im_K, geometry, constants)
    air_per_cyl_kg_s = total_air_kg_s / geometry.num_cylinders
    air_per_cycle_per_cyl_kg = air_mass_per_cycle_per_cyl_kg(op.N_rpm, op.MAP_Pa, T_im_K, geometry, constants)

    bearing_clearance_m = op.bearing_clearance_m
    if bearing_clearance_m is None:
        bearing_clearance_m = constants.bearing_clearance_nominal_m

    per_cylinder = []
    for i, m_f in enumerate(op.fuel_flow_kg_s_per_cyl):
        cyl_num = i + 1
        cyl_constants = per_cylinder_constants.get(cyl_num, constants)

        phi = equivalence_ratio(m_f, air_per_cyl_kg_s)
        egt_result = egt_steady_state_K(phi, T_im_K, atm["T_amb_K"], geometry, cyl_constants)
        egt_K = egt_result["EGT_ss_K"]

        cht_K = None
        p_max_Pa = None
        timing_correction_K = 0.0
        if run_full_cycle_for_cht:
            fuel_per_cycle_kg = m_f * 120.0 / op.N_rpm
            m_charge_kg = air_per_cycle_per_cyl_kg + fuel_per_cycle_kg
            cyc = simulate_cylinder_cycle(op.N_rpm, op.MAP_Pa, T_im_K, phi, m_charge_kg, geometry, cyl_constants)
            cht_K = cht_steady_state_K(cyc["Q_wall_per_cycle_J"], op.N_rpm, T_im_K, cyl_constants)
            p_max_Pa = cyc["p_max_Pa"]

            # Energy-conservation coupling for ignition-timing faults
            # (e.g. detonation): the algebraic Step-5 EGT route has no
            # concept of burn phasing on its own (Handbook 7.6 is a
            # deliberately timing-independent simplification), so a fault
            # that only advances ignition timing is otherwise invisible
            # to EGT. We do NOT replace the algebraic route with the
            # crank-angle EVO temperature -- verified separately (see
            # simulator commit notes) that T_evo does not reliably
            # reproduce the single-peaked mixture hill across the full
            # phi range, so using it as the primary EGT source would risk
            # the one shape that must never break. Instead: run a second,
            # nominal-timing reference cycle, and correct EGT by the
            # resulting change in wall heat loss, converted to an
            # equivalent temperature deficit. This is scoped strictly to
            # ignition_timing_deg_btdc deviations, so it cannot interfere
            # with misfire (combustion_efficiency_multiplier) or cooling
            # degradation (R_th), both already correct without it.
            if cyl_constants.ignition_timing_deg_btdc != constants.ignition_timing_deg_btdc:
                cyc_nominal = simulate_cylinder_cycle(
                    op.N_rpm, op.MAP_Pa, T_im_K, phi, m_charge_kg, geometry, constants
                )
                delta_Q_wall_J = cyc["Q_wall_per_cycle_J"] - cyc_nominal["Q_wall_per_cycle_J"]
                c_v = R_AIR / (GAMMA_HOT - 1.0)
                delta_T_K = delta_Q_wall_J / (m_charge_kg * c_v)
                timing_correction_K = -cyl_constants.k_egt * delta_T_K
                egt_K = egt_K + timing_correction_K

        per_cylinder.append({
            "cylinder": i + 1,
            "phi": phi,
            "EGT_K": egt_K,
            "T3_K_raw_combustion": egt_result["T3_K"],
            "timing_correction_K": timing_correction_K,
            "CHT_K": cht_K,
            "p_max_Pa": p_max_Pa,
        })

    p_oil_Pa = oil_pressure_Pa(op.N_rpm, op.T_oil_K, bearing_clearance_m, constants)
    T_oil_ss_K = oil_temp_steady_state_K(op.N_rpm, bearing_clearance_m, atm["T_amb_K"], constants=constants)

    return {
        "atmosphere": atm,
        "total_air_kg_s": total_air_kg_s,
        "per_cylinder": per_cylinder,
        "p_oil_Pa": p_oil_Pa,
        "T_oil_ss_K": T_oil_ss_K,
        "bearing_clearance_m": bearing_clearance_m,
    }
