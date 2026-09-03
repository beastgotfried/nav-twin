"""
Handbook Part 12 -- crankshaft torque and the order-domain misfire signature.

This module computes the ONE thing in the order-spectrum story that our
physics model can actually derive: gas torque from cylinder pressure, and
the half-order (0.5x) speed fluctuation a single-cylinder misfire produces
in a 4-cylinder engine. Everything else conventionally shown on an order
spectrum (1x unbalance, 2x misalignment, bearing defect frequencies) is a
STRUCTURAL/mechanical phenomenon our combustion-only model has no way to
produce -- see verify_order_spectrum.py and 09-Visuals/fig_order_spectrum.py
for how that distinction is kept visible rather than blurred.

Core relation (DERIVED, standard IC engine result, not fitted): the work
done by cylinder gas pressure over a crank angle increment equals torque
times that angle increment, so

    T_gas(theta) = p(theta) * dV/dtheta

Both p(theta) and dV/dtheta already exist and are verified elsewhere in
this package (cycle.simulate_cylinder_cycle, cycle.cylinder_volume_derivative_m3_per_rad).
This module adds nothing new to the thermodynamics -- it only combines
what is already there across four phased cylinders and looks at the
result in the frequency domain.

Documented simplification: only the closed portion of the cycle
(compression + power, -180 to +180 relative to each cylinder's own TDC)
is modelled, matching Handbook 7.7. Torque contribution during a
cylinder's own intake/exhaust strokes is treated as zero. This is
standard practice in simplified crank-torque/misfire literature (gas
exchange strokes contribute comparatively little torque ripple next to
compression/power) and is a documented omission, not a hidden one.
"""

import math
import numpy as np

from .engine_params import EngineGeometry, CalibrationConstants, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from .cycle import cylinder_volume_derivative_m3_per_rad, simulate_cylinder_cycle

# ASSUMED, ill-constrained pending Phase 2 rig data. Not a thermodynamic
# calibration constant (hence not in CalibrationConstants) -- this is a
# crankshaft/flywheel rotational inertia, a different physical subsystem.
I_CRANK_KGM2_DEFAULT = 0.02  # ASSUMED, typical order of magnitude for a small 4-cyl crank + flywheel


def single_cylinder_torque_trace(
    N_rpm: float, MAP_Pa: float, T_im_K: float, phi: float, m_charge_per_cycle_kg: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY, constants: CalibrationConstants = DEFAULT_CONSTANTS,
    n_points: int = 361,
) -> dict:
    """One cylinder's gas torque over its own -180..+180 degree window
    (relative to ITS TDC). Reuses simulate_cylinder_cycle's already-verified
    pressure trace; adds only T = p * dV/dtheta."""
    cyc = simulate_cylinder_cycle(N_rpm, MAP_Pa, T_im_K, phi, m_charge_per_cycle_kg,
                                    geometry, constants, n_points=n_points)
    theta_deg = cyc["theta_deg"]
    p_trace = cyc["p_trace_Pa"]
    dV_dtheta = np.array([cylinder_volume_derivative_m3_per_rad(math.radians(t), geometry) for t in theta_deg])
    torque = p_trace * dV_dtheta  # N*m, DERIVED (T = p * dV/dtheta)
    return {"theta_deg_rel": theta_deg, "torque_Nm": torque}


def combined_four_cylinder_torque(
    healthy_trace: dict,
    n_cycles: int,
    faulted_cylinder: int = None,
    faulted_trace: dict = None,
    firing_phase_deg: float = 180.0,
) -> dict:
    """Sum four phase-shifted copies of a single-cylinder torque trace into
    the combined crankshaft torque over n_cycles full 720-degree cycles.

    firing_phase_deg=180 matches a conventional 4-cylinder 4-stroke firing
    interval (four evenly-spaced power strokes per 720 degrees). If
    faulted_cylinder is given (1-4), that cylinder's contribution uses
    faulted_trace instead of healthy_trace -- this is how a misfire enters
    the combined signal: one of the four periodic pulses is replaced,
    which is exactly what breaks the healthy 180-degree periodicity and
    introduces energy at half the firing order.
    """
    theta_rel = healthy_trace["theta_deg_rel"]
    window = 180.0  # each cylinder's trace is nonzero only within +/-180 deg of its own TDC

    total_deg = 720.0 * n_cycles
    n_samples_per_deg = 2  # 0.5-degree resolution, matches simulate_cylinder_cycle's default density
    theta_global = np.arange(0, total_deg, 1.0 / n_samples_per_deg)
    torque_total = np.zeros_like(theta_global)

    for k in range(4):
        cyl_num = k + 1
        trace = faulted_trace if (faulted_cylinder == cyl_num) else healthy_trace
        theta_tdc_base = k * firing_phase_deg  # this cylinder's TDC within one 720-deg cycle

        for cycle_i in range(n_cycles):
            theta_tdc = theta_tdc_base + 720.0 * cycle_i
            rel = theta_global - theta_tdc
            in_window = np.abs(rel) <= window
            if not np.any(in_window):
                continue
            torque_total[in_window] += np.interp(rel[in_window], theta_rel, trace["torque_Nm"])

    return {"theta_global_deg": theta_global, "torque_Nm": torque_total, "n_cycles": n_cycles}


def angular_velocity_fluctuation(
    torque_signal: dict, N_rpm: float, I_crank_kgm2: float = I_CRANK_KGM2_DEFAULT,
) -> dict:
    """Small-perturbation crankshaft speed fluctuation from torque ripple.

    I * omega_mean * domega/dtheta = T(theta) - T_mean
    -> domega/dtheta = (T(theta) - T_mean) / (I * omega_mean)
    -> omega(theta) = omega_mean + cumulative_integral(domega/dtheta)

    Standard small-perturbation approximation used in real misfire-detection
    literature (the crankshaft's actual speed variation is a few percent of
    nominal, which is exactly the regime this linearisation is valid in).
    ASSUMED: I_crank_kgm2 is not measured -- see module docstring.
    """
    theta_deg = torque_signal["theta_global_deg"]
    torque = torque_signal["torque_Nm"]
    omega_mean = 2.0 * math.pi * N_rpm / 60.0  # rad/s

    T_mean = np.mean(torque)
    domega_dtheta = (torque - T_mean) / (I_crank_kgm2 * omega_mean)  # rad/s per radian
    dtheta_rad = np.radians(np.diff(theta_deg, prepend=theta_deg[0] - (theta_deg[1] - theta_deg[0])))
    domega = domega_dtheta * dtheta_rad
    omega_fluct = np.cumsum(domega)
    omega_fluct -= np.mean(omega_fluct)  # remove any residual DC drift from the cumulative sum

    return {"theta_global_deg": theta_deg, "omega_fluct_rad_s": omega_fluct, "omega_mean_rad_s": omega_mean}


def order_spectrum(signal: np.ndarray, n_cycles: int, samples_per_deg: float = 2.0) -> dict:
    """FFT-based order-domain spectrum of a signal sampled UNIFORMLY IN
    CRANK ANGLE (not time) -- this is what makes the frequency axis land
    directly in "orders" (cycles per shaft revolution) rather than needing
    a separate resampling/order-tracking step, since angle-domain sampling
    at a fixed points-per-degree rate is already angle-synchronous.

    order = 1.0 means "once per crankshaft revolution". order = 0.5 is
    the half-order misfire signature (Handbook 12.1): once per TWO
    revolutions, because a given cylinder in a four-stroke engine fires
    once every two revolutions.
    """
    n = len(signal)
    revolutions_total = n_cycles * 2.0  # each 720-degree engine cycle = 2 crankshaft revolutions
    windowed = signal * np.hanning(n)  # standard technique, reduces spectral leakage from a finite window
    spectrum = np.abs(np.fft.rfft(windowed))
    orders = np.fft.rfftfreq(n, d=revolutions_total / n)
    return {"orders": orders, "amplitude": spectrum}
