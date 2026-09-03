"""Handbook Part 6 -- Monte Carlo uncertainty propagation.

Implements the four sources of uncertainty from Handbook 6.5 EXCEPT
model-form error (source 4, the Kennedy-O'Hagan discrepancy term delta(x)
from Handbook 6.6): that needs healthy telemetry across the operating
envelope to fit against, which needs a mission-profile generator that
doesn't exist yet. This module covers sources 1-3: sensor noise, input
uncertainty, and parameter uncertainty (theta).

Why Monte Carlo and not linearisation: Handbook 6.3 argues linearisation
fails at the mixture peak because dEGT/dphi = 0 there, so first-order
propagation reports zero contribution from mixture uncertainty at exactly
the operating point where the model is least distinguishable from its
neighbours. verify_uncertainty.py checks this claim directly, it is not
asserted here.

Deliberate scoping decision, and the reason it matters: EGT (Handbook 7.6,
algebraic route) and CHT (Handbook 7.7, crank-angle route) do NOT share
the same set of calibration constants -- checked directly against the
source before writing this module, not assumed. EGT depends on exactly
ONE calibration constant (k_egt). CHT depends on seven (R_th,
ignition_timing_deg_btdc, woschni_C, burn_duration_deg_at_peak,
flame_speed_width_b, wiebe_a, wiebe_m). Perturbing parameters that don't
appear in the equation being propagated would be either wasted computation
or, worse, silently wrong if a future refactor accidentally makes them
matter without updating this file. So the two propagation functions below
sample deliberately different, disjoint parameter sets, not one blanket
"perturb everything" sampler.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from .engine_params import EngineGeometry, CalibrationConstants, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from .cycle import egt_steady_state_K, simulate_cylinder_cycle, cht_steady_state_K
from .combustion import equivalence_ratio


@dataclass(frozen=True)
class UncertaintySpec:
    """ASSUMED relative standard deviations (fraction of nominal value),
    pending Phase 2 rig calibration data. The point of this module is the
    propagation machinery and the qualitative band-breathing behaviour it
    produces, not these specific magnitudes -- exactly the theta/input
    split described in Handbook 6.5.

    Only fields actually consumed by the equation they perturb exist here;
    see module docstring.
    """

    # EGT-relevant parameter (theta)
    k_egt_rel: float = 0.15

    # CHT-relevant parameters (theta)
    R_th_rel: float = 0.20
    woschni_C_rel: float = 0.15
    wiebe_a_rel: float = 0.10
    wiebe_m_rel: float = 0.10
    ignition_timing_rel: float = 0.10
    burn_duration_rel: float = 0.15
    flame_speed_width_rel: float = 0.20

    # Shared input uncertainty (u)
    T_amb_sigma_K: float = 2.0     # ASSUMED sensor noise, absolute
    phi_rel: float = 0.03           # ASSUMED fuel-flow measurement uncertainty,
                                     # applied directly to phi as a proxy for
                                     # joint fuel-flow/air-flow measurement error


DEFAULT_UNCERTAINTY = UncertaintySpec()


def _perturb(rng: np.random.Generator, nominal: float, rel_sigma: float) -> float:
    """One relative-normal draw, clipped to stay positive."""
    return max(nominal * (1.0 + rng.normal(0.0, rel_sigma)), nominal * 1e-3)


def _summarize(samples: np.ndarray) -> dict:
    return {
        "samples": samples,
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples, ddof=1)),
        "p2_5": float(np.percentile(samples, 2.5)),
        "p50": float(np.percentile(samples, 50.0)),
        "p97_5": float(np.percentile(samples, 97.5)),
    }


def propagate_egt_uncertainty(
    phi_nominal: float,
    T_im_K: float,
    T_amb_K_nominal: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    spec: UncertaintySpec = DEFAULT_UNCERTAINTY,
    n_samples: int = 3000,
    rng: np.random.Generator = None,
) -> dict:
    """Empirical predictive distribution of EGT at one operating point.
    Cheap (algebraic route, Handbook 7.6): thousands of samples in well
    under a second, matching Handbook 6.3's "a few thousand draws per
    timestep is cheap" claim -- that claim is specifically about this
    route, not the crank-angle one (see propagate_cht_uncertainty).
    """
    rng = rng or np.random.default_rng()
    egts = np.empty(n_samples)
    for i in range(n_samples):
        k_egt_s = _perturb(rng, constants.k_egt, spec.k_egt_rel)
        T_amb_s = T_amb_K_nominal + rng.normal(0.0, spec.T_amb_sigma_K)
        phi_s = _perturb(rng, phi_nominal, spec.phi_rel)
        c_s = dataclasses.replace(constants, k_egt=k_egt_s)
        r = egt_steady_state_K(phi_s, T_im_K, T_amb_s, geometry, c_s)
        egts[i] = r["EGT_ss_K"]
    return _summarize(egts)


def propagate_cht_uncertainty(
    N_rpm: float,
    MAP_Pa: float,
    T_im_K: float,
    T_amb_K_nominal: float,
    phi_nominal: float,
    m_charge_per_cycle_kg: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    spec: UncertaintySpec = DEFAULT_UNCERTAINTY,
    n_samples: int = 200,
    rng: np.random.Generator = None,
) -> dict:
    """Empirical predictive distribution of CHT at one operating point.
    EXPENSIVE (crank-angle ODE integration per sample, Handbook 7.7):
    n_samples defaults much lower than the EGT route for this reason.
    This is exactly why Handbook 6.8 specifies fitting an offline
    surrogate for deployment rather than running Monte Carlo online --
    this function IS the offline side of that architecture.
    """
    rng = rng or np.random.default_rng()
    chts = np.empty(n_samples)
    for i in range(n_samples):
        c_s = dataclasses.replace(
            constants,
            R_th=_perturb(rng, constants.R_th, spec.R_th_rel),
            woschni_C=_perturb(rng, constants.woschni_C, spec.woschni_C_rel),
            wiebe_a=_perturb(rng, constants.wiebe_a, spec.wiebe_a_rel),
            wiebe_m=_perturb(rng, constants.wiebe_m, spec.wiebe_m_rel),
            ignition_timing_deg_btdc=_perturb(rng, constants.ignition_timing_deg_btdc, spec.ignition_timing_rel),
            burn_duration_deg_at_peak=_perturb(rng, constants.burn_duration_deg_at_peak, spec.burn_duration_rel),
            flame_speed_width_b=_perturb(rng, constants.flame_speed_width_b, spec.flame_speed_width_rel),
        )
        T_amb_s = T_amb_K_nominal + rng.normal(0.0, spec.T_amb_sigma_K)
        phi_s = _perturb(rng, phi_nominal, spec.phi_rel)
        cyc = simulate_cylinder_cycle(N_rpm, MAP_Pa, T_im_K, phi_s, m_charge_per_cycle_kg, geometry, c_s, n_points=240)
        chts[i] = cht_steady_state_K(cyc["Q_wall_per_cycle_J"], N_rpm, T_amb_s, c_s)
    return _summarize(chts)


def linearized_egt_std(
    phi_nominal: float,
    T_im_K: float,
    T_amb_K_nominal: float,
    geometry: EngineGeometry = DEFAULT_GEOMETRY,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    spec: UncertaintySpec = DEFAULT_UNCERTAINTY,
    h: float = 1e-4,
) -> float:
    """The delta-method comparison from Handbook 6.3: first-order
    (linearised) uncertainty propagation via finite-difference Jacobian.
    Exists ONLY so verify_uncertainty.py can demonstrate it failing at the
    mixture peak, where dEGT/dphi = 0 -- this function is not used
    anywhere else in the codebase and should not be trusted for anything.
    """
    def egt_of(phi, T_amb, k_egt):
        c = dataclasses.replace(constants, k_egt=k_egt)
        return egt_steady_state_K(phi, T_im_K, T_amb, geometry, c)["EGT_ss_K"]

    base = egt_of(phi_nominal, T_amb_K_nominal, constants.k_egt)

    d_phi = (egt_of(phi_nominal + h, T_amb_K_nominal, constants.k_egt) - base) / h
    d_Tamb = (egt_of(phi_nominal, T_amb_K_nominal + h, constants.k_egt) - base) / h
    d_kegt = (egt_of(phi_nominal, T_amb_K_nominal, constants.k_egt + h) - base) / h

    sigma_phi = phi_nominal * spec.phi_rel
    sigma_Tamb = spec.T_amb_sigma_K
    sigma_kegt = constants.k_egt * spec.k_egt_rel

    variance = (d_phi * sigma_phi) ** 2 + (d_Tamb * sigma_Tamb) ** 2 + (d_kegt * sigma_kegt) ** 2
    return float(np.sqrt(variance))


def normalized_residual(observed: float, predicted_mean: float, predicted_std: float) -> float:
    """Handbook 6.2: z = (observed - predicted) / sigma_predicted(operating point).
    The actual output object the whole architecture builds toward -- this
    is what feeds the anomaly detector, not the raw residual."""
    if predicted_std <= 0:
        raise ValueError("predicted_std must be positive")
    return (observed - predicted_mean) / predicted_std
