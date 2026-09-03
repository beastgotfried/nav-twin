"""
Engine geometry and calibration constants for the target engine class
(Rotax 915 iS / 916 iS class, fuel-injected, FADEC-equipped -- see
02-Research/engine-rotax.md for why this class was chosen over the 914).

Every field is tagged DERIVED or ASSUMED/EMPIRICAL per Handbook 7.10.
The ASSUMED/EMPIRICAL values are physically reasoned placeholders, not
measurements. They are exactly the parameters (theta) that Monte Carlo
uncertainty propagation samples over, and exactly what Phase 2 rig
calibration would replace with fitted values. Do not present them as
measured constants.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EngineGeometry:
    """DERIVED from published total displacement; bore/stroke split is
    ASSUMED (a square-engine simplification) since the manufacturer does
    not publish bore and stroke separately in the sources we verified."""

    num_cylinders: int = 4
    total_displacement_m3: float = 1352e-6      # 1352 cc, verified (02-Research/engine-rotax.md)
    compression_ratio: float = 10.0              # ASSUMED, typical for this engine class
    rod_to_crank_ratio: float = 3.3               # ASSUMED, typical small aero engine

    @property
    def displacement_per_cyl_m3(self) -> float:
        return self.total_displacement_m3 / self.num_cylinders

    @property
    def bore_m(self) -> float:
        # ASSUMED square engine (bore == stroke): V_d = (pi/4) * B^2 * S = (pi/4) * B^3
        return (self.displacement_per_cyl_m3 / (math.pi / 4.0)) ** (1.0 / 3.0)

    @property
    def stroke_m(self) -> float:
        return self.bore_m  # square engine assumption

    @property
    def clearance_volume_m3(self) -> float:
        # V_c = V_d_cyl / (r - 1)
        return self.displacement_per_cyl_m3 / (self.compression_ratio - 1.0)

    @property
    def crank_radius_m(self) -> float:
        return self.stroke_m / 2.0

    @property
    def rod_length_m(self) -> float:
        return self.rod_to_crank_ratio * self.crank_radius_m

    @property
    def piston_area_m2(self) -> float:
        return (math.pi / 4.0) * self.bore_m ** 2


@dataclass(frozen=True)
class CalibrationConstants:
    """ASSUMED/EMPIRICAL unless noted. This is exactly the theta vector of
    Handbook 6.6 (Kennedy-O'Hagan): the parameters whose uncertainty Monte
    Carlo propagation samples over, and what Phase 2 rig calibration fits."""

    # Step 5 (EGT, algebraic route)
    k_egt: float = 0.52              # blowdown/port-loss lumping factor, dimensionless, ASSUMED.
                                       # Tuned via verify_sanity_checks.py so EGT_ss lands at
                                       # 774-786 C at phi=1.05, matching the 700-820 C range real
                                       # piston aircraft EGT gauges read in cruise. Still a proxy
                                       # for real calibration data, not a measurement -- Phase 2.

    # Step 6 (CHT, crank-angle route)
    woschni_C: float = 3.26           # Woschni coefficient, ASSUMED (Heywood-typical order of magnitude)
    wiebe_a: float = 5.0               # Wiebe efficiency parameter, EMPIRICAL (standard textbook value)
    wiebe_m: float = 2.0               # Wiebe form factor, EMPIRICAL (standard textbook value)
    ignition_timing_deg_btdc: float = 20.0   # start of combustion, ASSUMED default
    burn_duration_deg_at_peak: float = 40.0   # combustion duration at flame-speed peak (phi=1.1), ASSUMED
    flame_speed_width_b: float = 5.0   # width of S_L(phi) Gaussian, ASSUMED
    R_th: float = 0.02                 # K per (J/s), lumped head thermal resistance, ASSUMED
    combustion_efficiency_multiplier: float = 1.0   # fault-injection hook, see combustion.py.
                                                       # 1.0=nominal, 0.0=complete misfire.

    # Step 7 (thermal lag)
    tau_egt_s: float = 1.5             # DERIVED range midpoint (Handbook 7.8: "1 to 2 s")
    tau_cht_s: float = 20.0            # DERIVED range midpoint (Handbook 7.8: "10 to 30 s")
    tau_oil_s: float = 60.0            # ASSUMED, oil thermal mass is larger than metal head

    # Step 2 (volumetric efficiency, constant baseline; a full map is a later refinement)
    eta_v_nominal: float = 0.85        # ASSUMED, typical turbocharged SI engine

    # Step 8 (oil subsystem)
    oil_pump_gain_pa_per_rpm: float = 900.0     # ASSUMED linear pump curve slope
    oil_pump_max_pa: float = 550_000.0           # ASSUMED pump relief pressure, ~5.5 bar
    oil_leak_coeff: float = 8.96e16               # ASSUMED. Recalibrated via
                                                     # verify_fault_signatures.py: clearance^3 is
                                                     # ~1e-14 m^3, so the coefficient must be this
                                                     # large to produce a physically meaningful
                                                     # pressure drop. Original placeholder (2.5e-4)
                                                     # was off by ~12 orders of magnitude and
                                                     # produced zero visible effect from bearing wear.
                                                     # Tuned so nominal clearance + hot oil gives
                                                     # ~480 kPa (healthy), 3x clearance drives it
                                                     # toward the pump floor (failed bearing).
    oil_mu_ref_pa_s: float = 0.02                # ASSUMED, reference viscosity at T_ref
    oil_mu_ref_T_K: float = 353.15                # 80 C reference temperature
    oil_mu_decay_per_K: float = 0.035             # ASSUMED exponential viscosity-temperature decay rate
    bearing_clearance_nominal_m: float = 25e-6    # ASSUMED nominal bearing clearance, 25 micron
    friction_heat_nominal_W: float = 300.0        # ASSUMED nominal friction heat at cruise rpm
    oil_mass_kg: float = 4.0                       # ASSUMED oil system mass
    oil_cp_J_per_kgK: float = 2000.0               # DERIVED, typical mineral/synthetic oil specific heat


DEFAULT_GEOMETRY = EngineGeometry()
DEFAULT_CONSTANTS = CalibrationConstants()
