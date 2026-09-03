"""
Verifies the one thing in the order-spectrum story that is actually
computed rather than cited: does a single-cylinder misfire genuinely
produce a half-order (0.5x) spike in simulated crankshaft speed, absent
in healthy operation? Handbook 12.1 argues this from first principles
(a cylinder fires once every two revolutions, so removing one pulse from
four breaks the 180-degree periodicity at half the firing frequency).
This script is where that argument gets tested against an actual
simulated signal, not just reasoned about.

Also checks an EXTERNAL fact, not just internal consistency: a healthy
4-cylinder 4-stroke engine's torque ripple should be dominated by order 2
and its harmonics (four evenly-spaced power pulses per two revolutions =
twice per revolution), which is textbook-established NVH knowledge, not
something we derived ourselves. If our simulation doesn't reproduce that,
something in the torque/order machinery is wrong regardless of what the
misfire check shows.

Run: python verify_order_spectrum.py
"""

import sys
import numpy as np

from physics.crankdynamics import (
    single_cylinder_torque_trace, combined_four_cylinder_torque,
    angular_velocity_fluctuation, order_spectrum,
)
from physics.intake import air_mass_per_cycle_per_cyl_kg
from physics.atmosphere import isa_atmosphere
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.constants import FA_STOICH
import dataclasses

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


ALT_M, N_RPM, MAP_PA, PHI = 3000.0, 5000.0, 130000.0, 1.15
N_CYCLES = 10

atm = isa_atmosphere(ALT_M)
T_im = atm["T_amb_K"]
air_per_cycle = air_mass_per_cycle_per_cyl_kg(N_RPM, MAP_PA, T_im, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)
m_fuel_cycle = PHI * air_per_cycle * FA_STOICH
m_charge = air_per_cycle + m_fuel_cycle

print("=" * 78)
print("Computing single-cylinder torque traces (healthy and misfired)...")
print("=" * 78)
healthy_trace = single_cylinder_torque_trace(N_RPM, MAP_PA, T_im, PHI, m_charge,
                                              DEFAULT_GEOMETRY, DEFAULT_CONSTANTS, n_points=361)
misfire_constants = dataclasses.replace(DEFAULT_CONSTANTS, combustion_efficiency_multiplier=0.0)
misfire_trace = single_cylinder_torque_trace(N_RPM, MAP_PA, T_im, PHI, m_charge,
                                              DEFAULT_GEOMETRY, misfire_constants, n_points=361)
print(f"  healthy peak torque:  {healthy_trace['torque_Nm'].max():.1f} N*m")
print(f"  misfired peak torque: {misfire_trace['torque_Nm'].max():.1f} N*m "
      f"(should be much lower -- no combustion push)")
check("misfired single-cylinder torque never exceeds healthy (no combustion to drive it)",
      misfire_trace["torque_Nm"].max() < healthy_trace["torque_Nm"].max() * 0.5)

print()
print("=" * 78)
print("HEALTHY engine: combining 4 phased cylinders, checking order spectrum")
print("=" * 78)
healthy_combined = combined_four_cylinder_torque(healthy_trace, n_cycles=N_CYCLES)
healthy_omega = angular_velocity_fluctuation(healthy_combined, N_RPM)
healthy_spec = order_spectrum(healthy_omega["omega_fluct_rad_s"], n_cycles=N_CYCLES)


def amplitude_near_order(spec, target_order, tol=0.08):
    mask = np.abs(spec["orders"] - target_order) <= tol
    return float(np.max(spec["amplitude"][mask])) if np.any(mask) else 0.0


a_order2_healthy = amplitude_near_order(healthy_spec, 2.0)
a_order4_healthy = amplitude_near_order(healthy_spec, 4.0)
a_order05_healthy = amplitude_near_order(healthy_spec, 0.5)
a_order1_healthy = amplitude_near_order(healthy_spec, 1.0)

print(f"  amplitude at order 0.5 (misfire signature): {a_order05_healthy:.4f}")
print(f"  amplitude at order 1.0:                      {a_order1_healthy:.4f}")
print(f"  amplitude at order 2.0 (firing frequency):   {a_order2_healthy:.4f}  <- should dominate")
print(f"  amplitude at order 4.0 (2nd harmonic):        {a_order4_healthy:.4f}")

check("order 2 (firing frequency) is the dominant peak in a healthy 4-cylinder engine "
      "-- textbook NVH fact, not our own claim",
      a_order2_healthy > a_order05_healthy * 5 and a_order2_healthy > a_order1_healthy * 3,
      f"order2={a_order2_healthy:.4f} vs order0.5={a_order05_healthy:.4f}, order1={a_order1_healthy:.4f}")
check("order 0.5 is negligible in the healthy engine (perfect 180-degree periodicity)",
      a_order05_healthy < a_order2_healthy * 0.05,
      f"order0.5={a_order05_healthy:.5f} vs order2 peak={a_order2_healthy:.4f}")

print()
print("=" * 78)
print("MISFIRING engine (cylinder 1): does order 0.5 appear?")
print("=" * 78)
misfire_combined = combined_four_cylinder_torque(healthy_trace, n_cycles=N_CYCLES,
                                                   faulted_cylinder=1, faulted_trace=misfire_trace)
misfire_omega = angular_velocity_fluctuation(misfire_combined, N_RPM)
misfire_spec = order_spectrum(misfire_omega["omega_fluct_rad_s"], n_cycles=N_CYCLES)

a_order05_misfire = amplitude_near_order(misfire_spec, 0.5)
a_order15_misfire = amplitude_near_order(misfire_spec, 1.5)
a_order2_misfire = amplitude_near_order(misfire_spec, 2.0)

print(f"  amplitude at order 0.5:  healthy={a_order05_healthy:.4f}   misfiring={a_order05_misfire:.4f}")
print(f"  amplitude at order 1.5:  misfiring={a_order15_misfire:.4f}  (odd multiple of 0.5, also expected)")
print(f"  amplitude at order 2.0:  healthy={a_order2_healthy:.4f}   misfiring={a_order2_misfire:.4f}")

check("*** the actual claim: order 0.5 rises sharply once cylinder 1 misfires ***",
      a_order05_misfire > a_order05_healthy * 8,
      f"healthy={a_order05_healthy:.4f} -> misfiring={a_order05_misfire:.4f} "
      f"({a_order05_misfire/max(a_order05_healthy,1e-9):.1f}x)")
check("order 0.5 becomes clearly visible against the misfiring spectrum's own order-2 peak "
      "(not just larger than before, but a real feature of the spectrum)",
      a_order05_misfire > a_order2_misfire * 0.15,
      f"order0.5={a_order05_misfire:.4f} vs order2={a_order2_misfire:.4f}")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED")
    sys.exit(0)
