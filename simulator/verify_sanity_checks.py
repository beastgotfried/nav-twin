"""
Verification script, NOT a demo. Two jobs:

1. Reproduce the three worked examples from Handbook Part 7 exactly, so we
   know the code matches the equations we published in the document the
   team is reading. Any mismatch here means the handbook or the code is
   wrong, and must be fixed before either is trusted.

2. Check the qualitative claims the entire pitch depends on actually
   emerge from the code: EGT is non-monotonic in phi (the "hill"), and
   CHT peaks at a richer mixture than EGT (the "offset"). These are not
   asserted anywhere in the code. If they don't come out, the physics is
   wrong regardless of what the handbook says.

Run: python verify_sanity_checks.py
"""

import sys
import math
import numpy as np

from physics.atmosphere import isa_atmosphere
from physics.intake import air_mass_flow_kg_s
from physics.combustion import heat_release_per_kg_charge, equivalence_ratio
from physics.cycle import egt_steady_state_K, simulate_cylinder_cycle, cht_steady_state_K
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.constants import FA_STOICH

FAILURES = []


def check(label, actual, expected, tol_frac=0.02):
    ok = abs(actual - expected) / abs(expected) <= tol_frac
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: got {actual:,.4f}, expected ~{expected:,.4f} (tol {tol_frac:.0%})")
    if not ok:
        FAILURES.append(label)
    return ok


def check_true(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


print("=" * 78)
print("WORKED EXAMPLE 1 -- Handbook 7.2: ISA atmosphere at 7600 m")
print("=" * 78)
atm = isa_atmosphere(7600.0)
check("T_amb [K]", atm["T_amb_K"], 238.75, tol_frac=0.001)
check("p_amb [Pa]", atm["p_amb_Pa"], 37710.0, tol_frac=0.01)
check("rho_amb [kg/m3]", atm["rho_amb_kgm3"], 0.550, tol_frac=0.01)
print(f"  (for reference: sea-level density ratio = {atm['rho_amb_kgm3']/1.225:.3f}, handbook says ~0.45)")

print()
print("=" * 78)
print("WORKED EXAMPLE 2 -- Handbook 7.3: air mass flow at 5500 rpm")
print("=" * 78)
m_air = air_mass_flow_kg_s(N_rpm=5500, MAP_Pa=140000, T_im_K=320,
                            geometry=DEFAULT_GEOMETRY, constants=DEFAULT_CONSTANTS, eta_v=0.85)
check("m_air [kg/s]", m_air, 0.0803, tol_frac=0.01)

phi_check = 1.1
m_fuel_total = phi_check * m_air * FA_STOICH
m_fuel_kg_hr = m_fuel_total * 3600.0
check("m_fuel at phi=1.1 [kg/hr]", m_fuel_kg_hr, 21.6, tol_frac=0.01)
litres_per_hr = m_fuel_kg_hr / 0.72  # approx petrol density kg/L
print(f"  -> approx {litres_per_hr:.1f} L/hr (handbook: ~30 L/hr, published-consumption sanity check)")

print()
print("=" * 78)
print("WORKED EXAMPLE 3 -- Handbook 7.5: heat release at phi=1.1")
print("=" * 78)
q = heat_release_per_kg_charge(1.1)
check("q(phi=1.1) [J/kg charge]", q, 2.753e6, tol_frac=0.005)

print()
print("=" * 78)
print("QUALITATIVE CHECK A -- EGT is non-monotonic in phi (the 'hill')")
print("=" * 78)
phis = np.linspace(0.7, 1.4, 15)
egts = []
for phi in phis:
    r = egt_steady_state_K(phi, T_im_K=320, T_amb_K=238.75)
    egts.append(r["EGT_ss_K"])
egts = np.array(egts)
i_peak = int(np.argmax(egts))
phi_peak_egt = phis[i_peak]

is_hill = 0 < i_peak < len(phis) - 1
check_true("EGT has an interior peak (not monotonic)", is_hill,
           f"peak at phi={phi_peak_egt:.3f}, EGT={egts[i_peak]:.1f} K")

if is_hill:
    # Interpolate rather than grid-search: the lean segment (phi <= peak) is
    # monotonically increasing, so for any rich-side EGT value we can solve
    # directly for the lean-side phi that gives the same reading.
    lean_phis = phis[: i_peak + 1]
    lean_egts = egts[: i_peak + 1]
    rich_index = min(i_peak + 4, len(phis) - 1)  # a clearly rich-side point
    rich_phi = phis[rich_index]
    target_egt = egts[rich_index]

    if lean_egts[0] <= target_egt <= lean_egts[-1]:
        lean_phi_match = float(np.interp(target_egt, lean_egts, lean_phis))
        print(f"  -> concrete ambiguity: phi={lean_phi_match:.3f} (lean) and phi={rich_phi:.3f} (rich) "
              f"BOTH give EGT ~ {target_egt:.1f} K")
        check_true("Two distinct mixtures give the same EGT (the ambiguity claim)", True)
    else:
        check_true("Two distinct mixtures give the same EGT (the ambiguity claim)", False,
                   f"rich-side EGT {target_egt:.1f} K falls outside the lean-side range "
                   f"[{lean_egts[0]:.1f}, {lean_egts[-1]:.1f}] K -- widen the phi sweep")

print()
print(f"  Full EGT(phi) sweep:")
for phi, egt in zip(phis, egts):
    bar = "#" * int((egt - egts.min()) / (egts.max() - egts.min() + 1e-9) * 40)
    marker = " <- PEAK" if phi == phi_peak_egt else ""
    print(f"    phi={phi:.3f}  EGT={egt:7.1f} K  {bar}{marker}")

print()
print("=" * 78)
print("QUALITATIVE CHECK B -- CHT peaks richer than EGT (the 'offset')")
print("=" * 78)
print("  (this runs the full crank-angle cycle integration -- slower)")

phis_cht = np.linspace(0.85, 1.35, 11)
chts = []
air_per_cycle = None
from physics.intake import air_mass_per_cycle_per_cyl_kg
air_per_cycle = air_mass_per_cycle_per_cyl_kg(5500, 140000, 320, DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)

for phi in phis_cht:
    m_fuel_per_cyl = phi * (air_per_cycle) * FA_STOICH  # approx per-cycle fuel mass directly
    m_charge = air_per_cycle + m_fuel_per_cyl
    try:
        cyc = simulate_cylinder_cycle(5500, 140000, 320, phi, m_charge,
                                       DEFAULT_GEOMETRY, DEFAULT_CONSTANTS, n_points=360)
        cht = cht_steady_state_K(cyc["Q_wall_per_cycle_J"], 5500, 320, DEFAULT_CONSTANTS)
    except Exception as e:
        print(f"  !! integration failed at phi={phi:.3f}: {e}")
        cht = float("nan")
    chts.append(cht)

chts = np.array(chts)
valid = ~np.isnan(chts)
if valid.sum() > 2:
    i_peak_cht = int(np.argmax(chts[valid]))
    phi_peak_cht = phis_cht[valid][i_peak_cht]

    print()
    for phi, cht in zip(phis_cht, chts):
        if np.isnan(cht):
            print(f"    phi={phi:.3f}  CHT=  FAILED")
            continue
        bar = "#" * int((cht - np.nanmin(chts)) / (np.nanmax(chts) - np.nanmin(chts) + 1e-9) * 40)
        marker = " <- PEAK" if phi == phi_peak_cht else ""
        print(f"    phi={phi:.3f}  CHT={cht:7.1f} K  {bar}{marker}")

    print()
    check_true("CHT peaks at a richer (higher) phi than EGT", phi_peak_cht > phi_peak_egt,
               f"phi_peak_CHT={phi_peak_cht:.3f} vs phi_peak_EGT={phi_peak_egt:.3f}")
else:
    check_true("CHT cycle integration produced enough valid points", False)

print()
print("=" * 78)
print("QUALITATIVE CHECK C -- raw combustion temperature overprediction")
print("(handbook 7.6 explicitly warns this happens; confirming it's the")
print(" expected magnitude, not a sign something else is broken)")
print("=" * 78)
r = egt_steady_state_K(1.05, T_im_K=320, T_amb_K=238.75)
print(f"  T3 (raw air-standard combustion temp) = {r['T3_K']:.0f} K")
print(f"  EGT_ss (after k_egt scaling)           = {r['EGT_ss_K']:.0f} K")
check_true("T3 overpredicts into the 2500-4500 K range (as the handbook warns)",
           2500 < r["T3_K"] < 4500)
check_true("EGT_ss (post-scaling) lands in a plausible piston-aircraft range (900-1250 K, ~630-980 C)",
           900 < r["EGT_ss_K"] < 1250,
           "if this fails, k_egt needs adjusting -- see engine_params.py")

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
