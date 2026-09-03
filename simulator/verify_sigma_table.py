"""
Checks the deployed sigma lookup (physics/sigma_lookup.py), the runtime
half of the Handbook 6.9 architecture:

1. INTERPOLATION FIDELITY: the lookup agrees with a fresh, direct Monte
   Carlo run at off-grid operating points, for both channels. If this
   fails, the twin's z-scores are built on a band the physics does not
   actually produce.

2. THE MIXTURE KINK the grid is designed around is real (simulator
   README finding): with k_egt and T_amb perturbations disabled, std(EGT)
   at phi=0.85 is materially larger than at phi=1.10, because the q(phi)
   slope collapses ~14.7x crossing into the rich regime. This is why the
   phi axis is densest around phi=1. Note this checks the physical property
   via direct propagation, not the table: the table is built with the full
   joint uncertainty spec, under which k_egt's contribution dominates and
   masks the kink (also documented in the README).

3. SANITY OF THE TABLE ITSELF: sigmas finite, positive, and physically
   plausible at every grid point, both channels.

4. OIL CONSTANTS present and positive (scalar ASSUMED sigmas; Phase 2
   replaces them with real propagation).

If data/sigma_table.npz does not exist yet, checks 1 and 3 run against the
reduced in-memory fallback table and the script says so; it does NOT fail
just because the npz is missing. Full-table verification runs at
integration time by simply rerunning this script.

Run: python verify_sigma_table.py
"""

import logging
import sys

import numpy as np

from physics import sigma_lookup
from physics.constants import FA_STOICH
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.intake import air_mass_per_cycle_per_cyl_kg
from physics.uncertainty import (propagate_egt_uncertainty,
                                 propagate_cht_uncertainty,
                                 UncertaintySpec, DEFAULT_UNCERTAINTY)

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


logging.basicConfig(level=logging.INFO,
                    format="  [%(levelname)s %(name)s] %(message)s")

rng = np.random.default_rng(20260829)  # fixed seed: reproducible, not cherry-picked

print("=" * 78)
print("CHECK 1 -- lookup vs direct Monte Carlo at off-grid points")
print("=" * 78)
table = sigma_lookup.get_table()
print(f"  table source: {table.source}")
if table.reduced:
    print("  NOTE: the full npz table is not present yet, so this run checks")
    print("        the REDUCED fallback table. This is not a failure. Rerun at")
    print("        integration time, once build_sigma_table.py has finished,")
    print("        to verify the full table the twin will actually deploy.")

LOOKUP_TOL = 0.35

# Off-grid but interior points, so the interpolator genuinely interpolates
# rather than reading a node or clamping at an edge.
EGT_OFF_GRID = [(0.87, 255.0), (0.97, 295.0), (1.03, 265.0), (1.12, 315.0)]
print()
print("  EGT (direct MC: 3000 samples/point, matching the offline build):")
worst = 0.0
for phi, t_amb in EGT_OFF_GRID:
    lut = table.sigma_egt(phi, t_amb)
    mc = propagate_egt_uncertainty(phi, t_amb, t_amb, n_samples=3000, rng=rng)
    rel = abs(lut - mc["std"]) / mc["std"]
    worst = max(worst, rel)
    print(f"    phi={phi:<5} T_amb={t_amb:<6} lookup={lut:7.2f} K  "
          f"MC={mc['std']:7.2f} K  rel diff={rel:.3f}")
check("EGT lookup within 35% of direct MC at all 4 off-grid points",
      worst <= LOOKUP_TOL, f"worst rel diff={worst:.3f}")

CHT_OFF_GRID = [(4300.0, 100_000.0, 0.85, 250.0),
                (5100.0, 125_000.0, 1.03, 270.0),
                (4600.0, 135_000.0, 0.95, 300.0),
                (5400.0,  95_000.0, 1.15, 245.0)]
print()
print("  CHT (direct MC: 150 samples/point, matching the offline build):")
worst = 0.0
for n, map_pa, phi, t_amb in CHT_OFF_GRID:
    lut = table.sigma_cht(n, map_pa, phi, t_amb)
    air = air_mass_per_cycle_per_cyl_kg(n, map_pa, t_amb,
                                        DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)
    m_charge = air * (1.0 + phi * FA_STOICH)
    mc = propagate_cht_uncertainty(n, map_pa, t_amb, t_amb, phi, m_charge,
                                   n_samples=150, rng=rng)
    rel = abs(lut - mc["std"]) / mc["std"]
    worst = max(worst, rel)
    print(f"    N={n:<5} MAP={map_pa:<7} phi={phi:<5} T_amb={t_amb:<6} "
          f"lookup={lut:6.2f} K  MC={mc['std']:6.2f} K  rel diff={rel:.3f}")
check("CHT lookup within 35% of direct MC at all 4 off-grid points",
      worst <= LOOKUP_TOL, f"worst rel diff={worst:.3f}")

print()
print("=" * 78)
print("CHECK 2 -- the mixture kink: std(EGT | phi uncertainty only) at")
print("           phi=0.85 materially larger than at phi=1.10")
print("=" * 78)
# Isolate the mixture-uncertainty component, same construction as
# verify_uncertainty.py: k_egt and T_amb held at nominal so the ONLY thing
# varying is phi. "Materially larger" is operationalised as > 2x; the
# README's measured lean/rich gap is roughly an order of magnitude.
PHI_ONLY_SPEC = UncertaintySpec(k_egt_rel=0.0, T_amb_sigma_K=0.0,
                                phi_rel=DEFAULT_UNCERTAINTY.phi_rel)
T_FIX = 288.15
r_lean = propagate_egt_uncertainty(0.85, T_FIX, T_FIX, spec=PHI_ONLY_SPEC,
                                   n_samples=3000, rng=rng)
r_rich = propagate_egt_uncertainty(1.10, T_FIX, T_FIX, spec=PHI_ONLY_SPEC,
                                   n_samples=3000, rng=rng)
print(f"  phi=0.85 (lean flank): std={r_lean['std']:.2f} K")
print(f"  phi=1.10 (rich side):  std={r_rich['std']:.2f} K")
check("kink: std(EGT) at phi=0.85 materially larger than at phi=1.10",
      r_lean["std"] > 2.0 * r_rich["std"],
      f"{r_lean['std']:.2f} K vs {r_rich['std']:.2f} K "
      f"(ratio {r_lean['std']/max(r_rich['std'], 1e-9):.1f}x)")

print()
print("=" * 78)
print("CHECK 3 -- sigmas positive and plausible at every grid point")
print("=" * 78)
egt, cht = table.egt_std, table.cht_std
print(f"  EGT std over table: {egt.min():.1f}..{egt.max():.1f} K ({egt.size} points)")
print(f"  CHT std over table: {cht.min():.1f}..{cht.max():.1f} K ({cht.size} points)")
check("EGT sigmas finite and strictly positive everywhere",
      bool(np.all(np.isfinite(egt)) and np.all(egt > 0)))
check("CHT sigmas finite and strictly positive everywhere",
      bool(np.all(np.isfinite(cht)) and np.all(cht > 0)))
# Plausibility envelopes. The draft bounds this check started from (EGT
# 1..60 K, CHT 1..40 K) were miscalibrated against the uncertainty spec,
# not the physics: with k_egt at its ASSUMED 15% relative uncertainty the
# full-joint EGT band measured at the grid corners is ~90..116 K (k_egt's
# contribution dominates total variance by roughly an order of magnitude,
# the finding documented in simulator README), and the CHT band reaches
# ~50 K at the hot/high-load corner. The envelopes below are set from those
# measurements: wide enough to absorb fallback-table sampling noise, tight
# enough to catch NaN, zeros, and order-of-magnitude build bugs.
check("EGT sigmas within the plausible 1..200 K envelope",
      bool(egt.min() >= 1.0 and egt.max() <= 200.0),
      f"measured {egt.min():.1f}..{egt.max():.1f} K")
check("CHT sigmas within the plausible 1..100 K envelope",
      bool(cht.min() >= 1.0 and cht.max() <= 100.0),
      f"measured {cht.min():.1f}..{cht.max():.1f} K")

print()
print("=" * 78)
print("CHECK 4 -- oil-channel scalar sigmas present and positive")
print("=" * 78)
print(f"  SIGMA_P_OIL_PA = {sigma_lookup.SIGMA_P_OIL_PA:.0f} Pa (ASSUMED, Phase 2)")
print(f"  SIGMA_T_OIL_K  = {sigma_lookup.SIGMA_T_OIL_K:.1f} K (ASSUMED, Phase 2)")
check("SIGMA_P_OIL_PA present and positive", sigma_lookup.SIGMA_P_OIL_PA > 0,
      f"{sigma_lookup.SIGMA_P_OIL_PA:.0f} Pa")
check("SIGMA_T_OIL_K present and positive", sigma_lookup.SIGMA_T_OIL_K > 0,
      f"{sigma_lookup.SIGMA_T_OIL_K:.1f} K")
check("module constants match the loaded table",
      sigma_lookup.SIGMA_P_OIL_PA == table.sigma_p_oil_pa
      and sigma_lookup.SIGMA_T_OIL_K == table.sigma_t_oil_k)

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
