"""
The falsification test described in Handbook Part 9 and claimed (before
this script existed) in 03-Design/fault-injection.md: "all 7 reproduce
correctly." That claim was written before any of this was run. This is
where it actually gets checked.

Every fault below is injected as ONE parameter perturbation (faults.py).
No temperature direction is hand-coded anywhere in this codebase. If a
row here fails, the physics model is wrong and must be fixed -- the
handbook's Table 1 does not get edited to match broken code.

Run: python verify_fault_signatures.py
"""

import sys
import copy

from physics.engine_model import OperatingPoint, predict_steady_state
from physics.faults import FaultSpec, apply_fault, apply_sensor_drift
from physics.intake import air_mass_flow_kg_s
from physics.atmosphere import isa_atmosphere
from physics.constants import FA_STOICH
from physics.engine_params import DEFAULT_CONSTANTS, DEFAULT_GEOMETRY

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def cyl(prediction, n):
    return next(c for c in prediction["per_cylinder"] if c["cylinder"] == n)


# ---------------------------------------------------------------------------
# Baseline operating point: cruise, deliberately rich of peak (phi ~ 1.15)
# so the injector-restriction severity-inversion test has room to show the
# EGT rise before it crosses the peak. This precondition is the whole point
# of that test -- see Handbook 8.4.
# ---------------------------------------------------------------------------
ALT_M = 3000.0
N_RPM = 5000.0
MAP_PA = 130000.0
PHI_BASELINE = 1.15

atm = isa_atmosphere(ALT_M)
total_air = air_mass_flow_kg_s(N_RPM, MAP_PA, atm["T_amb_K"], DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)
air_per_cyl = total_air / DEFAULT_GEOMETRY.num_cylinders
fuel_per_cyl_nominal = PHI_BASELINE * air_per_cyl * FA_STOICH

baseline_op = OperatingPoint(
    N_rpm=N_RPM, MAP_Pa=MAP_PA, altitude_m=ALT_M,
    fuel_flow_kg_s_per_cyl=[fuel_per_cyl_nominal] * 4,
)

print("=" * 78)
print(f"BASELINE: N={N_RPM} rpm, MAP={MAP_PA/1000:.0f} kPa, alt={ALT_M:.0f} m, phi={PHI_BASELINE}")
print("=" * 78)
baseline = predict_steady_state(baseline_op)
b1 = cyl(baseline, 1)
print(f"  cyl1: phi={b1['phi']:.3f}  EGT={b1['EGT_K']:.1f} K  CHT={b1['CHT_K']:.1f} K")
print(f"  oil: p_oil={baseline['p_oil_Pa']/1000:.1f} kPa  T_oil_target={baseline['T_oil_ss_K']:.1f} K")


def run_fault(label, fault, expect):
    """expect: dict of checks to run on cylinder-1 vs baseline, plus optional
    cross-cylinder isolation check."""
    print()
    print("-" * 78)
    print(label)
    print("-" * 78)
    new_op, per_cyl_const = apply_fault(baseline_op, fault, DEFAULT_CONSTANTS, ALT_M)
    pred = predict_steady_state(new_op, per_cylinder_constants=per_cyl_const)
    c1 = cyl(pred, fault.cylinder) if fault.cylinder else None

    if c1 is not None:
        print(f"  cyl{fault.cylinder}: phi={c1['phi']:.3f}  EGT={c1['EGT_K']:.1f} K "
              f"(baseline {b1['EGT_K']:.1f})  CHT={c1['CHT_K']:.1f} K (baseline {b1['CHT_K']:.1f})")

    for check_label, fn in expect.items():
        check(check_label, fn(pred, c1))

    return pred


# ---------------------------------------------------------------------------
# 1. MISFIRE (cyl 1): combustion efficiency -> 0. Table 1: EGT down, CHT down.
# ---------------------------------------------------------------------------
run_fault(
    "1. MISFIRE, cylinder 1, severity 1.0 (complete)",
    FaultSpec(kind="misfire", cylinder=1, severity=1.0),
    {
        "EGT falls vs baseline (dead cylinder runs cold)": lambda p, c: c["EGT_K"] < b1["EGT_K"] - 5.0,
        "CHT falls vs baseline": lambda p, c: c["CHT_K"] < b1["CHT_K"] - 1.0,
        "cylinders 2-4 unaffected": lambda p, c: all(
            abs(cyl(p, n)["EGT_K"] - baseline["per_cylinder"][n-1]["EGT_K"]) < 0.01 for n in (2, 3, 4)
        ),
    },
)

# ---------------------------------------------------------------------------
# 2 & 3. INJECTOR RESTRICTION -- THE SEVERITY INVERSION TEST
# Table 1 + Handbook 8.4: partial restriction (leaning a rich cylinder
# toward peak) -> EGT UP. Full blockage -> dead cylinder -> EGT DOWN.
# Same fault, same parameter, opposite sign, purely from severity.
# ---------------------------------------------------------------------------
partial = run_fault(
    "2. INJECTOR RESTRICTION, cylinder 1, severity 0.08 (partial)",
    # NOTE: severity here is a fuel-flow fraction removed, not a free choice.
    # Baseline phi=1.15; the peak is at phi=1.0. A 30% cut was tried first
    # and OVERSHOT the peak entirely (phi crashed to 0.805, past the peak
    # into leaner-than-baseline territory), so EGT fell instead of rising --
    # correct physics against a badly chosen test input, not a bug. 0.08
    # (8% fuel cut) moves phi from 1.15 to ~1.06, staying rich of peak,
    # which is the regime Handbook 8.4 actually describes. This also means
    # the "EGT rises" regime is narrower in fuel-cut terms than intuition
    # suggests -- a real, useful finding, not just a test-tuning detail.
    FaultSpec(kind="injector_restriction", cylinder=1, severity=0.08),
    {
        "EGT RISES vs baseline (leaning a rich cylinder toward peak)":
            lambda p, c: c["EGT_K"] > b1["EGT_K"] + 1.0,
    },
)

full = run_fault(
    "3. INJECTOR RESTRICTION, cylinder 1, severity 1.0 (full blockage)",
    FaultSpec(kind="injector_restriction", cylinder=1, severity=1.0),
    {
        "EGT FALLS vs baseline (starved cylinder is now dead)":
            lambda p, c: c["EGT_K"] < b1["EGT_K"] - 5.0,
    },
)

c1_partial = cyl(partial, 1)
c1_full = cyl(full, 1)
check(
    "*** SEVERITY INVERSION: same fault, opposite EGT sign, no hand-coded switch ***",
    (c1_partial["EGT_K"] > b1["EGT_K"]) and (c1_full["EGT_K"] < b1["EGT_K"]),
    f"partial={c1_partial['EGT_K']:.1f} K (up), full={c1_full['EGT_K']:.1f} K (down), "
    f"baseline={b1['EGT_K']:.1f} K",
)

# ---------------------------------------------------------------------------
# 4. DETONATION (cyl 1): ignition advanced sharply.
# Table 1: CHT up sharply, EGT down. Opposite-sign fingerprint.
# ---------------------------------------------------------------------------
run_fault(
    "4. DETONATION, cylinder 1, severity 1.0",
    FaultSpec(kind="detonation", cylinder=1, severity=1.0),
    {
        "CHT rises vs baseline (energy dumped into the head)": lambda p, c: c["CHT_K"] > b1["CHT_K"] + 5.0,
        "EGT falls vs baseline (less energy reaches the exhaust)": lambda p, c: c["EGT_K"] < b1["EGT_K"],
        "*** OPPOSITE-SIGN fingerprint: CHT up AND EGT down, same cause ***":
            lambda p, c: (c["CHT_K"] > b1["CHT_K"]) and (c["EGT_K"] < b1["EGT_K"]),
    },
)

# ---------------------------------------------------------------------------
# 5. COOLING DEGRADATION (cyl 1): R_th up.
# Table 1: CHT up, EGT flat (cycle untouched).
# ---------------------------------------------------------------------------
run_fault(
    "5. COOLING DEGRADATION, cylinder 1, severity 1.0",
    FaultSpec(kind="cooling_degradation", cylinder=1, severity=1.0),
    {
        "CHT rises vs baseline": lambda p, c: c["CHT_K"] > b1["CHT_K"] + 5.0,
        "EGT essentially unchanged (cycle untouched)": lambda p, c: abs(c["EGT_K"] - b1["EGT_K"]) < 0.5,
    },
)

# ---------------------------------------------------------------------------
# 6. BEARING WEAR: clearance up.
# Table 1: oil pressure down, oil temp target up. Engine-wide, not per-cyl.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("6. BEARING WEAR, severity 1.0")
print("-" * 78)
new_op, _ = apply_fault(baseline_op, FaultSpec(kind="bearing_wear", severity=1.0), DEFAULT_CONSTANTS, ALT_M)
pred = predict_steady_state(new_op, run_full_cycle_for_cht=False)
print(f"  p_oil: {pred['p_oil_Pa']/1000:.1f} kPa (baseline {baseline['p_oil_Pa']/1000:.1f} kPa)")
print(f"  T_oil target: {pred['T_oil_ss_K']:.1f} K (baseline {baseline['T_oil_ss_K']:.1f} K)")
check("oil pressure falls vs baseline", pred["p_oil_Pa"] < baseline["p_oil_Pa"] - 1000.0)
check("oil temperature target rises vs baseline (more friction heat)",
      pred["T_oil_ss_K"] > baseline["T_oil_ss_K"] + 1.0)
check("*** joint signature: pressure down AND temp up, together ***",
      (pred["p_oil_Pa"] < baseline["p_oil_Pa"]) and (pred["T_oil_ss_K"] > baseline["T_oil_ss_K"]))

# ---------------------------------------------------------------------------
# 7. SENSOR DRIFT (cyl 1, EGT channel): reading biased, physics untouched.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("7. SENSOR DRIFT, cylinder 1, EGT channel, +80 K bias")
print("-" * 78)
drift_fault = FaultSpec(kind="sensor_drift", cylinder=1, sensor_channel="EGT_K", bias_K=80.0)
drifted = apply_sensor_drift(baseline, drift_fault)
d1 = cyl(drifted, 1)
print(f"  cyl1 EGT reported: {d1['EGT_K']:.1f} K (true physics value: {b1['EGT_K']:.1f} K)")
check("reported EGT shifted by exactly the bias", abs(d1["EGT_K"] - (b1["EGT_K"] + 80.0)) < 1e-6)
check("CHT on the SAME cylinder is untouched (physics never saw this fault)",
      abs(d1["CHT_K"] - b1["CHT_K"]) < 1e-6)
check("other cylinders completely untouched", all(
    cyl(drifted, n) == baseline["per_cylinder"][n - 1] for n in (2, 3, 4)
))
check("*** the tell: EGT moved but CHT and every other cylinder did not ***",
      abs(d1["EGT_K"] - b1["EGT_K"]) > 1.0 and abs(d1["CHT_K"] - b1["CHT_K"]) < 1e-6)

# ---------------------------------------------------------------------------
# 8. TURBO DEGRADATION (bonus, Handbook 7.2): deficit should GROW with altitude.
# ---------------------------------------------------------------------------
print()
print("-" * 78)
print("8. TURBO DEGRADATION, severity 1.0, at two altitudes")
print("-" * 78)
for alt in (2000.0, 8000.0):
    op_at_alt = copy.deepcopy(baseline_op)
    op_at_alt.altitude_m = alt
    new_op, _ = apply_fault(op_at_alt, FaultSpec(kind="turbo_degradation", severity=1.0), DEFAULT_CONSTANTS, alt)
    deficit_kPa = (op_at_alt.MAP_Pa - new_op.MAP_Pa) / 1000.0
    print(f"  altitude {alt/1000:.0f} km: MAP deficit = {deficit_kPa:.2f} kPa")

op_low = copy.deepcopy(baseline_op); op_low.altitude_m = 2000.0
op_high = copy.deepcopy(baseline_op); op_high.altitude_m = 8000.0
_, _ = None, None
new_low, _ = apply_fault(op_low, FaultSpec(kind="turbo_degradation", severity=1.0), DEFAULT_CONSTANTS, 2000.0)
new_high, _ = apply_fault(op_high, FaultSpec(kind="turbo_degradation", severity=1.0), DEFAULT_CONSTANTS, 8000.0)
deficit_low = op_low.MAP_Pa - new_low.MAP_Pa
deficit_high = op_high.MAP_Pa - new_high.MAP_Pa
check("*** MAP deficit grows with altitude (turbo health indicator, Handbook 7.2) ***",
      deficit_high > deficit_low, f"low={deficit_low/1000:.2f} kPa, high={deficit_high/1000:.2f} kPa")

# ---------------------------------------------------------------------------
print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("RESULT: ALL CHECKS PASSED -- all 7 Table 2 faults (+turbo) reproduce correctly")
    sys.exit(0)
