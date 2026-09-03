"""Fault injection: Handbook Part 9 / Table 2 / 02-Research/fault-signatures.md.

Every fault is expressed as a change to ONE physical parameter already in
the equations, then the thermodynamics runs. We never hand-code a
temperature direction anywhere in this file. If the signatures in
fault-signatures.md Table 1 (EGT/CHT up or down) don't fall out of running
these perturbations through engine_model.py, the physics is wrong --
that's the actual test, run by verify_fault_signatures.py, not an
assertion made here.

Two distinct injection mechanisms, matching how each fault is physically
caused:

  - Faults that change a CALIBRATION CONSTANT for one cylinder (misfire,
    detonation, cooling degradation) go through per_cylinder_constants,
    using dataclasses.replace() so only the affected cylinder deviates
    from the shared nominal constants.
  - Faults that change an OPERATING INPUT (injector restriction/blockage,
    bearing wear, turbo degradation) go through a modified OperatingPoint.
  - Sensor drift touches NEITHER. It is applied after prediction, to the
    reported reading only -- the physics never sees it. This is the
    concrete implementation of "physics unchanged, only the reading is
    biased" from fault-signatures.md.
"""

import copy
import dataclasses
from dataclasses import dataclass

from .engine_model import OperatingPoint
from .engine_params import CalibrationConstants, DEFAULT_CONSTANTS


VALID_KINDS = {
    "none",
    "misfire",
    "injector_restriction",   # severity in [0,1]; severity=1.0 IS the "blockage" case
    "detonation",
    "cooling_degradation",
    "bearing_wear",
    "sensor_drift",
    "turbo_degradation",
}


@dataclass(frozen=True)
class FaultSpec:
    """One injected fault.

    kind: one of VALID_KINDS.
    cylinder: 1-indexed cylinder number for single-cylinder faults
        (misfire, injector_restriction, detonation, cooling_degradation,
        sensor_drift). None for engine-wide faults (bearing_wear,
        turbo_degradation).
    severity: meaning is fault-specific, documented at each branch in
        apply_fault() below. Always in [0, 1] except where noted.
    sensor_channel: for sensor_drift only -- "EGT_K" or "CHT_K".
    bias_K: for sensor_drift only -- the bias in Kelvin added to the
        reported reading. Physical units, not severity, because "how much
        a lying sensor lies by" isn't naturally a 0-1 scale.
    """
    kind: str
    cylinder: int | None = None
    severity: float = 1.0
    sensor_channel: str | None = None
    bias_K: float = 0.0

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}, must be one of {VALID_KINDS}")


# ASSUMED fault-magnitude constants. These set how hard each fault pushes
# its parameter at severity=1.0. Physically reasoned, not measured --
# exactly the kind of thing Phase 2 rig data would calibrate.
DETONATION_MAX_ADVANCE_DEG = 25.0        # extra crank degrees BTDC at severity=1.0
COOLING_DEGRADATION_MAX_R_TH_MULT = 3.0   # R_th multiplier at severity=1.0
BEARING_WEAR_MAX_CLEARANCE_MULT = 3.0     # clearance multiplier at severity=1.0
TURBO_DEGRADATION_MAP_DEFICIT_PA_PER_KM = 3000.0  # Pa deficit per km altitude, at severity=1.0


def apply_fault(
    op: OperatingPoint,
    fault: FaultSpec,
    constants: CalibrationConstants = DEFAULT_CONSTANTS,
    altitude_m: float = None,
) -> tuple:
    """Returns (new_op, per_cylinder_constants). Does not mutate inputs.

    altitude_m is only used by turbo_degradation, to compute the MAP
    deficit; if not given, op.altitude_m is used.
    """
    if fault.kind == "none":
        return op, {}

    if altitude_m is None:
        altitude_m = op.altitude_m

    per_cyl_constants = {}
    new_fuel = list(op.fuel_flow_kg_s_per_cyl)
    new_map_Pa = op.MAP_Pa
    new_clearance_m = op.bearing_clearance_m

    if fault.kind == "misfire":
        # Table 2: combustion efficiency of that cylinder set to zero.
        # severity=1.0 -> multiplier=0.0, complete misfire. Partial
        # severity models a weak/intermittent misfire (fouled plug
        # rather than dead plug).
        if fault.cylinder is None:
            raise ValueError("misfire requires a cylinder")
        cyl_constants = dataclasses.replace(
            constants, combustion_efficiency_multiplier=1.0 - fault.severity
        )
        per_cyl_constants[fault.cylinder] = cyl_constants

    elif fault.kind == "injector_restriction":
        # Table 2: fuel mass flow to that cylinder reduced by fraction c.
        # severity IS c. severity=1.0 is the "injector fully blocked"
        # case in fault-signatures.md -- same parameter, same mechanism,
        # the sign inversion (EGT up when partial, down when total) is
        # NOT coded here, it falls out of crossing the phi=1 peak.
        if fault.cylinder is None:
            raise ValueError("injector_restriction requires a cylinder")
        idx = fault.cylinder - 1
        new_fuel[idx] = new_fuel[idx] * (1.0 - fault.severity)

    elif fault.kind == "detonation":
        # Table 2: burn phasing theta_0 advanced sharply.
        if fault.cylinder is None:
            raise ValueError("detonation requires a cylinder")
        advance = fault.severity * DETONATION_MAX_ADVANCE_DEG
        cyl_constants = dataclasses.replace(
            constants, ignition_timing_deg_btdc=constants.ignition_timing_deg_btdc + advance
        )
        per_cyl_constants[fault.cylinder] = cyl_constants

    elif fault.kind == "cooling_degradation":
        # Table 2: thermal resistance R_th increased.
        if fault.cylinder is None:
            raise ValueError("cooling_degradation requires a cylinder")
        mult = 1.0 + fault.severity * (COOLING_DEGRADATION_MAX_R_TH_MULT - 1.0)
        cyl_constants = dataclasses.replace(constants, R_th=constants.R_th * mult)
        per_cyl_constants[fault.cylinder] = cyl_constants

    elif fault.kind == "bearing_wear":
        # Table 2: bearing clearance and friction coefficient increased.
        # Engine-wide (one oil system), not per-cylinder.
        nominal = constants.bearing_clearance_nominal_m
        mult = 1.0 + fault.severity * (BEARING_WEAR_MAX_CLEARANCE_MULT - 1.0)
        new_clearance_m = nominal * mult

    elif fault.kind == "turbo_degradation":
        # Not in the original Table 2 but derived in Handbook 7.2: a
        # degrading turbo shows as a growing gap between commanded and
        # achieved MAP, and that gap grows with ALTITUDE specifically,
        # because the turbo has to work harder to maintain boost as
        # ambient pressure falls. No compressor map is modelled; this
        # is a direct MAP deficit, documented as a simplification.
        deficit = fault.severity * TURBO_DEGRADATION_MAP_DEFICIT_PA_PER_KM * (altitude_m / 1000.0)
        new_map_Pa = max(op.MAP_Pa - deficit, 10000.0)

    elif fault.kind == "sensor_drift":
        pass  # handled entirely in apply_sensor_drift(), not here

    else:
        raise AssertionError(f"unhandled fault kind {fault.kind!r}")

    new_op = dataclasses.replace(
        op,
        fuel_flow_kg_s_per_cyl=new_fuel,
        MAP_Pa=new_map_Pa,
        bearing_clearance_m=new_clearance_m,
    )
    return new_op, per_cyl_constants


def apply_sensor_drift(prediction: dict, fault: FaultSpec) -> dict:
    """Post-hoc bias on a REPORTED reading. The physics prediction this
    was computed from is untouched -- deep-copy first, then bias only the
    named channel on the named cylinder. This is what makes sensor drift
    distinguishable from a real fault downstream: every OTHER channel
    still agrees with physics, because physics never saw this fault."""
    if fault.kind != "sensor_drift":
        raise ValueError("apply_sensor_drift called with a non-sensor_drift fault")
    if fault.cylinder is None or fault.sensor_channel is None:
        raise ValueError("sensor_drift requires cylinder and sensor_channel")

    result = copy.deepcopy(prediction)
    for cyl in result["per_cylinder"]:
        if cyl["cylinder"] == fault.cylinder:
            if cyl.get(fault.sensor_channel) is None:
                raise ValueError(f"channel {fault.sensor_channel!r} not present or is None "
                                  f"on cylinder {fault.cylinder}")
            cyl[fault.sensor_channel] = cyl[fault.sensor_channel] + fault.bias_K
    return result
