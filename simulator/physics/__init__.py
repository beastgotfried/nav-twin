"""
Physics twin for a fuel-injected aero piston engine.

Implements the equation chain atmosphere through cylinder head temperature.
Every empirical constant is flagged DERIVED (first principles) or
ASSUMED/EMPIRICAL (needs calibration against a real engine).

This module intentionally does NOT implement:
- Monte Carlo uncertainty propagation (separate module, next build step)
- Fault injection (separate module, next build step)
- The oil subsystem's exact Vogel viscosity law (simplified Andrade form used;
  flagged inline)

Module map:
    constants.py     universal constants and fuel properties
    engine_params.py engine geometry and calibration constants (theta)
    atmosphere.py    Step 1: ISA atmosphere
    intake.py        Step 2: air mass flow
    combustion.py    Step 3-4: equivalence ratio, heat release, Wiebe, flame speed
    cycle.py         Step 5-6: EGT (algebraic) and CHT (crank-angle Woschni/Wiebe)
    thermal.py       Step 7: first-order thermal lag
    oil.py           Step 8: oil pressure/temperature subsystem
    engine_model.py  orchestrates all of the above into one prediction call
    faults.py        Table 2 fault injection: parameter perturbation, not
                      hand-coded signatures -- see verify_fault_signatures.py
"""
