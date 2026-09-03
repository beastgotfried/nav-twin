"""Handbook Part 7.2 -- Step 1: ISA atmosphere (troposphere, valid to 11 km)."""

from .constants import T0_ISA, P0_ISA, L_ISA, R_AIR, ISA_EXPONENT


def isa_atmosphere(altitude_m: float) -> dict:
    """Return ambient temperature, pressure and density at the given altitude.

    DERIVED: standard ISA troposphere model, no fitted constants.
    Reproduces Handbook 7.2 worked example at h=7600 m.
    """
    t_amb = T0_ISA - L_ISA * altitude_m
    p_amb = P0_ISA * (t_amb / T0_ISA) ** ISA_EXPONENT
    rho_amb = p_amb / (R_AIR * t_amb)
    return {"T_amb_K": t_amb, "p_amb_Pa": p_amb, "rho_amb_kgm3": rho_amb}
