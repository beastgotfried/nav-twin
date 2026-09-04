"""bridge/mavlink.py -- ArduPilot/PX4-style engine telemetry adapter.

Target schema: the EFI (electronic fuel injection) telemetry real UAV
engine ECUs report on the MAVLink bus, exemplified by ArduPilot's
EFI_STATUS message, plus an altitude source. This is the payload shape a
production UAV autopilot already emits, so connecting a real airframe is
a field-mapping exercise, not a research project.

Accepted payload (JSON envelope):

    {
      "t_s": 123.0,                    # optional, seconds; default: wall time delta
      "altitude_m": 4200.0,            # from the autopilot (GLOBAL_POSITION_INT
                                       # relative alt, or caller-computed)
      "efi": {
        "rpm": 5100.0,                 # engine RPM
        "intake_manifold_pressure_kpa": 118.0,
        "fuel_flow_g_min": 240.0,      # ASSUMED unit: grams per minute, as
                                       # ArduPilot reports efi_fuel_flow
        "cylinder_head_temperature_c": [145.0, 147.0, 144.0, 146.0],
        "exhaust_gas_temperature_c":   [700.0, 705.0, 698.0, 702.0],
        "oil_pressure_kpa": 480.0,
        "oil_temperature_c": 82.0
      }
    }

Unit conversions are stated where they happen. Anything missing that the
twin needs is rejected with a FrameError naming the field; we never
fabricate a channel.
"""

from .frames import normalize, FrameError

_G_MIN_TO_KG_S = 1.0 / 60000.0     # g/min -> kg/s
_N_CYL = 4


def _need(d, key, where):
    if key not in d:
        raise FrameError(f"mavlink payload missing {where}.{key}")
    return d[key]


def from_mavlink(payload: dict) -> dict:
    """One MAVLink-style envelope -> one internal frame."""
    if not isinstance(payload, dict):
        raise FrameError("mavlink payload must be an object")
    efi = _need(payload, "efi", "payload")
    if not isinstance(efi, dict):
        raise FrameError("payload.efi must be an object")

    cht_c = _need(efi, "cylinder_head_temperature_c", "efi")
    egt_c = _need(efi, "exhaust_gas_temperature_c", "efi")
    if len(cht_c) != _N_CYL or len(egt_c) != _N_CYL:
        raise FrameError(f"efi cylinder arrays must have {_N_CYL} entries")

    fuel_total_g_min = _need(efi, "fuel_flow_g_min", "efi")
    # EFI reports ONE total flow; per-cylinder split is even. That is an
    # assumption (real distribution is uneven; that unevenness is itself a
    # health signal we cannot see without per-cylinder flow), flagged here
    # rather than hidden.
    fuel_per_cyl = [fuel_total_g_min * _G_MIN_TO_KG_S / _N_CYL] * _N_CYL

    frame = {
        "t_s": payload.get("t_s"),
        "scenario": payload.get("scenario", "mavlink"),
        "N_rpm": _need(efi, "rpm", "efi"),
        "MAP_Pa": _need(efi, "intake_manifold_pressure_kpa", "efi") * 1000.0,
        "altitude_m": _need(payload, "altitude_m", "payload"),
        "fuel_flow_kg_s_per_cyl": fuel_per_cyl,
        "CHT_K": [c + 273.15 for c in cht_c],
        "EGT_K": [c + 273.15 for c in egt_c],
        "p_oil_Pa": _need(efi, "oil_pressure_kpa", "efi") * 1000.0,
        "T_oil_K": _need(efi, "oil_temperature_c", "efi") + 273.15,
    }
    if frame["t_s"] is None:
        raise FrameError("payload.t_s is required (seconds since start)")
    return normalize(frame)


def to_mavlink(frame: dict) -> dict:
    """The reverse map, for tests and for emitting twin-compatible
    telemetry back onto a MAVLink-shaped channel. Internal frame ->
    EFI-style envelope."""
    return {
        "t_s": frame["t_s"],
        "altitude_m": frame["altitude_m"],
        "efi": {
            "rpm": frame["N_rpm"],
            "intake_manifold_pressure_kpa": frame["MAP_Pa"] / 1000.0,
            "fuel_flow_g_min":
                sum(frame["fuel_flow_kg_s_per_cyl"]) / _G_MIN_TO_KG_S,
            "cylinder_head_temperature_c": [c - 273.15 for c in frame["CHT_K"]],
            "exhaust_gas_temperature_c": [c - 273.15 for c in frame["EGT_K"]],
            "oil_pressure_kpa": frame["p_oil_Pa"] / 1000.0,
            "oil_temperature_c": frame["T_oil_K"] - 273.15,
        },
    }
