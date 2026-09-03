"""Handbook Part 7.8 -- Step 7: first-order thermal lag.

DERIVED functional form (standard first-order lag), tau values are
ASSUMED/DERIVED-range per Handbook 7.8 ("1 to 2 s" for EGT, "10 to 30 s"
for CHT). This is what makes the lag-ratio sensor-drift check in Handbook
7.8 possible: a genuine combustion change must show up in EGT quickly and
CHT slowly; a sensor fault does not respect that timing.
"""


def first_order_lag_step(current_value: float, steady_state_target: float,
                          tau_s: float, dt_s: float) -> float:
    """One explicit-Euler step of tau*dy/dt + y = target.
    Stable for dt_s << tau_s (true for our simulated telemetry rates)."""
    return current_value + (dt_s / tau_s) * (steady_state_target - current_value)
