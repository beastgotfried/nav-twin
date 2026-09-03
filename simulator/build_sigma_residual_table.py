"""build_sigma_residual_table.py -- the band the twin actually alarms on.

Why a second table: the full-joint Monte Carlo band (build_sigma_table.py)
is dominated by systematic PARAMETER uncertainty (k_egt's ASSUMED 15% alone
drives EGT std to ~90-120 K; see simulator README and Handbook 6.7). That
uncertainty is a constant offset across a mission, and the frozen baseline
correction (Handbook 6.6, delta(x) fit on healthy data then frozen) removes
it. What remains for alarming is the TRANSITORY part: input measurement
uncertainty (ambient temperature, fuel flow / phi) plus sensor noise.

So this table propagates ONLY the input uncertainty (UncertaintySpec with
every parameter sigma zeroed), on the same grids as the full table. The twin
adds the ASSUMED sensor-noise sigmas from mission.py in quadrature at lookup
time (sensor noise is iid, not a function of operating point, so it does not
belong in the grid).

Run:  python build_sigma_residual_table.py
"""

import time
from pathlib import Path

import numpy as np

from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.intake import air_mass_per_cycle_per_cyl_kg
from physics.constants import FA_STOICH
from physics.uncertainty import (propagate_egt_uncertainty,
                                 propagate_cht_uncertainty,
                                 UncertaintySpec)
from build_sigma_table import (PHI_GRID, T_AMB_GRID, N_GRID, MAP_GRID,
                               CHT_PHI_GRID, CHT_T_AMB_GRID)

OUT = Path(__file__).parent / "data" / "sigma_residual_table.npz"

# Input-only uncertainty: every calibration parameter held at nominal, only
# the transitory measurement uncertainties (ambient temperature, fuel flow
# via phi) propagate. Handbook 6.5 sources 1-2, with 3 absorbed by the
# frozen baseline and 4 explicitly out of scope for the band.
INPUT_ONLY = UncertaintySpec(
    k_egt_rel=0.0, R_th_rel=0.0, woschni_C_rel=0.0, wiebe_a_rel=0.0,
    wiebe_m_rel=0.0, ignition_timing_rel=0.0, burn_duration_rel=0.0,
    flame_speed_width_rel=0.0,
)


def build():
    rng = np.random.default_rng(20260829)
    egt_std = np.empty((len(PHI_GRID), len(T_AMB_GRID)))
    t0 = time.time()
    for i, phi in enumerate(PHI_GRID):
        for j, t_amb in enumerate(T_AMB_GRID):
            r = propagate_egt_uncertainty(phi, t_amb, t_amb, spec=INPUT_ONLY,
                                          n_samples=2000, rng=rng)
            egt_std[i, j] = r["std"]
    print(f"EGT residual grid: {egt_std.size} points in {time.time()-t0:.1f}s, "
          f"std range {egt_std.min():.1f}..{egt_std.max():.1f} K")

    shape = (len(N_GRID), len(MAP_GRID), len(CHT_PHI_GRID), len(CHT_T_AMB_GRID))
    cht_std = np.empty(shape)
    t0 = time.time()
    done = 0
    for a, n in enumerate(N_GRID):
        for b, map_pa in enumerate(MAP_GRID):
            for c, phi in enumerate(CHT_PHI_GRID):
                for d, t_amb in enumerate(CHT_T_AMB_GRID):
                    air = air_mass_per_cycle_per_cyl_kg(n, map_pa, t_amb,
                                                        DEFAULT_GEOMETRY,
                                                        DEFAULT_CONSTANTS)
                    m_charge = air * (1.0 + phi * FA_STOICH)
                    r = propagate_cht_uncertainty(
                        n, map_pa, t_amb, t_amb, phi, m_charge,
                        spec=INPUT_ONLY, n_samples=60, rng=rng)
                    cht_std[a, b, c, d] = r["std"]
                    done += 1
                    if done % 27 == 0:
                        el = time.time() - t0
                        print(f"  CHT {done}/{cht_std.size} ({el:.0f}s)", flush=True)
    print(f"CHT residual grid: {cht_std.size} points in {time.time()-t0:.1f}s, "
          f"std range {cht_std.min():.1f}..{cht_std.max():.1f} K")
    return egt_std, cht_std


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    egt_std, cht_std = build()
    np.savez_compressed(
        OUT,
        phi_grid=PHI_GRID, t_amb_grid=T_AMB_GRID, egt_std=egt_std,
        n_grid=N_GRID, map_grid=MAP_GRID,
        cht_phi_grid=CHT_PHI_GRID, cht_t_amb_grid=CHT_T_AMB_GRID,
        cht_std=cht_std,
        note="input-uncertainty-only propagation; sensor noise added in "
             "quadrature at lookup time by the twin",
    )
    print(f"saved {OUT} ({OUT.stat().st_size/1024:.0f} KiB)")
