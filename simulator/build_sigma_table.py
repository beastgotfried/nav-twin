"""build_sigma_table.py -- the OFFLINE side of Handbook 6.9.

Monte Carlo propagation is far too slow to run live (verify_uncertainty.py
measures ~5.5 s per CHT sample set; the integration note in
02-Research/ideas/ puts a full mission at hours). So the uncertainty band
is precomputed here ONCE over a grid of operating points, saved to
data/sigma_table.npz, and deployed as an interpolated lookup
(physics/sigma_lookup.py). Runtime cost collapses to a table lookup plus a
subtraction, exactly as Handbook 6.9 specifies.

Grids:
- EGT: (phi, T_amb), algebraic route, cheap at 3000 samples/point.
- CHT: (N, MAP, phi, T_amb), crank-angle route, expensive (~10-15 min for
  the whole grid on a workstation). Coarse on purpose: the band varies
  smoothly away from the phi kink, and the phi axis is densest where the
  kink lives.
- Oil channels: no MC propagation exists for the oil subsystem yet, so the
  table carries scalar ASSUMED sigmas (documented in sigma_lookup.py).

Run:  python build_sigma_table.py
"""

import time
from pathlib import Path

import numpy as np

from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.intake import air_mass_per_cycle_per_cyl_kg
from physics.constants import FA_STOICH
from physics.uncertainty import (propagate_egt_uncertainty,
                                 propagate_cht_uncertainty,
                                 DEFAULT_UNCERTAINTY)

OUT = Path(__file__).parent / "data" / "sigma_table.npz"

# Grid axes. phi is dense around the kink at phi=1 (simulator README:
# slope collapses ~14.7x crossing into the rich regime).
PHI_GRID = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.02, 1.05, 1.08,
                     1.10, 1.15, 1.20, 1.25])
T_AMB_GRID = np.array([230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 288.15,
                       300.0, 310.0, 320.0])
N_GRID = np.array([4000.0, 4750.0, 5500.0])
MAP_GRID = np.array([90_000.0, 115_000.0, 140_000.0])
CHT_PHI_GRID = np.array([0.80, 0.90, 1.00, 1.05, 1.10, 1.20])
CHT_T_AMB_GRID = np.array([240.0, 288.15, 310.0])

# Scalar ASSUMED sigmas for channels without MC propagation (oil subsystem)
# and for the oil channels generally, pending Phase 2. Documented as ASSUMED
# in sigma_lookup.py; they exist so z is defined on every channel.
SIGMA_P_OIL_PA = 15_000.0
SIGMA_T_OIL_K = 3.0


def build_egt(rng):
    std = np.empty((len(PHI_GRID), len(T_AMB_GRID)))
    mean = np.empty_like(std)
    t0 = time.time()
    for i, phi in enumerate(PHI_GRID):
        for j, t_amb in enumerate(T_AMB_GRID):
            r = propagate_egt_uncertainty(phi, t_amb, t_amb, n_samples=3000,
                                          rng=rng)
            std[i, j], mean[i, j] = r["std"], r["mean"]
    print(f"EGT grid: {std.size} points in {time.time()-t0:.1f}s, "
          f"std range {std.min():.1f}..{std.max():.1f} K")
    return mean, std


def build_cht(rng):
    shape = (len(N_GRID), len(MAP_GRID), len(CHT_PHI_GRID), len(CHT_T_AMB_GRID))
    std = np.empty(shape)
    mean = np.empty(shape)
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
                        n_samples=150, rng=rng)
                    std[a, b, c, d], mean[a, b, c, d] = r["std"], r["mean"]
                    done += 1
                    if done % 18 == 0:
                        el = time.time() - t0
                        print(f"  CHT {done}/{std.size} "
                              f"({el:.0f}s elapsed, ~{el/done*(std.size-done):.0f}s left)",
                              flush=True)
    print(f"CHT grid: {std.size} points in {time.time()-t0:.1f}s, "
          f"std range {std.min():.1f}..{std.max():.1f} K")
    return mean, std


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260829)
    egt_mean, egt_std = build_egt(rng)
    cht_mean, cht_std = build_cht(rng)
    np.savez_compressed(
        OUT,
        phi_grid=PHI_GRID, t_amb_grid=T_AMB_GRID,
        egt_mean=egt_mean, egt_std=egt_std,
        n_grid=N_GRID, map_grid=MAP_GRID,
        cht_phi_grid=CHT_PHI_GRID, cht_t_amb_grid=CHT_T_AMB_GRID,
        cht_mean=cht_mean, cht_std=cht_std,
        sigma_p_oil_pa=np.float64(SIGMA_P_OIL_PA),
        sigma_t_oil_k=np.float64(SIGMA_T_OIL_K),
        uncertainty_spec=str(DEFAULT_UNCERTAINTY),
    )
    print(f"saved {OUT} ({OUT.stat().st_size/1024:.0f} KiB)")
