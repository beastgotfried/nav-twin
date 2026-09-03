"""physics/sigma_lookup.py -- the DEPLOYED side of Handbook 6.9.

build_sigma_table.py precomputes Monte Carlo predictive stds over a grid of
operating points and saves them to data/sigma_table.npz (the offline side of
Handbook 6.9). This module is what runtime code actually calls: a linear
interpolation into that table (scipy RegularGridInterpolator) with queries
clamped to the grid edges, i.e. nearest-value filling outside the grid. The
per-timestep cost of the uncertainty band collapses from a Monte Carlo run
(verify_uncertainty.py measures ~5.5 s per CHT sample set) to one
interpolation, which is the whole point of the Handbook 6.9 deployment
architecture.

Channels:
- sigma_egt(phi, T_amb_K): 2D over (phi, T_amb), from the cheap algebraic
  EGT route (Handbook 7.6).
- sigma_cht(N_rpm, MAP_Pa, phi, T_amb_K): 4D over (N, MAP, phi, T_amb),
  from the crank-angle route (Handbook 7.7).
- The oil channels have no MC propagation yet (Phase 2), so the table
  carries scalar ASSUMED sigmas, exposed here as the module constants
  SIGMA_P_OIL_PA and SIGMA_T_OIL_K. Once the npz is loaded the values come
  from the file; until then they hold the same ASSUMED values mirrored from
  build_sigma_table.py (identical by construction, see below).

Fallback while the npz does not exist: the offline build takes ~10-15 min
(build_sigma_table.py docstring), so on first use without the file a
REDUCED table is built in memory over the same grids with fewer samples per
point (500 EGT / 40 CHT instead of 3000 / 150) and a warning is logged.
Downstream code always works; the band is just noisier until the full table
lands. verify_sigma_table.py reports which table it actually checked.
"""

import logging
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import FA_STOICH
from .engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from .intake import air_mass_per_cycle_per_cyl_kg
from .uncertainty import propagate_egt_uncertainty, propagate_cht_uncertainty

log = logging.getLogger(__name__)

TABLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sigma_table.npz"

# Sample counts for the reduced in-memory fallback (full table: 3000 / 150,
# see build_sigma_table.py). Chosen so the fallback builds in a few minutes
# rather than the ~10-15 min of the offline build, while keeping per-point
# std sampling error near 3% (EGT) / 11% (CHT): 1/sqrt(2*(n-1)) for n
# Gaussian samples.
REDUCED_EGT_SAMPLES = 500
REDUCED_CHT_SAMPLES = 40

# Grid axes and the ASSUMED oil sigmas live in build_sigma_table.py (single
# source: it is what writes them into the npz). Imported here so the module
# constants exist before first use; the npz values overwrite them on load.
try:
    import build_sigma_table as _bst
except ImportError:  # simulator not on sys.path (standalone physics use)
    _bst = None

if _bst is not None:
    SIGMA_P_OIL_PA = float(_bst.SIGMA_P_OIL_PA)  # ASSUMED, see build_sigma_table.py
    SIGMA_T_OIL_K = float(_bst.SIGMA_T_OIL_K)    # ASSUMED, see build_sigma_table.py
else:
    # Mirror of build_sigma_table.py; keep in sync. Overwritten from the npz
    # on first table load, identical by construction.
    SIGMA_P_OIL_PA = 15_000.0  # ASSUMED
    SIGMA_T_OIL_K = 3.0        # ASSUMED


class SigmaTable:
    """Interpolated view of one sigma table (full npz or reduced fallback).

    Linear interpolation inside the grid; queries outside the grid are
    clamped to the nearest edge value, so the lookup never extrapolates a
    noise band into operating regions the offline build never sampled.
    """

    def __init__(self, *, phi_grid, t_amb_grid, egt_mean, egt_std,
                 n_grid, map_grid, cht_phi_grid, cht_t_amb_grid,
                 cht_mean, cht_std, sigma_p_oil_pa, sigma_t_oil_k,
                 source, reduced):
        self.source = source
        self.reduced = reduced
        self.phi_grid = np.asarray(phi_grid, dtype=float)
        self.t_amb_grid = np.asarray(t_amb_grid, dtype=float)
        self.egt_mean = np.asarray(egt_mean, dtype=float)
        self.egt_std = np.asarray(egt_std, dtype=float)
        self.n_grid = np.asarray(n_grid, dtype=float)
        self.map_grid = np.asarray(map_grid, dtype=float)
        self.cht_phi_grid = np.asarray(cht_phi_grid, dtype=float)
        self.cht_t_amb_grid = np.asarray(cht_t_amb_grid, dtype=float)
        self.cht_mean = np.asarray(cht_mean, dtype=float)
        self.cht_std = np.asarray(cht_std, dtype=float)
        self.sigma_p_oil_pa = float(sigma_p_oil_pa)
        self.sigma_t_oil_k = float(sigma_t_oil_k)
        self._egt_interp = RegularGridInterpolator(
            (self.phi_grid, self.t_amb_grid), self.egt_std,
            method="linear", bounds_error=False, fill_value=np.nan)
        self._cht_interp = RegularGridInterpolator(
            (self.n_grid, self.map_grid, self.cht_phi_grid, self.cht_t_amb_grid),
            self.cht_std, method="linear", bounds_error=False, fill_value=np.nan)

    @staticmethod
    def _clamp(x, grid):
        """Nearest-edge fill: fold an out-of-grid query onto the boundary."""
        return float(min(max(float(x), grid[0]), grid[-1]))

    def sigma_egt(self, phi, t_amb_k):
        pt = (self._clamp(phi, self.phi_grid),
              self._clamp(t_amb_k, self.t_amb_grid))
        return float(self._egt_interp(pt))

    def sigma_cht(self, n_rpm, map_pa, phi, t_amb_k):
        pt = (self._clamp(n_rpm, self.n_grid),
              self._clamp(map_pa, self.map_grid),
              self._clamp(phi, self.cht_phi_grid),
              self._clamp(t_amb_k, self.cht_t_amb_grid))
        return float(self._cht_interp(pt))


def _load_from_npz():
    with np.load(TABLE_PATH) as d:
        table = SigmaTable(
            phi_grid=d["phi_grid"], t_amb_grid=d["t_amb_grid"],
            egt_mean=d["egt_mean"], egt_std=d["egt_std"],
            n_grid=d["n_grid"], map_grid=d["map_grid"],
            cht_phi_grid=d["cht_phi_grid"], cht_t_amb_grid=d["cht_t_amb_grid"],
            cht_mean=d["cht_mean"], cht_std=d["cht_std"],
            sigma_p_oil_pa=d["sigma_p_oil_pa"],
            sigma_t_oil_k=d["sigma_t_oil_k"],
            source=str(TABLE_PATH), reduced=False)
    log.info("sigma table loaded from %s", TABLE_PATH)
    return table


def _build_reduced():
    """Same grids and propagation calls as build_sigma_table.py, fewer samples."""
    if _bst is None:
        raise RuntimeError(
            f"{TABLE_PATH} is missing and build_sigma_table.py is not "
            "importable (simulator not on sys.path): cannot build the "
            "reduced fallback table.")
    rng = np.random.default_rng(20260830)  # fixed seed: reproducible, not cherry-picked
    t0 = time.time()

    egt_std = np.empty((len(_bst.PHI_GRID), len(_bst.T_AMB_GRID)))
    egt_mean = np.empty_like(egt_std)
    for i, phi in enumerate(_bst.PHI_GRID):
        for j, t_amb in enumerate(_bst.T_AMB_GRID):
            r = propagate_egt_uncertainty(phi, t_amb, t_amb,
                                          n_samples=REDUCED_EGT_SAMPLES, rng=rng)
            egt_std[i, j], egt_mean[i, j] = r["std"], r["mean"]

    shape = (len(_bst.N_GRID), len(_bst.MAP_GRID),
             len(_bst.CHT_PHI_GRID), len(_bst.CHT_T_AMB_GRID))
    cht_std = np.empty(shape)
    cht_mean = np.empty(shape)
    for a, n in enumerate(_bst.N_GRID):
        for b, map_pa in enumerate(_bst.MAP_GRID):
            for c, phi in enumerate(_bst.CHT_PHI_GRID):
                for d, t_amb in enumerate(_bst.CHT_T_AMB_GRID):
                    air = air_mass_per_cycle_per_cyl_kg(n, map_pa, t_amb,
                                                        DEFAULT_GEOMETRY,
                                                        DEFAULT_CONSTANTS)
                    m_charge = air * (1.0 + phi * FA_STOICH)
                    r = propagate_cht_uncertainty(n, map_pa, t_amb, t_amb, phi,
                                                  m_charge,
                                                  n_samples=REDUCED_CHT_SAMPLES,
                                                  rng=rng)
                    cht_std[a, b, c, d], cht_mean[a, b, c, d] = r["std"], r["mean"]

    log.warning("reduced in-memory sigma table built in %.0f s "
                "(EGT std %.1f..%.1f K, CHT std %.1f..%.1f K); "
                "run build_sigma_table.py for the full table",
                time.time() - t0, egt_std.min(), egt_std.max(),
                cht_std.min(), cht_std.max())
    return SigmaTable(
        phi_grid=_bst.PHI_GRID, t_amb_grid=_bst.T_AMB_GRID,
        egt_mean=egt_mean, egt_std=egt_std,
        n_grid=_bst.N_GRID, map_grid=_bst.MAP_GRID,
        cht_phi_grid=_bst.CHT_PHI_GRID, cht_t_amb_grid=_bst.CHT_T_AMB_GRID,
        cht_mean=cht_mean, cht_std=cht_std,
        sigma_p_oil_pa=_bst.SIGMA_P_OIL_PA, sigma_t_oil_k=_bst.SIGMA_T_OIL_K,
        source=(f"reduced in-memory fallback ({REDUCED_EGT_SAMPLES} EGT / "
                f"{REDUCED_CHT_SAMPLES} CHT samples per grid point)"),
        reduced=True)


_TABLE = None


def get_table():
    """The process-wide table, loaded or built on first call (lazy)."""
    global _TABLE, SIGMA_P_OIL_PA, SIGMA_T_OIL_K
    if _TABLE is None:
        if TABLE_PATH.exists():
            _TABLE = _load_from_npz()
        else:
            log.warning("%s not found; building the REDUCED in-memory "
                        "fallback table (%d EGT / %d CHT samples per grid "
                        "point, takes a few minutes). Bands are noisier than "
                        "the offline table; use them for plumbing checks "
                        "only, not for trusting z-scores.",
                        TABLE_PATH, REDUCED_EGT_SAMPLES, REDUCED_CHT_SAMPLES)
            _TABLE = _build_reduced()
        SIGMA_P_OIL_PA = _TABLE.sigma_p_oil_pa
        SIGMA_T_OIL_K = _TABLE.sigma_t_oil_k
    return _TABLE


def sigma_egt(phi, T_amb_K):
    """Predictive std of EGT (K) at one operating point. Handbook 6.9."""
    return get_table().sigma_egt(phi, T_amb_K)


def sigma_cht(N_rpm, MAP_Pa, phi, T_amb_K):
    """Predictive std of CHT (K) at one operating point. Handbook 6.9."""
    return get_table().sigma_cht(N_rpm, MAP_Pa, phi, T_amb_K)


# --- Residual (input-uncertainty-only) table -------------------------------
# The band the twin ALARMS on. The full table above includes systematic
# parameter uncertainty, which the frozen baseline correction absorbs
# (Handbook 6.6/6.7); what remains is input measurement uncertainty, built
# offline by build_sigma_residual_table.py. Sensor noise is iid and added in
# quadrature by the caller, not stored here.

RESIDUAL_TABLE_PATH = (Path(__file__).resolve().parent.parent / "data"
                       / "sigma_residual_table.npz")

_RESIDUAL = None


def _load_residual():
    with np.load(RESIDUAL_TABLE_PATH) as d:
        # Reuse SigmaTable's interpolators; means and oil scalars are not
        # stored in the residual npz and are irrelevant to it.
        return SigmaTable(
            phi_grid=d["phi_grid"], t_amb_grid=d["t_amb_grid"],
            egt_mean=np.zeros((len(d["phi_grid"]), len(d["t_amb_grid"]))),
            egt_std=d["egt_std"],
            n_grid=d["n_grid"], map_grid=d["map_grid"],
            cht_phi_grid=d["cht_phi_grid"], cht_t_amb_grid=d["cht_t_amb_grid"],
            cht_mean=np.zeros((len(d["n_grid"]), len(d["map_grid"]),
                               len(d["cht_phi_grid"]), len(d["cht_t_amb_grid"]))),
            cht_std=d["cht_std"],
            sigma_p_oil_pa=SIGMA_P_OIL_PA, sigma_t_oil_k=SIGMA_T_OIL_K,
            source=str(RESIDUAL_TABLE_PATH), reduced=False)


def get_residual_table():
    """The process-wide residual table, loaded on first call. Unlike the
    full table there is no in-memory fallback: alarming on a reduced band
    would silently change demo behaviour, so a missing file is an error."""
    global _RESIDUAL
    if _RESIDUAL is None:
        if not RESIDUAL_TABLE_PATH.exists():
            raise FileNotFoundError(
                f"{RESIDUAL_TABLE_PATH} missing; run "
                "build_sigma_residual_table.py first")
        _RESIDUAL = _load_residual()
    return _RESIDUAL


def sigma_egt_residual(phi, T_amb_K):
    """Input-uncertainty-only predictive std of EGT (K). Handbook 6.7."""
    return get_residual_table().sigma_egt(phi, T_amb_K)


def sigma_cht_residual(N_rpm, MAP_Pa, phi, T_amb_K):
    """Input-uncertainty-only predictive std of CHT (K). Handbook 6.7."""
    return get_residual_table().sigma_cht(N_rpm, MAP_Pa, phi, T_amb_K)
