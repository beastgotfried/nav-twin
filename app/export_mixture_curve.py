"""export_mixture_curve.py -- precompute the EGT/CHT mixture curves for the UI.

The mixture hill is the project's central physics argument: EGT is
non-monotonic in equivalence ratio, so the SAME exhaust temperature occurs
on both sides of the peak and a threshold on EGT is ambiguous by
construction (00-STREAM 2.4, Handbook 3.5). The dashboard needs to show it,
and the browser cannot run the physics, so it is computed here by the same
functions 09-Visuals/fig_mixture_hill.py uses and shipped as static JSON.

This is the same contract as export_canned.py: every point comes out of the
model. Nothing here is drawn, sketched or fitted to look right. Regenerate,
do not edit.

The curve is evaluated at ONE stated reference operating point, because the
curve's position shifts with intake and ambient temperature. The UI labels
it as such and places each cylinder on the curve by its own live phi, which
answers "which side of the peak is this cylinder on". It does not claim the
curve's EGT equals that cylinder's observed EGT.
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "simulator") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "simulator"))

from physics.atmosphere import isa_atmosphere            # noqa: E402
from physics.cycle import egt_steady_state_K              # noqa: E402
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS  # noqa: E402

OUT = Path(__file__).resolve().parent / "dashboard" / "public" / "canned"

# Reference operating point. Cruise at altitude, which is where these engines
# actually spend a mission and where the lean/rich decision is made.
REF_ALTITUDE_M = 4000.0
REF_T_IM_K = 320.0     # intake manifold temperature after the turbo and cooler
N_POINTS = 140
PHI_MIN, PHI_MAX = 0.68, 1.62


def build():
    atm = isa_atmosphere(REF_ALTITUDE_M)
    t_amb = atm["T_amb_K"] if isinstance(atm, dict) else atm.T_amb_K

    step = (PHI_MAX - PHI_MIN) / (N_POINTS - 1)
    phis = [PHI_MIN + i * step for i in range(N_POINTS)]
    egts = [
        egt_steady_state_K(phi, REF_T_IM_K, t_amb,
                           DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)["EGT_ss_K"]
        for phi in phis
    ]

    peak_i = max(range(len(egts)), key=lambda i: egts[i])

    payload = {
        "note": "computed by physics.combustion.egt_steady_state_K, not drawn",
        "reference": {
            "altitude_m": REF_ALTITUDE_M,
            "T_amb_K": round(t_amb, 2),
            "T_im_K": REF_T_IM_K,
        },
        "phi": [round(p, 4) for p in phis],
        "EGT_K": [round(e, 2) for e in egts],
        "peak": {"phi": round(phis[peak_i], 4), "EGT_K": round(egts[peak_i], 2)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mixture_curve.json"
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"  mixture_curve: {len(phis)} points, peak at phi="
          f"{payload['peak']['phi']}, EGT={payload['peak']['EGT_K']} K, "
          f"{path.stat().st_size} B")
    return payload


if __name__ == "__main__":
    print("computing the mixture curve from the real model")
    p = build()
    # The load-bearing claim, asserted here so a bad regeneration is caught
    # at build time rather than on a slide: the curve must be a hill, with
    # its peak strictly inside the swept range and both flanks falling away.
    egts = p["EGT_K"]
    i = egts.index(max(egts))
    assert 0 < i < len(egts) - 1, "peak is at an edge; the sweep is too narrow"
    assert egts[0] < egts[i] and egts[-1] < egts[i], "curve is not a hill"
    print("  OK: EGT is non-monotonic in phi, peak strictly interior")
