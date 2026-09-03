"""twin/diagnose.py -- rule-based differential diagnosis.

Implements the z sign-pattern table from
03-Design/uncertainty-quantification.md ("How faults become sign patterns"),
which is the system-level encoding of 02-Research/fault-signatures.md.
Every rule is a researched fault signature; no rule invents a direction.

Output is the ranked, confidence-scored differential diagnosis of
ml-layer.md section 2: candidate causes with evidence strings, never a bare
anomaly score. Confidence is a softmax over per-candidate scores derived
from |z| magnitudes: it ranks, it is not a calibrated probability
(consistent with the project's no-overclaiming rule, the label says
"confidence", not "probability").

The supervised classifier of ml-layer.md is the post-MVP upgrade of this
module; the rule set here remains its sanity baseline.
"""

from collections import deque

import numpy as np

Z_STRONG = 3.0     # a channel counts as "moved" beyond this (== alarm threshold)
Z_FLAT = 1.0       # a channel counts as "uncorroborated" below this
LAG_CONFIRM_S = 60.0   # ~2x tau_CHT upper range: a real combustion change
                       # must reach CHT within this window (Handbook 7.8)

# Turbo: commanded-vs-achieved MAP deficit (Handbook 7.2). The monitor keeps
# a window of (altitude, deficit) and looks for the documented behaviour: a
# deficit that is present and grows with altitude.
TURBO_WINDOW_S = 60.0
TURBO_MIN_DEFICIT_PA = 3000.0


class Diagnoser:
    def __init__(self):
        self._map_window = deque(maxlen=int(TURBO_WINDOW_S))
        self._out_since = {}   # (channel, cylinder) -> t_s when it left the band

    def reset(self):
        self._map_window.clear()
        self._out_since.clear()

    def step(self, state: dict) -> list:
        """Twin state in, ranked candidate list out (may be empty)."""
        cands = []
        physical_cyls = set()
        t_now = state["t_s"]
        for c in state["cylinders"]:
            n, ze, zc = c["n"], c["z_EGT"], c["z_CHT"]
            # Deviation in Kelvin as well as in z: the researched distinction
            # between a dead cylinder (a collapse of hundreds of K) and a
            # severe lean-out (tens of K) is magnitude, and z hides magnitude
            # behind the band width.
            dev_e = c["EGT_K"] - c["EGT_pred_K"]
            ev = [f"z_EGT({n}) = {ze:+.1f}", f"z_CHT({n}) = {zc:+.1f}",
                  f"EGT deviation {dev_e:+.0f} K"]
            # Lag-ratio timing (fault-signatures.md section 7): a real
            # combustion change shows in EGT within seconds and MUST reach
            # CHT within about two CHT time constants. An EGT excursion older
            # than that with CHT still flat is sensor behaviour, not
            # combustion.
            egt_out = abs(ze) >= Z_STRONG
            cht_out = abs(zc) >= Z_STRONG
            for ch, out in (("EGT_K", egt_out), ("CHT_K", cht_out)):
                key = (ch, n)
                if out:
                    self._out_since.setdefault(key, t_now)
                else:
                    self._out_since.pop(key, None)
            egt_age = (t_now - self._out_since.get(("EGT_K", n), t_now))
            egt_unconfirmed = egt_out and abs(zc) < Z_FLAT and \
                egt_age > LAG_CONFIRM_S
            if ze < -Z_STRONG and zc < -Z_STRONG and dev_e <= -150.0:
                # Thermal collapse on one cylinder: both channels fall hard
                # (a dead cylinder runs cold, fault-signatures.md section 1).
                # The Kelvin bound is the researched distinction between this
                # and a severe lean-out, which moves the same direction but
                # by tens of K, not hundreds.
                cands.append((f"misfire / dead cylinder, cyl {n}",
                              (abs(ze) + abs(zc)) / 2, ev))
                physical_cyls.add(n)
            elif ze < -Z_STRONG and zc < -Z_FLAT * 2 and dev_e > -150.0:
                # Same sign pattern, modest magnitude: the cylinder leaned
                # PAST the mixture peak, so EGT fell (fault-signatures.md
                # section 3: severity inverts the sign of the same fault).
                cands.append((f"injector restriction (severe), cyl {n}",
                              (abs(ze) + abs(zc)) / 2,
                              ev + ["both channels down moderately: mixture "
                                    "driven past peak, not a dead cylinder"]))
                physical_cyls.add(n)
            elif zc > Z_STRONG and ze < -Z_STRONG:
                # Opposite-sign pair from one cause (fault-signatures.md
                # section 2): heat into the head, not out the exhaust.
                cands.append((f"detonation / advanced timing, cyl {n}",
                              (zc + abs(ze)) / 2, ev))
                physical_cyls.add(n)
            elif ze > Z_STRONG and zc < Z_STRONG and not egt_unconfirmed:
                # One cylinder's EGT rises without a matched CHT move:
                # leaning toward the mixture peak (partial restriction).
                # Only while young or CHT-corroborated: an old EGT-only
                # excursion is the sensor-drift case below.
                cands.append((f"injector restriction (partial), cyl {n}",
                              ze, ev + ["CHT corroboration weak, consistent "
                                        "with a lean-ward mixture shift"]))
                physical_cyls.add(n)
            elif zc > Z_STRONG and abs(ze) < Z_FLAT:
                # Head hot, cycle untouched (fault-signatures.md row 5).
                cands.append((f"cooling degradation, cyl {n}", zc, ev))
                physical_cyls.add(n)

        cands += self._oil(state)
        cands += self._turbo(state)
        cands += self._sensor_drift(state, physical_cyls)

        if not cands:
            return []
        scores = np.array([s for _, s, _ in cands], dtype=float)
        conf = np.exp(scores - scores.max())
        conf /= conf.sum()
        order = np.argsort(-conf)
        return [{"rank": r + 1,
                 "label": cands[i][0],
                 "confidence": round(float(conf[i]), 2),
                 "evidence": cands[i][2]}
                for r, i in enumerate(order)]

    def _oil(self, state):
        zt, zp = state["oil"]["z_T"], state["oil"]["z_p"]
        if zt > Z_STRONG and zp < -Z_STRONG:
            return [("bearing wear (oil temp up, pressure down, warm engine)",
                     (zt + abs(zp)) / 2,
                     [f"z_oil_temp = {zt:+.1f}", f"z_oil_press = {zp:+.1f}"])]
        return []

    def _turbo(self, state):
        cmd = state["inputs"].get("MAP_commanded_Pa", state["inputs"]["MAP_Pa"])
        deficit = cmd - state["inputs"]["MAP_Pa"]
        alt = state["inputs"]["altitude_m"]
        self._map_window.append((alt, deficit))
        if len(self._map_window) < 10 or deficit < TURBO_MIN_DEFICIT_PA:
            return []
        alts = np.array([w[0] for w in self._map_window])
        defs = np.array([w[1] for w in self._map_window])
        grows = (np.corrcoef(alts, defs)[0, 1] > 0.5
                 if np.ptp(alts) > 500 else False)
        score = deficit / TURBO_MIN_DEFICIT_PA + (0.5 if grows else 0.0)
        ev = [f"achieved MAP {deficit/1000:.1f} kPa below commanded at "
              f"{alt:,.0f} m".replace(",", " ")]
        if grows:
            ev.append("deficit grows with altitude over the last minute")
        return [("turbocharger degradation", score, ev)]

    def _sensor_drift(self, state, physical_cyls):
        """One channel out of band, nothing physically coupled corroborates
        (Handbook 7.8: a real combustion change must propagate EGT -> CHT;
        fault-signatures.md row 7). Drift is a diagnosis of EXCLUSION: while
        a physical fault still explains the same channel (young excursion,
        CHT may simply not have caught up), drift stays on the list but
        damped, exactly like the 0.06-confidence row in ml-layer.md. Once
        the excursion has outlived the lag-confirmation window with CHT
        still flat, drift becomes the leading explanation."""
        out = []
        t_now = state["t_s"]
        for c in state["cylinders"]:
            ze, zc, n = c["z_EGT"], c["z_CHT"], c["n"]
            egt_only = abs(ze) >= Z_STRONG and abs(zc) < Z_FLAT
            cht_only = abs(zc) >= Z_STRONG and abs(ze) < Z_FLAT
            others_quiet = all(
                max(abs(o["z_EGT"]), abs(o["z_CHT"])) < Z_STRONG
                for o in state["cylinders"] if o["n"] != n)
            egt_age = t_now - self._out_since.get(("EGT_K", n), t_now)
            cht_age = t_now - self._out_since.get(("CHT_K", n), t_now)
            if egt_only and others_quiet:
                old = egt_age > LAG_CONFIRM_S
                damp = 1.0 if old else (0.5 if n in physical_cyls else 0.8)
                ev = [f"z_EGT({n}) = {ze:+.1f}",
                      "no CHT response on the same cylinder",
                      "no other channel corroborates"]
                if old:
                    ev.append(f"EGT-only for {egt_age:.0f} s, past the "
                              "lag-confirmation window")
                out.append((f"sensor drift, EGT cyl {n}", abs(ze) * damp, ev))
            elif cht_only and others_quiet:
                old = cht_age > LAG_CONFIRM_S
                damp = 1.0 if old else (0.5 if n in physical_cyls else 0.8)
                out.append((f"sensor drift, CHT cyl {n}", abs(zc) * damp,
                            [f"z_CHT({n}) = {zc:+.1f}",
                             "no EGT move precedes it",
                             "no other channel corroborates"]))
        return out
