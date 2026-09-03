"""twin/anomaly.py -- band-exit detection with persistence.

The anomaly signal is the normalised residual leaving the band (Handbook
6.2, uncertainty-quantification.md), not a learned detector. The isolation
forest of ml-layer.md remains the post-MVP upgrade; for the demo the
computed band IS the detector, and it is the thing no threshold system can
replicate.

Persistence (demo-script Acts 3/4 need this to be stable): an alarm goes
active only after ALARM_ON_STEPS consecutive out-of-band steps on the same
channel, and clears after ALARM_OFF_STEPS consecutive in-band steps.
Single-sample noise excursions never alarm; a real excursion persists.
"""

ALARM_ON_STEPS = 3
ALARM_OFF_STEPS = 10
Z_CAUTION = 2.0
Z_WARNING = 3.0


class AnomalyMonitor:
    """Stateful per-channel persistence counters.

    Channels are (name, cylinder) pairs, e.g. ("EGT_K", 3), plus
    ("p_oil_Pa", None) and ("T_oil_K", None).
    """

    def __init__(self):
        self._on = {}        # channel -> consecutive out-of-band count
        self._off = {}       # channel -> consecutive in-band count
        self._active = {}    # channel -> True while alarming
        self._since = None   # t_s when the current alarm episode began
        self.reset()

    def reset(self):
        self._on = {}
        self._off = {}
        self._active = {}
        self._since = None

    def step(self, state: dict) -> dict:
        """Consume one twin state, return the alarm block for it."""
        zs = {}
        for c in state["cylinders"]:
            zs[("EGT_K", c["n"])] = c["z_EGT"]
            zs[("CHT_K", c["n"])] = c["z_CHT"]
        zs[("p_oil_Pa", None)] = state["oil"]["z_p"]
        zs[("T_oil_K", None)] = state["oil"]["z_T"]

        for ch, z in zs.items():
            if abs(z) >= Z_WARNING:
                self._on[ch] = self._on.get(ch, 0) + 1
                self._off[ch] = 0
            else:
                self._off[ch] = self._off.get(ch, 0) + 1
                self._on[ch] = 0
            if self._on[ch] >= ALARM_ON_STEPS:
                self._active[ch] = True
            if self._off[ch] >= ALARM_OFF_STEPS:
                self._active.pop(ch, None)

        worst = 0.0
        worst_ch = None
        for ch, z in zs.items():
            if abs(z) > worst:
                worst, worst_ch = abs(z), ch

        active = bool(self._active)
        if active and self._since is None:
            self._since = state["t_s"]
        if not active:
            self._since = None
        if active:
            level = "warning"
        elif worst >= Z_CAUTION:
            level = "caution"
        else:
            level = "nominal"

        name, cyl = worst_ch if worst_ch is not None else (None, None)
        return {
            "active": active,
            "level": level,
            "since_t_s": self._since,
            "channel": name if active or level != "nominal" else None,
            "cylinder": cyl if active or level != "nominal" else None,
            "active_channels": [f"{n} cyl {c}" if c else n
                                for (n, c) in sorted(self._active, key=str)],
        }
