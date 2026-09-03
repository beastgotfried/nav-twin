"""twin -- the twin core. Facade plus the pipeline stages.

Twin.step(frame) runs residual -> anomaly -> diagnose and returns the state
dict specified in app/README.md. The server streams it live; replay.py
runs it over logged missions; verify_twin.py asserts the demo acts on it.
"""

import sys
from pathlib import Path

# The physics and the mission generator live in simulator.
_SIM = str(Path(__file__).resolve().parent.parent.parent / "simulator")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from .residual import ResidualEngine
from .anomaly import AnomalyMonitor
from .diagnose import Diagnoser

# The first seconds of a mission are assumed healthy and used to fit the
# frozen baseline delta (Handbook 6.6). ASSUMPTION, documented: a real
# deployment fits delta on missions confirmed healthy after inspection, not
# on the current mission's opening seconds (02-Research/ideas/
# integration-plug-in-architecture.md, "The continuous-training trap").
DEFAULT_CALIBRATE_S = 30.0


class Twin:
    def __init__(self, calibrate_s: float = DEFAULT_CALIBRATE_S):
        self.calibrate_s = calibrate_s
        self.residual = ResidualEngine()
        self.anomaly = AnomalyMonitor()
        self.diagnoser = Diagnoser()

    def reset(self):
        self.residual.reset()
        self.anomaly.reset()
        self.diagnoser.reset()

    def step(self, frame: dict) -> dict:
        state = self.residual.step(frame)
        if not self.residual.calibrated and frame["t_s"] >= self.calibrate_s:
            self.residual.freeze_baseline()
        state["alarm"] = self.anomaly.step(state)
        state["diagnosis"] = self.diagnoser.step(state)
        if state["alarm"]["active"] and not state["diagnosis"]:
            # The band fired but no researched signature matches: exactly the
            # case the unsupervised layer exists for (ml-layer.md section 1).
            state["diagnosis"] = [{
                "rank": 1, "label": "unexplained residual excursion",
                "confidence": 1.0,
                "evidence": ["outside the computed band on "
                             + (state["alarm"]["channel"] or "?"),
                             "no known fault signature matches"]}]
        return state


from .replay import replay_frames  # noqa: E402  (after Twin, avoids a cycle)
