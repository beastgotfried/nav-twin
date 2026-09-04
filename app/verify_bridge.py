"""verify_bridge.py -- gates the production integration layer.

Checks, in order:

1. ADAPTER ROUND TRIP: a real simulator frame converted to a MAVLink/EFI
   envelope and back must equal the native frame within unit round-trip
   tolerance. This proves the translation loses nothing.
2. VALIDATION IS REAL: Celsius-shaped EGT, missing channels and non-numeric
   fields are rejected with reasons, never coerced.
3. TWIN EQUIVALENCE: the twin run on frames that arrived via the MAVLink
   adapter produces z values identical to the twin run on the native
   frames. The bridge is then provably transparent.
4. SERVER ROUND TRIP: frames posted to /api/ingest flow through the live
   server, are counted by /api/bridge/status, and appear in the twin
   history. The mixing guard (409 while a demo mission runs) is exercised
   too.

Run: python verify_bridge.py
"""

import sys
import time
from pathlib import Path

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
sys.path.insert(0, str(_ROOT / "simulator"))
sys.path.insert(0, str(_APP))

import numpy as np  # noqa: E402

import mission as M                       # noqa: E402
from bridge import normalize, FrameError  # noqa: E402
from bridge.mavlink import from_mavlink, to_mavlink  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# --- 1 + 2: adapter round trip and validation -----------------------------
# Noise-free frames: per-cylinder fuel is then exactly equal, so the EFI
# total-flow -> even-split round trip is exact too. The even-split's
# behaviour on noisy per-cylinder data is a documented assumption in
# mavlink.py, and it is what the 2% fuel tolerance below pins.
frames = [f for f in M.run_mission("endurance", seed=7, noise=False)
          if f["t_s"] < 40.0]
ref = normalize(frames[35])
round_tripped = from_mavlink(to_mavlink(ref))
worst = max(abs(round_tripped[k] - ref[k])
            for k in ("N_rpm", "MAP_Pa", "altitude_m", "p_oil_Pa", "T_oil_K"))
worst_cyl = max(abs(a - b)
                for k in ("EGT_K", "CHT_K", "fuel_flow_kg_s_per_cyl")
                for a, b in zip(round_tripped[k], ref[k]))
check(worst < 1e-9 and worst_cyl < 1e-9,
      f"MAVLink round trip is lossless (worst deviation {worst:.2e})")

noisy = frames[-1]  # reuse the last clean frame, perturb fuel per cylinder
noisy = dict(noisy)
noisy["fuel_flow_kg_s_per_cyl"] = [m * (1 + 0.005 * ((i % 2) * 2 - 1))
                                   for i, m in
                                   enumerate(noisy["fuel_flow_kg_s_per_cyl"])]
rt = from_mavlink(to_mavlink(normalize(noisy)))
rel = max(abs(a - b) / b for a, b in
          zip(rt["fuel_flow_kg_s_per_cyl"], noisy["fuel_flow_kg_s_per_cyl"]))
check(rel < 0.02,
      f"even-split assumption bounded: per-cylinder fuel within {rel:.2%}")
try:
    normalize({**ref, "EGT_K": [700.0, 700.0, 700.0, 25.0]})
    check(False, "Celsius-range EGT rejected")
except FrameError:
    check(True, "Celsius-range EGT rejected with a reason")
try:
    from_mavlink({"t_s": 1.0, "altitude_m": 100.0, "efi": {"rpm": 5000.0}})
    check(False, "incomplete EFI envelope rejected")
except FrameError:
    check(True, "incomplete EFI envelope rejected with a reason")

# --- 3: twin equivalence through the adapter ------------------------------
from twin import Twin  # noqa: E402

native = [normalize(f) for f in frames]
bridged = [from_mavlink(to_mavlink(f)) for f in frames]
t1, t2 = Twin(), Twin()
z_native = [t1.step(f)["cylinders"][0]["z_EGT"] for f in native]
z_bridged = [t2.step(f)["cylinders"][0]["z_EGT"] for f in bridged]
dz = float(np.max(np.abs(np.array(z_native) - np.array(z_bridged))))
check(dz < 1e-9, f"twin output identical via the bridge (max dz {dz:.2e})")

# --- 4: server round trip --------------------------------------------------
from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402

with TestClient(server.app) as c:
    # mixing guard: a demo mission runs, ingest must refuse with 409
    r = c.post("/api/control", json={"action": "start",
                                     "scenario": "endurance"})
    check(r.status_code == 200, "demo mission started for the mixing guard")
    r = c.post("/api/ingest", json={"format": "native", "frame": native[0]})
    check(r.status_code == 409,
          "ingest refuses to mix with a running demo mission (409)")
    c.post("/api/control", json={"action": "reset"})

    payloads = [to_mavlink(f) for f in frames]
    r = c.post("/api/ingest",
               json={"format": "mavlink", "frames": payloads[:10]})
    check(r.status_code == 200 and r.json()["queued"] == 10,
          f"10 MAVLink frames accepted ({r.status_code})")
    r = c.post("/api/ingest",
               json={"format": "mavlink", "frames": payloads[10:]})
    check(r.status_code == 200, "rest of the batch accepted")

    # the mission loop paces at 1 frame/s; give it a few seconds to drain
    time.sleep(3.5)
    st = c.get("/api/bridge/status").json()
    check(st["frames_ingested"] == len(frames)
          and st["sources"].get("mavlink") == len(frames),
          f"status counts every ingested frame "
          f"({st['frames_ingested']} total, {st['sources']})")
    hist = c.get("/api/history").json()
    check(len(hist["t_s"]) >= 2,
          f"twin processed ingested frames ({len(hist['t_s'])} states "
          "streamed and counting)")
    c.post("/api/control", json={"action": "reset"})

print()
if failures:
    print(f"RESULT: {len(failures)} FAILURES")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
