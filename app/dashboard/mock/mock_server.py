"""Mock twin server for dashboard development (NOT the MVP backend).

Stands in for app/server.py while the twin core is built in parallel.
It speaks the exact protocol from app/README.md (WS /ws hello + state,
POST /api/control, GET /api/history) so the dashboard's data path is
identical to production, and it serves dashboard/dist/ statically so the
built app can be exercised end to end.

Number provenance, per the repo rules:

- Observed telemetry comes from simulator/mission.py run_mission, the
  verified mission generator, faults included.
- Predicted values come from predict_steady_state on the reported inputs
  with no faults, which is what app/twin/residual.py will do.
- Sigmas come from simulator/data/sigma_table.npz (the offline Monte
  Carlo table), interpolated with scipy.
- The alarm persistence window (PERSIST_TICKS below) and the diagnosis
  ranking are MOCK heuristics, explicitly ASSUMED, existing only so every
  dashboard component renders against real-shaped data. The real ones live
  in twin/anomaly.py and twin/diagnose.py and replace this file's guesses;
  they are not part of this dashboard deliverable.

Stdlib plus numpy/scipy only, so the project venv runs it unchanged.

Run:  python mock_server.py            (port 8734, 4x playback)
Env:  MOCK_PORT, MOCK_TICK_S, MOCK_SEED override the defaults.
"""

import base64
import hashlib
import json
import os
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = DASHBOARD_DIR / "dist"
SIM_DIR = Path(__file__).resolve().parents[3] / "simulator"
sys.path.insert(0, str(SIM_DIR))

from mission import run_mission, MISSION_PROFILES  # noqa: E402
from physics.atmosphere import isa_atmosphere  # noqa: E402
from physics.constants import FA_STOICH  # noqa: E402
from physics.engine_model import OperatingPoint, predict_steady_state  # noqa: E402
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS  # noqa: E402
from physics.faults import FaultSpec  # noqa: E402
from physics.intake import air_mass_flow_kg_s  # noqa: E402
from physics.uncertainty import normalized_residual  # noqa: E402
from mission import FaultEvent  # noqa: E402

PORT = int(os.environ.get("MOCK_PORT", "8734"))
TICK_S = float(os.environ.get("MOCK_TICK_S", "0.25"))
SEED = int(os.environ.get("MOCK_SEED", "42"))

# ASSUMED mock persistence: an exceedance must hold this many consecutive
# ticks before the alarm goes active (anomaly.py owns the real window).
PERSIST_TICKS = 3

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _load_sigma():
    d = np.load(SIM_DIR / "data" / "sigma_table.npz")
    egt = RegularGridInterpolator(
        (d["phi_grid"], d["t_amb_grid"]), d["egt_std"])
    cht = RegularGridInterpolator(
        (d["n_grid"], d["map_grid"], d["cht_phi_grid"], d["cht_t_amb_grid"]),
        d["cht_std"])
    grids = {k: d[k] for k in (
        "phi_grid", "t_amb_grid", "n_grid", "map_grid", "cht_phi_grid",
        "cht_t_amb_grid")}
    return egt, cht, grids, float(d["sigma_p_oil_pa"]), float(d["sigma_t_oil_k"])


SIGMA_EGT, SIGMA_CHT, GRIDS, SIGMA_P_OIL, SIGMA_T_OIL = _load_sigma()


def _clip(v, grid):
    return float(np.clip(v, grid.min(), grid.max()))


def sigma_egt(phi, t_amb):
    return float(SIGMA_EGT([(_clip(phi, GRIDS["phi_grid"]),
                             _clip(t_amb, GRIDS["t_amb_grid"]))])[0])


def sigma_cht(n_rpm, map_pa, phi, t_amb):
    return float(SIGMA_CHT([(_clip(n_rpm, GRIDS["n_grid"]),
                             _clip(map_pa, GRIDS["map_grid"]),
                             _clip(phi, GRIDS["cht_phi_grid"]),
                             _clip(t_amb, GRIDS["cht_t_amb_grid"]))])[0])


def z_class(z):
    a = abs(z)
    if a >= 3:
        return "warning"
    if a >= 2:
        return "caution"
    return "nominal"


def mock_diagnosis(cylinders):
    """ASSUMED stand-in for twin/diagnose.py: classify the per-cylinder
    z sign pattern, rank by a monotone map of the worst |z|. Exists so the
    Fault Alert card renders real-shaped entries; confidence here is a
    heuristic, not the twin's rule-table output."""
    entries = []
    for c in cylinders:
        ze, zc, n = c["z_EGT"], c["z_CHT"], c["n"]
        worst = max(abs(ze), abs(zc))
        if worst < 2:
            continue
        if zc >= 2 and ze <= -2:
            label = f"detonation / advanced timing, cyl {n}"
        elif zc >= 2 and ze >= 2:
            label = f"cooling degradation, cyl {n}"
        elif ze <= -2 and zc <= -2:
            label = f"misfire, cyl {n}"
        elif ze <= -2 and abs(zc) < 2:
            label = f"injector restriction, cyl {n}"
        elif ze >= 2 and abs(zc) < 2:
            label = f"combustion instability, cyl {n}"
        else:
            ch = "CHT" if abs(zc) >= abs(ze) else "EGT"
            label = f"sensor drift ({ch}), cyl {n}"
        confidence = round(min(0.95, worst / (worst + 1.5)), 2)
        evidence = [
            f"z_CHT({n}) = {zc:+.1f}",
            f"z_EGT({n}) = {ze:+.1f}",
        ]
        entries.append({"label": label, "confidence": confidence,
                        "evidence": evidence})
    entries.sort(key=lambda e: -e["confidence"])
    return [{"rank": i + 1, **e} for i, e in enumerate(entries[:3])]


class TwinMock:
    """Holds the running mission and turns telemetry frames into the twin
    state JSON of app/README.md."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scenario = "endurance"
        self.fault_events = []      # shared with the running generator
        self.gen = None
        self.thread = None
        self.running = False
        self.paused = False
        self.latest = None
        self.history = None
        self.exceed = {}            # channel key -> consecutive |z|>=2 ticks
        self.alarm_since = None
        self.clients = []
        self._clear_history()

    def _clear_history(self):
        self.history = {"t_s": [], "cylinders": {
            str(n): {"EGT_K": [], "EGT_pred_K": [], "sigma_EGT_K": [],
                     "CHT_K": [], "CHT_pred_K": [], "sigma_CHT_K": []}
            for n in range(1, DEFAULT_GEOMETRY.num_cylinders + 1)}}

    # --- mission lifecycle ------------------------------------------------

    def start(self, scenario, fault_events=()):
        self.stop()
        with self.lock:
            self.scenario = scenario
            self.fault_events = list(fault_events)
            self._clear_history()
            self.exceed = {}
            self.alarm_since = None
            self.latest = None
            self.gen = run_mission(scenario, self.fault_events, seed=SEED)
            self.running = True
            self.paused = False
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self.thread = None
        self.gen = None

    def reset(self):
        self.stop()
        with self.lock:
            self.fault_events = []
            self._clear_history()
            self.exceed = {}
            self.alarm_since = None
            self.latest = None

    def inject(self, fault: FaultSpec, ramp_s: float):
        with self.lock:
            if self.latest is None:
                return False
            self.fault_events.append(
                FaultEvent(t_start_s=self.latest["t_s"], fault=fault,
                           ramp_s=ramp_s))
            return True

    def clear_faults(self):
        with self.lock:
            self.fault_events.clear()

    # --- per-tick state ----------------------------------------------------

    def _loop(self):
        while self.running:
            with self.lock:
                paused = self.paused
            if paused:
                time.sleep(0.05)
                continue
            t0 = time.time()
            try:
                frame = next(self.gen)
            except StopIteration:
                self.running = False
                break
            except Exception as exc:  # surface, then stop cleanly
                print(f"mock: mission generator failed: {exc}", flush=True)
                self.running = False
                break
            state = self._build_state(frame)
            with self.lock:
                self.latest = state
                self._append_history(state)
            self._broadcast({"type": "state", "state": state})
            dt = time.time() - t0
            time.sleep(max(0.0, TICK_S - dt))

    def _build_state(self, frame):
        alt = frame["altitude_m"]
        t_im = isa_atmosphere(alt)["T_amb_K"]
        t_amb = t_im
        air_total = air_mass_flow_kg_s(frame["N_rpm"], frame["MAP_Pa"], t_im,
                                       DEFAULT_GEOMETRY, DEFAULT_CONSTANTS)
        air_per_cyl = air_total / DEFAULT_GEOMETRY.num_cylinders

        op = OperatingPoint(
            N_rpm=frame["N_rpm"], MAP_Pa=frame["MAP_Pa"], altitude_m=alt,
            fuel_flow_kg_s_per_cyl=list(frame["fuel_flow_kg_s_per_cyl"]),
            T_oil_K=frame["T_oil_K"])
        pred = predict_steady_state(op)

        n_cyl = DEFAULT_GEOMETRY.num_cylinders
        fuel_total_kg_h = float(sum(frame["fuel_flow_kg_s_per_cyl"])) * 3600.0

        cylinders = []
        for i in range(n_cyl):
            m_f = frame["fuel_flow_kg_s_per_cyl"][i]
            phi = (m_f / air_per_cyl) / FA_STOICH
            pc = pred["per_cylinder"][i]
            se = sigma_egt(phi, t_amb)
            sc = sigma_cht(frame["N_rpm"], frame["MAP_Pa"], phi, t_amb)
            ze = normalized_residual(frame["EGT_K"][i], pc["EGT_K"], se)
            zc = normalized_residual(frame["CHT_K"][i], pc["CHT_K"], sc)
            cylinders.append({
                "n": i + 1,
                "EGT_K": float(frame["EGT_K"][i]),
                "CHT_K": float(frame["CHT_K"][i]),
                "EGT_pred_K": float(pc["EGT_K"]),
                "CHT_pred_K": float(pc["CHT_K"]),
                "sigma_EGT_K": se,
                "sigma_CHT_K": sc,
                "z_EGT": float(ze),
                "z_CHT": float(zc),
                "status": z_class(ze if abs(ze) >= abs(zc) else zc),
            })

        oil = {
            "p_Pa": float(frame["p_oil_Pa"]),
            "T_K": float(frame["T_oil_K"]),
            "p_pred_Pa": float(pred["p_oil_Pa"]),
            "T_pred_K": float(pred["T_oil_ss_K"]),
            "z_p": float(normalized_residual(
                frame["p_oil_Pa"], pred["p_oil_Pa"], SIGMA_P_OIL)),
            "z_T": float(normalized_residual(
                frame["T_oil_K"], pred["T_oil_ss_K"], SIGMA_T_OIL)),
        }

        alarm = self._update_alarm(frame["t_s"], cylinders, oil)

        return {
            "t_s": float(frame["t_s"]),
            "scenario": self.scenario,
            "inputs": {
                "N_rpm": float(frame["N_rpm"]),
                "MAP_Pa": float(frame["MAP_Pa"]),
                "altitude_m": float(alt),
                "fuel_flow_total_kg_h": fuel_total_kg_h,
            },
            "cylinders": cylinders,
            "oil": oil,
            "alarm": alarm,
            "diagnosis": mock_diagnosis(cylinders) if alarm["active"] else [],
        }

    def _update_alarm(self, t_s, cylinders, oil):
        zs = {}
        for c in cylinders:
            zs[f"EGT_{c['n']}"] = (c["z_EGT"], c["n"])
            zs[f"CHT_{c['n']}"] = (c["z_CHT"], c["n"])
        zs["p_oil"] = (oil["z_p"], None)
        zs["T_oil"] = (oil["z_T"], None)

        for key, (z, _) in zs.items():
            self.exceed[key] = self.exceed.get(key, 0) + 1 if abs(z) >= 2 else 0

        active = {k: zs[k] for k, v in self.exceed.items() if v >= PERSIST_TICKS}
        if not active:
            self.alarm_since = None
            return {"active": False, "level": "nominal", "since_t_s": None,
                    "channel": None, "cylinder": None}
        worst_key = max(active, key=lambda k: abs(active[k][0]))
        worst_z, worst_cyl = active[worst_key]
        level = "warning" if abs(worst_z) >= 3 else "caution"
        if self.alarm_since is None:
            self.alarm_since = t_s
        return {"active": True, "level": level, "since_t_s": self.alarm_since,
                "channel": worst_key, "cylinder": worst_cyl}

    def _append_history(self, state):
        self.history["t_s"].append(state["t_s"])
        for c in state["cylinders"]:
            h = self.history["cylinders"][str(c["n"])]
            h["EGT_K"].append(c["EGT_K"])
            h["EGT_pred_K"].append(c["EGT_pred_K"])
            h["sigma_EGT_K"].append(c["sigma_EGT_K"])
            h["CHT_K"].append(c["CHT_K"])
            h["CHT_pred_K"].append(c["CHT_pred_K"])
            h["sigma_CHT_K"].append(c["sigma_CHT_K"])

    # --- websocket fanout ---------------------------------------------------

    def add_client(self, sock):
        entry = {"sock": sock, "wlock": threading.Lock(), "dead": False}
        with self.lock:
            self.clients.append(entry)
        return entry

    def drop_client(self, entry):
        entry["dead"] = True
        with self.lock:
            if entry in self.clients:
                self.clients.remove(entry)
        try:
            entry["sock"].close()
        except OSError:
            pass

    def _broadcast(self, msg):
        data = json.dumps(msg).encode("utf-8")
        frame = ws_text_frame(data)
        with self.lock:
            clients = list(self.clients)
        for entry in clients:
            try:
                with entry["wlock"]:
                    entry["sock"].sendall(frame)
            except OSError:
                self.drop_client(entry)


MOCK = TwinMock()


# --- minimal websocket framing ---------------------------------------------


def ws_text_frame(payload: bytes) -> bytes:
    head = bytearray([0x81])
    n = len(payload)
    if n < 126:
        head.append(n)
    elif n <= 0xFFFF:
        head.append(126)
        head += struct.pack(">H", n)
    else:
        head.append(127)
        head += struct.pack(">Q", n)
    return bytes(head) + payload


def ws_read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def ws_pump(sock, entry):
    """Consume client frames so closes and pings are handled; the protocol
    has no client-to-server messages (control goes over REST)."""
    try:
        while not entry["dead"]:
            hdr = ws_read_exact(sock, 2)
            opcode = hdr[0] & 0x0F
            masked = hdr[1] & 0x80
            length = hdr[1] & 0x7F
            if length == 126:
                length = struct.unpack(">H", ws_read_exact(sock, 2))[0]
            elif length == 127:
                length = struct.unpack(">Q", ws_read_exact(sock, 8))[0]
            mask = ws_read_exact(sock, 4) if masked else b"\x00" * 4
            payload = bytearray(ws_read_exact(sock, length)) if length else bytearray()
            for i in range(len(payload)):
                payload[i] ^= mask[i % 4]
            if opcode == 8:
                break
            if opcode == 9:
                with entry["wlock"]:
                    sock.sendall(bytes([0x8A, 0]))
    except (ConnectionError, OSError):
        pass
    MOCK.drop_client(entry)


# --- HTTP + websocket server --------------------------------------------------

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def parse_fault(d):
    return FaultSpec(
        kind=d["kind"],
        cylinder=d.get("cylinder"),
        severity=float(d.get("severity", 1.0)),
        sensor_channel=d.get("sensor_channel"),
        bias_K=float(d.get("bias_K", 0.0)),
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ws" and "upgrade" in self.headers.get("Connection", "").lower():
            return self._websocket()
        if self.path == "/api/history":
            with MOCK.lock:
                hist = MOCK.history
            return self._send_json(hist)
        return self._static()

    def do_POST(self):
        if self.path != "/api/control":
            return self._send_json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "bad json"}, 400)

        action = body.get("action")
        try:
            if action == "start":
                scenario = body.get("scenario", "endurance")
                if scenario not in MISSION_PROFILES:
                    return self._send_json({"error": "unknown scenario"}, 400)
                events = [FaultEvent(t_start_s=0.0, fault=parse_fault(f),
                                     ramp_s=float(f.get("ramp_s", 0.0)))
                          for f in body.get("fault_events", [])]
                MOCK.start(scenario, events)
            elif action == "pause":
                MOCK.paused = True
            elif action == "resume":
                MOCK.paused = False
            elif action == "reset":
                MOCK.reset()
            elif action == "inject":
                ok = MOCK.inject(parse_fault(body["fault"]),
                                 float(body.get("ramp_s", 0.0)))
                if not ok:
                    return self._send_json({"error": "no mission running"}, 409)
            elif action == "clear_faults":
                MOCK.clear_faults()
            elif action == "replay":
                return self._send_json(
                    {"error": "mock does not implement replay"}, 501)
            else:
                return self._send_json({"error": "unknown action"}, 400)
        except (KeyError, ValueError) as exc:
            return self._send_json({"error": str(exc)}, 400)
        return self._send_json({"ok": True, "action": action})

    def _static(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", ""):
            path = "/index.html"
        target = (DIST_DIR / path.lstrip("/")).resolve()
        if not str(target).startswith(str(DIST_DIR.resolve())) or not target.is_file():
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        data = target.read_bytes()
        ctype = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _websocket(self):
        try:
            self._websocket_inner()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _websocket_inner(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()

        sock = self.connection
        entry = MOCK.add_client(sock)
        hello = json.dumps({"type": "hello",
                            "scenarios": list(MISSION_PROFILES)}).encode()
        try:
            sock.sendall(ws_text_frame(hello))
            if MOCK.latest is not None:
                sock.sendall(ws_text_frame(json.dumps(
                    {"type": "state", "state": MOCK.latest}).encode()))
        except OSError:
            MOCK.drop_client(entry)
            return
        ws_pump(sock, entry)
        self.close_connection = True


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"mock twin server on http://127.0.0.1:{PORT} "
          f"(tick {TICK_S}s, seed {SEED})", flush=True)
    print("POST /api/control {\"action\": \"start\", \"scenario\": \"endurance\"} "
          "to run a mission", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
