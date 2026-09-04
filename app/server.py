"""FastAPI server for the twin app (app/README.md "Server interface";
domain-primer.md section 8: "Software with an interface, demonstrated live").

Serves the built dashboard from dashboard/dist, streams twin states over
WS /ws at one state per telemetry frame, exposes GET /api/history for trend
charts, and takes mission controls on POST /api/control (start, pause,
resume, reset, inject, clear_faults, replay).

Telemetry comes from the simulator (simulator/mission.py), which pretends
to be the engine; each frame is stepped through the twin (app/twin/) and
the resulting state is broadcast to every connected websocket client. The
mission generator reads its fault_events list lazily at each timestep
(mission._active_faults), so inject/clear_faults simply mutate that one list
and the running mission picks the change up on the next tick. No simulator
code is modified to make that work.

Replay reads a logged mission CSV back through mission.frames_from_csv and
streams the frames at the same paced rate; the twin is reset first, then
frames older than seek_s are dropped so the twin never sees them.

The optional "speed" field on start/replay multiplies the tick rate for
rehearsal: {"speed": 4} streams four frames per second instead of one.
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

# The simulator and the twin package are imported from sibling directories,
# so put both on sys.path before importing them. Derived from __file__ so the
# server starts from any working directory.
_TWIN_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TWIN_DIR.parent
_SIM_DIR = _REPO_ROOT / "simulator"
for _p in (str(_SIM_DIR), str(_TWIN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mission  # noqa: E402
from physics import faults  # noqa: E402
from physics.engine_params import DEFAULT_GEOMETRY  # noqa: E402
from bridge import normalize, FrameError  # noqa: E402
from bridge.mavlink import from_mavlink  # noqa: E402
from bridge import mqtt as bridge_mqtt  # noqa: E402

try:
    from twin import Twin  # noqa: E402
except ImportError:
    # ------------------------------------------------------------------
    # FALLBACK STUB, development only. Used only when the real twin
    # package (app/twin/) is not importable, so this server always
    # runs. It fabricates a zero-residual state straight from the
    # telemetry frame: predictions equal readings, every z is zero,
    # everything nominal, no alarm, no diagnosis. It runs no physics and
    # detects nothing. The real Twin wins whenever it can be imported.
    # ------------------------------------------------------------------
    _STUB_SIGMA_EGT_K = 1.0  # ASSUMED: unit scale so the z fields are defined; the stub never flags anyway
    _STUB_SIGMA_CHT_K = 1.0  # ASSUMED: same

    class Twin:  # type: ignore[no-redef]
        """Zero-residual stand-in for the real twin; see the banner above."""

        def reset(self):
            """Stateless, so nothing to clear."""

        def step(self, frame):
            cylinders = []
            for i, (egt, cht) in enumerate(
                    zip(frame["EGT_K"], frame["CHT_K"]), start=1):
                cylinders.append({
                    "n": i,
                    "EGT_K": egt, "CHT_K": cht,
                    "EGT_pred_K": egt, "CHT_pred_K": cht,
                    "sigma_EGT_K": _STUB_SIGMA_EGT_K,
                    "sigma_CHT_K": _STUB_SIGMA_CHT_K,
                    "z_EGT": 0.0, "z_CHT": 0.0,
                    "status": "nominal",
                })
            return {
                "t_s": frame["t_s"],
                "scenario": frame["scenario"],
                "inputs": {
                    "N_rpm": frame["N_rpm"],
                    "MAP_Pa": frame["MAP_Pa"],
                    "altitude_m": frame["altitude_m"],
                    # kg/s summed over cylinders, times seconds per hour.
                    "fuel_flow_total_kg_h": sum(frame["fuel_flow_kg_s_per_cyl"]) * 3600.0,
                },
                "cylinders": cylinders,
                "oil": {
                    "p_Pa": frame["p_oil_Pa"], "T_K": frame["T_oil_K"],
                    "p_pred_Pa": frame["p_oil_Pa"], "T_pred_K": frame["T_oil_K"],
                    "z_p": 0.0, "z_T": 0.0,
                },
                "alarm": {"active": False, "level": "nominal", "since_t_s": None,
                          "channel": None, "cylinder": None},
                "diagnosis": [],
            }


log = logging.getLogger("twin.server")

DIST_DIR = _TWIN_DIR / "dashboard" / "dist"

# History channel list is exactly the shape app/README.md specifies for
# GET /api/history.
_HISTORY_CHANNELS = ("EGT_K", "EGT_pred_K", "sigma_EGT_K",
                     "CHT_K", "CHT_pred_K", "sigma_CHT_K")

# Single-cylinder kinds, from the FaultSpec docstring in physics/faults.py.
_CYLINDER_KINDS = {"misfire", "injector_restriction", "detonation",
                   "cooling_degradation", "sensor_drift"}

# Channels a sensor_drift fault may bias, per app/README.md.
_SENSOR_CHANNELS = ("EGT_K", "CHT_K")

# Poll interval while a session is paused; an engineering choice, not a
# model number.
_PAUSE_POLL_S = 0.05

_PLACEHOLDER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nav-twin</title>
</head>
<body>
<h1>nav-twin digital twin</h1>
<p>The operator dashboard has not been built into app/dashboard/dist yet.
This placeholder is served so the twin API is usable in the meantime.</p>
<ul>
<li>WS /ws: twin state stream</li>
<li>GET /api/history: per-cylinder trend history since the last reset</li>
<li>POST /api/control: start, pause, resume, reset, inject, clear_faults, replay</li>
</ul>
</body>
</html>
"""

app = FastAPI(title="nav-twin")

TWIN = Twin()
_TWIN_LOCK = asyncio.Lock()
_CLIENTS: set = set()
_SESSION = None       # active _Session or None
_RUNNER = None        # asyncio.Task running _mission_loop, or None
_HISTORY = {"t_s": [], "cylinders": {}}

# External telemetry ingest (bridge/). One queue feeds the external
# session's generator; counters back GET /api/bridge/status.
import queue as _queue  # noqa: E402
_INGEST_Q = _queue.Queue()
_INGEST_STATS = {"frames": 0, "rejected": 0, "sources": {},
                 "last_frame_monotonic": None}
_EXTERNAL_STALL_S = 30.0   # no frames for this long -> the external session ends


def _external_gen(q):
    """Frame generator fed by the ingest queue. Runs inside the mission
    loop's executor thread, so a blocking get with a stall timeout is
    correct here. None on stall ends the session, same as a mission CSV
    running out."""
    while True:
        try:
            yield q.get(timeout=_EXTERNAL_STALL_S)
        except _queue.Empty:
            return


def _push_ingest_frame(frame, source):
    """One validated frame into the ingest queue, with counters."""
    import time
    _INGEST_Q.put(frame)
    _INGEST_STATS["frames"] += 1
    _INGEST_STATS["sources"][source] = \
        _INGEST_STATS["sources"].get(source, 0) + 1
    _INGEST_STATS["last_frame_monotonic"] = time.monotonic()


@dataclass
class _Session:
    """One running mission (live or replay) and its mutable control state."""
    kind: str            # "live" or "replay"
    gen: object          # frame generator from mission.py
    fault_events: list   # the same list object the live generator reads
    speed: float
    paused: bool = False
    ended: bool = False
    t_now: float | None = None   # t_s of the last streamed frame


def _jsonable(obj):
    """Convert numpy scalars and friends to plain JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    item = getattr(obj, "item", None)
    if callable(item):
        return obj.item()
    return str(obj)


def _next_or_none(gen):
    """next() that maps StopIteration to None, safe for run_in_executor."""
    try:
        return next(gen)
    except StopIteration:
        return None


def _append_history(state):
    _HISTORY["t_s"].append(state["t_s"])
    for cyl in state.get("cylinders", []):
        slot = _HISTORY["cylinders"].setdefault(
            str(cyl["n"]), {ch: [] for ch in _HISTORY_CHANNELS})
        for ch in _HISTORY_CHANNELS:
            slot[ch].append(cyl.get(ch))


def _reset_history():
    global _HISTORY
    _HISTORY = {"t_s": [], "cylinders": {}}


async def _broadcast(msg):
    if not _CLIENTS:
        return
    text = json.dumps(msg)
    stale = []
    for ws in list(_CLIENTS):
        try:
            await ws.send_text(text)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _CLIENTS.discard(ws)


async def _mission_loop(sess):
    """Stream one twin state per telemetry frame, paced to 1 Hz x speed."""
    loop = asyncio.get_running_loop()
    try:
        while _SESSION is sess:
            if sess.paused:
                await asyncio.sleep(_PAUSE_POLL_S)
                continue
            try:
                frame = await loop.run_in_executor(None, _next_or_none, sess.gen)
            except Exception as exc:
                log.warning("mission source failed, ending session: %s", exc)
                break
            if frame is None:
                break
            async with _TWIN_LOCK:
                state = await loop.run_in_executor(None, TWIN.step, frame)
            state = _jsonable(state)
            sess.t_now = frame["t_s"]
            _append_history(state)
            await _broadcast({"type": "state", "state": state})
            await asyncio.sleep(mission.DT_S / sess.speed)
    except asyncio.CancelledError:
        raise
    finally:
        if _SESSION is sess:
            sess.ended = True


async def _stop_session():
    """Detach and cancel the current session's runner task, if any."""
    global _SESSION, _RUNNER
    _SESSION = None
    task, _RUNNER = _RUNNER, None
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def _begin_session(kind, gen, fault_events, speed):
    """Stop whatever is running, reset twin and history, start a new loop."""
    global _SESSION, _RUNNER
    await _stop_session()
    async with _TWIN_LOCK:
        TWIN.reset()
        _reset_history()
    sess = _Session(kind=kind, gen=gen, fault_events=fault_events, speed=speed)
    _SESSION = sess
    _RUNNER = asyncio.create_task(_mission_loop(sess))
    return sess


def _require_running():
    sess = _SESSION
    if sess is None or sess.ended:
        raise HTTPException(400, "no mission is running")
    return sess


def _as_float(value, name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{name!r} must be a number, got {value!r}")


def _as_speed(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(400, f"'speed' must be a positive number, got {value!r}")
    speed = float(value)
    if speed <= 0.0:
        raise HTTPException(400, f"'speed' must be positive, got {value!r}")
    return speed


def _parse_fault_spec(d):
    """Validate a fault object and build a physics.faults.FaultSpec."""
    if not isinstance(d, dict):
        raise HTTPException(400, "fault entries must be objects")
    kind = d.get("kind")
    if kind not in faults.VALID_KINDS:
        raise HTTPException(
            400, f"unknown fault kind {kind!r}; "
                 f"valid kinds: {sorted(faults.VALID_KINDS)}")
    cylinder = d.get("cylinder")
    if kind in _CYLINDER_KINDS:
        n_cyl = DEFAULT_GEOMETRY.num_cylinders
        if (isinstance(cylinder, bool) or not isinstance(cylinder, int)
                or not 1 <= cylinder <= n_cyl):
            raise HTTPException(
                400, f"{kind} requires 'cylinder' as an int in 1..{n_cyl}, "
                     f"got {cylinder!r}")
    if kind == "sensor_drift" and d.get("sensor_channel") not in _SENSOR_CHANNELS:
        raise HTTPException(
            400, f"sensor_drift requires 'sensor_channel' one of "
                 f"{list(_SENSOR_CHANNELS)}, got {d.get('sensor_channel')!r}")
    try:
        return faults.FaultSpec(
            kind=kind,
            cylinder=cylinder,
            severity=_as_float(d.get("severity", 1.0), "severity"),
            sensor_channel=d.get("sensor_channel"),
            bias_K=_as_float(d.get("bias_K", 0.0), "bias_K"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _parse_fault_event(d):
    """Parse one start-time fault event into a mission.FaultEvent.

    Accepts either a flat object ({t_start_s, kind, cylinder, ...}) or the
    nested form ({t_start_s, fault: {...}, ramp_s}), matching the inject
    body's shape where ramp_s rides inside the fault object.
    """
    if not isinstance(d, dict):
        raise HTTPException(400, "each fault_events entry must be an object")
    fault_body = d.get("fault", d)
    ramp_default = fault_body.get("ramp_s", 0.0) if isinstance(fault_body, dict) else 0.0
    return mission.FaultEvent(
        t_start_s=_as_float(d.get("t_start_s", 0.0), "t_start_s"),
        fault=_parse_fault_spec(fault_body),
        ramp_s=_as_float(d.get("ramp_s", ramp_default), "ramp_s"),
    )


@app.get("/api/history")
async def api_history():
    return _HISTORY


@app.post("/api/control")
async def api_control(body: dict = Body(...)):
    action = body.get("action")
    if not isinstance(action, str):
        raise HTTPException(400, "body must include an 'action' string")

    if action == "start":
        scenario = body.get("scenario", "endurance")
        if scenario not in mission.MISSION_PROFILES:
            raise HTTPException(
                400, f"unknown scenario {scenario!r}; "
                     f"valid scenarios: {list(mission.MISSION_PROFILES)}")
        speed = _as_speed(body.get("speed", 1.0))
        events_body = body.get("fault_events", [])
        if not isinstance(events_body, list):
            raise HTTPException(400, "'fault_events' must be a list")
        events = [_parse_fault_event(e) for e in events_body]
        # The generator holds on to this exact list; inject/clear_faults
        # mutate it and the running mission sees the change next tick.
        gen = mission.run_mission(scenario, events)
        await _begin_session("live", gen, events, speed)
        return {"ok": True, "scenario": scenario, "speed": speed,
                "fault_events": len(events)}

    if action == "pause":
        _require_running().paused = True
        return {"ok": True, "state": "paused"}

    if action == "resume":
        _require_running().paused = False
        return {"ok": True, "state": "running"}

    if action == "reset":
        await _stop_session()
        async with _TWIN_LOCK:
            TWIN.reset()
            _reset_history()
        return {"ok": True}

    if action == "inject":
        sess = _require_running()
        if sess.kind != "live":
            raise HTTPException(
                400, "inject only applies to a live mission, not a replay")
        fault_body = body.get("fault")
        if not isinstance(fault_body, dict):
            raise HTTPException(400, "inject requires a 'fault' object")
        spec = _parse_fault_spec(fault_body)
        ramp_s = _as_float(fault_body.get("ramp_s", 0.0), "ramp_s")
        t_start = sess.t_now if sess.t_now is not None else 0.0
        sess.fault_events.append(
            mission.FaultEvent(t_start_s=t_start, fault=spec, ramp_s=ramp_s))
        return {"ok": True, "active_faults": len(sess.fault_events)}

    if action == "clear_faults":
        sess = _require_running()
        sess.fault_events.clear()
        return {"ok": True}

    if action == "speed":
        # Repace the running mission. The guided demo posts this on every
        # speed button and only moves its highlight when we agree, so the
        # validation here is the contract the UI relies on.
        sess = _require_running()
        sess.speed = _as_speed(body.get("speed"))
        return {"ok": True, "speed": sess.speed}

    if action == "replay":
        path = body.get("path")
        if not isinstance(path, str) or not path:
            raise HTTPException(400, "replay requires a 'path' to a mission CSV")
        csv_path = Path(path)
        if not csv_path.is_absolute():
            csv_path = _REPO_ROOT / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            raise HTTPException(400, f"no such mission CSV: {path!r}")
        seek_s = _as_float(body.get("seek_s", 0.0), "seek_s")
        if seek_s < 0.0:
            raise HTTPException(400, f"'seek_s' must be >= 0, got {seek_s}")
        speed = _as_speed(body.get("speed", 1.0))
        # Twin is reset in _begin_session; frames before seek_s are dropped
        # here so the twin never sees them.
        gen = (fr for fr in mission.frames_from_csv(str(csv_path))
               if fr["t_s"] >= seek_s)
        await _begin_session("replay", gen, [], speed)
        return {"ok": True, "path": str(csv_path), "seek_s": seek_s,
                "speed": speed}

    raise HTTPException(
        400, f"unknown action {action!r}; valid actions: start, pause, "
             "resume, reset, inject, clear_faults, replay, speed")


@app.post("/api/ingest")
async def api_ingest(body: dict = Body(...)):
    """External telemetry in. Accepts one frame or a batch, in our native
    schema or a MAVLink/EFI-style envelope (bridge/). The first frame of a
    stream starts an external session; frames then flow through the twin
    and out the websocket exactly like a simulated mission."""
    global _SESSION
    fmt = body.get("format", "native")
    if fmt not in ("native", "mavlink"):
        raise HTTPException(400, "'format' must be 'native' or 'mavlink'")
    raw = body.get("frames", body.get("frame"))
    if raw is None:
        raise HTTPException(400, "body must include 'frame' or 'frames'")
    items = raw if isinstance(raw, list) else [raw]

    frames = []
    try:
        for item in items:
            frames.append(from_mavlink(item) if fmt == "mavlink"
                          else normalize(item))
    except FrameError as e:
        _INGEST_STATS["rejected"] += 1
        raise HTTPException(422, f"frame rejected: {e}")

    if _SESSION is not None and not _SESSION.ended \
            and _SESSION.kind != "external":
        raise HTTPException(409, "a mission is running; reset before "
                                 "ingesting external telemetry")
    if _SESSION is None or _SESSION.ended:
        await _begin_session("external", _external_gen(_INGEST_Q), [], 1.0)

    for f in frames:
        _push_ingest_frame(f, fmt)
    return {"ok": True, "queued": len(frames),
            "frames_total": _INGEST_STATS["frames"]}


@app.get("/api/bridge/status")
async def api_bridge_status():
    """What the integration layer is doing right now, for the dashboard's
    integration card and for operators wiring up a feed."""
    import time
    last = _INGEST_STATS["last_frame_monotonic"]
    return {
        "listening": True,
        "session": _SESSION.kind if _SESSION and not _SESSION.ended else None,
        "frames_ingested": _INGEST_STATS["frames"],
        "frames_rejected": _INGEST_STATS["rejected"],
        "sources": _INGEST_STATS["sources"],
        "queue_depth": _INGEST_Q.qsize(),
        "last_frame_s_ago": (round(time.monotonic() - last, 1)
                             if last is not None else None),
        "endpoints": {
            "rest": "POST /api/ingest  {format: native|mavlink, frame|frames}",
            "mqtt": ("set NAVTWIN_MQTT=host[:port] and "
                     "NAVTWIN_MQTT_TOPIC to enable"),
            "outbound": "WS /ws streams twin states; GET /api/history "
                        "for trends",
        },
    }


@app.websocket("/ws")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    _CLIENTS.add(websocket)
    try:
        await websocket.send_text(json.dumps(
            {"type": "hello", "scenarios": list(mission.MISSION_PROFILES)}))
        # States are pushed from _mission_loop; this loop only keeps the
        # socket open and detects disconnects. Client messages are ignored.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _CLIENTS.discard(websocket)


def _serve_dashboard(rel: str):
    """Serve the built dashboard if dashboard/dist exists, else a placeholder.

    Unknown paths fall back to index.html so client-side routing works; the
    check runs per request so a build appearing later is picked up without a
    restart.
    """
    if rel == "api" or rel.startswith("api/"):
        raise HTTPException(404, "not found")
    if DIST_DIR.is_dir():
        base = DIST_DIR.resolve()
        candidate = (base / rel).resolve() if rel else base / "index.html"
        try:
            candidate.relative_to(base)
        except ValueError:
            raise HTTPException(404, "not found")
        if candidate.is_file():
            return FileResponse(candidate)
        index = base / "index.html"
        if index.is_file():
            return FileResponse(index)
    return HTMLResponse(_PLACEHOLDER_HTML)


@app.get("/")
async def dashboard_index():
    return _serve_dashboard("")


@app.get("/{full_path:path}")
async def dashboard_files(full_path: str):
    return _serve_dashboard(full_path)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    # Optional MQTT bridge, only if NAVTWIN_MQTT is set (bridge/mqtt.py).
    bridge_mqtt.start_listener(_push_ingest_frame)
    uvicorn.run(app, host="127.0.0.1", port=8000)
