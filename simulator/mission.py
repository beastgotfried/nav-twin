"""Mission generator: the "engine" side of the MVP (domain-primer.md section 8:
"a program pretending to be a real engine, with switches to inject faults").

Produces time-series telemetry at 1 Hz for the four mission scenarios the
problem statement's Section E names verbatim: high altitude, endurance
mission, hot-weather operation, rapid throttle transitions.

At each timestep this runs the FULL physics model (predict_steady_state with
any active faults applied via faults.py), then applies the Handbook 7.8
first-order thermal lags and ASSUMED sensor noise. The twin (app/) only
ever sees the resulting telemetry frames, never the fault schedule, which is
what keeps the demo honest.

Two deliberate simplifications, both documented here rather than hidden:

- Hot-weather operation is modelled as sustained high-load low-altitude
  operation, because OperatingPoint derives T_amb from ISA altitude and has
  no ambient-temperature offset field. Adding one is an interface extension
  to the physics package, which is a maintainer decision; post-MVP.
- Sensor noise sigmas below are ASSUMED (STREAM section 7: real sensor noise
  characteristics of the target engine are unknown, Phase 2 question).
- Fuel flow telemetry is the ECU-COMMANDED value, matching how FADEC
  telemetry actually works (the ECU reports what it asked the injectors to
  deliver; the problem statement's "injection timing parameters" are the
  same channel). A restricted injector DELIVERS less than commanded, which
  is exactly the residual the twin detects. Likewise MAP telemetry is the
  ACHIEVED manifold pressure (what the sensor reads), and the frame carries
  the commanded value alongside it, because the commanded-vs-achieved gap
  is the turbo health indicator (Handbook 7.2).

Fuel flow is derived from a commanded equivalence ratio via the Step 2 air
mass flow, matching how the handbook's worked example anchors the model
(phi=1.1 at 5500 rpm, MAP 140 kPa -> ~30 L/hr).
"""

import csv
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from physics.atmosphere import isa_atmosphere
from physics.constants import FA_STOICH
from physics.engine_model import OperatingPoint, predict_steady_state
from physics.engine_params import DEFAULT_GEOMETRY, DEFAULT_CONSTANTS
from physics.faults import FaultSpec, apply_fault, apply_sensor_drift
from physics.intake import air_mass_flow_kg_s
from physics.thermal import first_order_lag_step

# ASSUMED sensor noise (1-sigma), per the STREAM assumption table. Used to
# make telemetry realistic; the twin's uncertainty band is much wider than
# these, which is exactly the point of the uncertainty layer.
NOISE_SIGMA = {
    "EGT_K": 2.0,          # K, per cylinder
    "CHT_K": 1.0,          # K, per cylinder
    "p_oil_Pa": 5000.0,    # Pa
    "T_oil_K": 1.0,        # K
    "N_rpm": 5.0,          # rpm
    "MAP_Pa": 500.0,       # Pa
    "fuel_flow_rel": 0.01, # 1% of reading
}

DT_S = 1.0


@dataclass(frozen=True)
class MissionPoint:
    """One commanded operating point: what the pilot/autopilot demands."""
    t_s: float
    N_rpm: float
    MAP_Pa: float
    altitude_m: float
    phi: float          # commanded equivalence ratio, same for all cylinders


@dataclass(frozen=True)
class FaultEvent:
    """A fault switched on at t_start_s. ramp_s > 0 linearly grows severity
    from 0 to the FaultSpec's severity over that many seconds (progressive
    degradation); ramp_s = 0 is a step fault (demo-script detection latency).
    t_end_s removes the fault at that time (used by the full_mission demo
    profile to separate fault phases)."""
    t_start_s: float
    fault: FaultSpec
    ramp_s: float = 0.0
    t_end_s: float | None = None


def _ramp(t_s: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Linear interpolation with clamping, the only curve a profile needs."""
    if t_s <= t0:
        return v0
    if t_s >= t1:
        return v1
    return v0 + (v1 - v0) * (t_s - t0) / (t1 - t0)


def _profile_points(name: str):
    """Yield MissionPoints for a named scenario. Durations are demo-scale
    (minutes), not real-mission scale (hours); the physics is time-scale
    free and the demo cannot wait 20 hours for an endurance leg."""
    if name == "high_altitude":
        # Climb to 7600 m (25,000 ft, the handbook's worked example), cruise,
        # descend. Boost holds MAP nearly constant as ambient falls, which is
        # the entire reason these engines are turbocharged (Handbook 7.2).
        pts = []
        t = 0.0
        while t <= 300.0:
            alt = _ramp(t, 0, 90, 500.0, 7600.0) if t <= 90 else \
                _ramp(t, 210, 300, 7600.0, 500.0)
            n = _ramp(t, 0, 20, 4800.0, 5500.0)
            pts.append(MissionPoint(t, n, 140_000.0, alt, 1.08))
            t += DT_S
        return pts

    if name == "endurance":
        # Long loiter at moderate altitude, economy mixture, steady state.
        # This is the MALE UAV's actual job (domain-primer.md section 1).
        return [MissionPoint(t, 5000.0, 120_000.0, 5500.0, 0.92)
                for t in np.arange(0.0, 360.0 + DT_S, DT_S)]

    if name == "hot_weather":
        # Sustained high-load low-altitude operation: poor cooling on the
        # ground and in the climb, the condition the PS scenario names.
        # See module docstring for why there is no T_amb offset.
        pts = []
        t = 0.0
        while t <= 240.0:
            alt = _ramp(t, 60, 180, 300.0, 2000.0)
            n = _ramp(t, 0, 15, 5200.0, 5800.0)
            pts.append(MissionPoint(t, n, 135_000.0, alt, 1.10))
            t += DT_S
        return pts

    if name == "rapid_throttle":
        # Step the throttle up and down; exercises the thermal lags, which is
        # what makes the lag-ratio sensor-drift discriminator observable.
        pts = []
        t = 0.0
        while t <= 240.0:
            phase = int(t // 30.0) % 2
            n = 5500.0 if phase == 0 else 3800.0
            map_pa = 140_000.0 if phase == 0 else 90_000.0
            pts.append(MissionPoint(t, n, map_pa, 3000.0, 1.05))
            t += DT_S
        return pts

    # The two cruise points the demo acts 3/4 live at, chosen by probing the
    # model (see verify_twin.py): deep-rich, where the band is narrow and an
    # 8% leaning stays inside it, and lean-of-kink, where the same cut
    # drives the cylinder onto the steep flank and outside the band.
    if name == "cruise_rich":
        return [MissionPoint(t, 5000.0, 120_000.0, 3000.0, 1.15)
                for t in np.arange(0.0, 300.0 + DT_S, DT_S)]

    if name == "cruise_lean":
        return [MissionPoint(t, 5000.0, 120_000.0, 3000.0, 0.85)
                for t in np.arange(0.0, 300.0 + DT_S, DT_S)]

    if name == "full_mission":
        # One continuous takeoff-to-landing flight that exercises every
        # capability of the system, rules and learned layer alike, across
        # all eight fault kinds. Phases and the fault schedule are
        # documented in FULL_MISSION_PHASES and full_mission_faults below;
        # every behaviour shown is pinned by verify_mission.py. Pacing
        # rule: engine faults that heat the head get at least 60 s of
        # clear time afterwards so the next phase starts clean
        # (tau_CHT ~ 20 s).
        pts = []
        t = 0.0
        while t <= 1390.0:
            if t < 90.0:
                # Phase 1: takeoff and climb to 7600 m.
                n = _ramp(t, 0, 20, 4800.0, 5500.0)
                alt = _ramp(t, 0, 90, 500.0, 7600.0)
                map_pa, phi = 140_000.0, 1.08
            elif t < 210.0:
                # Phase 2: descent to cruise, deep rich. Small injector
                # restriction begins at t=150 (see full_mission_faults).
                n, map_pa = 5000.0, 120_000.0
                alt = _ramp(t, 90, 120, 7600.0, 3000.0)
                phi = _ramp(t, 90, 120, 1.08, 1.15)
            elif t < 330.0:
                # Phase 3: mixture leaned to 0.85, same fault active.
                n, map_pa, alt = 5000.0, 120_000.0, 3000.0
                phi = _ramp(t, 210, 240, 1.15, 0.85)
            elif t < 450.0:
                # Phase 4: restriction ramps to full blockage.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 3000.0, 0.95
            elif t < 600.0:
                # Phase 5: cylinder 3 recovers after its fault clears at
                # 420, then detonation on cylinder 1 from t=510.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 4000.0, 1.05
            elif t < 720.0:
                # Phase 6: sensor drift on cylinder 2's EGT from t=630.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 4000.0, 1.00
            elif t < 840.0:
                # Phase 7: misfire on cylinder 2 from t=750.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 4000.0, 1.05
            elif t < 960.0:
                # Phase 8: bearing wear ramped from t=870 (oil channels).
                # Ends at 960 because oil temperature needs ~3 min to
                # settle (tau_oil = 60 s) and the turbo leg must start
                # clean; verify_mission.py pins this.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 4000.0, 1.05
            elif t < 1080.0:
                # Phase 9: cooling degradation on cylinder 4, ramped from
                # t=990; a progressive fault for the RUL layer.
                n, map_pa, alt, phi = 5000.0, 120_000.0, 4000.0, 1.05
            elif t < 1260.0:
                # Phase 10: climb back to altitude with a fading turbo
                # from t=1110; the MAP gap grows with altitude. The band
                # has no channel for this; the learned forest does.
                n = _ramp(t, 1080, 1110, 5000.0, 5500.0)
                map_pa = 140_000.0
                alt = _ramp(t, 1080, 1200, 4000.0, 7600.0) if t <= 1200 \
                    else 7600.0
                phi = 1.08
            else:
                # Phase 11: faults cleared, descend home.
                n = _ramp(t, 1260, 1300, 5000.0, 3800.0)
                map_pa = _ramp(t, 1260, 1300, 120_000.0, 90_000.0)
                alt = _ramp(t, 1260, 1390, 4000.0, 800.0)
                phi = 0.95
            pts.append(MissionPoint(t, n, map_pa, alt, phi))
            t += DT_S
        return pts

    raise ValueError(f"unknown mission profile {name!r}; expected one of "
                     "high_altitude, endurance, hot_weather, rapid_throttle, "
                     "cruise_rich, cruise_lean, full_mission")


MISSION_PROFILES = ("high_altitude", "endurance", "hot_weather",
                    "rapid_throttle", "cruise_rich", "cruise_lean",
                    "full_mission")


# The guided flight's narrative structure. Captions are plain language for
# the dashboard's guided demo; every claim in them is verified behaviour of
# the pipeline, asserted in verify_twin.py / verify_mission.py.
FULL_MISSION_PHASES = [
    {"start_s": 0, "name": "Takeoff and climb",
     "caption": "The twin is predicting every gauge. The shaded band is its "
                "own computed uncertainty, not a tuned threshold."},
    {"start_s": 90, "name": "Cruise, and a small fault",
     "caption": "An 8 percent injector restriction begins on cylinder 3 at "
                "t=150. Watch the system stay quiet: the deviation sits "
                "inside its own uncertainty band."},
    {"start_s": 210, "name": "Same fault, leaner mixture",
     "caption": "The mixture leans and the cylinder slides past the "
                "heat-release peak. The residual steps outside the band and "
                "the alarm fires; the diagnosis settles on the injector as "
                "the slow head-temperature channel confirms it."},
    {"start_s": 330, "name": "Escalation to full blockage",
     "caption": "The same parameter worsens. The exhaust temperature sign "
                "flips on its own and the diagnosis becomes a dead "
                "cylinder. One knob, two signatures."},
    {"start_s": 450, "name": "Detonation",
     "caption": "Cylinder 3 recovers as its fault clears. Then a different "
                "fault on cylinder 1: head temperature climbs while exhaust "
                "temperature falls. Opposite directions from one cause; no "
                "single channel sees it."},
    {"start_s": 630, "name": "A lying sensor",
     "caption": "Cylinder 2's exhaust probe starts lying at t=630, "
                "instantly, with no engine change. Physics says a real "
                "combustion change must reach the head within about a "
                "minute; when it never does, the system calls it what it "
                "is: sensor drift, not an engine fault."},
    {"start_s": 750, "name": "A dead cylinder",
     "caption": "Ignition fails on cylinder 2. Watch it run COLD, not hot: "
                "no combustion means no heat. Both its temperatures "
                "collapse while the other three stay in band."},
    {"start_s": 870, "name": "Bearings wearing out",
     "caption": "Oil temperature rising and pressure falling together, "
                "gradually. The particle filter tracks the wear and "
                "projects a bounded time to the failure threshold. The "
                "bounds are the honest part: a spread, not a date."},
    {"start_s": 990, "name": "Cooling degrades slowly",
     "caption": "A baffle crack on cylinder 4 worsens over a minute and a "
                "half. Exhaust stays flat, head temperature climbs, and "
                "the RUL tracker follows the severity rising with its "
                "uncertainty bounds."},
    {"start_s": 1110, "name": "Climb, and a fading turbo",
     "caption": "The turbo loses efficiency as we climb. Every temperature "
                "channel stays in band, because physics says they should. "
                "But the twin watches more than temperatures: the gap "
                "between commanded and achieved manifold pressure grows "
                "with altitude, and the diagnosis names the fading turbo "
                "while the gauges read normal."},
    {"start_s": 1260, "name": "Recovery and descent",
     "caption": "Faults cleared. Watch the flagged channels settle back "
                "into the band as the engine cools, then the twin returns "
                "to nominal for the descent home."},
]


def full_mission_faults():
    """The fault schedule for the full_mission profile. Built here so the
    canned exporter, the verify scripts and any live run all share exactly
    one schedule."""
    return [
        # Phase 2: 8 percent restriction on cylinder 3. Clears at t=420 so
        # the cylinder finishes recovering before the detonation phase
        # (the CHT channel needs about a minute, tau_CHT ~20 s).
        FaultEvent(150.0, FaultSpec("injector_restriction", cylinder=3,
                                    severity=0.10), t_end_s=420.0),
        # Phase 4: ramps toward full blockage over 60 s. Applied on top of
        # the restriction above, so the effective cut deepens smoothly to
        # total (fuel multiplier (1-0.10)*(1-s)).
        FaultEvent(330.0, FaultSpec("injector_restriction", cylinder=3,
                                    severity=1.0), ramp_s=60.0,
                   t_end_s=420.0),
        # Phase 5: detonation on cylinder 1.
        FaultEvent(510.0, FaultSpec("detonation", cylinder=1, severity=1.0),
                   t_end_s=600.0),
        # Phase 6: +40 K bias on cylinder 2's EGT reading. Instant, applied
        # to the reading only; physics never sees it (faults.py).
        FaultEvent(630.0, FaultSpec("sensor_drift", cylinder=2,
                                    sensor_channel="EGT_K", bias_K=40.0),
                   t_end_s=720.0),
        # Phase 7: complete misfire on cylinder 2.
        FaultEvent(750.0, FaultSpec("misfire", cylinder=2, severity=1.0),
                   t_end_s=840.0),
        # Phase 8: bearing wear ramped over 90 s (oil temperature up,
        # pressure down, progressively). Ends at 960 so the oil channel
        # settles before the turbo leg (tau_oil = 60 s).
        FaultEvent(870.0, FaultSpec("bearing_wear", severity=1.0),
                   ramp_s=90.0, t_end_s=960.0),
        # Phase 9: cooling degradation on cylinder 4, ramped over 60 s so
        # the RUL tracker has a progressive severity to follow. Ends at
        # 1050 because a z~40 CHT excursion needs over a minute to settle
        # (tau_CHT = 20 s), and the turbo leg's "band sees nothing" claim
        # requires a clean band; verify_mission.py pins this.
        FaultEvent(990.0, FaultSpec("cooling_degradation", cylinder=4,
                                    severity=1.0), ramp_s=60.0,
                   t_end_s=1050.0),
        # Phase 10: turbo degradation for the whole high-altitude leg.
        # Severity 0.8 keeps achieved MAP physically sensible at 7600 m.
        FaultEvent(1110.0, FaultSpec("turbo_degradation", severity=0.8),
                   t_end_s=1260.0),
    ]


def _active_faults(t_s: float, events):
    """Faults active at time t, with ramp severity applied."""
    out = []
    for ev in events:
        if t_s < ev.t_start_s:
            continue
        if ev.t_end_s is not None and t_s >= ev.t_end_s:
            continue
        if ev.ramp_s > 0.0:
            frac = min((t_s - ev.t_start_s) / ev.ramp_s, 1.0)
            out.append(dataclasses.replace(ev.fault, severity=ev.fault.severity * frac))
        else:
            out.append(ev.fault)
    return out


def _fuel_flow_per_cyl(pt: MissionPoint, geometry=DEFAULT_GEOMETRY,
                       constants=DEFAULT_CONSTANTS):
    """Commanded fuel flow per cylinder from the commanded phi, via the
    Step 2 air mass flow relation (Handbook 7.3)."""
    T_im_K = isa_atmosphere(pt.altitude_m)["T_amb_K"]
    total_air = air_mass_flow_kg_s(pt.N_rpm, pt.MAP_Pa, T_im_K, geometry, constants)
    air_per_cyl = total_air / geometry.num_cylinders
    m_f = pt.phi * FA_STOICH * air_per_cyl
    return [m_f] * geometry.num_cylinders


def run_mission(profile, fault_events=(), dt_s: float = DT_S,
                seed: int = 42, noise: bool = True):
    """Generate telemetry frames for a mission. Yields one dict per timestep.

    `profile` is a scenario name from MISSION_PROFILES, or an explicit list
    of MissionPoints (used by verification to command exact step inputs).

    Frame keys: t_s, scenario, N_rpm, MAP_Pa, altitude_m,
    fuel_flow_kg_s_per_cyl (list), EGT_K (list per cyl), CHT_K (list),
    p_oil_Pa, T_oil_K. Temperatures are lagged and noisy; inputs carry their
    own small measurement noise, as real telemetry would.
    """
    rng = np.random.default_rng(seed)
    if isinstance(profile, str):
        points = _profile_points(profile)
        scenario = profile
    else:
        points = list(profile)
        scenario = "custom"
    n_cyl = DEFAULT_GEOMETRY.num_cylinders
    lag = {"EGT_K": None, "CHT_K": None, "T_oil_K": None}

    for pt in points:
        fuel = _fuel_flow_per_cyl(pt)
        op = OperatingPoint(
            N_rpm=pt.N_rpm, MAP_Pa=pt.MAP_Pa, altitude_m=pt.altitude_m,
            fuel_flow_kg_s_per_cyl=fuel,
            T_oil_K=lag["T_oil_K"] if lag["T_oil_K"] is not None else 353.15,
        )

        per_cyl_constants = {}
        sensor_faults = []
        for f in _active_faults(pt.t_s, fault_events):
            if f.kind == "sensor_drift":
                sensor_faults.append(f)
            else:
                op, pc = apply_fault(op, f)
                per_cyl_constants.update(pc)

        pred = predict_steady_state(op, per_cylinder_constants=per_cyl_constants)

        egt_ss = [c["EGT_K"] for c in pred["per_cylinder"]]
        cht_ss = [c["CHT_K"] for c in pred["per_cylinder"]]
        t_oil_ss = pred["T_oil_ss_K"]

        if lag["EGT_K"] is None:
            lag["EGT_K"], lag["CHT_K"], lag["T_oil_K"] = egt_ss, cht_ss, t_oil_ss
        else:
            lag["EGT_K"] = [first_order_lag_step(v, s, DEFAULT_CONSTANTS.tau_egt_s, dt_s)
                            for v, s in zip(lag["EGT_K"], egt_ss)]
            lag["CHT_K"] = [first_order_lag_step(v, s, DEFAULT_CONSTANTS.tau_cht_s, dt_s)
                            for v, s in zip(lag["CHT_K"], cht_ss)]
            lag["T_oil_K"] = first_order_lag_step(lag["T_oil_K"], t_oil_ss,
                                                  DEFAULT_CONSTANTS.tau_oil_s, dt_s)

        frame = {
            "t_s": pt.t_s,
            "scenario": scenario,
            "N_rpm": pt.N_rpm + (rng.normal(0, NOISE_SIGMA["N_rpm"]) if noise else 0),
            # Achieved MAP is what the manifold sensor reads (faults applied);
            # commanded MAP is what the FADEC asked for. The gap is the turbo
            # health indicator.
            "MAP_Pa": op.MAP_Pa + (rng.normal(0, NOISE_SIGMA["MAP_Pa"]) if noise else 0),
            "MAP_commanded_Pa": pt.MAP_Pa,
            "altitude_m": pt.altitude_m,
            # ECU-commanded fuel flow (module docstring): the twin predicts
            # from the command, the engine ran on what was delivered.
            "fuel_flow_kg_s_per_cyl": [
                m * (1 + (rng.normal(0, NOISE_SIGMA["fuel_flow_rel"]) if noise else 0))
                for m in fuel],
            "EGT_K": [v + (rng.normal(0, NOISE_SIGMA["EGT_K"]) if noise else 0)
                      for v in lag["EGT_K"]],
            "CHT_K": [v + (rng.normal(0, NOISE_SIGMA["CHT_K"]) if noise else 0)
                      for v in lag["CHT_K"]],
            "p_oil_Pa": pred["p_oil_Pa"] + (rng.normal(0, NOISE_SIGMA["p_oil_Pa"]) if noise else 0),
            "T_oil_K": lag["T_oil_K"] + (rng.normal(0, NOISE_SIGMA["T_oil_K"]) if noise else 0),
        }

        # Sensor drift is applied to the REPORTED reading only, after
        # everything else; the physics never saw it (faults.py docstring).
        for f in sensor_faults:
            idx = f.cylinder - 1
            frame[f.sensor_channel] = list(frame[f.sensor_channel])
            frame[f.sensor_channel][idx] += f.bias_K

        yield frame


def log_mission(path: str, profile: str, fault_events=(), seed: int = 42,
                noise: bool = True):
    """Run a mission and write telemetry to CSV, one row per timestep.
    Per-cylinder channels are flattened to EGT_K_1 .. EGT_K_4 style columns.
    This CSV is what mission replay (app/twin/replay.py) consumes."""
    frames = list(run_mission(profile, fault_events, seed=seed, noise=noise))
    n_cyl = DEFAULT_GEOMETRY.num_cylinders
    cols = (["t_s", "scenario", "N_rpm", "MAP_Pa", "MAP_commanded_Pa", "altitude_m"]
            + [f"fuel_flow_kg_s_per_cyl_{i}" for i in range(1, n_cyl + 1)]
            + [f"EGT_K_{i}" for i in range(1, n_cyl + 1)]
            + [f"CHT_K_{i}" for i in range(1, n_cyl + 1)]
            + ["p_oil_Pa", "T_oil_K"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for fr in frames:
            w.writerow([fr["t_s"], fr["scenario"], fr["N_rpm"], fr["MAP_Pa"],
                        fr["MAP_commanded_Pa"], fr["altitude_m"]]
                       + list(fr["fuel_flow_kg_s_per_cyl"])
                       + list(fr["EGT_K"]) + list(fr["CHT_K"])
                       + [fr["p_oil_Pa"], fr["T_oil_K"]])
    return len(frames)


def frames_from_csv(path: str):
    """Inverse of log_mission: replay a logged mission as telemetry frames,
    identical in shape to what run_mission yields live."""
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_cyl = DEFAULT_GEOMETRY.num_cylinders
            yield {
                "t_s": float(row["t_s"]),
                "scenario": row["scenario"],
                "N_rpm": float(row["N_rpm"]),
                "MAP_Pa": float(row["MAP_Pa"]),
                "MAP_commanded_Pa": float(row["MAP_commanded_Pa"]),
                "altitude_m": float(row["altitude_m"]),
                "fuel_flow_kg_s_per_cyl": [float(row[f"fuel_flow_kg_s_per_cyl_{i}"])
                                           for i in range(1, n_cyl + 1)],
                "EGT_K": [float(row[f"EGT_K_{i}"]) for i in range(1, n_cyl + 1)],
                "CHT_K": [float(row[f"CHT_K_{i}"]) for i in range(1, n_cyl + 1)],
                "p_oil_Pa": float(row["p_oil_Pa"]),
                "T_oil_K": float(row["T_oil_K"]),
            }
