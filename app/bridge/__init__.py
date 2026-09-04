"""bridge/ -- the production integration layer.

The twin's frame contract (t_s, N_rpm, MAP_Pa, altitude_m, per-cylinder
fuel flow, EGT, CHT, oil pressure and temperature) is internal. Real
systems do not speak it. This package owns the translation:

- frames.normalize(): validate any accepted payload and produce the exact
  internal frame dict the twin consumes
- mavlink: ArduPilot/PX4-style EFI telemetry, the bus schema real UAV
  engine ECUs actually report on
- native: our own schema, for systems that can meet it directly
- mqtt: an optional listener for MQTT-brokered telemetry (the transport
  aerospace and IIoT already run), guarded so the server runs fine
  without the dependency

Every conversion documents its unit assumptions inline. A frame that
fails validation is rejected with a reason, never silently coerced: the
twin's claims are only as good as what crosses this boundary.
"""

from .frames import normalize, FrameError, FRAME_FIELDS  # noqa: F401
