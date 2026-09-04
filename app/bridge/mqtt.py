"""bridge/mqtt.py -- optional MQTT telemetry listener.

MQTT is the transport production telemetry already runs on (and the one
architecture.md names for the ingest layer). This listener subscribes to a
topic, parses each message as either a native frame or a MAVLink-style
envelope, and pushes frames into the same ingest queue the REST endpoint
uses.

Deliberately optional: the dependency (paho-mqtt) is imported lazily and
the server runs without it. Enabled by setting NAVTWIN_MQTT to
host[:port] and optionally NAVTWIN_MQTT_TOPIC (default
"navtwin/telemetry").
"""

import json
import logging
import os
import threading

from .frames import FrameError, normalize
from .mavlink import from_mavlink

log = logging.getLogger("twin.bridge.mqtt")


def start_listener(push_frame):
    """Start the listener thread if configured. push_frame(frame) is the
    server's ingest callback. Returns True if the listener is running."""
    host = os.environ.get("NAVTWIN_MQTT")
    if not host:
        return False
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        log.warning("NAVTWIN_MQTT is set but paho-mqtt is not installed; "
                    "MQTT listener disabled")
        return False

    topic = os.environ.get("NAVTWIN_MQTT_TOPIC", "navtwin/telemetry")
    hostname, _, port = host.partition(":")

    def on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            frame = from_mavlink(payload) if "efi" in payload \
                else normalize(payload)
            push_frame(frame)
        except (FrameError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning("rejected MQTT frame on %s: %s", topic, e)

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(hostname, int(port or 1883))
    client.subscribe(topic)
    threading.Thread(target=client.loop_forever, daemon=True).start()
    log.info("MQTT bridge listening on %s topic %s", host, topic)
    return True
