"""
cloud_bridge.py

Wireless bridge - lets the dashboard and the Raspberry Pi talk over the
INTERNET instead of a USB cable, using a free Firebase Realtime Database
as the "middleman." This is what makes the dashboard checkable from a
phone/browser anywhere, not just on the machine wired to the Pi.

Architecture:
    Pi (camera + vision model) --WiFi--> Firebase <--internet-- Dashboard

Data stored in Firebase (all under one node, e.g. "seed_sorting"):
    good           <int>            cumulative good-seed count
    bad            <int>            cumulative bad-seed count
    status         "Idle"/"Running"/"Calibration"/"Fault" - mirrors the
                   Pi/ESP's own reported state, not just what was requested
    diverter       "Ready"/"Diverting" - the reject servo's own state,
                   independent of the machine status above
    command        "START"/"STOP"/"CALIBRATE"/"SPD1"/"SPD2"/"SPD3"/
                   "SERVO50"/"SERVO100"/"SERVO150"/"SERVO180"/""
                   <- dashboard writes, Pi reads + clears
    camera_frame   "<base64 jpeg>"     <- Pi writes, dashboard displays
    camera_ts      <unix timestamp>    <- when the frame was last updated

=== ONE-TIME FIREBASE SETUP (5 minutes, free) ===
1. Go to https://console.firebase.google.com -> Add project (any name).
2. In the project, open "Realtime Database" (left menu, under Build) -> Create Database.
3. Choose "Start in test mode" (fine for a student project demo -
   it's open read/write for 30 days; tighten rules later if needed).
4. Copy the database URL shown at the top, e.g.:
       https://your-project-id-default-rtdb.firebaseio.com
5. Paste that URL into the dashboard sidebar (Cloud mode) - that's it.

=== WHAT THE PI / VISION SCRIPT NEEDS TO DO ===
Using plain Python + the `requests` library (no special SDK needed),
the Pi should periodically PATCH its readings, e.g.:

    import requests
    URL = "https://your-project-id-default-rtdb.firebaseio.com/seed_sorting.json"

    requests.patch(URL, json={
        "good": good_count,
        "bad": bad_count,
        "status": "Running",
        "camera_frame": base64_jpeg_string,   # optional, from the camera
    })

And to receive START/STOP commands, it should poll:

    r = requests.get(URL + "?shallow=false")
    cmd = r.json().get("command")
    # act on cmd, then clear it:
    requests.patch(URL, json={"command": ""})
"""

import time
import requests


class CloudBridge:
    """Talks to the Pi over the internet via a Firebase Realtime Database."""

    def __init__(self, db_url: str, node: str = "seed_sorting", timeout: float = 5.0):
        self.db_url = db_url.rstrip("/")
        self.node = node
        self.timeout = timeout
        self.connected = False
        self.error = None

    def _url(self) -> str:
        return f"{self.db_url}/{self.node}.json"

    def connect(self):
        try:
            r = requests.get(self._url(), timeout=self.timeout)
            r.raise_for_status()
            self.connected = True
            self.error = None
        except Exception as exc:
            self.connected = False
            self.error = f"Could not reach Firebase: {exc}"

    def disconnect(self):
        self.connected = False

    def send_command(self, cmd: str) -> bool:
        try:
            r = requests.patch(self._url(), json={"command": cmd}, timeout=self.timeout)
            r.raise_for_status()
            return True
        except Exception as exc:
            self.error = f"Could not send command: {exc}"
            return False

    def get_snapshot(self) -> dict:
        try:
            r = requests.get(self._url(), timeout=self.timeout)
            r.raise_for_status()
            data = r.json() or {}
            self.connected = True
            self.error = None
            return {
                "good": int(data.get("good", 0) or 0),
                "bad": int(data.get("bad", 0) or 0),
                "status": data.get("status", "Idle") or "Idle",
                "diverter": data.get("diverter", "Ready") or "Ready",
                "belt_speed": int(data.get("belt_speed", 1) or 1),
                "connected": True,
                "error": None,
                "camera_frame": data.get("camera_frame"),
                "camera_ts": data.get("camera_ts"),
            }
        except Exception as exc:
            self.connected = False
            self.error = f"Could not read from Firebase: {exc}"
            return {
                "good": 0, "bad": 0, "status": "Idle", "diverter": "Ready", "belt_speed": 1,
                "connected": False, "error": self.error,
                "camera_frame": None, "camera_ts": None,
            }
