"""cloud_link.py

Pi-side counterpart to the dashboard's cloud_bridge.py. This plays the
"Pi / vision script" role described in that file's own docstring: plain
HTTPS GET/PATCH against the same Firebase Realtime Database node
("seed_sorting" by default) — no firebase-admin package, no service
account. Nothing on the dashboard side (app.py, cloud_bridge.py) needs
to change for this to work.

Data this writes:
    good, bad       cumulative counters (matches the dashboard's running totals)
    status          "Idle" / "Running" / "Calibration" / "Fault" — reflects the
                    Pi agent's own synthesized state (ESP motor echoes plus the
                    Pi's own reject-FIFO health), not just what was requested
    diverter        "Ready" / "Diverting" — the reject servo's own current state,
                    independent of the overall machine status above
    camera_frame    base64 JPEG string
    camera_ts       unix timestamp of that frame

Data this reads, then clears:
    command         "START" / "STOP" / "CALIBRATE" / "SPD1" / "SPD2" / "SPD3" /
                    "SERVO50" / "SERVO100" / "SERVO150" / "SERVO180" / "" — the
                    dashboard writes it when the operator clicks a button; we
                    act on it once and PATCH it back to "" so the same click
                    doesn't fire twice.

This matches cloud_bridge.py's documented setup exactly: create a
Firebase Realtime Database in test mode, put its URL in both the
dashboard sidebar (Cloud mode) and CLOUD_DB_URL below. Test mode is
open read/write for 30 days — fine for a demo, tighten the rules
before this needs to be public long-term.
"""
from __future__ import annotations

import base64
import logging
import queue
import threading
import time

import requests

LOG = logging.getLogger("sorter_agent.cloud")


class CloudLink:
    def __init__(self, db_url: str, node: str = "seed_sorting",
                 poll_interval: float = 1.0, timeout: float = 5.0):
        self.url = f"{db_url.rstrip('/')}/{node}.json"
        self.poll_interval = poll_interval
        self.timeout = timeout

        self.outgoing_frames: queue.Queue = queue.Queue(maxsize=1)
        self.outgoing_status: queue.Queue = queue.Queue(maxsize=1)
        self.incoming_commands: queue.Queue = queue.Queue()

        threading.Thread(target=self._writer_loop, daemon=True).start()
        threading.Thread(target=self._poller_loop, daemon=True).start()
        LOG.info("Cloud link polling %s", self.url)

    # --- interface used by sorter_pi_agent.py ---

    def on_seed(self, is_good: bool) -> None:
        pass  # cloud_bridge.py reads cumulative totals via push_status, not per-seed events

    def push_frame(self, jpeg_bytes: bytes) -> None:
        self._replace(self.outgoing_frames, jpeg_bytes)

    def push_status(self, status: dict) -> None:
        payload = {
            "good": status["good"],
            "bad": status["bad"],
            "status": status["state"],      # "Idle" / "Running" / "Calibration" / "Fault"
            "diverter": status["diverter"], # "Ready" / "Diverting"
        }
        self._replace(self.outgoing_status, payload)

    def poll_command(self):
        try:
            return self.incoming_commands.get_nowait()
        except queue.Empty:
            return None

    # --- internals ---

    @staticmethod
    def _replace(q: "queue.Queue", item) -> None:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _writer_loop(self) -> None:
        while True:
            patch = {}
            try:
                frame = self.outgoing_frames.get_nowait()
                patch["camera_frame"] = base64.b64encode(frame).decode("ascii")
                patch["camera_ts"] = time.time()
            except queue.Empty:
                pass
            try:
                patch.update(self.outgoing_status.get_nowait())
            except queue.Empty:
                pass
            wrote = False
            if patch:
                try:
                    requests.patch(self.url, json=patch, timeout=self.timeout)
                    wrote = True
                except Exception as exc:
                    LOG.warning("Cloud write failed: %s", exc)
            time.sleep(0.02 if wrote else 0.1)

    def _poller_loop(self) -> None:
        while True:
            try:
                r = requests.get(self.url, timeout=self.timeout)
                r.raise_for_status()
                data = r.json() or {}
                cmd = data.get("command")
                recognized = (
                    "START", "STOP", "CALIBRATE", "SPD1", "SPD2", "SPD3",
                    "SERVO50", "SERVO100", "SERVO150", "SERVO180",
                )
                if cmd in recognized:
                    self.incoming_commands.put(cmd)
                    requests.patch(self.url, json={"command": ""}, timeout=self.timeout)
            except Exception as exc:
                LOG.warning("Cloud poll failed: %s", exc)
            time.sleep(self.poll_interval)
