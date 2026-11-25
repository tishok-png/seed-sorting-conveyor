"""websocket_bridge.py - dashboard-side WebSocket client.

Same interface as the other bridges used by app.py:
    connect() / disconnect() / get_snapshot() / send_command(cmd)

A background thread keeps the connection alive and reconnects automatically,
so Streamlit reruns never block on the network.

Requires:  pip install "websockets>=13"
"""
from __future__ import annotations

import base64
import json
import socket
import threading
import time

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

FRAME_TIMEOUT = 3.0      # seconds without a frame before the camera shows NO SIGNAL
DEAD_LINK_TIMEOUT = 6.0  # seconds without any message before we reconnect


class _AuthError(Exception):
    pass


def _describe(exc: Exception) -> str:
    if isinstance(exc, ConnectionRefusedError):
        return "Connection refused. Is sorter_agent.py running on the Pi?"
    if isinstance(exc, socket.gaierror):
        return "Cannot find that host. Check the Pi address."
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "Timed out reaching the Pi. Are both devices on the same WiFi?"
    if isinstance(exc, ConnectionClosed):
        return "Connection closed by the Pi."
    return str(exc) or type(exc).__name__


class WebSocketBridge:
    def __init__(self, host, port=8765, access_key="", connect_wait=4.0):
        self.host = str(host).strip()
        self.port = int(port)
        self.access_key = str(access_key)
        self.connect_wait = float(connect_wait)
        self._url = f"ws://{self.host}:{self.port}"

        self._lock = threading.Lock()
        self._state = {
            "good": 0, "bad": 0, "status": "Idle", "diverter": "Ready",
        }
        self._frame_b64 = None
        self._frame_at = 0.0
        self._connected = False
        self._error = None

        self._ws = None
        self._stop = threading.Event()
        self._first_attempt = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------
    # Bridge interface
    # ------------------------------------------------------------------
    def connect(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._first_attempt.clear()
        self._thread = threading.Thread(target=self._run, name="ws-bridge", daemon=True)
        self._thread.start()
        # Give the first attempt a moment so the sidebar shows a real result.
        self._first_attempt.wait(self.connect_wait)

    def disconnect(self):
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._connected = False
            self._frame_b64 = None

    def get_snapshot(self) -> dict:
        with self._lock:
            snap = dict(self._state)
            fresh = self._frame_b64 is not None and (time.monotonic() - self._frame_at) < FRAME_TIMEOUT
            snap["camera_frame"] = self._frame_b64 if fresh else None
            snap["connected"] = self._connected
            snap["error"] = self._error
        return snap

    def send_command(self, command: str) -> bool:
        ws = self._ws
        if ws is None or not self._connected:
            return False
        try:
            ws.send(json.dumps({"type": "cmd", "cmd": command}))
            return True
        except Exception as exc:
            with self._lock:
                self._error = _describe(exc)
            return False

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------
    def _run(self):
        delay = 0.5
        while not self._stop.is_set():
            was_connected = False
            try:
                was_connected = self._session()
            except _AuthError as exc:
                with self._lock:
                    self._error = str(exc)
                self._first_attempt.set()
                return  # wrong key: retrying will not help
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._error = _describe(exc)
            finally:
                self._ws = None
                with self._lock:
                    self._connected = False
                self._first_attempt.set()
            if self._stop.wait(delay):
                break
            delay = 0.5 if was_connected else min(delay * 2, 5.0)

    def _session(self) -> bool:
        """One connection. Returns True if it got as far as being authenticated."""
        authed = False
        with ws_connect(
            self._url, open_timeout=3.0, close_timeout=1.0,
            max_size=4 * 1024 * 1024, compression=None,
        ) as ws:
            self._ws = ws
            ws.send(json.dumps({"type": "auth", "key": self.access_key}))
            last_rx = time.monotonic()
            while not self._stop.is_set():
                try:
                    message = ws.recv(timeout=0.5)
                except TimeoutError:
                    if time.monotonic() - last_rx > DEAD_LINK_TIMEOUT:
                        raise ConnectionError("No data from the Pi; reconnecting.")
                    continue
                last_rx = time.monotonic()
                if self._handle(message):
                    authed = True
        return authed

    def _handle(self, message) -> bool:
        """Process one message. Returns True when it confirms authentication."""
        if isinstance(message, (bytes, bytearray)):
            encoded = base64.b64encode(message).decode("ascii")
            with self._lock:
                self._frame_b64 = encoded
                self._frame_at = time.monotonic()
            return False
        try:
            data = json.loads(message)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        kind = data.get("type")
        if kind == "auth_ok":
            with self._lock:
                self._connected = True
                self._error = None
            self._first_attempt.set()
            return True
        if kind == "auth_fail":
            raise _AuthError("Access key rejected by the Pi. Check the key in the sidebar.")
        if kind == "status":
            data.pop("type", None)
            try:
                data["good"] = int(data.get("good", 0))
                data["bad"] = int(data.get("bad", 0))
            except (TypeError, ValueError):
                data["good"], data["bad"] = 0, 0
            with self._lock:
                self._state.update(data)
        return False
