#!/usr/bin/env python3
"""Pi-side seed sorter agent: live camera, on-demand classifier, reject FIFO."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from reject_actuator import ServoReject

PROJECT_DIR = Path(__file__).resolve().parent

# ---- Everyday operation ----
DRY_RUN = False
ESP_LINK = "wifi"                 # TCP over WiFi (matches ESP8266 firmware)
ESP_SERIAL_PORT = "/dev/ttyUSB0"
ESP_SERIAL_BAUD = 115200
ESP_HOST = "seedsorter.local"    # ESP8266 mDNS hostname
ESP_PORT = 8266
CAMERA = "picamera2"              # or camera index, e.g. 0
WIDTH, HEIGHT = 640, 480
HEADLESS = True

# ---- Dashboard link ----
# Both dashboard transports are available simultaneously. app.py chooses one.
WS_HOST = "0.0.0.0"
WS_PORT = 8765
WS_ACCESS_KEY = "seedsorter"       # Must match DEFAULT_WS_KEY in app.py
CLOUD_DB_URL = "https://maize-sorting-default-rtdb.firebaseio.com"
CLOUD_NODE = "seed_sorting"
DASHBOARD_SERIAL_PORT = "/dev/ttyUSB0"
DASHBOARD_SERIAL_BAUD = 9600
STREAM_FPS = 4.0
STREAM_JPEG_QUALITY = 60

# ---- Pi GPIO reject servo ----
SERVO_GPIO_PIN = 18
SERVO_PASS_ANGLE = 20
SERVO_DIVERT_ANGLE = 110
SERVO_HOLD_SECONDS = 0.65
SERVO_MOVE_SECONDS = 0.35
REJECT_MAX_IN_FLIGHT = 16
SERVO_MANUAL_SWEEP_SECONDS = 1.5
CAMERA_TO_SERVO_MM = 500.0
BELT_SPEED_CM_S_BY_LEVEL = {1: 3.75, 2: 6.2, 3: 7.4}  # Must match ESP firmware

# ---- Detection timing ----
SEED_OBSERVATION_SECONDS = 6.0
STABLE_DECISION_SECONDS = 3.0
EMPTY_FRAMES = 3
EMPTY_SECONDS = 0.4
MAX_FRAME_AGE = 0.75
MAX_FRAME_GAP = 1.0
MIN_MAIZE_FRAMES = 3

# ---- Model and camera ----
MODEL_PATH = PROJECT_DIR / "model_outputs/maize_mobilenetv3large_float32.tflite"
METADATA_PATH = PROJECT_DIR / "model_outputs/deployment_metadata_v2_recovered.json"
THREADS = 2

LOG = logging.getLogger("sorter_agent")
CLASSES = ["BAD_SEED", "GOOD_SEED", "NO_MAIZE"]


def prepare_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(
        rgb.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    )[None, ...]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    eligible: bool
    empty: bool


def interpret_probabilities(values, metadata: dict) -> Prediction:
    probs = np.asarray(values, dtype=np.float64).reshape(-1)
    if probs.shape != (3,) or not np.isfinite(probs).all():
        raise ValueError("Model must return three finite class probabilities.")
    if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1.0) > 0.01:
        raise ValueError("Model output is not a three-class softmax probability vector.")
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    margin = confidence - float(np.sort(probs)[-2])
    label = CLASSES[index]
    policy = metadata["decision_policy"]
    threshold = policy["class_thresholds"].get(label, {})
    eligible = (
        label != "NO_MAIZE"
        and bool(policy["actuation_enabled"])
        and confidence >= float(threshold.get("confidence", 1.0))
        and margin >= float(threshold.get("margin", 1.0))
    )
    return Prediction(label, confidence, margin, eligible, label == "NO_MAIZE")


def validate_metadata(metadata: dict) -> None:
    if metadata.get("schema_version") != 2:
        raise ValueError("Expected schema-version-2 deployment metadata.")
    if metadata.get("class_names") != CLASSES:
        raise ValueError("Metadata class_names must be BAD_SEED, GOOD_SEED, NO_MAIZE.")
    if metadata.get("class_to_index") != dict(zip(CLASSES, range(3))):
        raise ValueError("Metadata class_to_index does not match the model class order.")
    if (
        metadata.get("input_color_order") != "RGB"
        or metadata.get("input_dtype") != "float32"
        or metadata.get("external_input_range") != [0.0, 255.0]
        or metadata.get("resize_method") != "pillow_bilinear"
    ):
        raise ValueError("Unsupported model preprocessing contract.")
    policy = metadata.get("decision_policy")
    if not isinstance(policy, dict) or "class_thresholds" not in policy:
        raise ValueError("Metadata is missing decision_policy.class_thresholds.")


class Classifier:
    def __init__(self, model_path: Path, metadata_path: Path, threads: int = 2):
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        validate_metadata(self.metadata)
        model_bytes = model_path.read_bytes()
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite import Interpreter
        self.interpreter = Interpreter(model_content=model_bytes, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.size = tuple(int(x) for x in self.metadata["image_size"])
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError(f"metadata image_size must be [height, width], got {self.size!r}")
        if self.input["dtype"] != np.float32:
            raise ValueError(f"Model input dtype is {self.input['dtype']}; expected float32.")
        if tuple(self.input["shape"])[-1] != 3:
            raise ValueError(f"Model input must have 3 RGB channels; got {self.input['shape']!r}")
        expected = tuple(self.input["shape"])
        if len(expected) == 4 and expected[1:3] != self.size:
            raise ValueError(
                f"Metadata image_size {self.size} does not match model input {expected}."
            )

    def predict(self, rgb: Image.Image) -> Prediction:
        tensor = prepare_rgb(rgb, self.size)
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        return interpret_probabilities(
            self.interpreter.get_tensor(self.output["index"]), self.metadata
        )


@dataclass(frozen=True)
class PassageDetection:
    first_seen: float
    left_frame_at: float
    classification: str
    confidence: float
    margin: float
    stable: bool


class PassageSeedDetector:
    """Collects evidence while a seed is visible, then emits one result on exit."""

    def __init__(
        self, observation_seconds, stable_seconds, empty_frames,
        empty_seconds, min_maize_frames, max_gap
    ):
        self.observation_seconds = float(observation_seconds)
        self.stable_seconds = float(stable_seconds)
        self.empty_frames = int(empty_frames)
        self.empty_seconds = float(empty_seconds)
        self.min_maize_frames = int(min_maize_frames)
        self.max_gap = float(max_gap)
        self.last_frame = None
        self.armed = True
        self.in_seed = False
        self.first_seen = 0.0
        self.empty_since = 0.0
        self.empty_count = 0
        self.candidate = None
        self.candidate_since = 0.0
        self.locked = None
        self.best = {"BAD_SEED": None, "GOOD_SEED": None}
        self.scores = {"BAD_SEED": 0.0, "GOOD_SEED": 0.0}
        self.maize_frames = 0
        self.observation_closed = False

    def reset(self):
        self.last_frame = None
        self.armed = True
        self.empty_since = 0.0
        self.empty_count = 0
        self._reset_seed()

    def _reset_seed(self):
        self.in_seed = False
        self.first_seen = 0.0
        self.candidate = None
        self.candidate_since = 0.0
        self.locked = None
        self.best = {"BAD_SEED": None, "GOOD_SEED": None}
        self.scores = {"BAD_SEED": 0.0, "GOOD_SEED": 0.0}
        self.maize_frames = 0
        self.observation_closed = False

    def _finish(self, left_at):
        if self.maize_frames < self.min_maize_frames:
            LOG.warning(
                "Passage ignored: %d maize frames; need at least %d.",
                self.maize_frames, self.min_maize_frames
            )
            self._reset_seed()
            return None
        chosen = self.locked
        stable = chosen is not None
        if chosen is None:
            chosen = max(self.scores, key=self.scores.get)
            if self.scores[chosen] <= 0:
                chosen = "BAD_SEED"
        evidence = self.best.get(chosen)
        result = PassageDetection(
            self.first_seen, left_at, chosen,
            evidence.confidence if evidence else 0.0,
            evidence.margin if evidence else 0.0,
            stable,
        )
        self._reset_seed()
        return result

    def observe(self, prediction: Prediction, now: float):
        if self.last_frame is not None and now <= self.last_frame:
            return None
        if self.last_frame is not None and now - self.last_frame > self.max_gap:
            if self.in_seed:
                raise RuntimeError("Camera tracking gap during seed passage; clear the belt.")
            self.armed = True
            self.empty_count = 0
        self.last_frame = now

        if prediction.empty:
            if self.empty_count == 0:
                self.empty_since = now
            self.empty_count += 1
            if self.empty_count < self.empty_frames or now - self.empty_since < self.empty_seconds:
                return None
            left_at = self.empty_since
            if self.in_seed:
                result = self._finish(left_at)
                self.armed = True
                return result
            self.armed = True
            return None

        self.empty_count = 0
        if not self.in_seed:
            if not self.armed:
                return None
            self.in_seed = True
            self.armed = False
            self.first_seen = now

        if now - self.first_seen >= self.observation_seconds:
            self.observation_closed = True
        if self.observation_closed or prediction.label not in ("BAD_SEED", "GOOD_SEED"):
            return None

        self.maize_frames += 1
        self.scores[prediction.label] += prediction.confidence
        previous = self.best[prediction.label]
        if previous is None or prediction.confidence > previous.confidence:
            self.best[prediction.label] = prediction

        if not prediction.eligible:
            self.candidate = None
        elif prediction.label != self.candidate:
            self.candidate = prediction.label
            self.candidate_since = now
        elif self.locked is None and now - self.candidate_since >= self.stable_seconds:
            self.locked = self.candidate
        return None


class LatestCamera:
    """Continuously reads the camera in a background thread; latest frame wins."""

    def __init__(self, source, width, height):
        self.capture = None
        self.picamera = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.error = None

        if source == "picamera2":
            from picamera2 import Picamera2
            self.picamera = Picamera2()
            config = self.picamera.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                buffer_count=2,
            )
            self.picamera.configure(config)
            self.picamera.start()
        else:
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"Could not open camera {source!r}.")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self.thread.start()

    def _reader(self):
        failures = 0
        try:
            while not self.stop_event.is_set():
                if self.picamera is not None:
                    frame = self.picamera.capture_array("main")
                    ok = frame is not None
                else:
                    ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError("Camera stopped delivering frames.")
                    self.stop_event.wait(0.05)
                    continue
                failures = 0
                with self.lock:
                    self.frame = frame
                    self.timestamp = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = exc

    def latest(self):
        with self.lock:
            if self.error:
                raise RuntimeError(f"Camera reader failed: {self.error}") from self.error
            return (None if self.frame is None else self.frame.copy()), self.timestamp

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            LOG.error("Camera reader did not stop before shutdown.")
        if self.picamera is not None:
            try:
                self.picamera.stop()
                self.picamera.close()
            except Exception:
                LOG.exception("Error closing Picamera2.")
        elif self.capture is not None:
            self.capture.release()


class ESPWiFiLink:
    def __init__(self, host, port, connect_timeout=3.0, reconnect_interval=3.0):
        self.host, self.port = host, int(port)
        self.connect_timeout = connect_timeout
        self.reconnect_interval = reconnect_interval
        self.sock = None
        self._recv_buffer = b""
        self._pending_lines = []
        self._ever_connected = False
        self._last_connect_attempt = 0.0
        self._try_connect()

    @staticmethod
    def _frame(payload):
        checksum = 0
        for byte in payload.encode("ascii"):
            checksum ^= byte
        return f"{payload}*{checksum:02X}\n".encode("ascii")

    def _close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _try_connect(self):
        now = time.monotonic()
        if now - self._last_connect_attempt < self.reconnect_interval:
            return
        self._last_connect_attempt = now
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            sock.settimeout(0.0)
            self.sock = sock
            self._recv_buffer = b""
            reconnect = self._ever_connected
            self._ever_connected = True
            LOG.info("Connected to ESP at %s:%d.", self.host, self.port)
            self._wait_ready(suppress_ready=not reconnect)
        except OSError as exc:
            self._close_socket()
            LOG.warning("ESP connection failed: %s", exc)

    def _wait_ready(self, suppress_ready, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.sock is not None:
            for line in self._read_socket_lines():
                if line == "READY":
                    if not suppress_ready:
                        self._pending_lines.append("READY")
                    return
                self._pending_lines.append(line)
            time.sleep(0.02)
        LOG.warning("No ESP READY banner received.")

    def _read_socket_lines(self):
        lines = []
        if self.sock is None:
            return lines
        try:
            chunk = self.sock.recv(4096)
            if chunk == b"":
                self._close_socket()
                return lines
            self._recv_buffer += chunk
        except BlockingIOError:
            pass
        except OSError:
            self._close_socket()
            return lines
        while b"\n" in self._recv_buffer:
            raw, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            line = raw.decode("ascii", "ignore").strip()
            if line:
                lines.append(line)
        return lines

    def send(self, payload):
        if self.sock is None:
            self._try_connect()
        if self.sock is None:
            LOG.error("ESP link down; command %s was not sent.", payload)
            return
        try:
            self.sock.sendall(self._frame(payload))
        except OSError as exc:
            LOG.warning("ESP send failed: %s", exc)
            self._close_socket()

    def read_lines(self):
        if self.sock is None:
            self._try_connect()
        lines = list(self._pending_lines)
        self._pending_lines.clear()
        lines.extend(self._read_socket_lines())
        return lines

    def run(self): self.send("RUN")
    def stop(self): self.send("STOP")
    def calibrate(self): self.send("CALIBRATE")
    def set_speed(self, level): self.send(f"SPD{int(level)}")

    def close(self):
        if self.sock is not None:
            self.send("STOP")
        self._close_socket()


class ESPSerialLink:
    def __init__(self, port, baud=115200, reconnect_interval=3.0, ready_timeout=5.0):
        import serial
        self._serial = serial
        self.port, self.baud = port, int(baud)
        self.reconnect_interval = reconnect_interval
        self.ready_timeout = ready_timeout
        self.ser = None
        self._recv_buffer = b""
        self._pending_lines = []
        self._ever_connected = False
        self._last_connect_attempt = 0.0
        self._try_connect()

    @staticmethod
    def _frame(payload):
        checksum = 0
        for byte in payload.encode("ascii"):
            checksum ^= byte
        return f"{payload}*{checksum:02X}\n".encode("ascii")

    def _close_port(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def _try_connect(self):
        now = time.monotonic()
        if now - self._last_connect_attempt < self.reconnect_interval:
            return
        self._last_connect_attempt = now
        try:
            self.ser = self._serial.Serial(
                self.port, self.baud, timeout=0, write_timeout=1.0
            )
            self._recv_buffer = b""
            reconnect = self._ever_connected
            self._ever_connected = True
            LOG.info("Opened ESP serial port %s @ %d.", self.port, self.baud)
            self._wait_ready(suppress_ready=not reconnect)
        except (self._serial.SerialException, OSError) as exc:
            self._close_port()
            LOG.warning("Could not open ESP serial port %s: %s", self.port, exc)

    def _wait_ready(self, suppress_ready):
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline and self.ser is not None:
            for line in self._read_serial_lines():
                if line == "READY":
                    if not suppress_ready:
                        self._pending_lines.append("READY")
                    return
                self._pending_lines.append(line)
            time.sleep(0.02)
        LOG.warning("No ESP READY banner received on serial.")

    def _read_serial_lines(self):
        lines = []
        if self.ser is None:
            return lines
        try:
            chunk = self.ser.read(4096)
            if chunk:
                self._recv_buffer += chunk
        except (self._serial.SerialException, OSError):
            self._close_port()
            return lines
        while b"\n" in self._recv_buffer:
            raw, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            line = raw.decode("ascii", "ignore").strip()
            if line:
                lines.append(line)
        return lines

    def send(self, payload):
        if self.ser is None:
            self._try_connect()
        if self.ser is None:
            LOG.error("ESP link down; command %s was not sent.", payload)
            return
        try:
            self.ser.write(self._frame(payload))
        except (self._serial.SerialException, OSError) as exc:
            LOG.warning("ESP serial send failed: %s", exc)
            self._close_port()

    def read_lines(self):
        if self.ser is None:
            self._try_connect()
        lines = list(self._pending_lines)
        self._pending_lines.clear()
        lines.extend(self._read_serial_lines())
        return lines

    def run(self): self.send("RUN")
    def stop(self): self.send("STOP")
    def calibrate(self): self.send("CALIBRATE")
    def set_speed(self, level): self.send(f"SPD{int(level)}")

    def close(self):
        if self.ser is not None:
            self.send("STOP")
        self._close_port()


class DryRunLink:
    def run(self): LOG.info("[DRY RUN] RUN")
    def stop(self): LOG.info("[DRY RUN] STOP")
    def calibrate(self): LOG.info("[DRY RUN] CALIBRATE")
    def set_speed(self, level): LOG.info("[DRY RUN] SPD%d", level)
    def read_lines(self): return []
    def close(self): pass


class WebSocketDashboardServer:
    """Pi-side WebSocket server used by app.py's WebSocketBridge.

    Protocol: dashboard sends auth JSON first, then {"type":"cmd","cmd":"START"}.
    Pi sends {"type":"status", ...} JSON and binary JPEG frames.
    """
    def __init__(self, host=WS_HOST, port=WS_PORT, access_key=WS_ACCESS_KEY):
        self.host = host
        self.port = int(port)
        self.access_key = access_key
        self.commands = queue.Queue()
        self.clients = set()
        self.clients_lock = None
        self.loop = None
        self.server = None
        self.thread = None
        self.ready = threading.Event()
        self.stop_requested = False
        self.good = 0
        self.bad = 0
        self._last_error_log = 0.0
        self.thread = threading.Thread(target=self._thread_main, name="dashboard-websocket", daemon=True)
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("WebSocket server did not start within 5 seconds.")

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except Exception:
            LOG.exception("WebSocket dashboard server stopped.")
            self.ready.set()

    async def _serve(self):
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("WebSocket mode requires the 'websockets' package: pip install websockets") from exc

        self.loop = asyncio.get_running_loop()
        self.clients_lock = asyncio.Lock()

        async def handler(websocket, path=None):
            try:
                first = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                if isinstance(first, bytes):
                    return
                auth = json.loads(first)
                if auth.get("type") != "auth" or auth.get("key") != self.access_key:
                    await websocket.close(code=4001, reason="Authentication failed")
                    return
                async with self.clients_lock:
                    self.clients.add(websocket)
                LOG.info("Dashboard connected by WebSocket: %s", getattr(websocket, "remote_address", None))
                await websocket.send(json.dumps({
                    "type": "status", "status": "Idle", "state": "Idle",
                    "diverter": "Ready", "good": self.good, "bad": self.bad,
                }))
                async for message in websocket:
                    if not isinstance(message, str):
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("type") == "cmd":
                        command = payload.get("cmd")
                        if isinstance(command, str) and command:
                            self.commands.put(command.strip())
                            LOG.info("WebSocket dashboard command received: %s", command.strip().split(":", 1)[0])
            except Exception as exc:
                # Normal disconnects are common when the dashboard is closed.
                if exc.__class__.__name__ not in ("ConnectionClosedOK", "ConnectionClosedError", "ConnectionClosed"):
                    LOG.debug("WebSocket client ended: %s", exc)
            finally:
                if self.clients_lock is not None:
                    async with self.clients_lock:
                        self.clients.discard(websocket)

        self.server = await websockets.serve(handler, self.host, self.port, max_size=2**20, ping_interval=20, ping_timeout=20)
        LOG.info("WebSocket dashboard server listening on %s:%d", self.host, self.port)
        self.ready.set()
        await self.server.wait_closed()

    async def _broadcast(self, payload):
        if self.clients_lock is None:
            return
        async with self.clients_lock:
            clients = list(self.clients)
        if not clients:
            return
        stale = []
        for client in clients:
            try:
                await client.send(payload)
            except Exception:
                stale.append(client)
        if stale:
            async with self.clients_lock:
                for client in stale:
                    self.clients.discard(client)

    def _send(self, payload):
        if self.loop is None or not self.loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)
        except RuntimeError:
            pass

    def poll_command(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def on_seed(self, is_good):
        if is_good:
            self.good += 1
        else:
            self.bad += 1

    def push_frame(self, jpeg_bytes):
        self._send(bytes(jpeg_bytes))

    def push_status(self, status):
        payload = dict(status)
        payload["type"] = "status"
        payload.setdefault("status", payload.get("state", "Idle"))
        payload.setdefault("good", self.good)
        payload.setdefault("bad", self.bad)
        self._send(json.dumps(payload, separators=(",", ":")))

    def close(self):
        if self.loop is not None and self.loop.is_running() and self.server is not None:
            self.loop.call_soon_threadsafe(self.server.close)
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class DashboardMux:
    """Fan out telemetry to both transports and accept commands from either."""
    def __init__(self, cloud_link, websocket_link):
        self.cloud = cloud_link
        self.websocket = websocket_link
        self._last_cloud_error = 0.0

    def poll_command(self):
        # The UI only connects through one transport at a time. Polling both
        # means changing the dashboard selector no longer requires a Pi restart.
        command = self.websocket.poll_command() if self.websocket else None
        if command:
            return command
        if self.cloud:
            try:
                return self.cloud.poll_command()
            except Exception as exc:
                now = time.monotonic()
                if now - self._last_cloud_error > 10:
                    LOG.warning("Cloud command polling failed; WebSocket remains available: %s", exc)
                    self._last_cloud_error = now
        return None

    def on_seed(self, is_good):
        for link in (self.cloud, self.websocket):
            if link:
                try: link.on_seed(is_good)
                except Exception: LOG.exception("Dashboard seed-count update failed.")

    def push_frame(self, jpeg_bytes):
        for link in (self.cloud, self.websocket):
            if link:
                try: link.push_frame(jpeg_bytes)
                except Exception: LOG.exception("Dashboard frame publish failed.")

    def push_status(self, status):
        for link in (self.cloud, self.websocket):
            if link:
                try: link.push_status(status)
                except Exception: LOG.exception("Dashboard status publish failed.")

    def close(self):
        for link in (self.cloud, self.websocket):
            if link:
                try:
                    close = getattr(link, "close", None) or getattr(link, "disconnect", None)
                    if close: close()
                except Exception:
                    LOG.exception("Dashboard link shutdown failed.")


def build_dashboard_link():
    """Start both dashboard transports; Streamlit chooses which one to use."""
    cloud = websocket_link = None
    try:
        from cloud_link import CloudLink
        cloud = CloudLink(CLOUD_DB_URL, CLOUD_NODE)
        LOG.info("Cloud dashboard link ready (Firebase node: %s).", CLOUD_NODE)
    except Exception:
        LOG.exception("Cloud dashboard link could not start; WebSocket can still be used.")

    try:
        websocket_link = WebSocketDashboardServer()
    except Exception:
        LOG.exception("WebSocket dashboard server could not start; Cloud mode can still be used.")

    if cloud is None and websocket_link is None:
        raise RuntimeError("Neither dashboard transport could be started.")
    return DashboardMux(cloud, websocket_link)


def build_esp_link():
    if DRY_RUN:
        return DryRunLink()
    if ESP_LINK == "serial":
        return ESPSerialLink(ESP_SERIAL_PORT, ESP_SERIAL_BAUD)
    if ESP_LINK == "wifi":
        return ESPWiFiLink(ESP_HOST, ESP_PORT)
    raise ValueError(f"Unknown ESP_LINK: {ESP_LINK!r}")


def parse_servo_command(command):
    if not isinstance(command, str) or not command.startswith("SERVO"):
        return None
    rest = command[5:]
    if not rest.isdigit():
        return None
    angle = int(rest)
    return angle if 0 <= angle <= 180 else None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    link = camera = reject = web = None
    classifier = None
    try:
        link = build_esp_link()
        web = build_dashboard_link()
        camera = LatestCamera(CAMERA, WIDTH, HEIGHT)

        try:
            reject = ServoReject(
                SERVO_GPIO_PIN, SERVO_PASS_ANGLE, SERVO_DIVERT_ANGLE,
                SERVO_HOLD_SECONDS, SERVO_MOVE_SECONDS, REJECT_MAX_IN_FLIGHT,
                manual_slew_full_sweep_seconds=SERVO_MANUAL_SWEEP_SECONDS,
            )
        except Exception:
            LOG.exception("Reject servo unavailable. Camera will stream, but START is disabled.")
            reject = None

        detector = PassageSeedDetector(
            SEED_OBSERVATION_SECONDS, STABLE_DECISION_SECONDS,
            EMPTY_FRAMES, EMPTY_SECONDS, MIN_MAIZE_FRAMES, MAX_FRAME_GAP,
        )
        machine_state = "Idle" if reject is not None else "Fault"
        in_fault = reject is None
        current_speed_level = 1
        good_count = bad_count = 0
        last_stream = 0.0
        stream_interval = 1.0 / max(0.5, STREAM_FPS)
        last_prediction_log = 0.0
        last_stale_log = 0.0

        LOG.info("Camera is live. Model is idle; waiting for START.")

        while True:
            command = web.poll_command()

            # Handle dashboard commands. START is deliberately accepted only
            # from Idle; it cannot restart the machine during Calibration.
            if command == "START":
                if machine_state != "Idle":
                    LOG.warning("START ignored while machine state is %s.", machine_state)
                elif reject is None or in_fault:
                    LOG.warning("START ignored: reject servo is unavailable or FAULT is active.")
                elif reject.manual:
                    LOG.warning("START ignored: switch servo to Autonomous mode first.")
                else:
                    try:
                        if classifier is None:
                            LOG.info("Loading TFLite model: %s", MODEL_PATH)
                            classifier = Classifier(MODEL_PATH, METADATA_PATH, THREADS)
                            LOG.info(
                                "Model loaded. input=%s output=%s image_size=%s",
                                classifier.input["shape"], classifier.output["shape"],
                                classifier.size,
                            )
                        detector.reset()
                        reject.clear()
                        if reject.fault:
                            raise RuntimeError(f"Servo fault: {reject.fault}")
                        link.set_speed(current_speed_level)
                        link.run()
                        machine_state = "Running"
                        LOG.info("START received. Classification is active.")
                    except Exception:
                        LOG.exception("START failed; stopping motors.")
                        link.stop()
                        machine_state = "Fault"
                        in_fault = True

            elif command == "STOP":
                link.stop()
                if reject is not None:
                    try:
                        reject.clear()
                    except Exception:
                        LOG.exception("Could not park reject gate during STOP.")
                        in_fault = True
                detector.reset()
                machine_state = "Idle" if reject is not None and not in_fault else "Fault"
                LOG.info("STOP received. Classification paused; camera keeps streaming.")

            elif command == "CALIBRATE":
                if machine_state not in ("Idle", "Calibration") or in_fault:
                    LOG.warning("CALIBRATE ignored while state is %s.", machine_state)
                else:
                    link.stop()
                    link.calibrate()
                    current_speed_level = 1
                    detector.reset()
                    machine_state = "Calibration"
                    LOG.info("CALIBRATE received.")

            elif command in ("SPD1", "SPD2", "SPD3"):
                new_level = int(command[-1])
                if machine_state == "Running" and reject is not None:
                    # Re-time pending events before applying the new speed.
                    reject.requeue_with_speed(
                        BELT_SPEED_CM_S_BY_LEVEL[new_level], CAMERA_TO_SERVO_MM
                    )
                link.set_speed(new_level)
                current_speed_level = new_level
                LOG.info("%s received.", command)

            elif command in ("MANUAL", "AUTO", "AUTONOMOUS"):
                if reject is None:
                    LOG.warning("Mode command ignored: servo hardware unavailable.")
                elif command == "MANUAL" and machine_state == "Running":
                    link.stop()
                    detector.reset()
                    try:
                        reject.set_manual(True)
                    finally:
                        machine_state = "Idle"
                    LOG.warning("Manual mode selected; sorting stopped for safety.")
                else:
                    try:
                        reject.set_manual(command == "MANUAL")
                    except Exception:
                        LOG.exception("Could not change servo mode.")
                        in_fault = True
                        machine_state = "Fault"

            else:
                angle = parse_servo_command(command)
                if angle is not None:
                    if reject is None:
                        LOG.warning("Servo command ignored: servo unavailable.")
                    else:
                        try:
                            moved = reject.move_to(angle)
                            LOG.info("Servo -> %d degrees; accepted=%s.", angle, moved)
                        except Exception:
                            LOG.exception("Manual servo movement failed.")
                            in_fault = True
                            machine_state = "Fault"
                            link.stop()

            # Read ESP messages and detect a reboot during an active operation.
            for line in link.read_lines():
                LOG.info("ESP: %s", line)
                if line == "READY" and machine_state in ("Running", "Calibration"):
                    LOG.error("ESP reboot detected during %s; entering FAULT.", machine_state)
                    link.stop()
                    if reject is not None:
                        try:
                            reject.clear()
                        except Exception:
                            LOG.exception("Could not park gate after ESP reboot.")
                    detector.reset()
                    in_fault = True
                    machine_state = "Fault"
                elif line.startswith("STATE ") and not in_fault:
                    esp_state = line[len("STATE "):].strip().capitalize()
                    if esp_state in ("Idle", "Running", "Calibration"):
                        # Do not let an ESP status line override a local fault.
                        if esp_state == "Running" and machine_state != "Running":
                            LOG.warning("Ignoring unexpected ESP RUNNING state.")
                        elif machine_state != "Running" or esp_state != "Idle":
                            machine_state = esp_state

            running = machine_state == "Running"
            prediction = None
            frame = None
            captured = 0.0

            try:
                frame, captured = camera.latest()
            except Exception:
                LOG.exception("Camera reader failed; stopping machine.")
                link.stop()
                detector.reset()
                in_fault = True
                machine_state = "Fault"

            if frame is None:
                # Keep the agent alive so the dashboard can still report FAULT.
                if machine_state == "Running":
                    link.stop()
                    detector.reset()
                    in_fault = True
                    machine_state = "Fault"
                now = time.monotonic()
                if now - last_stream >= 1.0:
                    web.push_status({
                        "state": machine_state,
                        "camera": "Unavailable",
                        "model": "Idle",
                        "diverter": reject.state() if reject else "Unavailable",
                        "reject_queue": reject.depth() if reject else 0,
                        "good": good_count,
                        "bad": bad_count,
                        "label": "IDLE",
                        "confidence": 0.0,
                        "servo_mode": reject.mode if reject else "Unavailable",
                        "servo_angle": round(reject.current_angle_deg(), 1) if reject else 0.0,
                    })
                    last_stream = now
                time.sleep(0.05)
                continue

            frame_age = time.monotonic() - captured
            if frame_age > MAX_FRAME_AGE:
                if running:
                    LOG.error("Camera frame stale (%.2fs); stopping for safety.", frame_age)
                    link.stop()
                    detector.reset()
                    in_fault = True
                    machine_state = "Fault"
                    running = False
                if time.monotonic() - last_stale_log >= 2.0:
                    LOG.warning("Camera frame is stale (%.2fs).", frame_age)
                    last_stale_log = time.monotonic()

            running = machine_state == "Running"
            if running:
                if classifier is None:
                    LOG.critical("Running state has no classifier; stopping.")
                    link.stop()
                    machine_state = "Fault"
                    in_fault = True
                    detector.reset()
                    running = False
                else:
                    try:
                        rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        prediction = classifier.predict(rgb)
                        log_now = time.monotonic()
                        if log_now - last_prediction_log >= 1.0:
                            LOG.info(
                                "MODEL: %s confidence=%.3f margin=%.3f eligible=%s",
                                prediction.label, prediction.confidence,
                                prediction.margin, prediction.eligible,
                            )
                            last_prediction_log = log_now

                        detection = detector.observe(prediction, captured)
                        if detection is not None:
                            if detection.classification == "GOOD_SEED":
                                good_count += 1
                                web.on_seed(True)
                                LOG.info(
                                    "GOOD seed confidence=%.3f stable=%s",
                                    detection.confidence, detection.stable,
                                )
                            elif detection.classification == "BAD_SEED":
                                belt_speed = BELT_SPEED_CM_S_BY_LEVEL[current_speed_level]
                                travel_seconds = (CAMERA_TO_SERVO_MM / 10.0) / belt_speed
                                reject.enqueue(detection.left_frame_at, travel_seconds)
                                bad_count += 1
                                web.on_seed(False)
                                LOG.info(
                                    "BAD seed queued: travel=%.2fs queue=%d confidence=%.3f",
                                    travel_seconds, reject.depth(), detection.confidence,
                                )
                    except Exception:
                        LOG.exception("Classification/detection/reject processing failed.")
                        link.stop()
                        detector.reset()
                        in_fault = True
                        machine_state = "Fault"
                        running = False
            else:
                detector.reset()

            if reject is not None and reject.fault and machine_state != "Fault":
                LOG.error("Reject servo fault: %s", reject.fault)
                link.stop()
                detector.reset()
                in_fault = True
                machine_state = "Fault"
                running = False

            display = frame.copy()
            if running and prediction is not None:
                overlay = (
                    f"RUNNING | {prediction.label} {prediction.confidence:.2f} "
                    f"| SERVO {reject.state() if reject else 'UNAVAILABLE'}"
                )
            else:
                overlay = f"{machine_state.upper()} | LIVE CAMERA | MODEL {'READY' if classifier else 'IDLE'}"
            cv2.putText(
                display, overlay, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if running else (0, 200, 255), 2, cv2.LINE_AA,
            )

            now = time.monotonic()
            if now - last_stream >= stream_interval:
                ok, jpeg = cv2.imencode(
                    ".jpg", display,
                    [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY],
                )
                if ok:
                    web.push_frame(jpeg.tobytes())
                web.push_status({
                    "state": machine_state,
                    "camera": "Live" if frame_age <= MAX_FRAME_AGE else "Stale",
                    "model": "Running" if running else ("Loaded" if classifier else "Idle"),
                    "diverter": reject.state() if reject else "Unavailable",
                    "reject_queue": reject.depth() if reject else 0,
                    "good": good_count,
                    "bad": bad_count,
                    "label": prediction.label if prediction else "IDLE",
                    "confidence": round(prediction.confidence, 3) if prediction else 0.0,
                    "servo_mode": reject.mode if reject else "Unavailable",
                    "servo_angle": round(reject.current_angle_deg(), 1) if reject else 0.0,
                })
                last_stream = now

            if not HEADLESS:
                cv2.imshow("Sorter agent (q to quit)", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        return 0

    except KeyboardInterrupt:
        LOG.info("Stopped by user.")
        return 0
    except Exception:
        LOG.exception("Agent stopped due to an unrecoverable error.")
        return 1
    finally:
        if link is not None:
            try:
                link.stop()
                link.close()
            except Exception:
                LOG.exception("Error closing ESP link.")
        if web is not None:
            try:
                web.close()
            except Exception:
                LOG.exception("Error closing dashboard links.")
        if reject is not None:
            try:
                reject.close()
            except Exception:
                LOG.exception("Error closing reject servo.")
        if camera is not None:
            try:
                camera.close()
            except Exception:
                LOG.exception("Error closing camera.")
        if not HEADLESS:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
