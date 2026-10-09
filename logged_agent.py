#!/usr/bin/env python3
"""Pi-side agent: see, classify, and own the reject servo + its FIFO.

(same docstring as before — unchanged architecture)

EDIT THESE SETTINGS, THEN RUN: python sorter_pi_agent.py
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps

from reject_actuator import ServoReject

PROJECT_DIR = Path(__file__).resolve().parent

# ---- Logging ----
LOG_FILE = PROJECT_DIR / "log"    # every log line is ALSO appended to this file
LOG_LEVEL = logging.DEBUG         # DEBUG = everything; use INFO once things work

# ---- Everyday operation ----
DRY_RUN = True                    # True = classify + stream only; no commands sent to the ESP.
ESP_HOST = "seedsorter.local"
ESP_PORT = 8266
CAMERA = "picamera2"
WIDTH, HEIGHT = 640, 480
HEADLESS = True

# ---- Dashboard link ----
CONTROL_MODE = "cloud"
CLOUD_DB_URL = "https://maize-sorting-default-rtdb.firebaseio.com"
CLOUD_NODE = "seed_sorting"
DASHBOARD_SERIAL_PORT = "/dev/ttyUSB0"
DASHBOARD_SERIAL_BAUD = 9600
STREAM_FPS = 4.0
STREAM_JPEG_QUALITY = 60

# ---- Reject servo (Pi GPIO, via pigpio) ----
SERVO_GPIO_PIN = 18
SERVO_PASS_ANGLE = 20
SERVO_DIVERT_ANGLE = 110
SERVO_HOLD_SECONDS = 0.3
SERVO_MOVE_SECONDS = 0.35
REJECT_MAX_IN_FLIGHT = 16

CAMERA_TO_SERVO_MM = 500.0

BELT_SPEED_CM_S_BY_LEVEL = {1: 3.75, 2: 6.2, 3: 7.4}

# ---- Detection timing ----
SEED_OBSERVATION_SECONDS = 6.0
STABLE_DECISION_SECONDS = 3.0
EMPTY_FRAMES = 3
EMPTY_SECONDS = 0.4
MAX_FRAME_AGE = 0.75
MAX_FRAME_GAP = 1.0
MIN_MAIZE_FRAMES = 3

# ---- Model and camera settings ----
MODEL_PATH = PROJECT_DIR / "model_outputs/maize_mobilenetv3large_float32.tflite"
METADATA_PATH = PROJECT_DIR / "model_outputs/deployment_metadata_v2_recovered.json"
THREADS = 2

LOG = logging.getLogger("sorter_agent")
CLASSES = ["BAD_SEED", "GOOD_SEED", "NO_MAIZE"]


def setup_logging() -> None:
    """Log to BOTH the console and a file named 'log' in the project dir."""
    LOG.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(fmt)

    LOG.handlers.clear()
    LOG.addHandler(console)
    LOG.addHandler(file_handler)
    LOG.propagate = False
    LOG.info("Logging initialised. Console + file: %s (level=%s)",
             LOG_FILE, logging.getLevelName(LOG_LEVEL))


# =====================================================================
# Model contract
# =====================================================================

def prepare_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    arr = np.asarray(rgb.resize((width, height), Image.Resampling.BILINEAR),
                     dtype=np.float32)[None]
    LOG.debug("prepare_rgb: input PIL size=%s -> model input shape=%s dtype=%s range=[%.1f, %.1f]",
              image.size, arr.shape, arr.dtype, float(arr.min()), float(arr.max()))
    return arr


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float
    eligible: bool
    empty: bool


def interpret_probabilities(values, metadata: dict) -> Prediction:
    probs = np.asarray(values, dtype=np.float64).reshape(-1)
    LOG.debug("Raw model output: %s", np.array2string(probs, precision=4))
    if probs.shape != (3,) or not np.isfinite(probs).all():
        raise ValueError("Model must return three finite class probabilities.")
    if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1) > 0.01:
        raise ValueError("Model output is not a three-class softmax probability vector.")
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    margin = confidence - float(np.sort(probs)[-2])
    label = CLASSES[index]
    policy = metadata["decision_policy"]
    threshold = policy["class_thresholds"].get(label, {})
    conf_req = threshold.get("confidence", 1.0)
    margin_req = threshold.get("margin", 1.0)
    eligible = label != "NO_MAIZE" and bool(policy["actuation_enabled"]) and (
        confidence >= conf_req and margin >= margin_req
    )
    empty = label == "NO_MAIZE" and confidence >= policy["empty_confidence"] and margin >= policy["empty_margin"]

    # --- explain the decision ---
    if label == "NO_MAIZE":
        LOG.debug("Threshold check: NO_MAIZE conf=%.3f>=%.3f? margin=%.3f>=%.3f? -> empty=%s",
                  confidence, policy["empty_confidence"], margin, policy["empty_margin"], empty)
    else:
        LOG.debug("Threshold check: %s conf=%.3f (req %.3f, %s) margin=%.3f (req %.3f, %s) actuation_enabled=%s -> eligible=%s",
                  label, confidence, conf_req, "PASS" if confidence >= conf_req else "FAIL",
                  margin, margin_req, "PASS" if margin >= margin_req else "FAIL",
                  policy["actuation_enabled"], eligible)
    return Prediction(label, confidence, margin, eligible, empty)


def validate_metadata(metadata: dict) -> None:
    if metadata.get("schema_version") != 2:
        raise ValueError("Use schema-version-2 deployment_metadata.json from the notebook.")
    if metadata.get("class_names") != CLASSES or metadata.get("class_to_index") != dict(zip(CLASSES, range(3))):
        raise ValueError("Model class mapping does not match BAD_SEED, GOOD_SEED, NO_MAIZE.")
    if (metadata.get("input_color_order") != "RGB" or metadata.get("input_dtype") != "float32"
            or metadata.get("external_input_range") != [0.0, 255.0]
            or metadata.get("resize_method") != "pillow_bilinear"):
        raise ValueError("Unsupported model preprocessing contract.")


class Classifier:
    def __init__(self, model_path: Path, metadata_path: Path, threads: int = 2):
        self.metadata = json.loads(metadata_path.read_text())
        validate_metadata(self.metadata)
        LOG.info("Metadata OK: image_size=%s decision_policy=%s",
                 self.metadata.get("image_size"),
                 json.dumps(self.metadata.get("decision_policy"), sort_keys=True))
        model_bytes = model_path.read_bytes()
        LOG.info("Model loaded: %s (%.1f KB)", model_path, len(model_bytes) / 1024)
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
        self.size = tuple(self.metadata["image_size"])
        LOG.info("Interpreter ready: input shape=%s dtype=%s | output shape=%s",
                 self.input["shape"], self.input["dtype"], self.output["shape"])

    def predict(self, rgb: Image.Image) -> Prediction:
        LOG.debug("predict() called with PIL image size=%s", rgb.size)
        tensor = prepare_rgb(rgb, self.size)
        self.interpreter.set_tensor(self.input["index"], tensor)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        LOG.debug("invoke() took %.1f ms", elapsed_ms)
        pred = interpret_probabilities(self.interpreter.get_tensor(self.output["index"]), self.metadata)
        LOG.info("PREDICTION: label=%s confidence=%.3f margin=%.3f eligible=%s empty=%s (%.0f ms)",
                 pred.label, pred.confidence, pred.margin, pred.eligible, pred.empty, elapsed_ms)
        return pred


# =====================================================================
# Passage detector — now with a log line for EVERY state transition.
# =====================================================================

@dataclass(frozen=True)
class PassageDetection:
    first_seen: float
    left_frame_at: float
    classification: str
    confidence: float
    margin: float
    stable: bool


class PassageSeedDetector:
    def __init__(self, observation_seconds, stable_seconds, empty_frames,
                 empty_seconds, min_maize_frames, max_gap):
        self.observation_seconds = observation_seconds
        self.stable_seconds = stable_seconds
        self.empty_frames = empty_frames
        self.empty_seconds = empty_seconds
        self.min_maize_frames = min_maize_frames
        self.max_gap = max_gap
        self.last_frame = None
        self.armed = False
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
        LOG.info("Detector init: observation=%.1fs stable=%.1fs empty_frames=%d empty_seconds=%.1fs "
                 "min_maize_frames=%d max_gap=%.1fs",
                 observation_seconds, stable_seconds, empty_frames,
                 empty_seconds, min_maize_frames, max_gap)

    def _reset_seed(self):
        LOG.debug("detector: seed state reset (scores were BAD=%.2f GOOD=%.2f, maize_frames=%d)",
                  self.scores["BAD_SEED"], self.scores["GOOD_SEED"], self.maize_frames)
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
            LOG.error("detector: FINISH ABORTED — only %d maize frames (need %d). "
                      "Tracking is ambiguous; treating as error.",
                      self.maize_frames, self.min_maize_frames)
            raise RuntimeError("Seed passage had too few maize frames; tracking is ambiguous.")
        chosen = self.locked
        stable = chosen is not None
        if chosen is None:
            chosen = max(self.scores, key=self.scores.get)
            if self.scores[chosen] <= 0:
                chosen = "BAD_SEED"
            LOG.info("detector: FINISH (no lock) — fell back to highest summed confidence: %s "
                     "(BAD=%.2f GOOD=%.2f)", chosen, self.scores["BAD_SEED"], self.scores["GOOD_SEED"])
        else:
            LOG.info("detector: FINISH (locked) — %s held for %.1fs (BAD=%.2f GOOD=%.2f)",
                     chosen, self.stable_seconds, self.scores["BAD_SEED"], self.scores["GOOD_SEED"])
        evidence = self.best.get(chosen)
        confidence = evidence.confidence if evidence else 0.0
        margin = evidence.margin if evidence else 0.0
        result = PassageDetection(self.first_seen, left_at, chosen, confidence, margin, stable)
        LOG.info("detector: DECISION -> %s confidence=%.3f margin=%.3f stable=%s "
                 "(seed entered at t=%.2f, left frame at t=%.2f, duration=%.2fs)",
                 result.classification, confidence, margin, stable,
                 self.first_seen, left_at, left_at - self.first_seen)
        self._reset_seed()
        return result

    def observe(self, prediction: Prediction, now: float) -> PassageDetection | None:
        if self.last_frame is not None and now <= self.last_frame:
            LOG.debug("detector: out-of-order/stale frame ignored (now=%.3f <= last=%.3f)", now, self.last_frame)
            return None
        if self.last_frame is not None and now - self.last_frame > self.max_gap:
            LOG.warning("detector: frame gap %.2fs exceeded max_gap=%.2fs",
                        now - self.last_frame, self.max_gap)
            if self.in_seed:
                LOG.error("detector: gap occurred DURING a seed passage — raising error; clear the belt.")
                raise RuntimeError("Camera tracking gap during a seed passage; clear the belt.")
            LOG.info("detector: was not in a seed; disarming and clearing empty counter.")
            self.armed = False
            self.empty_count = 0
        self.last_frame = now

        if prediction.empty:
            if not self.empty_count:
                self.empty_since = now
                LOG.debug("detector: empty frame #1 at t=%.3f", now)
            self.empty_count += 1
            if self.empty_count < self.empty_frames or now - self.empty_since < self.empty_seconds:
                LOG.debug("detector: empty frame %d (need %d frames / %.1fs before passage can close)",
                          self.empty_count, self.empty_frames, self.empty_seconds)
                return None
            left_at = self.empty_since
            if self.in_seed:
                LOG.info("detector: seed left the frame at t=%.3f (empty for %.1fs) — finishing passage",
                         left_at, now - self.empty_since)
                result = self._finish(left_at)
                self.armed = True
                LOG.debug("detector: re-armed, waiting for next seed")
                return result
            if not self.armed:
                LOG.debug("detector: belt empty and armed — waiting for a seed")
            self.armed = True
            return None

        # A maize-looking frame — reset the empty streak.
        if self.empty_count:
            LOG.debug("detector: empty streak of %d frame(s) broken by %s", self.empty_count, prediction.label)
        self.empty_count = 0

        if not self.in_seed:
            if not self.armed:
                LOG.debug("detector: %s frame but NOT armed (belt was never empty) — ignoring", prediction.label)
                return None
            self.in_seed = True
            self.armed = False
            self.first_seen = now
            LOG.info("detector: SEED ENTERED frame at t=%.3f (label=%s conf=%.3f)",
                     now, prediction.label, prediction.confidence)

        if now - self.first_seen >= self.observation_seconds and not self.observation_closed:
            self.observation_closed = True
            LOG.info("detector: observation window closed after %.1fs (maize_frames=%d, scores BAD=%.2f GOOD=%.2f)",
                     self.observation_seconds, self.maize_frames,
                     self.scores["BAD_SEED"], self.scores["GOOD_SEED"])

        if self.observation_closed or prediction.label not in CLASSES[:2]:
            LOG.debug("detector: frame skipped for scoring (observation_closed=%s label=%s)",
                      self.observation_closed, prediction.label)
            return None

        self.maize_frames += 1
        self.scores[prediction.label] += prediction.confidence
        previous = self.best[prediction.label]
        if previous is None or prediction.confidence > previous.confidence:
            self.best[prediction.label] = prediction
        LOG.debug("detector: scored frame #%d as %s (conf=%.3f). Totals: BAD=%.2f GOOD=%.2f | best BAD=%s best GOOD=%s",
                  self.maize_frames, prediction.label, prediction.confidence,
                  self.scores["BAD_SEED"], self.scores["GOOD_SEED"],
                  "%.3f" % self.best["BAD_SEED"].confidence if self.best["BAD_SEED"] else "none",
                  "%.3f" % self.best["GOOD_SEED"].confidence if self.best["GOOD_SEED"] else "none")

        if not prediction.eligible:
            if self.candidate is not None:
                LOG.debug("detector: candidate %s CLEARED — frame not eligible (conf/margin below threshold)",
                          self.candidate)
            self.candidate = None
        elif prediction.label != self.candidate:
            if self.candidate is not None:
                LOG.info("detector: candidate switched %s -> %s at t=%.3f (stability timer restarted)",
                         self.candidate, prediction.label, now)
            else:
                LOG.debug("detector: new candidate %s at t=%.3f (needs %.1fs to lock)",
                          prediction.label, now, self.stable_seconds)
            self.candidate = prediction.label
            self.candidate_since = now
        elif self.locked is None:
            held = now - self.candidate_since
            if held >= self.stable_seconds:
                self.locked = self.candidate
                LOG.info("detector: LOCKED on %s after %.1fs of stable eligible frames", self.locked, held)
            else:
                LOG.debug("detector: candidate %s held for %.1fs/%.1fs — not locked yet",
                          self.candidate, held, self.stable_seconds)
        return None


# =====================================================================
# Camera reader
# =====================================================================

class LatestCamera:
    def __init__(self, source, width, height):
        self.capture = self.picamera = None
        LOG.info("Opening camera source=%r at %dx%d", source, width, height)
        if source == "picamera2":
            from picamera2 import Picamera2
            self.picamera = Picamera2()
            config = self.picamera.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                buffer_count=2, queue=False,
            )
            self.picamera.configure(config)
            self.picamera.start()
            LOG.info("picamera2 started.")
        else:
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                self.capture.release()
                raise RuntimeError(f"Could not open camera {source!r}.")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            LOG.info("cv2.VideoCapture opened: %r", source)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        failures = 0
        frame_count = 0
        try:
            while not self.stop.is_set():
                if self.picamera:
                    frame = self.picamera.capture_array("main")
                    ok = frame is not None
                else:
                    ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    LOG.warning("camera: failed to grab frame (%d consecutive failures)", failures)
                    if failures >= 5:
                        raise RuntimeError("Camera stopped delivering frames.")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                frame_count += 1
                with self.lock:
                    self.frame, self.timestamp = frame, time.monotonic()
                if frame_count % 100 == 0:
                    LOG.debug("camera: %d frames grabbed so far", frame_count)
        except Exception as exc:
            with self.lock:
                self.error = exc
            LOG.exception("camera: reader thread died")

    def latest(self):
        with self.lock:
            if self.error:
                raise self.error
            return (None if self.frame is None else self.frame.copy()), self.timestamp

    def close(self):
        LOG.info("camera: closing")
        self.stop.set()
        self.thread.join(timeout=1.0)
        if self.picamera:
            self.picamera.stop()
            self.picamera.close()
        elif not self.thread.is_alive():
            self.capture.release()


# =====================================================================
# Checksummed WiFi (TCP) link to the ESP
# =====================================================================

class ESPWiFiLink:
    def __init__(self, host: str, port: int, connect_timeout: float = 5.0,
                 reconnect_interval: float = 3.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.reconnect_interval = reconnect_interval
        self.sock: socket.socket | None = None
        self._recv_buffer = b""
        self._last_connect_attempt = 0.0
        self._try_connect()

    def _try_connect(self) -> None:
        now = time.monotonic()
        if now - self._last_connect_attempt < self.reconnect_interval:
            return
        self._last_connect_attempt = now
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            self.sock.settimeout(0.0)
            self._recv_buffer = b""
            LOG.info("Connected to ESP at %s:%d.", self.host, self.port)
            self._wait_ready()
        except OSError as exc:
            self.sock = None
            LOG.warning("Could not connect to ESP at %s:%d (%s).", self.host, self.port, exc)

    def _wait_ready(self, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for line in self.read_lines():
                if line == "READY":
                    LOG.info("ESP ready.")
                    return
            time.sleep(0.02)
        LOG.warning("No READY banner seen from the ESP; continuing anyway.")

    def _frame(self, payload: str) -> bytes:
        checksum = 0
        for b in payload.encode("ascii"):
            checksum ^= b
        framed = f"{payload}*{checksum:02X}\n".encode("ascii")
        LOG.debug("ESP TX: %r", framed.decode("ascii").strip())
        return framed

    def _close_socket(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def send(self, payload: str) -> None:
        if self.sock is None:
            self._try_connect()
            if self.sock is None:
                LOG.debug("ESP TX dropped (no connection): %s", payload)
                return
        try:
            self.sock.sendall(self._frame(payload))
        except OSError as exc:
            LOG.warning("Lost the ESP connection while sending (%s); will reconnect.", exc)
            self._close_socket()

    def read_lines(self) -> list[str]:
        if self.sock is None:
            self._try_connect()
            return []
        try:
            chunk = self.sock.recv(4096)
            if chunk == b"":
                LOG.warning("ESP closed the connection; will reconnect.")
                self._close_socket()
                return []
            self._recv_buffer += chunk
        except BlockingIOError:
            pass
        except OSError as exc:
            LOG.warning("Lost the ESP connection while reading (%s); will reconnect.", exc)
            self._close_socket()
            return []

        lines = []
        while b"\n" in self._recv_buffer:
            raw, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            text = raw.decode("ascii", "ignore").strip()
            if text:
                lines.append(text)
        return lines

    def run(self):
        LOG.info("COMMAND -> RUN")
        self.send("RUN")

    def stop(self):
        LOG.info("COMMAND -> STOP")
        self.send("STOP")

    def calibrate(self):
        LOG.info("COMMAND -> CALIBRATE")
        self.send("CALIBRATE")

    def set_speed(self, level: int):
        LOG.info("COMMAND -> SPD%d", level)
        self.send(f"SPD{level}")

    def close(self):
        try:
            self.stop()
        finally:
            self._close_socket()


class DryRunLink:
    """Stand-in for ESPWiFiLink so the pipeline can run with no hardware attached."""

    def run(self):
        LOG.info("[DRY RUN] RUN")

    def stop(self):
        LOG.info("[DRY RUN] STOP")

    def calibrate(self):
        LOG.info("[DRY RUN] CALIBRATE")

    def set_speed(self, level: int):
        LOG.info("[DRY RUN] SPD%d", level)

    def read_lines(self):
        return []

    def close(self):
        pass


# =====================================================================
# Dashboard link
# =====================================================================

class NullLink:
    def on_seed(self, is_good: bool) -> None:
        LOG.debug("NullLink.on_seed(%s) — no dashboard configured", is_good)

    def push_frame(self, jpeg_bytes: bytes) -> None:
        pass

    def push_status(self, status: dict) -> None:
        pass

    def poll_command(self):
        return None


def build_dashboard_link():
    LOG.info("Building dashboard link for CONTROL_MODE=%r", CONTROL_MODE)
    if CONTROL_MODE == "cloud":
        from cloud_link import CloudLink
        LOG.info("Using CloudLink: url=%s node=%s", CLOUD_DB_URL, CLOUD_NODE)
        return CloudLink(CLOUD_DB_URL, CLOUD_NODE)
    if CONTROL_MODE == "dashboard_serial":
        from dashboard_serial_link import DashboardSerialLink
        LOG.info("Using DashboardSerialLink: port=%s baud=%d", DASHBOARD_SERIAL_PORT, DASHBOARD_SERIAL_BAUD)
        return DashboardSerialLink(DASHBOARD_SERIAL_PORT, DASHBOARD_SERIAL_BAUD)
    if CONTROL_MODE == "none":
        LOG.info("No dashboard link (NullLink).")
        return NullLink()
    raise ValueError(f"Unknown CONTROL_MODE: {CONTROL_MODE!r}")


# =====================================================================
# Main loop
# =====================================================================

def main() -> int:
    setup_logging()
    link = camera = reject = None
    try:
        LOG.info("=== Agent starting up ===")
        LOG.info("Config: DRY_RUN=%s CAMERA=%r CONTROL_MODE=%r STREAM_FPS=%.1f "
                 "MODEL=%s", DRY_RUN, CAMERA, CONTROL_MODE, STREAM_FPS, MODEL_PATH)
        classifier = Classifier(MODEL_PATH, METADATA_PATH, THREADS)
        detector = PassageSeedDetector(SEED_OBSERVATION_SECONDS, STABLE_DECISION_SECONDS,
                                        EMPTY_FRAMES, EMPTY_SECONDS, MIN_MAIZE_FRAMES, MAX_FRAME_GAP)
        link = DryRunLink() if DRY_RUN else ESPWiFiLink(ESP_HOST, ESP_PORT)
        LOG.info("ESP link: %s", "DRY RUN (no hardware)" if DRY_RUN else f"WiFi {ESP_HOST}:{ESP_PORT}")
        reject = ServoReject(SERVO_GPIO_PIN, SERVO_PASS_ANGLE, SERVO_DIVERT_ANGLE,
                              SERVO_HOLD_SECONDS, SERVO_MOVE_SECONDS, REJECT_MAX_IN_FLIGHT)
        LOG.info("Reject servo ready: GPIO=%d pass=%d deg divert=%d deg hold=%.2fs move=%.2fs max_in_flight=%d",
                 SERVO_GPIO_PIN, SERVO_PASS_ANGLE, SERVO_DIVERT_ANGLE,
                 SERVO_HOLD_SECONDS, SERVO_MOVE_SECONDS, REJECT_MAX_IN_FLIGHT)
        web = build_dashboard_link()
        camera = LatestCamera(CAMERA, WIDTH, HEIGHT)

        machine_state = "Idle"
        in_fault = False
        current_speed_level = 1
        good_count = bad_count = 0
        last_stream = 0.0
        stream_interval = 1.0 / STREAM_FPS
        prediction = None
        loop_count = 0

        LOG.info("Agent ready. Waiting for a command from the website.")

        while True:
            loop_count += 1
            command = web.poll_command()
            if command is not None:
                LOG.info("DASHBOARD RX: command=%r (state=%s in_fault=%s)", command, machine_state, in_fault)
            if command == "START" and machine_state != "Running":
                if in_fault:
                    LOG.warning("START ignored — still in FAULT; press STOP to clear it first.")
                else:
                    link.run()
                    machine_state = "Running"
                    LOG.info("State transition -> Running. Detection is now ACTIVE.")
            elif command == "STOP":
                link.stop()
                reject.clear()
                in_fault = False
                machine_state = "Idle"
                LOG.info("State transition -> Idle. Detection paused.")
            elif command == "CALIBRATE" and not in_fault:
                link.calibrate()
                machine_state = "Calibration"
                LOG.info("State transition -> Calibration.")
            elif command in ("SPD1", "SPD2", "SPD3"):
                current_speed_level = int(command[-1])
                link.set_speed(current_speed_level)
                LOG.info("Speed level -> %d (belt speed %.2f cm/s)",
                         current_speed_level, BELT_SPEED_CM_S_BY_LEVEL[current_speed_level])
            elif command in ("SERVO50", "SERVO100", "SERVO150", "SERVO180"):
                angle = int(command[len("SERVO"):])
                if machine_state == "Idle":
                    reject.move_to(angle)
                    LOG.info("Servo manually moved to %d degrees.", angle)
                else:
                    LOG.warning("%s ignored — only allowed while Idle.", command)
            elif command is not None:
                LOG.warning("Unknown dashboard command ignored: %r", command)

            for line in link.read_lines():
                LOG.info("ESP RX: %s", line)
                if line.startswith("STATE ") and not in_fault:
                    new_state = line[len("STATE "):].strip().capitalize()
                    if new_state != machine_state:
                        LOG.info("State transition (ESP echo) %s -> %s", machine_state, new_state)
                    machine_state = new_state

            running = (machine_state == "Running")

            frame, captured = camera.latest()
            if frame is None:
                if loop_count % 200 == 0:
                    LOG.debug("main: no camera frame yet (loop #%d)", loop_count)
                time.sleep(0.005)
                continue
            frame_age = time.monotonic() - captured
            LOG.debug("main: frame captured t=%.3f age=%.3fs running=%s", captured, frame_age, running)
            if frame_age > MAX_FRAME_AGE:
                LOG.error("main: frame too stale (age=%.3fs > %.2fs) — stopping.", frame_age, MAX_FRAME_AGE)
                raise TimeoutError("Camera frame is stale; stopping.")

            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = classifier.predict(rgb)

            detection = detector.observe(prediction, captured) if running else None
            if not running and prediction.label != "NO_MAIZE":
                LOG.debug("main: seed-like frame seen but machine_state=%s — detection not active "
                          "(press START on the dashboard)", machine_state)
            if detection:
                if detection.classification == "GOOD_SEED":
                    good_count += 1
                    LOG.info("DECISION: GOOD_SEED (stable=%s) — total good=%d", detection.stable, good_count)
                    web.on_seed(True)
                    LOG.info("Dashboard notified: on_seed(good=True)")
                else:
                    belt_speed = BELT_SPEED_CM_S_BY_LEVEL[current_speed_level]
                    travel_seconds = (CAMERA_TO_SERVO_MM / 10.0) / belt_speed
                    LOG.info("DECISION: BAD_SEED (stable=%s) — total bad=%d. Enqueueing reject: "
                             "speed_level=%d belt=%.2f cm/s distance=%.0f mm travel=%.2fs",
                             detection.stable, bad_count + 1,
                             current_speed_level, belt_speed, CAMERA_TO_SERVO_MM, travel_seconds)
                    try:
                        reject.enqueue(detection.left_frame_at, travel_seconds)
                        bad_count += 1
                        LOG.info("Reject enqueued OK. FIFO depth=%d", reject.depth())
                        web.on_seed(False)
                        LOG.info("Dashboard notified: on_seed(good=False)")
                    except RuntimeError as exc:
                        LOG.error("Reject FIFO overflow (%s) — entering FAULT.", exc)
                        in_fault = True
                        machine_state = "Fault"
                        link.stop()
                        reject.clear()
                        LOG.error("State transition -> Fault. Belt stopped; rejects cleared. "
                                  "Press STOP to clear the fault.")

            now = time.monotonic()
            if now - last_stream >= stream_interval:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    LOG.debug("Streaming frame to dashboard: %d bytes JPEG", len(jpeg))
                    web.push_frame(jpeg.tobytes())
                else:
                    LOG.warning("JPEG encode FAILED — frame not streamed this tick")
                status_payload = {
                    "state": machine_state,
                    "diverter": reject.state(),
                    "good": good_count,
                    "bad": bad_count,
                    "label": prediction.label if prediction else "none",
                    "confidence": round(prediction.confidence, 3) if prediction else 0.0,
                }
                LOG.debug("Pushing status to dashboard: %s", json.dumps(status_payload))
                web.push_status(status_payload)
                last_stream = now

            if not HEADLESS:
                cv2.imshow("Sorter agent (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOG.info("Local preview: quit key pressed.")
                    break
        return 0
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")
        return 0
    except Exception:
        LOG.exception("Agent stopped due to an error.")
        return 1
    finally:
        LOG.info("=== Agent shutting down ===")
        if link:
            link.close()
        if reject:
            reject.close()
        if camera:
            camera.close()
            if not HEADLESS:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())