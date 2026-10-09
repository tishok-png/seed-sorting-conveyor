#!/usr/bin/env python3
"""Pi-side agent: see, classify, and own the reject servo + its FIFO.

The servo moved off the ESP8266 onto a Pi GPIO pin (see
reject_actuator.py) because the ESP8266's single core was running
stepper pulse generation and servo pulse generation at the same time —
a known source of servo jitter on that chip. Moving the servo here
means the reject FIFO had to move here too: whatever decides WHEN to
fire the servo has to be the thing that can actually fire it. The ESP
now only drives the belt, the singulator, and the LED.

This script's jobs are:

  1. Classify each seed as it passes the camera, using the same TFLite
     model contract as before.
  2. On a BAD seed, enqueue a reject with reject_actuator.ServoReject —
     no message is sent to the ESP for this at all anymore. On a GOOD
     seed, nothing happens; the gate's default position already lets
     it through.
  3. Send RUN / STOP / CALIBRATE / SPD1-3 to the ESP over WiFi (not USB
     serial anymore — see ESPWiFiLink below), for the belt, singulator,
     and LED only.
  4. Stream a low-rate JPEG preview and status to the Streamlit
     dashboard (app.py), and turn its button presses into the commands
     above, plus a few manual servo-angle test commands. Two dashboard
     link modes are supported, matching serial_bridge.py and
     cloud_bridge.py exactly: see CONTROL_MODE below.

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

# ---- Everyday operation ----
DRY_RUN = True                    # True = classify + stream only; no commands sent to the ESP.
ESP_HOST = "seedsorter.local"     # mDNS hostname the ESP advertises — see its own
                                   # serial-monitor output at boot if this doesn't resolve;
                                   # fall back to its printed IP address there instead.
ESP_PORT = 8266
CAMERA = "picamera2"              # CSI ribbon camera; use 0 or "/dev/video0" for USB.
WIDTH, HEIGHT = 640, 480
HEADLESS = True                   # False shows a local OpenCV preview window too.

# ---- Dashboard link — pick ONE, matching the mode you'll select in app.py's sidebar ----
# "cloud"             -> matches cloud_bridge.py (Firebase REST, WiFi/anywhere)
# "dashboard_serial"  -> matches serial_bridge.py's Live/USB mode (wired, no WiFi)
# "none"              -> no dashboard link; runs standalone (e.g. for bench testing)
CONTROL_MODE = "cloud"
CLOUD_DB_URL = "https://maize-sorting-default-rtdb.firebaseio.com"  # from Firebase console
CLOUD_NODE = "seed_sorting"                    # must match app.py's sidebar / cloud_bridge.py
DASHBOARD_SERIAL_PORT = "/dev/ttyUSB0"         # USB serial to the dashboard machine — unrelated
                                                # to the ESP link above, which is now WiFi
DASHBOARD_SERIAL_BAUD = 9600
STREAM_FPS = 4.0
STREAM_JPEG_QUALITY = 60

# ---- Reject servo (Pi GPIO, via pigpio) ----
SERVO_GPIO_PIN = 18               # BCM numbering = physical header pin 12
SERVO_PASS_ANGLE = 20             # gate closed / lets a good seed continue straight
SERVO_DIVERT_ANGLE = 110          # gate open / ejects a bad seed
SERVO_HOLD_SECONDS = 0.3          # how long the gate stays open per bad seed
SERVO_MOVE_SECONDS = 0.35         # measured worst-case 0<->divert swing time
REJECT_MAX_IN_FLIGHT = 16         # FIFO overflow beyond this -> FAULT

# Distance from the camera's effective capture point to the servo gate, in mm.
# *** MEASURE THIS *** — travel time is computed from this divided by whichever
# belt speed is active, so an inaccurate distance biases every reject's timing.
CAMERA_TO_SERVO_MM = 500.0

# These are the three levels the SPD1/SPD2/SPD3 dashboard buttons select on
# the ESP, but the VALUES here are your own physically measured belt speed
# per level, not the ESP's internal target — that's the whole point of
# measuring rather than trusting the pulley/gearbox math. Level 1 updated
# from your measurement (~3.75 cm/s); re-measure 2 and 3 the same way when
# you get a chance, since they're still the original computed estimates.
BELT_SPEED_CM_S_BY_LEVEL = {1: 3.75, 2: 6.2, 3: 7.4}

# ---- Detection timing (seconds unless the name says otherwise) ----
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


# =====================================================================
# Model contract — unchanged from the original single-Pi script.
# =====================================================================

def prepare_rgb(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(rgb.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)[None]


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
    if np.any(probs < 0) or np.any(probs > 1) or abs(float(probs.sum()) - 1) > 0.01:
        raise ValueError("Model output is not a three-class softmax probability vector.")
    index = int(np.argmax(probs))
    confidence = float(probs[index])
    margin = confidence - float(np.sort(probs)[-2])
    label = CLASSES[index]
    policy = metadata["decision_policy"]
    threshold = policy["class_thresholds"].get(label, {})
    eligible = label != "NO_MAIZE" and bool(policy["actuation_enabled"]) and (
        confidence >= threshold.get("confidence", 1.0)
        and margin >= threshold.get("margin", 1.0)
    )
    empty = label == "NO_MAIZE" and confidence >= policy["empty_confidence"] and margin >= policy["empty_margin"]
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
        self.size = tuple(self.metadata["image_size"])

    def predict(self, rgb: Image.Image) -> Prediction:
        self.interpreter.set_tensor(self.input["index"], prepare_rgb(rgb, self.size))
        self.interpreter.invoke()
        return interpret_probabilities(self.interpreter.get_tensor(self.output["index"]), self.metadata)


# =====================================================================
# Passage detector — same behaviour as before: one verdict per seed,
# issued once the seed has left the frame.
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
            raise RuntimeError("Seed passage had too few maize frames; tracking is ambiguous.")
        chosen = self.locked
        stable = chosen is not None
        if chosen is None:
            chosen = max(self.scores, key=self.scores.get)
            if self.scores[chosen] <= 0:
                chosen = "BAD_SEED"
        evidence = self.best.get(chosen)
        confidence = evidence.confidence if evidence else 0.0
        margin = evidence.margin if evidence else 0.0
        result = PassageDetection(self.first_seen, left_at, chosen, confidence, margin, stable)
        self._reset_seed()
        return result

    def observe(self, prediction: Prediction, now: float) -> PassageDetection | None:
        if self.last_frame is not None and now <= self.last_frame:
            return None
        if self.last_frame is not None and now - self.last_frame > self.max_gap:
            if self.in_seed:
                raise RuntimeError("Camera tracking gap during a seed passage; clear the belt.")
            self.armed = False
            self.empty_count = 0
        self.last_frame = now
        if prediction.empty:
            if not self.empty_count:
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
        if self.observation_closed or prediction.label not in CLASSES[:2]:
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


# =====================================================================
# Camera reader
# =====================================================================

class LatestCamera:
    def __init__(self, source, width, height):
        self.capture = self.picamera = None
        if source == "picamera2":
            from picamera2 import Picamera2
            self.picamera = Picamera2()
            config = self.picamera.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                buffer_count=2, queue=False,
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
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        failures = 0
        try:
            while not self.stop.is_set():
                if self.picamera:
                    frame = self.picamera.capture_array("main")
                    ok = frame is not None
                else:
                    ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError("Camera stopped delivering frames.")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                with self.lock:
                    self.frame, self.timestamp = frame, time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = exc

    def latest(self):
        with self.lock:
            if self.error:
                raise self.error
            return (None if self.frame is None else self.frame.copy()), self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)
        if self.picamera:
            self.picamera.stop()
            self.picamera.close()
        elif not self.thread.is_alive():
            self.capture.release()


# =====================================================================
# Checksummed WiFi (TCP) link to the ESP — belt, singulator, and LED
# only now. No GOOD/BAD, no age_ms; the ESP doesn't know or care which
# seeds are good or bad anymore. The reject FIFO is reject_actuator.py,
# below. This replaced a USB-serial link; WiFi is meaningfully more
# likely to blip than a cable was, so this reconnects on its own rather
# than treating a dropped connection as fatal.
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
            return  # don't hammer a reconnect attempt every main-loop iteration
        self._last_connect_attempt = now
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            self.sock.settimeout(0.0)  # non-blocking from here on
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
        return f"{payload}*{checksum:02X}\n".encode("ascii")

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
        self.send("RUN")

    def stop(self):
        self.send("STOP")

    def calibrate(self):
        self.send("CALIBRATE")

    def set_speed(self, level: int):
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
# Dashboard link: NullLink (no dashboard), or see cloud_link.py /
# dashboard_serial_link.py, both of which implement the same
# on_seed / push_frame / push_status / poll_command interface so the
# main loop below never needs to know which one is in use.
# =====================================================================

class NullLink:
    def on_seed(self, is_good: bool) -> None:
        pass

    def push_frame(self, jpeg_bytes: bytes) -> None:
        pass

    def push_status(self, status: dict) -> None:
        pass

    def poll_command(self):
        return None


def build_dashboard_link():
    if CONTROL_MODE == "cloud":
        from cloud_link import CloudLink
        return CloudLink(CLOUD_DB_URL, CLOUD_NODE)
    if CONTROL_MODE == "dashboard_serial":
        from dashboard_serial_link import DashboardSerialLink
        return DashboardSerialLink(DASHBOARD_SERIAL_PORT, DASHBOARD_SERIAL_BAUD)
    if CONTROL_MODE == "none":
        return NullLink()
    raise ValueError(f"Unknown CONTROL_MODE: {CONTROL_MODE!r}")


# =====================================================================
# Main loop
# =====================================================================

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    link = camera = reject = None
    try:
        classifier = Classifier(MODEL_PATH, METADATA_PATH, THREADS)
        detector = PassageSeedDetector(SEED_OBSERVATION_SECONDS, STABLE_DECISION_SECONDS,
                                        EMPTY_FRAMES, EMPTY_SECONDS, MIN_MAIZE_FRAMES, MAX_FRAME_GAP)
        link = DryRunLink() if DRY_RUN else ESPWiFiLink(ESP_HOST, ESP_PORT)
        reject = ServoReject(SERVO_GPIO_PIN, SERVO_PASS_ANGLE, SERVO_DIVERT_ANGLE,
                              SERVO_HOLD_SECONDS, SERVO_MOVE_SECONDS, REJECT_MAX_IN_FLIGHT)
        web = build_dashboard_link()
        camera = LatestCamera(CAMERA, WIDTH, HEIGHT)

        # machine_state is now synthesized on the Pi, not just mirrored from the
        # ESP: the ESP's echoed "STATE ..." lines (Idle/Running/Calibration)
        # reflect the motors, but Fault is a Pi-level concept now — it's set
        # only by our own reject FIFO overflowing (in_fault below), and an ESP
        # echo is never allowed to downgrade us out of it. Only an explicit
        # operator STOP clears a fault.
        machine_state = "Idle"
        in_fault = False
        current_speed_level = 1
        good_count = bad_count = 0
        last_stream = 0.0
        stream_interval = 1.0 / STREAM_FPS
        prediction = None

        LOG.info("Agent ready. Waiting for a command from the website.")

        while True:
            command = web.poll_command()
            if command == "START" and machine_state != "Running":
                if in_fault:
                    LOG.warning("START ignored — still in FAULT; press STOP to clear it first.")
                else:
                    link.run()
                    machine_state = "Running"
                    LOG.info("START received.")
            elif command == "STOP":
                link.stop()
                reject.clear()
                in_fault = False
                machine_state = "Idle"
                LOG.info("STOP received.")
            elif command == "CALIBRATE" and not in_fault:
                link.calibrate()
                machine_state = "Calibration"
                LOG.info("CALIBRATE received.")
            elif command in ("SPD1", "SPD2", "SPD3"):
                current_speed_level = int(command[-1])
                link.set_speed(current_speed_level)
                LOG.info("%s received.", command)
            elif command in ("SERVO50", "SERVO100", "SERVO150", "SERVO180"):
                angle = int(command[len("SERVO"):])
                if machine_state == "Idle":
                    reject.move_to(angle)
                    LOG.info("Servo manually moved to %d degrees.", angle)
                else:
                    LOG.warning("%s ignored — only allowed while Idle.", command)

            for line in link.read_lines():
                LOG.info("ESP: %s", line)
                if line.startswith("STATE ") and not in_fault:
                    machine_state = line[len("STATE "):].strip().capitalize()

            running = (machine_state == "Running")

            frame, captured = camera.latest()
            if frame is None:
                time.sleep(0.005)
                continue
            if time.monotonic() - captured > MAX_FRAME_AGE:
                raise TimeoutError("Camera frame is stale; stopping.")

            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = classifier.predict(rgb)
            LOG.info(
                "LIVE PREDICTION: label=%s confidence=%.3f "
                "margin=%.3f eligible=%s empty=%s",
                prediction.label,
                prediction.confidence,
                prediction.margin,
                prediction.eligible,
                prediction.empty,
                )

            detection = detector.observe(prediction, captured) if running else None
            if detection:
                if detection.classification == "GOOD_SEED":
                    good_count += 1
                    web.on_seed(True)
                    LOG.info("Seed: GOOD (stable=%s)", detection.stable)
                else:
                    belt_speed = BELT_SPEED_CM_S_BY_LEVEL[current_speed_level]
                    travel_seconds = (CAMERA_TO_SERVO_MM / 10.0) / belt_speed
                    try:
                        reject.enqueue(detection.left_frame_at, travel_seconds)
                        bad_count += 1
                        web.on_seed(False)
                        LOG.info("Seed: BAD, travel=%.2fs (stable=%s, queue=%d)",
                                 travel_seconds, detection.stable, reject.depth())
                    except RuntimeError:
                        LOG.error("Reject FIFO overflow — entering FAULT.")
                        in_fault = True
                        machine_state = "Fault"
                        link.stop()
                        reject.clear()

            now = time.monotonic()
            if now - last_stream >= stream_interval:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    web.push_frame(jpeg.tobytes())
                web.push_status({
                    "state": machine_state,
                    "diverter": reject.state(),  # "Ready" or "Diverting"
                    "good": good_count,
                    "bad": bad_count,
                    "label": prediction.label,
                    "confidence": round(prediction.confidence, 3),
                })
                last_stream = now

            if not HEADLESS:
                cv2.imshow("Sorter agent (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        return 0
    except KeyboardInterrupt:
        LOG.info("Stopped by user.")
        return 0
    except Exception:
        LOG.exception("Agent stopped due to an error.")
        return 1
    finally:
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
