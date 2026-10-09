"""
serial_bridge.py

Handles all communication between the Streamlit dashboard and the
Raspberry Pi controller (per Chapter 3 methodology):

  - Sends "START\n" / "STOP\n" / "CALIBRATE\n" / "SPD1\n" / "SPD2\n" /
    "SPD3\n" / "SERVO50\n" / "SERVO100\n" / "SERVO150\n" / "SERVO180\n"
    to the Pi over USB serial at 9600 baud when the operator uses the
    dashboard controls.
  - Listens for lines coming back from the Pi / vision script and
    updates live counters + machine status.

Incoming line protocol (agree on this with your Pi / vision-model
partners so their code matches):

    GOOD                -> increments the "good seed" counter by 1
    BAD                  -> increments the "bad seed" counter by 1
    COUNTS:<good>,<bad>  -> sets counters directly, e.g. COUNTS:120,8
    STATUS:IDLE          -> sets machine status to Idle
    STATUS:RUNNING       -> sets machine status to Running
    STATUS:CALIBRATION   -> sets machine status to Calibration
    STATUS:FAULT         -> sets machine status to Fault (the Pi/ESP
                            entered this on its own, e.g. a FIFO
                            overflow — not something the dashboard asked for)
    DIVERTER:READY       -> sets the reject servo's own state to Ready
    DIVERTER:DIVERTING   -> sets the reject servo's own state to Diverting

If your partners send a different format, only _process_line() below
needs to change - the rest of the dashboard doesn't care.

A SimulationBridge is also provided with the exact same interface so
you can build/demo the dashboard on a laptop with no Raspberry Pi or
cable attached yet.
"""

import threading
import time
import random

try:
    import serial  # pyserial
except ImportError:
    serial = None


class SerialBridge:
    """Real bridge - talks to an actual Raspberry Pi over USB serial."""

    def __init__(self, port: str, baud: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout

        self.ser = None
        self.connected = False
        self.error = None

        self.good_count = 0
        self.bad_count = 0
        self.machine_status = "Idle"
        self.diverter_state = "Ready"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def connect(self):
        if serial is None:
            self.error = "pyserial is not installed. Run: pip install pyserial"
            self.connected = False
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            self.connected = True
            self.error = None
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.connected = False
            self.error = f"Could not open {self.port}: {exc}"

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if line:
                    self._process_line(line)
            except Exception as exc:
                self.error = f"Serial read error: {exc}"
                time.sleep(0.5)

    def _process_line(self, line: str):
        with self._lock:
            if line == "GOOD" or line == "1":
                # "1" = good seed, per the vision model's labelling (0/1 scheme)
                self.good_count += 1
            elif line == "BAD" or line == "0":
                # "0" = bad/defective seed
                self.bad_count += 1
            elif line.startswith("LABEL:"):
                val = line.split(":", 1)[1].strip()
                if val == "1":
                    self.good_count += 1
                elif val == "0":
                    self.bad_count += 1
            elif line.startswith("COUNTS:"):
                try:
                    good_str, bad_str = line.split(":", 1)[1].split(",")
                    self.good_count = int(good_str)
                    self.bad_count = int(bad_str)
                except (ValueError, IndexError):
                    pass
            elif line.startswith("STATUS:"):
                status = line.split(":", 1)[1].strip().upper()
                status_map = {
                    "IDLE": "Idle", "RUNNING": "Running",
                    "CALIBRATION": "Calibration", "FAULT": "Fault",
                }
                if status in status_map:
                    self.machine_status = status_map[status]
                # an unrecognized value leaves machine_status unchanged rather
                # than guessing
            elif line.startswith("DIVERTER:"):
                diverter = line.split(":", 1)[1].strip().upper()
                diverter_map = {"READY": "Ready", "DIVERTING": "Diverting"}
                if diverter in diverter_map:
                    self.diverter_state = diverter_map[diverter]

    def send_command(self, cmd: str) -> bool:
        """Send START / STOP / CALIBRATE / SPD1-3 / SERVO50-180 to the Pi."""
        if not (self.connected and self.ser):
            return False
        try:
            self.ser.write(f"{cmd}\n".encode("utf-8"))
            # Optimistic local echo for instant UI feedback — the real
            # STATUS: line above will correct this shortly after, and is
            # what actually catches a Fault.
            if cmd == "START":
                self.machine_status = "Running"
            elif cmd == "STOP":
                self.machine_status = "Idle"
            elif cmd == "CALIBRATE":
                self.machine_status = "Calibration"
            return True
        except Exception as exc:
            self.error = f"Serial write error: {exc}"
            return False

    def disconnect(self):
        self._stop_event.set()
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.connected = False

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "good": self.good_count,
                "bad": self.bad_count,
                "status": self.machine_status,
                "diverter": self.diverter_state,
                "connected": self.connected,
                "error": self.error,
                "camera_frame": None,
                "camera_ts": None,
            }


class SimulationBridge:
    """
    Fake bridge with the identical interface as SerialBridge.
    Lets you build and demo the dashboard with no hardware attached -
    generates believable good/bad seed events on its own.
    """

    def __init__(self, *_, **__):
        self.connected = False
        self.error = None
        self.good_count = 0
        self.bad_count = 0
        self.machine_status = "Idle"
        self.diverter_state = "Ready"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def connect(self):
        self.connected = True
        self.error = None
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._simulate_loop, daemon=True)
        self._thread.start()

    def _simulate_loop(self):
        while not self._stop_event.is_set():
            time.sleep(random.uniform(0.4, 1.2))
            diverted = False
            with self._lock:
                if self.machine_status == "Running":
                    if random.random() < 0.88:
                        self.good_count += 1
                    else:
                        self.bad_count += 1
                        self.diverter_state = "Diverting"
                        diverted = True
            if diverted:
                time.sleep(0.3)  # mirrors the real servo's hold time
                with self._lock:
                    self.diverter_state = "Ready"

    def send_command(self, cmd: str) -> bool:
        if cmd == "START":
            self.machine_status = "Running"
        elif cmd == "STOP":
            self.machine_status = "Idle"
        elif cmd == "CALIBRATE":
            self.machine_status = "Calibration"
        # SPD1-3 and SERVO50-180 have no simulated effect — accepted, no-op
        return True

    def disconnect(self):
        self._stop_event.set()
        self.connected = False

    def get_snapshot(self) -> dict:
        with self._lock:
            return {
                "good": self.good_count,
                "bad": self.bad_count,
                "status": self.machine_status,
                "diverter": self.diverter_state,
                "connected": self.connected,
                "error": self.error,
                "camera_frame": None,
                "camera_ts": None,
            }
