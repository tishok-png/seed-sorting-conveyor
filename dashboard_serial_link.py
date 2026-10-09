"""dashboard_serial_link.py

Optional link matching the dashboard's "Live (Raspberry Pi over USB)"
mode — serial_bridge.py's SerialBridge. This is a SEPARATE physical
connection from the checksummed, 115200-baud link to the ESP8266
(sorter_esp8266.ino): a second, plain-text, 9600-baud serial link
straight to whichever computer is running the Streamlit dashboard, for
a wired, no-WiFi demo.

Only use this if the Pi actually has a spare serial connection to the
dashboard machine (a second USB-serial adapter, or the Pi's GPIO UART
wired out) — most setups will want cloud_link.py instead.

Protocol (matches serial_bridge.py's _process_line exactly):
    out (Pi -> dashboard): "GOOD\n" / "BAD\n" per seed, "STATUS:IDLE\n" /
                            "STATUS:RUNNING\n" / "STATUS:CALIBRATION\n" /
                            "STATUS:FAULT\n" on state changes — reflects the
                            Pi agent's own synthesized state (ESP motor echoes
                            plus the Pi's own reject-FIFO health), not just
                            what was requested — plus "DIVERTER:READY\n" /
                            "DIVERTER:DIVERTING\n" for the reject servo's own
                            state, independent of the line above.
    in  (dashboard -> Pi):  "START\n" / "STOP\n" / "CALIBRATE\n" / "SPD1\n" /
                            "SPD2\n" / "SPD3\n" / "SERVO50\n" / "SERVO100\n" /
                            "SERVO150\n" / "SERVO180\n"

Note: SerialBridge.get_snapshot() always reports camera_frame as None,
so this mode never shows a camera feed in the dashboard by design —
push_frame() below is a no-op on purpose, not an oversight.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import serial

LOG = logging.getLogger("sorter_agent.dashboard_serial")


class DashboardSerialLink:
    def __init__(self, port: str, baud: int = 9600):
        self.serial = serial.Serial(port, baudrate=baud, timeout=0)
        self.incoming_commands: queue.Queue = queue.Queue()
        self._buffer = ""
        threading.Thread(target=self._reader_loop, daemon=True).start()
        LOG.info("Dashboard serial link open on %s @ %d baud.", port, baud)

    # --- interface used by sorter_pi_agent.py ---

    def on_seed(self, is_good: bool) -> None:
        self._write_line("GOOD" if is_good else "BAD")

    def push_frame(self, jpeg_bytes: bytes) -> None:
        pass  # this mode has no camera feed — see module docstring

    def push_status(self, status: dict) -> None:
        self._write_line(f"STATUS:{status['state'].upper()}")
        self._write_line(f"DIVERTER:{status['diverter'].upper()}")

    def poll_command(self):
        try:
            return self.incoming_commands.get_nowait()
        except queue.Empty:
            return None

    # --- internals ---

    def _write_line(self, line: str) -> None:
        try:
            self.serial.write(f"{line}\n".encode("ascii"))
        except Exception as exc:
            LOG.warning("Dashboard serial write failed: %s", exc)

    def _reader_loop(self) -> None:
        while True:
            try:
                waiting = self.serial.in_waiting
                if not waiting:
                    time.sleep(0.02)
                    continue
                self._buffer += self.serial.read(waiting).decode("ascii", "ignore")
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    recognized = (
                        "START", "STOP", "CALIBRATE", "SPD1", "SPD2", "SPD3",
                        "SERVO50", "SERVO100", "SERVO150", "SERVO180",
                    )
                    if line in recognized:
                        self.incoming_commands.put(line)
            except Exception as exc:
                LOG.warning("Dashboard serial read failed: %s", exc)
                time.sleep(0.5)
