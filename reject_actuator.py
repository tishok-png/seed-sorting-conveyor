"""reject_actuator.py — Pi-owned reject servo and FIFO.

The diverter servo now lives here, driven directly from a Pi GPIO pin
via pigpio (hardware-timed pulses), because the ESP8266's single core
was running stepper pulse generation and servo pulse generation at the
same time — a known source of servo jitter on that chip. The ESP now
only drives the belt, the singulator, and the LED (see
sorter_esp8266.ino); it has no servo and no FIFO anymore.

Moving the servo onto the Pi means the reject FIFO has to move here
too — whatever decides WHEN to fire the servo has to be the same
thing that can actually fire it with real timing. This is a direct
port of the FIFO design the ESP used to own (age-compensated arrival
scheduling, a close-spaced-reject warning, an overflow guard), adapted
to run as a Python thread instead of a microcontroller's main loop.

One genuine improvement from this move: because classification and
actuation are now the same process, there is no serial round-trip to
compensate for anymore. enqueue() takes the exact left_frame_at
timestamp straight from the detector, so arrival time is computed as
left_frame_at + travel_seconds with no unmeasured gap at all — more
accurate than the old age_ms-over-serial scheme, not just simpler.

The residual source of timing error here is Python/OS scheduling of
the service thread below (5 ms poll interval) — smaller than what the
ESP8266's servo+stepper contention was causing, but not zero. If this
ever needs tighter guarantees, the fix is the same advice as before:
measure, don't assume.

Requires the pigpio daemon running:
    sudo apt install pigpio       # usually already present on Raspberry Pi OS
    sudo systemctl enable --now pigpiod
    pip install pigpio
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import pigpio

LOG = logging.getLogger("sorter_agent.reject")


@dataclass
class RejectEvent:
    number: int
    due_at: float    # time.monotonic() value when the gate should open
    clear_at: float  # when the gate should be closed again


class ServoReject:
    """Owns the GPIO servo and the FIFO of pending reject events."""

    def __init__(self, gpio_pin: int, pass_angle: float, divert_angle: float,
                 hold_seconds: float, move_seconds: float, max_in_flight: int = 16,
                 service_interval: float = 0.005):
        self.gpio_pin = gpio_pin
        self.pass_us = self._angle_to_us(pass_angle)
        self.divert_us = self._angle_to_us(divert_angle)
        self.hold_seconds = hold_seconds
        self.move_seconds = move_seconds
        self.max_in_flight = max_in_flight
        self.service_interval = service_interval

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(
                "Could not connect to pigpiod — is it running? "
                "Try: sudo systemctl enable --now pigpiod"
            )
        self.pi.set_servo_pulsewidth(self.gpio_pin, self.pass_us)

        self.queue: deque[RejectEvent] = deque()
        self.lock = threading.Lock()
        self.next_number = 1
        self.diverting = False
        self.divert_until = 0.0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._service_loop, daemon=True)
        self._thread.start()
        LOG.info("Servo ready on GPIO%d (pigpio).", gpio_pin)

    @staticmethod
    def _angle_to_us(angle_deg: float) -> int:
        # Standard hobby servo range: 0 deg ~ 500us, 180 deg ~ 2500us.
        return int(500 + (angle_deg / 180.0) * 2000)

    def enqueue(self, left_frame_at: float, travel_seconds: float) -> None:
        """Schedule a reject. Raises RuntimeError if the FIFO is full — the
        caller is expected to treat that as a FAULT condition, same as the
        ESP used to."""
        with self.lock:
            if len(self.queue) >= self.max_in_flight:
                raise RuntimeError("Reject FIFO overflow — too many bad seeds queued.")
            due_at = left_frame_at + travel_seconds
            event = RejectEvent(self.next_number, due_at, due_at + self.hold_seconds)
            if self.queue:
                min_gap = self.hold_seconds + self.move_seconds
                if due_at < self.queue[-1].clear_at + min_gap:
                    LOG.warning("Reject %d is close-spaced behind the previous one; "
                                "it may fire late.", self.next_number)
            self.queue.append(event)
            self.next_number += 1

    def depth(self) -> int:
        with self.lock:
            return len(self.queue)

    def state(self) -> str:
        """"Ready" when the gate is at rest waiting for the next reject (or
        nothing is queued), "Diverting" while it's actively open."""
        with self.lock:
            return "Diverting" if self.diverting else "Ready"

    def move_to(self, angle_deg: float) -> bool:
        """Manual bring-up control — moves the gate to an arbitrary angle,
        bypassing the FIFO entirely. Only safe to call with nothing queued
        and no reject in progress (the Pi agent only allows this while the
        overall machine is Idle); returns False and does nothing otherwise."""
        with self.lock:
            if self.queue or self.diverting:
                LOG.warning("move_to(%.0f) refused — a reject is queued or in progress.",
                            angle_deg)
                return False
        self.pi.set_servo_pulsewidth(self.gpio_pin, self._angle_to_us(angle_deg))
        return True

    def clear(self) -> None:
        with self.lock:
            self.queue.clear()
            self.diverting = False
        self.pi.set_servo_pulsewidth(self.gpio_pin, self.pass_us)

    def _service_loop(self) -> None:
        # This thread must never die silently — if it does, every future
        # reject looks like it "worked" (enqueue() succeeds, FIFO depth
        # climbs) while the servo never actually moves again. Every
        # hardware call is guarded and logged for exactly that reason.
        while not self._stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                if self.diverting and now >= self.divert_until:
                    self.diverting = False
                    close_gate = True
                else:
                    close_gate = False
                open_gate = False
                if not self.diverting and self.queue and now >= self.queue[0].due_at:
                    popped = self.queue.popleft()
                    self.diverting = True
                    self.divert_until = now + self.hold_seconds
                    open_gate = True
                else:
                    popped = None
            # pigpio calls happen outside the lock — they're I/O, not state changes.
            try:
                if close_gate:
                    self.pi.set_servo_pulsewidth(self.gpio_pin, self.pass_us)
                    LOG.info("Gate closed (Ready).")
                if open_gate:
                    self.pi.set_servo_pulsewidth(self.gpio_pin, self.divert_us)
                    LOG.info("Gate opened for reject %d.", popped.number)
            except Exception as exc:
                # A failed pigpio call must not kill this thread — log it and
                # keep servicing the FIFO so the next reject still gets a
                # fresh attempt, instead of the servo going dark permanently.
                LOG.error("Servo hardware call failed (%s) — will keep retrying "
                          "on the next event. Check 'sudo systemctl status pigpiod'.",
                          exc)
            time.sleep(self.service_interval)

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.clear()
        self.pi.set_servo_pulsewidth(self.gpio_pin, 0)  # stop sending pulses entirely
        self.pi.stop()
