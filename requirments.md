# Seed Sorter: Requirements

Raspberry Pi vision agent + ESP8266 motor controller + Pi-driven reject servo,
with a Firebase-backed dashboard.

## 1. Files

| File | Runs on | Role |
|---|---|---|
| `sorter_pi_agent.py` | Pi | Main loop: camera, TFLite classifier, state machine, ESP link |
| `reject_actuator.py` | Pi | Reject servo (pigpio) and reject FIFO |
| `cloud_link.py` | Pi | Firebase Realtime Database link to the dashboard |
| `model_outputs/maize_mobilenetv3large_float32.tflite` | Pi | Classifier model |
| `model_outputs/deployment_metadata_v2_recovered.json` | Pi | Model metadata (must be schema version 2) |
| `sorter_esp8266.ino` | ESP8266 | Belt, singulator and LED only (no servo) |

`sorter_pi_agent.py`, `reject_actuator.py` and `cloud_link.py` must sit in the same folder.
`model_outputs/` must sit next to `sorter_pi_agent.py`.

## 2. Hardware

- **Raspberry Pi 4 / 3 / Zero 2** with Raspberry Pi OS (64-bit recommended).
  **Not a Pi 5:** `pigpio` does not support it. A Pi 5 needs a different servo driver (lgpio / hardware PWM).
- **CSI camera** (picamera2). For a USB webcam set `CAMERA = 0` in the agent.
- **Hobby servo** (reject gate) on **BCM GPIO18 = physical pin 12**, with its own 5 V supply and a **common ground** with the Pi.
- **NodeMCU ESP8266** connected to the Pi by **USB** (appears as `/dev/ttyUSB0`).
- **2x stepper + DRV8825 drivers** (belt, singulator) and a **MOSFET-driven LED**, wired to the ESP:
  belt STEP = D2, singulator STEP = D5, singulator DIR = D6, LED = D7.
- DRV8825 MS1/MS2/MS3 jumpers must match `STEPS_PER_REV` in the `.ino` (currently 200 = full step).

## 3. Raspberry Pi system setup

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-picamera2 pigpio python3-pigpio
sudo systemctl enable --now pigpiod        # reject servo will not work without this daemon
sudo usermod -aG dialout $USER             # serial access to the ESP; log out and in afterwards
```

## 4. Python packages

Python 3.9 or newer (Raspberry Pi OS Bookworm ships 3.11).

```bash
python3 -m venv --system-site-packages ~/sorter-venv   # system site packages needed for picamera2/libcamera
source ~/sorter-venv/bin/activate
pip install numpy pillow opencv-python-headless requests pyserial pigpio
pip install ai-edge-litert                              # TFLite runtime (see note)
```

| Package | Used by | Notes |
|---|---|---|
| `numpy`, `pillow`, `opencv-python-headless` | agent | Set `HEADLESS = True` (default). A local preview window needs a full OpenCV build. |
| `ai-edge-litert` | agent | The agent tries `ai_edge_litert`, then `tflite_runtime`, then `tensorflow.lite`. One of them is enough. |
| `picamera2` | agent | Installed by apt (above), not pip. |
| `pigpio` | `reject_actuator.py` | Needs the `pigpiod` daemon running. |
| `pyserial` | agent | USB link to the ESP (`ESP_LINK = "serial"`). |
| `requests` | `cloud_link.py` | Firebase REST calls. |

## 5. ESP8266 firmware

- Arduino IDE or PlatformIO with the **ESP8266 board package**, board **NodeMCU 1.0**.
- Library: **AccelStepper**.
- Serial baud **115200**, which must equal `ESP_SERIAL_BAUD` in the agent.
- Link is **USB serial**, framed as `COMMAND*XX\n` (XX = XOR checksum of the command, uppercase hex).
  Commands: `RUN`, `STOP`, `CALIBRATE`, `SPD1`-`SPD3`.
  Replies: `READY`, `STATE RUNNING|IDLE|CALIBRATION`, `ERR ...`.
- The firmware has **no WiFi code**, so keep `ESP_LINK = "serial"`. Only use `"wifi"` after adding a TCP server to the firmware.
- Set `MOTOR_PULLEY_DIAMETER_MM` to the real measured value.

## 6. Cloud dashboard (Firebase Realtime Database)

- Create a Realtime Database and put its URL in `CLOUD_DB_URL` (agent) and in the dashboard sidebar.
- `CLOUD_NODE` (default `seed_sorting`) must match the dashboard.
- The Pi reads and clears `command`, and writes `good`, `bad`, `status`, `diverter`, `camera_frame`, `camera_ts`.
- Test-mode rules expire after 30 days and are open to **anyone with the URL**, including sending `START`. Tighten the rules before leaving it online.
- Frames are uploaded as base64 JPEG at `STREAM_FPS`. Watch the free tier's monthly download allowance; lower `STREAM_FPS` or `STREAM_JPEG_QUALITY` if it runs out.

## 7. Settings to check before first run (top of `sorter_pi_agent.py`)

| Setting | Check |
|---|---|
| `ESP_LINK`, `ESP_SERIAL_PORT` | `"serial"`, and the port the NodeMCU actually shows up as |
| `DASHBOARD_SERIAL_PORT` | Must differ from `ESP_SERIAL_PORT` if `CONTROL_MODE = "dashboard_serial"` |
| `SERVO_GPIO_PIN`, `SERVO_PASS_ANGLE`, `SERVO_DIVERT_ANGLE` | Match wiring and gate geometry |
| `SERVO_HOLD_SECONDS`, `SERVO_MOVE_SECONDS` | Hold must exceed the measured swing time |
| `CAMERA_TO_SERVO_MM` | Measured camera-to-gate distance on the belt |
| `BELT_SPEED_CM_S_BY_LEVEL` | **Measure each level.** Levels 2 and 3 are estimates, and the ESP firmware warns they exceed its safe step rate |
| `MODEL_PATH`, `METADATA_PATH` | Files exist; metadata is schema version 2 |

## 8. Run

1. Plug in the ESP and servo power; confirm `systemctl status pigpiod` is active.
2. `source ~/sorter-venv/bin/activate && python3 sorter_pi_agent.py`
3. Test with `DRY_RUN = True` first (no ESP needed), then switch to `False`.
4. Use the dashboard: servo test buttons while Idle, then CALIBRATE, then START.

## 9. Known open items

- **ESP has no watchdog.** If the Pi crashes or the USB cable drops while Running, the belt keeps running. Add a heartbeat timeout to the firmware.
- **`applySpeedLevel()` in the `.ino`** uses `- BELT_SPEED_MULTIPLIER` (and `- microScale`, `- SINGULATION_SPEED_MULTIPLIER`) where the boot diagnostics use `*`. Confirm which one is in the file you flash, since it changes belt speed and therefore reject timing.
- **Speed changes while Running** are not applied to rejects already queued.
- **Camera colour order:** picamera2 `BGR888` is RGB in memory. Check that yellow kernels look yellow in the dashboard stream.
- **Cloud polling** downloads the whole node (including the camera frame) every second. Polling only `.../seed_sorting/command.json` would cut that sharply.
