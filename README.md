Automated Maize Seed Sorting System

firebase link "https://console.firebase.google.com/u/0/project/maize-sorting/database/maize-sorting-default-rtdb/data/~2F"
streamlit app "https://cccgnakbcnhsp5mszbdh8q.streamlit.app/"

An automated maize seed sorting system that uses computer vision and a machine-learning model to identify good seeds, bad seeds, and objects that are not maize. A Raspberry Pi handles camera capture, classification, sorting decisions, and the reject servo, while an ESP8266 controls the conveyor and singulation motors. A Streamlit dashboard provides machine monitoring and operator controls.
Table of Contents
Project Overview
System Architecture
Main Features
Hardware Requirements
Software Requirements
Project Structure
Installation
Configuration
Wi-Fi and Device Credentials
Running the System
Dashboard Connection Modes
Operating Procedure
Safety and Troubleshooting
Limitations and Notes
Project Overview
The system automates the inspection and separation of maize seeds. A camera continuously observes seeds moving along the conveyor. When the operator starts a sorting session, the Raspberry Pi loads the TensorFlow Lite model and begins classifying the observed items.
The classifier uses three output classes:
Class	Meaning
`GOOD_SEED`	A seed classified as good
`BAD_SEED`	A seed classified as bad and scheduled for rejection
`NO_MAIZE`	An object that is not identified as maize
The Raspberry Pi schedules reject-gate actions using a timed FIFO queue so that a seed can travel from the camera position to the servo gate before the gate moves. The ESP8266 is responsible for motor actuation; the Raspberry Pi is responsible for image processing, classification, and the reject servo.
System Architecture
```text
                         ┌─────────────────────────┐
                         │    Streamlit Dashboard  │
                         │ Monitoring and Controls │
                         └────────────┬────────────┘
                                      │
                         Cloud / WebSocket connection
                                      │
                         ┌────────────▼────────────┐
                         │      Raspberry Pi       │
                         │                         │
                         │ Camera capture          │
                         │ TFLite classification   │
                         │ Sorting decision        │
                         │ Reject timing / FIFO    │
                         │ Servo control (GPIO 18) │
                         └───────┬─────────┬───────┘
                                 │         │
                         TCP/Wi-Fi│         │GPIO / pigpio
                                 │         │
                    ┌────────────▼───┐  ┌──▼────────────────┐
                    │    ESP8266     │  │ Reject gate servo │
                    │ Motor control  │  └───────────────────┘
                    └───────┬────────┘
                            │
                    Conveyor and
                    singulation motors
```
Component responsibilities
Raspberry Pi: Captures camera frames, runs the TensorFlow Lite classifier when a sorting session starts, makes sorting decisions, schedules reject actions, and communicates with the dashboard and ESP8266.
ESP8266: Controls the conveyor and singulation motors using commands from the Raspberry Pi.
Reject servo: Operated directly by the Raspberry Pi through `pigpio` on GPIO 18 in the current configuration.
Streamlit dashboard: Displays the camera feed, machine status, sorting counts, session information, and operator controls.
Firebase Realtime Database / WebSocket: Provides the dashboard-to-Pi communication path, depending on the selected connection mode.
Main Features
Live camera streaming while the machine is idle.
Model loading and classification begin when the operator sends `START`.
Three classification labels: `GOOD_SEED`, `BAD_SEED`, and `NO_MAIZE`.
Timed FIFO scheduling for reject-gate actions.
ESP8266 motor control over Wi-Fi TCP or serial, depending on configuration.
Dashboard connection through Firebase Cloud or direct WebSocket.
Dashboard controls for starting and stopping a sorting session, changing belt speed, calibration, and servo mode.
Live machine status, camera feed, good/bad counts, charts, session history, and CSV export.
Safety checks that prevent a sorting session from starting if the reject servo is unavailable or a fault is active.
Hardware Requirements
The exact mechanical design and wiring may vary. The current software configuration expects:
Raspberry Pi with Raspberry Pi OS.
Raspberry Pi camera supported by `Picamera2`, or a compatible camera index configured in the agent.
ESP8266 development board (for example, NodeMCU).
Conveyor belt motor and driver.
Singulation motor and driver.
Reject-gate servo connected to Raspberry Pi GPIO 18.
Suitable regulated power supplies for the Raspberry Pi, ESP8266, motors, and servo.
Mechanical conveyor, seed guide, camera mount, and reject chute.
A computer or device that can access the Streamlit dashboard.
Important: Do not power motors or a servo directly from a Raspberry Pi GPIO pin. Use appropriate motor drivers and a suitable external power supply. Ensure the control electronics share a compatible ground reference where required by the wiring design.
Software Requirements
Raspberry Pi
Python 3.
OpenCV (`cv2`).
NumPy.
Pillow.
`pigpio` Python package and the `pigpiod` daemon.
A TensorFlow Lite interpreter package compatible with the Raspberry Pi environment (`ai-edge-litert`, `tflite-runtime`, or a compatible TensorFlow installation).
`picamera2` when using the Raspberry Pi camera through Picamera2.
`websockets` for WebSocket dashboard communication.
`pyserial` if serial communication is selected.
The project’s `cloud_link.py` module and its required dependencies for Firebase communication.
Dashboard computer
Python 3.
Streamlit.
Pandas.
Altair.
The project’s `websocket_bridge.py` and `cloud_bridge.py` modules.
The required Firebase client/dependencies used by `cloud_bridge.py`.
Install packages in a virtual environment where possible. Raspberry Pi OS may prevent system-wide `pip` installation because its Python environment is externally managed.
Project Structure
Keep the project files arranged so that the Python modules and model files can be found by the application. A typical layout is:
```text
seed-sorting-system/
├── app.py
├── sorter_agent.py
├── reject_actuator.py
├── websocket_bridge.py
├── cloud_bridge.py
├── cloud_link.py
├── model_outputs/
│   ├── maize_mobilenetv3large_float32.tflite
│   └── deployment_metadata_v2_recovered.json
├── .streamlit/
│   └── config.toml              # optional dashboard theme configuration
└── README.md
```
The names of the model and metadata files above match the paths configured in `sorter_agent.py`. The Cloud modules are separate project dependencies imported by the dashboard and Raspberry Pi agent; include the corresponding working files in your repository/deployment.
Installation
1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```
Replace the URL with your GitHub repository URL.
2. Enter the Raspberry Pi project directory
The sorter project is stored on the Raspberry Pi at:
```bash
cd ~/project/project
```
3. Activate the existing Python virtual environment
The Raspberry Pi uses a virtual environment located at `~/venv_sorter`. Activate it before installing packages or running the sorter:
```bash
source ~/venv_sorter/bin/activate
```
After activation, your terminal prompt should show the environment name, usually `(venv_sorter)`. Upgrade pip if required:
```bash
python -m pip install --upgrade pip
```
4. Install Python dependencies
Install the packages required by the files in your project. A starting point for the dashboard and common Python dependencies is:
```bash
pip install streamlit pandas altair numpy opencv-python pillow websockets pyserial
```
Install Raspberry Pi-specific packages using the installation method recommended for your Raspberry Pi OS version. In particular, install and configure `picamera2`, `pigpio`, and a compatible TensorFlow Lite interpreter. Install the Firebase dependencies required by your supplied `cloud_link.py` and `cloud_bridge.py` modules.
> Package availability differs across Raspberry Pi OS and Python versions. Do not assume that the desktop `opencv-python` or a generic TensorFlow package is the correct installation for every Raspberry Pi image.
5. Install and start pigpio
The reject actuator uses the `pigpio` library. Install the package and daemon for your Raspberry Pi OS version, then start the daemon. On systems that provide the systemd service:
```bash
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```
Check that the daemon is running before starting the sorter agent:
```bash
sudo systemctl status pigpiod
```
6. Add the model files
Place the TensorFlow Lite model and its matching metadata file in `model_outputs/`:
```text
model_outputs/maize_mobilenetv3large_float32.tflite
model_outputs/deployment_metadata_v2_recovered.json
```
The metadata must correspond to the model. The agent validates the model input and metadata class order during initialization. The configured class order is `BAD_SEED`, `GOOD_SEED`, `NO_MAIZE`.
Configuration
Review the configuration constants near the beginning of `sorter_agent.py` before running the machine.
Setting	Current configured value	Purpose
`ESP_LINK`	`"wifi"`	Selects the ESP communication method
`ESP_HOST`	`"seedsorter.local"`	ESP8266 hostname for Wi-Fi TCP
`ESP_PORT`	`8266`	ESP8266 TCP port
`ESP_SERIAL_PORT`	`/dev/ttyUSB0`	Serial device when serial mode is selected
`CAMERA`	`"picamera2"`	Camera source; can be changed to a camera index such as `0`
`WS_HOST`	`0.0.0.0`	WebSocket server bind address
`WS_PORT`	`8765`	WebSocket server port
`WS_ACCESS_KEY`	`"seedsorter"`	WebSocket access key; must match the dashboard setting
`CLOUD_DB_URL`	Firebase Realtime Database URL	Firebase database endpoint
`CLOUD_NODE`	`"seed_sorting"`	Firebase node used by the agent
`SERVO_GPIO_PIN`	`18`	Raspberry Pi GPIO pin for reject servo
`SERVO_PASS_ANGLE`	`20`	Servo pass/park position in the current setup
`SERVO_DIVERT_ANGLE`	`110`	Servo reject/divert position in the current setup
`CAMERA_TO_SERVO_MM`	`500.0`	Distance between camera and reject servo
The dashboard’s WebSocket defaults are defined in `app.py`. Keep its host, port, and access key consistent with the Raspberry Pi agent. Change default credentials before deploying beyond a trusted test network; do not commit private credentials or Firebase secrets to a public repository.
The belt-speed values in the Raspberry Pi agent must remain consistent with the ESP8266 firmware. If you change the ESP firmware’s speed mapping, update the matching values in the Pi software as well.
Wi-Fi and Device Credentials
The ESP8266 and Raspberry Pi must be connected to the same local Wi-Fi network when the system is using Wi-Fi TCP communication or direct WebSocket communication over the local network.
1. Set the Wi-Fi details in the ESP8266 `.ino` firmware
Before uploading the firmware, open the ESP8266 Arduino sketch (`.ino`) in the Arduino IDE or your preferred editor. Find the Wi-Fi SSID and password definitions. They may look similar to this:
```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```
Replace the example values with the name and password of the Wi-Fi network that the machine will use:
```cpp
const char* ssid = "YourNetworkName";
const char* password = "YourNetworkPassword";
```
Use the same network details when configuring the Raspberry Pi. Save the sketch, select the correct ESP8266 board and port in the Arduino IDE, compile it, and upload it to the ESP8266. If the project contains more than one `.ino` sketch that connects to Wi-Fi, check and update the credentials in each relevant sketch.
Important: The SSID is the Wi-Fi network name, not the Raspberry Pi hostname. The password is the Wi-Fi password, not the Raspberry Pi login password.
2. Connect the Raspberry Pi to the same Wi-Fi network
Configure Wi-Fi in Raspberry Pi OS using the desktop network menu or the operating system's network configuration tools. Select the same SSID entered in the ESP8266 sketch and enter that network's Wi-Fi password. The Pi must be connected to the network before starting the sorter if the ESP8266 is being reached over Wi-Fi.
You can check the Pi's current connection and IP address with:
```bash
hostname -I
```
If the ESP8266 firmware uses a fixed IP address or hostname, make sure the values configured in `sorter_agent.py` match the ESP8266's actual network settings. For example, check `ESP_HOST` and `ESP_PORT` in the Pi code.
3. Raspberry Pi login
Use these login details when accessing the Raspberry Pi from a terminal or SSH client:
Setting	Value
Username	`autonomous`
Password	`autonomous`
Project directory	`~/project/project`
Python virtual environment	`~/venv_sorter`
For SSH, the connection command is:
```bash
ssh autonomous@<RASPBERRY_PI_IP>
```
Replace `<RASPBERRY_PI_IP>` with the address reported by `hostname -I`. When prompted, enter the Raspberry Pi login password. Change the default password before exposing the Pi to an untrusted network, and do not publish private Wi-Fi credentials or login details in a public repository.
4. Activate the environment and enter the project folder
After logging in, run:
```bash
cd ~/project/project
source ~/venv_sorter/bin/activate
```
Run the Raspberry Pi Python programs from this project directory with the `venv_sorter` environment active.
Running the System
Start the Raspberry Pi sorter agent
On the Raspberry Pi, enter the project directory and activate the existing environment:
```bash
cd ~/project/project
source ~/venv_sorter/bin/activate
python sorter_agent.py
```
The agent starts the camera stream and dashboard communication. The classifier is intentionally idle until a `START` command is received. If the reject servo cannot initialize or the system is in a fault state, `START` is disabled by the agent’s safety checks.
Start the Streamlit dashboard
On the computer that will operate the machine:
```bash
cd ~/project/project
source ~/venv_sorter/bin/activate
streamlit run app.py
```
Open the local URL printed by Streamlit in your browser. Select the desired connection mode in the dashboard sidebar, enter the Raspberry Pi host or IP address when using WebSocket, and connect.
Optional syntax check
Before running the agent, check the Python files for syntax errors:
```bash
python -m py_compile sorter_agent.py reject_actuator.py websocket_bridge.py
```
Dashboard Connection Modes
The dashboard supports two communication modes. Select the mode in the dashboard; the Raspberry Pi agent can provide both services.
1. WebSocket — same local network
Connect the dashboard computer and Raspberry Pi to the same Wi-Fi/LAN.
Enter the Raspberry Pi hostname or IP address in the dashboard.
Use port `8765` unless you changed `WS_PORT`.
Ensure the dashboard access key matches `WS_ACCESS_KEY` in `sorter_agent.py`.
This mode provides a direct connection and is intended for low-latency local operation.
2. Cloud — Firebase Realtime Database
Configure the Firebase Realtime Database URL and the required access settings in the Cloud modules.
Ensure both the Raspberry Pi and dashboard computer can reach the database over the internet.
Confirm that the Pi and dashboard use the same database and expected data node.
Protect database access with appropriate Firebase security rules. Do not publish service-account keys, tokens, or other secrets.
Cloud mode is intended for operation where the dashboard and Raspberry Pi are not necessarily on the same local network. It may have more latency than a direct WebSocket connection.
Operating Procedure
Check the mechanical assembly, wiring, power supplies, camera view, and reject-gate movement before loading seeds.
Power the ESP8266, Raspberry Pi, camera, motor drivers, and servo supply.
Confirm that the Raspberry Pi can communicate with the ESP8266 and initialize the reject servo.
Start `sorter_agent.py` and open the Streamlit dashboard.
Connect using either Cloud or WebSocket mode.
Confirm that the live camera feed and machine status are visible. The camera can stream while the classifier is idle.
Set the servo to Autonomous mode for automatic sorting.
Press START to begin classification and the sorting session.
Monitor the camera feed, counts, machine status, and reject-gate status during operation.
Press STOP to stop classification and end the session. The camera stream remains available while idle.
Use the dashboard’s manual servo controls only when the machine is stopped and it is safe to test the reject mechanism. Keep hands clear of moving belts, motors, linkages, and the reject gate.
Safety and Troubleshooting
Problem	Things to check
Dashboard cannot connect through WebSocket	Confirm the Pi agent is running, both devices are on the same network, the Pi address is correct, port `8765` is reachable, and the access keys match.
Cloud mode does not update	Check internet access, Firebase URL, database rules, configured node, and the Cloud bridge modules on both devices.
Camera feed is unavailable	Check camera connection, Picamera2 installation, camera permissions, and the `CAMERA` setting.
Model fails to load	Confirm that both model files exist at the configured paths and that the TensorFlow Lite runtime is installed and compatible.
`START` is rejected	Check that the servo initialized, no fault is active, and the servo is in Autonomous mode. Review the agent logs for the specific reason.
Servo is unavailable	Check GPIO wiring, servo power, `pigpiod` status, and the `pigpio` Python package. Do not power the servo from a GPIO pin.
ESP8266 does not respond	Confirm the ESP is powered, the configured Wi-Fi hostname/IP and port are correct, and the firmware communication settings match the Pi configuration.
Servo timing is inaccurate	Recheck the camera-to-servo distance, belt speed mapping, seed travel assumptions, and mechanical response. Test with the machine safely unloaded.
For service logs when the agent is managed by systemd, use the actual service name configured on your Raspberry Pi. For example:
```bash
sudo systemctl status YOUR_SERVICE_NAME
journalctl -u YOUR_SERVICE_NAME -f
```
Limitations and Notes
Classification performance depends on the training data, camera quality, lighting, seed presentation, and model accuracy. Validate the model with representative seeds before relying on automated decisions.
The camera-to-servo distance and belt-speed configuration affect rejection timing. Calibrate and test these values on the physical machine.
The current GPIO pin, servo angles, ESP network settings, and model paths are software configuration values; confirm that they match your actual wiring and deployment.
This README describes the software architecture and configuration represented by the project files. Hardware wiring diagrams, model training details, Firebase security configuration, and ESP8266 firmware source should be documented alongside the repository if you want contributors to reproduce the complete system.
Contributing
Contributions, bug reports, and suggestions are welcome. When submitting changes, include the hardware configuration affected, the software environment used for testing, and the steps needed to reproduce the change.
License
Add the license that applies to this project before publishing or accepting contributions. Do not assume that third-party model files, datasets, or dependencies are covered by the same license as your source code.
