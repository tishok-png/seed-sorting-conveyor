/* =====================================================================
   AUTOMATED SEED SORTING SYSTEM
   ESP8266 (NodeMCU) actuation firmware — motors + LED, over WiFi
   =====================================================================

   ROLE IN THE SYSTEM
   -------------------
   The Raspberry Pi does camera capture, seed classification, and the
   reject servo + its FIFO timing (see reject_actuator.py, driven from
   a Pi GPIO pin). This board's job is exactly: the conveyor belt
   motor, the singulation motor, and the indicator LED. It has no
   servo, no diverter, no FIFO, and no longer needs to know which
   seeds were GOOD or BAD — that lives entirely on the Pi.

   TRANSPORT: WIFI, NOT USB SERIAL
   -------------------------------------------------------------------
   The Pi no longer talks to this board over a USB cable. This board
   connects to your WiFi (WIFI_SSID/WIFI_PASSWORD below) and runs a
   small TCP server; the Pi connects to it as a client. The command
   protocol itself (checksummed lines, same verbs) is unchanged from
   the serial version — only the wire it travels over changed.

   Two real tradeoffs versus USB serial, worth knowing before you rely
   on this: (1) this board's single core now runs the WiFi stack and
   the stepper pulse generation (AccelStepper) at the same time, which
   can introduce small belt/singulator timing jitter that wasn't there
   over serial — WIFI_NONE_SLEEP below reduces but doesn't eliminate
   this. (2) if the WiFi link drops while RUNNING or in CALIBRATION,
   the Pi has no way to send STOP until it reconnects — this firmware
   stops the motors on its own the moment it notices the link is down
   (see checkWiFi()), but a physical E-stop on motor power is still
   the real safety backstop, same as it always was.

   LIBRARIES REQUIRED
   -------------------------------------------------------------------
   AccelStepper (by Mike McCauley) — both stepper motors.
   ESP8266WiFi and ESP8266mDNS — bundled with the ESP8266 Arduino core.

   PROTOCOL (from the Pi, one line per command, checksummed)
   -------------------------------------------------------------------
     Every line has the form   PAYLOAD*CK\n
     where CK is a 2-digit uppercase hex XOR checksum of every byte in
     PAYLOAD. A line that fails the checksum is ignored and answered
     with "ERR CHECKSUM" so you can see it happening on the Pi's log.

     RUN            -> start belt + singulator + LED
     STOP           -> stop everything. Always allowed, from any state.
     CALIBRATE      -> run belt + singulator continuously at level 1 for
                       physical alignment work.
     SPD1 / SPD2 / SPD3 -> set the speed level directly (matches the
                       dashboard's 3 speed buttons). Also applied live if
                       already RUNNING or in CALIBRATION.
     SPD+ / SPD-    -> step the speed level up/down by one (kept for
                       compatibility; same live-apply behaviour as above).
     TESTBELT       -> hardware bring-up only: spins ONLY the belt motor
                       for 3 seconds at a fixed diagnostic speed. Rejected
                       with ERR STOP_FIRST unless currently IDLE.
     TESTSING       -> same, but for ONLY the singulator motor.
     TESTLED        -> same, but for the indicator LED.

   MACHINE STATE
   -------------------------------------------------------------------
   Every state transition is echoed back as "STATE IDLE" / "STATE
   RUNNING" / "STATE CALIBRATION" so the Pi always reflects what the
   motors actually did, not just what they were asked to do. There is
   no FAULT state here — the only thing that used to trigger one (the
   reject FIFO overflowing) no longer happens on this board. If the
   Pi's own reject FIFO overflows, the Pi sends this board a STOP and
   reports its own FAULT upstream.

   HOW TO CHANGE THINGS
   -------------------------------------------------------------------
   Everything you're likely to tune lives in the USER CONFIGURATION
   block right below.
   ===================================================================== */

#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <AccelStepper.h>

/* =====================================================================
   USER CONFIGURATION — change these to match your build
   ===================================================================== */

// ---- WiFi ----
const char* WIFI_SSID = "G";
const char* WIFI_PASSWORD = "66666622";
const char* MDNS_HOSTNAME = "seedsorter";  // Pi connects to seedsorter.local
const uint16_t TCP_PORT = 8266;

// ---- Pins (as wired) ----
#define PIN_M1_STEP       D5   // Motor 1 (conveyor belt) STEP
#define PIN_M1_DIR        D6   // Motor 1 DIR
#define PIN_M2_STEP       D2   // Motor 2 (singulator) STEP — DIR is hardwired to 3V3
#define PIN_M2_DIR_UNUSED D0   // Not physically connected; AccelStepper's DRIVER mode
                                // needs a dir-pin argument even though motor 2's real
                                // DIR pin is hardwired to 3V3. Has no hardware effect.
#define PIN_LED_MOSFET    D7   // MOSFET gate driving the indicator LED

// ---- Conveyor mechanics (used to convert "belt speed" into a step rate) ----
// Drivetrain: motor -> 3:1 gearbox -> small pulley -> belt -> big pulley on
// roller shaft -> roller -> conveyor belt.
const float ROLLER_DIAMETER_MM       = 40.0;
const float DRIVE_PULLEY_DIAMETER_MM = 90.0;
const float MOTOR_PULLEY_DIAMETER_MM = 12.0;  // *** CONFIRM THIS *** placeholder —
                                               // measure the motor/gearbox output
                                               // pulley and update before trusting speed.
const float GEARBOX_RATIO            = 3.0;
const int   STEPS_PER_REV            = 200;   // 1.8 deg/step, full-step. Multiply here
                                               // if your DRV8825 MSx pins are set for
                                               // microstepping (e.g. 1600 for 1/8).

// Desired conveyor belt speeds for the 3 run levels, in cm/s.
// IMPORTANT: this is the TARGET the ESP commands the motor to reach from
// the pulley/gearbox math above — it is not necessarily the speed the belt
// actually achieves (that depends on MOTOR_PULLEY_DIAMETER_MM being right,
// plus slip/friction). reject_actuator.py on the Pi uses the separately
// MEASURED real speed for reject timing, which is why the two lists don't
// have to be identical, only level 1 vs 2 vs 3 meaning the same thing.
const float BELT_SPEED_CM_S[3] = { 3.14, 6.2, 7.4 };

// Singulation motor step rate for each level, in steps/sec.
const float SINGULATION_STEPS_PER_SEC[3] = { 150.0, 250.0, 350.0 };

/* =====================================================================
   Below here is logic — you shouldn't need to edit this for normal tuning
   ===================================================================== */

WiFiServer tcpServer(TCP_PORT);
WiFiClient piClient;

enum SystemState { STATE_IDLE, STATE_RUNNING, STATE_CALIBRATION };
SystemState currentState = STATE_IDLE;

// --- Hardware bring-up test mode ---
// TESTBELT / TESTSING / TESTLED drive exactly one actuator for a few
// seconds, independent of RUN/STOP, so you can identify which physical
// part is wired to which pins one at a time instead of guessing from
// RUN (which drives both motors together). Only allowed while IDLE.
enum TestKind { TEST_NONE, TEST_BELT, TEST_SINGULATOR, TEST_LED };
TestKind activeTest = TEST_NONE;
unsigned long testEndAt = 0;
const unsigned long TEST_DURATION_MS = 3000;
const float TEST_STEPS_PER_SEC = 200.0; // fixed, modest diagnostic speed — not tied to speedLevel

int speedLevel = 1; // 1..3, index into the arrays above (arrays are 0-based)

// AccelStepper handles all pulse timing for both motors, non-blocking.
AccelStepper beltMotor(AccelStepper::DRIVER, PIN_M1_STEP, PIN_M1_DIR);
AccelStepper singulatorMotor(AccelStepper::DRIVER, PIN_M2_STEP, PIN_M2_DIR_UNUSED);

// Incoming line buffer (from the Pi's TCP connection)
String serialLine = "";

/* ---------------------------------------------------------------------
   Reply to the Pi (if connected) and to the USB serial monitor (if
   attached) at the same time — Serial is debug-only now, not the
   command channel, but it's still useful for watching what's happening
   with a cable plugged in during bring-up.
   --------------------------------------------------------------------- */
void reply(const String &text) {
  Serial.println(text);
  if (piClient && piClient.connected()) {
    piClient.println(text);
  }
}

/* ---------------------------------------------------------------------
   Speed math: turn a target belt speed (cm/s) into a step rate for
   AccelStepper.
   --------------------------------------------------------------------- */
float beltSpeedToStepsPerSec(float beltSpeedCmS) {
  float rollerCirc_cm = (PI * ROLLER_DIAMETER_MM) / 10.0;
  float rollerRPS     = beltSpeedCmS / rollerCirc_cm;
  float pulleyRatio   = DRIVE_PULLEY_DIAMETER_MM / MOTOR_PULLEY_DIAMETER_MM;
  float motorRPS      = rollerRPS * pulleyRatio * GEARBOX_RATIO;
  return motorRPS * STEPS_PER_REV;
}

void applySpeedLevel() {
  int i = speedLevel - 1;

  float beltStepsPerSec = beltSpeedToStepsPerSec(BELT_SPEED_CM_S[i]);
  beltMotor.setMaxSpeed(beltStepsPerSec * 1.2);
  beltMotor.setSpeed(beltStepsPerSec);

  float singStepsPerSec = SINGULATION_STEPS_PER_SEC[i];
  singulatorMotor.setMaxSpeed(singStepsPerSec * 1.2);
  singulatorMotor.setSpeed(singStepsPerSec);
}

/* ---------------------------------------------------------------------
   State transitions
   --------------------------------------------------------------------- */
void startRunning() {
  currentState = STATE_RUNNING;
  applySpeedLevel();
  digitalWrite(PIN_LED_MOSFET, HIGH);
  reply(F("STATE RUNNING"));
}

void stopRunning() {
  currentState = STATE_IDLE;
  beltMotor.setSpeed(0);
  singulatorMotor.setSpeed(0);
  digitalWrite(PIN_LED_MOSFET, LOW);
  reply(F("STATE IDLE"));
}

void startCalibration() {
  currentState = STATE_CALIBRATION;
  speedLevel = 1; // calibration always runs at the slowest, safest level
  applySpeedLevel();
  digitalWrite(PIN_LED_MOSFET, HIGH);
  reply(F("STATE CALIBRATION"));
}

/* ---------------------------------------------------------------------
   Hardware bring-up test mode — one actuator at a time, independent of
   RUN/STOP, so you can identify which physical part is on which pins.
   --------------------------------------------------------------------- */
void startTest(TestKind kind) {
  if (currentState != STATE_IDLE) {
    reply(F("ERR STOP_FIRST"));
    return;
  }
  activeTest = kind;
  testEndAt = millis() + TEST_DURATION_MS;
  switch (kind) {
    case TEST_BELT:
      beltMotor.setMaxSpeed(TEST_STEPS_PER_SEC * 1.2);
      beltMotor.setSpeed(TEST_STEPS_PER_SEC);
      reply(F("TEST BELT_STARTED"));
      break;
    case TEST_SINGULATOR:
      singulatorMotor.setMaxSpeed(TEST_STEPS_PER_SEC * 1.2);
      singulatorMotor.setSpeed(TEST_STEPS_PER_SEC);
      reply(F("TEST SINGULATOR_STARTED"));
      break;
    case TEST_LED:
      digitalWrite(PIN_LED_MOSFET, HIGH);
      reply(F("TEST LED_STARTED"));
      break;
    default:
      break;
  }
}

void serviceTest() {
  if (activeTest == TEST_NONE) return;
  if (activeTest == TEST_BELT) beltMotor.runSpeed();
  if (activeTest == TEST_SINGULATOR) singulatorMotor.runSpeed();
  if ((long)(millis() - testEndAt) >= 0) {
    beltMotor.setSpeed(0);
    singulatorMotor.setSpeed(0);
    digitalWrite(PIN_LED_MOSFET, LOW);
    activeTest = TEST_NONE;
    reply(F("TEST DONE"));
  }
}

/* ---------------------------------------------------------------------
   Checksum framing: PAYLOAD*CK  where CK is a 2-digit hex XOR of PAYLOAD.
   --------------------------------------------------------------------- */
bool checkAndStrip(String &line) {
  int star = line.lastIndexOf('*');
  if (star < 0 || (int)line.length() - star != 3) return false;
  byte calc = 0;
  for (int i = 0; i < star; i++) calc ^= (byte)line[i];
  byte given = (byte)strtol(line.substring(star + 1).c_str(), nullptr, 16);
  if (calc != given) return false;
  line = line.substring(0, star);
  return true;
}

/* ---------------------------------------------------------------------
   Command handling
   --------------------------------------------------------------------- */
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (!checkAndStrip(cmd)) {
    reply(F("ERR CHECKSUM"));
    return;
  }

  bool liveApply = (currentState == STATE_RUNNING || currentState == STATE_CALIBRATION);

  if (cmd == "RUN") {
    startRunning();
  } else if (cmd == "STOP") {
    stopRunning();
  } else if (cmd == "CALIBRATE") {
    startCalibration();
  } else if (cmd == "SPD1") {
    speedLevel = 1;
    if (liveApply) applySpeedLevel();
  } else if (cmd == "SPD2") {
    speedLevel = 2;
    if (liveApply) applySpeedLevel();
  } else if (cmd == "SPD3") {
    speedLevel = 3;
    if (liveApply) applySpeedLevel();
  } else if (cmd == "SPD+") {
    if (speedLevel < 3) speedLevel++;
    if (liveApply) applySpeedLevel();
  } else if (cmd == "SPD-") {
    if (speedLevel > 1) speedLevel--;
    if (liveApply) applySpeedLevel();
  } else if (cmd == "TESTBELT") {
    startTest(TEST_BELT);
  } else if (cmd == "TESTSING") {
    startTest(TEST_SINGULATOR);
  } else if (cmd == "TESTLED") {
    startTest(TEST_LED);
  } else {
    reply(F("ERR UNKNOWN_CMD"));
  }
}

/* ---------------------------------------------------------------------
   TCP client handling — accepts one Pi connection at a time. If the Pi
   disconnects and reconnects (its own retry logic, or a Wi-Fi blip),
   this just accepts the new connection; no reboot needed on this side.
   --------------------------------------------------------------------- */
void pollClient() {
  if (!piClient || !piClient.connected()) {
    WiFiClient newClient = tcpServer.available();
    if (newClient) {
      piClient = newClient;
      Serial.println(F("Pi connected."));
      piClient.println(F("READY"));
    }
    return;
  }
  while (piClient.available() > 0) {
    char c = piClient.read();
    if (c == '\n') {
      handleCommand(serialLine);
      serialLine = "";
    } else if (c != '\r') {
      if (serialLine.length() < 64) serialLine += c; // guard against a runaway line
    }
  }
}

/* ---------------------------------------------------------------------
   If Wi-Fi drops while the motors are moving, there is no way for the
   Pi to send STOP until it reconnects — so stop on our own rather than
   keep running blind. A physical E-stop on motor power is still the
   real backstop; this is a software safety net, not a replacement.
   --------------------------------------------------------------------- */
void checkWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    if (currentState != STATE_IDLE) {
      beltMotor.setSpeed(0);
      singulatorMotor.setSpeed(0);
      digitalWrite(PIN_LED_MOSFET, LOW);
      currentState = STATE_IDLE;
      Serial.println(F("WiFi lost while active — stopped for safety."));
    }
    WiFi.reconnect();
  }
}

/* =====================================================================
   Arduino entry points
   ===================================================================== */
void setup() {
  pinMode(PIN_LED_MOSFET, OUTPUT);
  digitalWrite(PIN_LED_MOSFET, LOW);

  // If the belt runs backward once tested, uncomment the next line
  // instead of rewiring: beltMotor.setPinsInverted(true, false, false);
  beltMotor.setSpeed(0);
  singulatorMotor.setSpeed(0);

  Serial.begin(115200); // debug output only now — the command channel is WiFi
  Serial.println();
  Serial.print(F("Connecting to WiFi: "));
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.setSleepMode(WIFI_NONE_SLEEP); // reduces (not eliminates) WiFi-induced
                                       // timing jitter in the stepper loop
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  Serial.print(F("WiFi connected. IP address: "));
  Serial.println(WiFi.localIP());

  if (MDNS.begin(MDNS_HOSTNAME)) {
    Serial.print(F("mDNS responder started — the Pi can connect to "));
    Serial.print(MDNS_HOSTNAME);
    Serial.println(F(".local"));
  } else {
    Serial.println(F("mDNS failed to start — connect using the IP address above instead."));
  }

  tcpServer.begin();
  Serial.print(F("TCP server listening on port "));
  Serial.println(TCP_PORT);
}

void loop() {
  checkWiFi();
  MDNS.update();
  pollClient();

  if (currentState == STATE_RUNNING || currentState == STATE_CALIBRATION) {
    beltMotor.runSpeed();       // non-blocking — call every loop for smooth, even steps
    singulatorMotor.runSpeed();
  }

  serviceTest(); // runs independently of RUN/STOP, only while a test is active
}
