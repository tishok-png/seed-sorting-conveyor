/* =====================================================================
   AUTOMATED SEED SORTING SYSTEM
   ESP8266 (NodeMCU) actuation firmware — motors + LED only
   =====================================================================

   ROLE IN THE SYSTEM
   -------------------
   The Raspberry Pi does camera capture, seed classification, AND the
   reject servo + its FIFO timing. This firmware's job is exactly:
   the conveyor belt motor, the singulation motor, and the indicator LED.

   TRANSPORT  (changed from USB serial to TCP/WiFi)
   -------------------
   The Pi connects as a TCP CLIENT to port 8266 on this board.
   This firmware exposes the board at:

       seedsorter.local   (mDNS hostname)   port 8266

   which matches ESP_HOST = "seedsorter.local" and ESP_PORT = 8266
   in sorter_pi_agent.py. USB serial is retained for local debug only
   (you can type the same commands into a serial monitor and you will
   see every state echo there too). Do NOT drive both links at once.

   LIBRARIES REQUIRED
   -------------------------------------------------------------------
     AccelStepper  (by Mike McCauley)    both stepper motors
     ESP8266WiFi   (bundled w/ core)     TCP transport
     ESP8266mDNS   (bundled w/ core)     seedsorter.local

   WIRE PROTOCOL  (unchanged; same as the old serial version)
   -------------------------------------------------------------------
     Every command line has the form   PAYLOAD*CK\n
     where CK is a 2-digit uppercase hex XOR checksum of every byte in
     PAYLOAD. A line that fails the checksum is answered "ERR CHECKSUM".

     RUN            -> start belt + singulator + LED
     STOP           -> stop everything. Always allowed, from any state.
     CALIBRATE      -> run belt + singulator at level 1 for alignment.
     SPD1 / SPD2 / SPD3 -> set the speed level directly.
     SPD+ / SPD-    -> step the speed level up/down by one.
     TESTBELT       -> IDLE only: spin ONLY belt for 3 s.
     TESTSING       -> IDLE only: spin ONLY singulator for 3 s.
     TESTLED        -> IDLE only: flash the LED for 3 s.

   Responses (READY / STATE ... / TEST ... / ERR ...) are plain lines
   with no checksum — the Pi's read_lines() does not verify them, and
   the Pi's _wait_ready() looks for the literal token "READY".

   MACHINE STATE
   -------------------------------------------------------------------
   Every state transition echoes "STATE IDLE" / "STATE RUNNING" /
   "STATE CALIBRATION". No FAULT state on this board anymore — FAULT
   is a Pi-level concept (reject FIFO overflow).

   POST-SWAP NOTES (belt on D2, singulator on D5/D6)
   -------------------------------------------------------------------
   * Belt DIR is hardwired to 3V3 — direction cannot be changed from
     firmware. Swap a motor coil pair at the driver to reverse.
   * Singulator DIR is now on D6 (was hardwired HIGH). Inverted below
     via setPinsInverted(true,false,false) to preserve the physical
     direction; flip that boolean if it spins the wrong way.
   * SINGULATION_STEPS_PER_SEC is normalized by STEPS_PER_REV / 200 so
     the nominal values stay meaningful regardless of microstepping.
   ===================================================================== */

#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <AccelStepper.h>

/* =====================================================================
   USER CONFIGURATION — EDIT THE TWO WIFI STRINGS
   ===================================================================== */

// *** EDIT THESE ***
const char* WIFI_SSID     = "G";
const char* WIFI_PASS     = "66666622";

// Hostname this board advertises via mDNS. "seedsorter" makes the board
// reachable at seedsorter.local — which is what the Pi's ESP_HOST is set to.
const char* MDNS_HOSTNAME = "seedsorter";

// TCP port the Pi connects to. Matches ESP_PORT in sorter_pi_agent.py.
const uint16_t TCP_PORT   = 8266;

// ---- Pins (as wired, after the motor swap) ----
#define PIN_M1_STEP       D2   // Belt STEP — DIR hardwired to 3V3
#define PIN_M1_DIR_UNUSED D0   // Dummy DIR for belt; not connected. No effect.
#define PIN_M2_STEP       D5   // Singulator STEP
#define PIN_M2_DIR        D6   // Singulator DIR (was hardwired 3V3 before swap)
#define PIN_LED_MOSFET    D7   // MOSFET gate for the indicator LED

// ---- Serial: local debug only now, NOT the Pi link ----
#define PI_BAUD_RATE      115200

// ---- Conveyor mechanics ----
const float ROLLER_DIAMETER_MM       = 40.0;
const float DRIVE_PULLEY_DIAMETER_MM = 90.0;
// *** MEASURE THIS *** Real motor/gearbox output pulley. 8.0 is a
// reasonable placeholder; lower = faster belt. Raise
// BELT_SPEED_MULTIPLIER instead if you don't want to touch this.
const float MOTOR_PULLEY_DIAMETER_MM = 8.0;
const float GEARBOX_RATIO            = 3.0;

// MUST match your DRV8825 MS1/MS2/MS3 jumpers:
//   full=200, 1/2=400, 1/4=800, 1/8=1600, 1/16=3200, 1/32=6400
const int   STEPS_PER_REV            = 200;

// Belt speed targets per level (cm/s). Keep in sync with
// BELT_SPEED_CM_S_BY_LEVEL in sorter_pi_agent.py.
const float BELT_SPEED_CM_S[3] = { 6.5, 12.0, 16.0 };
const float BELT_SPEED_MULTIPLIER = 1.5;

// Singulator step rate per level, expressed as full-step-equivalent
// steps/sec. In applySpeedLevel() these are scaled by
// (STEPS_PER_REV / 200.0) so microstepping doesn't slow the singulator.
const float SINGULATION_STEPS_PER_SEC[3] = { 150.0, 250.0, 350.0 };
// Extra fudge factor on top. Raise if the singulator still feels weak.
const float SINGULATION_SPEED_MULTIPLIER = 1.0;

/* =====================================================================
   Logic — no edits needed below for normal tuning
   ===================================================================== */

enum SystemState { STATE_IDLE, STATE_RUNNING, STATE_CALIBRATION };
SystemState currentState = STATE_IDLE;

enum TestKind { TEST_NONE, TEST_BELT, TEST_SINGULATOR, TEST_LED };
TestKind activeTest = TEST_NONE;
unsigned long testEndAt = 0;
const unsigned long TEST_DURATION_MS = 3000;
const float TEST_STEPS_PER_SEC = 200.0;

int speedLevel = 1; // 1..3

AccelStepper beltMotor(AccelStepper::DRIVER, PIN_M1_STEP, PIN_M1_DIR_UNUSED);
AccelStepper singulatorMotor(AccelStepper::DRIVER, PIN_M2_STEP, PIN_M2_DIR);

// ---- TCP server + client ----
WiFiServer server(TCP_PORT);
WiFiClient client;

// ---- Line buffers, one per input source so partial commands from the
//      serial monitor can never get spliced with TCP bytes ----
String serialLine = "";
String tcpLine    = "";

// ---- Emit one line to BOTH the local serial monitor and the Pi ----
void emit(const __FlashStringHelper* msg) {
  Serial.println(msg);
  if (client && client.connected()) client.println(msg);
}

void emit(const String& msg) {
  Serial.println(msg);
  if (client && client.connected()) client.println(msg);
}

// ---------- Motor helpers ----------

float beltSpeedToStepsPerSec(float beltSpeedCmS) {
  float rollerCirc_cm = (PI * ROLLER_DIAMETER_MM) / 10.0;
  float rollerRPS     = beltSpeedCmS / rollerCirc_cm;
  float pulleyRatio   = DRIVE_PULLEY_DIAMETER_MM / MOTOR_PULLEY_DIAMETER_MM;
  float motorRPS      = rollerRPS * pulleyRatio * GEARBOX_RATIO;
  return motorRPS * STEPS_PER_REV;
}

void applySpeedLevel() {
  int i = speedLevel - 1;

  float beltStepsPerSec = beltSpeedToStepsPerSec(BELT_SPEED_CM_S[i])
                          * BELT_SPEED_MULTIPLIER;
  beltMotor.setMaxSpeed(beltStepsPerSec * 2.0);
  beltMotor.setSpeed(beltStepsPerSec);

  // Scale singulator for microstepping so physical speed is stable.
  float microScale       = (float)STEPS_PER_REV / 200.0;
  float singStepsPerSec  = SINGULATION_STEPS_PER_SEC[i]
                           * microScale
                           * SINGULATION_SPEED_MULTIPLIER;
  singulatorMotor.setMaxSpeed(singStepsPerSec * 2.0);
  singulatorMotor.setSpeed(singStepsPerSec);
}

void startRunning() {
  currentState = STATE_RUNNING;
  applySpeedLevel();
  digitalWrite(PIN_LED_MOSFET, HIGH);
  emit(F("STATE RUNNING"));
}

void stopRunning() {
  currentState = STATE_IDLE;
  beltMotor.setSpeed(0);
  singulatorMotor.setSpeed(0);
  digitalWrite(PIN_LED_MOSFET, LOW);
  emit(F("STATE IDLE"));
}

void startCalibration() {
  currentState = STATE_CALIBRATION;
  speedLevel = 1;
  applySpeedLevel();
  digitalWrite(PIN_LED_MOSFET, HIGH);
  emit(F("STATE CALIBRATION"));
}

void startTest(TestKind kind) {
  if (currentState != STATE_IDLE) {
    emit(F("ERR STOP_FIRST"));
    return;
  }
  activeTest = kind;
  testEndAt  = millis() + TEST_DURATION_MS;
  // Same microstepping-aware scaling for the singulator test.
  float microScale = (float)STEPS_PER_REV / 200.0;
  switch (kind) {
    case TEST_BELT:
      beltMotor.setMaxSpeed(TEST_STEPS_PER_SEC * 1.2);
      beltMotor.setSpeed(TEST_STEPS_PER_SEC);
      emit(F("TEST BELT_STARTED"));
      break;
    case TEST_SINGULATOR:
      singulatorMotor.setMaxSpeed(TEST_STEPS_PER_SEC * microScale * 1.2);
      singulatorMotor.setSpeed(TEST_STEPS_PER_SEC * microScale);
      emit(F("TEST SINGULATOR_STARTED"));
      break;
    case TEST_LED:
      digitalWrite(PIN_LED_MOSFET, HIGH);
      emit(F("TEST LED_STARTED"));
      break;
    default:
      break;
  }
}

void serviceTest() {
  if (activeTest == TEST_NONE) return;
  if (activeTest == TEST_BELT)       beltMotor.runSpeed();
  if (activeTest == TEST_SINGULATOR) singulatorMotor.runSpeed();
  if ((long)(millis() - testEndAt) >= 0) {
    beltMotor.setSpeed(0);
    singulatorMotor.setSpeed(0);
    digitalWrite(PIN_LED_MOSFET, LOW);
    activeTest = TEST_NONE;
    emit(F("TEST DONE"));
  }
}

// ---------- Protocol ----------

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

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  if (!checkAndStrip(cmd)) {
    emit(F("ERR CHECKSUM"));
    return;
  }

  bool liveApply = (currentState == STATE_RUNNING || currentState == STATE_CALIBRATION);

  if      (cmd == "RUN")       startRunning();
  else if (cmd == "STOP")      stopRunning();
  else if (cmd == "CALIBRATE") startCalibration();
  else if (cmd == "SPD1") { speedLevel = 1; if (liveApply) applySpeedLevel(); }
  else if (cmd == "SPD2") { speedLevel = 2; if (liveApply) applySpeedLevel(); }
  else if (cmd == "SPD3") { speedLevel = 3; if (liveApply) applySpeedLevel(); }
  else if (cmd == "SPD+") { if (speedLevel < 3) speedLevel++; if (liveApply) applySpeedLevel(); }
  else if (cmd == "SPD-") { if (speedLevel > 1) speedLevel--; if (liveApply) applySpeedLevel(); }
  else if (cmd == "TESTBELT") startTest(TEST_BELT);
  else if (cmd == "TESTSING") startTest(TEST_SINGULATOR);
  else if (cmd == "TESTLED")  startTest(TEST_LED);
  else emit(F("ERR UNKNOWN_CMD"));
}

// ---------- Link plumbing ----------

void acceptClient() {
  if (client && client.connected()) return;
  if (client) client.stop();          // clean up a dead handle
  WiFiClient newClient = server.available();
  if (newClient) {
    client = newClient;
    client.setNoDelay(true);          // state echoes go out immediately
    tcpLine = "";                     // never splice with an old session
    Serial.println(F("TCP client connected"));
    client.println(F("READY"));       // Pi's _wait_ready() looks for this
  }
}

void pollLinks() {
  // --- Pi over TCP ---
  if (client && client.connected()) {
    while (client.available() > 0) {
      char c = client.read();
      if (c == '\n') {
        handleCommand(tcpLine);
        tcpLine = "";
      } else if (c != '\r') {
        if (tcpLine.length() < 64) tcpLine += c;
      }
    }
  }
  // --- local serial debug ---
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      handleCommand(serialLine);
      serialLine = "";
    } else if (c != '\r') {
      if (serialLine.length() < 64) serialLine += c;
    }
  }
}

// ---------- setup / loop ----------

void setup() {
  pinMode(PIN_LED_MOSFET, OUTPUT);
  digitalWrite(PIN_LED_MOSFET, LOW);

  pinMode(PIN_M1_STEP, OUTPUT);
  pinMode(PIN_M2_STEP, OUTPUT);
  pinMode(PIN_M2_DIR,  OUTPUT);

  // Singulator's DIR used to be hardwired HIGH; it's now driven LOW by
  // AccelStepper. Invert logical direction to restore the original
  // physical direction. Flip to false if it spins the wrong way.
  singulatorMotor.setPinsInverted(true, false, false);

  beltMotor.setSpeed(0);
  singulatorMotor.setSpeed(0);

  // DRV8825 needs >= 1.9 us step pulses. 3 gives margin.
  beltMotor.setMinPulseWidth(3);
  singulatorMotor.setMinPulseWidth(3);

  Serial.begin(PI_BAUD_RATE);
  Serial.println(F("READY (local)"));

  // ---- WiFi ----
  WiFi.persistent(false);               // don't wear out flash on every boot
  WiFi.mode(WIFI_STA);
  WiFi.hostname(MDNS_HOSTNAME);         // DHCP hostname -> seedsorter
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print(F("Connecting to WiFi"));
  unsigned long deadline = millis() + 20000;
  while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print(F("WiFi OK, IP: "));
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(F("WiFi not connected yet — will keep trying. "
                     "If you need control NOW, reset the board."));
  }
}

void loop() {
  // Start the TCP server + mDNS exactly once, after WiFi is actually up.
  static bool netReady = false;
  if (!netReady && WiFi.status() == WL_CONNECTED) {
    netReady = true;
    server.begin();
    server.setNoDelay(true);
    if (MDNS.begin(MDNS_HOSTNAME)) {
      MDNS.addService("seed_sorter", "tcp", TCP_PORT);
    }
    Serial.print(F("TCP listening on "));
    Serial.print(WiFi.localIP());
    Serial.print(F(":"));
    Serial.println(TCP_PORT);
    Serial.print(F("mDNS hostname: "));
    Serial.print(MDNS_HOSTNAME);
    Serial.println(F(".local"));
  }

  // Safety net: if we were connected and WiFi has been gone for >30 s,
  // restart — mid-run reconnects are flakier than a fresh boot.
  static unsigned long wifiLostAt = 0;
  if (WiFi.status() == WL_CONNECTED) {
    wifiLostAt = 0;
  } else if (netReady) {
    if (!wifiLostAt) wifiLostAt = millis();
    else if (millis() - wifiLostAt > 30000UL) {
      Serial.println(F("WiFi lost >30 s — restarting board"));
      ESP.restart();
    }
  }

  acceptClient();
  pollLinks();

  if (currentState == STATE_RUNNING || currentState == STATE_CALIBRATION) {
    beltMotor.runSpeed();
    singulatorMotor.runSpeed();
  }
  serviceTest();

  MDNS.update();
}
