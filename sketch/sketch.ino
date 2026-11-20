// ANTI MICROSLEEP - MCU side (STM32U585)
//
// Receives the alarm state from the Python app via Bridge and drives:
//   - Speaker / buzzer on SPEAKER_PIN (through an amplifier or transistor)
//   - RGB LED4 (status colour)
//   - 8x13 LED matrix (flashes on ALARM)
//
// LED4 colours:
//   green  solid  = OK, driver awake
//   cyan   blink  = calibrating (look straight, eyes open)
//   blue   blink  = no camera
//   blue   solid  = no face visible
//   purple solid  = too dark for the camera
//   yellow (no sound)               = WARNING
//   red flashing + 988 Hz tone      = ALARM (USB speaker plays sounds/beep.wav too)
//   white  blink  = no message from Python for 3 s (app not running)
//
// GPS (NEO-6M / NEO-M8N, NMEA at 9600 baud) on Serial1:
//   GPS TX -> D0 (RX), GPS RX -> D1 (TX, optional), VCC -> 3.3V, GND -> GND
//   Once per second the position is pushed to Python: Bridge.notify("gps", ...)

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// ---------------------------------------------------------------- GPS (NMEA)
const unsigned long GPS_BAUD = 9600;
const unsigned long GPS_SEND_MS = 1000;
const unsigned long GPS_SILENT_MS = 3000;    // no NMEA bytes this long = no module

struct GpsFix {
  bool valid = false;          // RMC status 'A'
  long latE6 = 0, lonE6 = 0;   // micro-degrees (int: no float precision loss over Bridge)
  long speedX10 = 0;           // km/h x10
  long courseX10 = -1;         // degrees x10, -1 = unknown
  int sats = 0;
  long hdopX10 = 0;
} gps;
unsigned long gpsLastByte = 0, gpsLastSend = 0;
char nmea[100];
int nmeaLen = 0;

// "ddmm.mmmm" + hemisphere -> micro-degrees
long nmeaToE6(const char *v, char hemi, int degDigits) {
  if (!v || !*v) return 0;
  long deg = 0;
  for (int i = 0; i < degDigits; i++) deg = deg * 10 + (v[i] - '0');
  double minutes = atof(v + degDigits);
  long e6 = deg * 1000000L + (long)(minutes / 60.0 * 1000000.0 + 0.5);
  return (hemi == 'S' || hemi == 'W') ? -e6 : e6;
}

bool nmeaChecksumOk(const char *s) {
  const char *star = strchr(s, '*');
  if (!star || star - s < 2) return false;
  uint8_t sum = 0;
  for (const char *p = s + 1; p < star; p++) sum ^= (uint8_t)*p;
  return sum == (uint8_t)strtol(star + 1, nullptr, 16);
}

void parseNmea(char *s) {
  if (s[0] != '$' || !nmeaChecksumOk(s)) return;
  *strchr(s, '*') = 0;
  char *f[20];
  int n = 0;
  for (char *p = s; n < 20; ) {          // split on ',' keeping empty fields
    f[n++] = p;
    char *c = strchr(p, ',');
    if (!c) break;
    *c = 0; p = c + 1;
  }
  const char *type = f[0] + 3;            // skip "$GP" / "$GN" / "$GL" / "$BD"
  if (strcmp(type, "RMC") == 0 && n >= 9) {
    if (f[2][0] == 'A' && f[3][0] && f[5][0]) {
      gps.valid = true;
      gps.latE6 = nmeaToE6(f[3], f[4][0], 2);
      gps.lonE6 = nmeaToE6(f[5], f[6][0], 3);
      gps.speedX10 = (long)(atof(f[7]) * 1.852 * 10.0 + 0.5);   // knots -> km/h
      gps.courseX10 = f[8][0] ? (long)(atof(f[8]) * 10.0 + 0.5) : -1;
    } else if (f[2][0] == 'V' && gps.sats < 3) {
      gps.valid = false;
    }
  } else if (strcmp(type, "GGA") == 0 && n >= 9) {
    gps.sats = atoi(f[7]);
    gps.hdopX10 = (long)(atof(f[8]) * 10.0 + 0.5);
    if (f[6][0] >= '1' && f[2][0] && f[4][0]) {
      gps.valid = true;
      gps.latE6 = nmeaToE6(f[2], f[3][0], 2);
      gps.lonE6 = nmeaToE6(f[4], f[5][0], 3);
    }
  } else if (strcmp(type, "GLL") == 0 && n >= 7) {
    if (f[6][0] == 'A' && f[1][0] && f[3][0]) {
      gps.valid = true;
      gps.latE6 = nmeaToE6(f[1], f[2][0], 2);
      gps.lonE6 = nmeaToE6(f[3], f[4][0], 3);
    }
  }
}

void gpsPoll(unsigned long now) {
  while (Serial1.available()) {
    char c = Serial1.read();
    gpsLastByte = now;
    if (c == '$') nmeaLen = 0;
    if (c == '\r' || c == '\n') {
      if (nmeaLen > 6) { nmea[nmeaLen] = 0; parseNmea(nmea); }
      nmeaLen = 0;
    } else if (nmeaLen < (int)sizeof(nmea) - 1) {
      nmea[nmeaLen++] = c;
    }
  }
  if (now - gpsLastSend >= GPS_SEND_MS) {
    gpsLastSend = now;
    // state: -1 = no module talking, 0 = module but no fix yet, 1 = fix
    int state = (gpsLastByte == 0 || now - gpsLastByte > GPS_SILENT_MS) ? -1 : (gps.valid ? 1 : 0);
    Bridge.notify("gps", state, gps.latE6, gps.lonE6, gps.speedX10, gps.courseX10,
                  gps.sats, gps.hdopX10);
  }
}

const int SPEAKER_PIN = 9;            // optional buzzer / amp (e.g. PAM8403)
const unsigned int ALARM_HZ = 988;    // matches sounds/beep.wav
const unsigned long LINK_TIMEOUT_MS = 3000;

enum { ST_OK, ST_CALIBRATING, ST_NO_CAMERA, ST_NO_FACE, ST_LOW_LIGHT };

volatile int g_level = 0;
volatile int g_status = ST_NO_CAMERA;
volatile unsigned long g_lastMsg = 0;

Arduino_LED_Matrix matrix;
const uint32_t MATRIX_FULL[4] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

// Called from Python: Bridge.notify("set_state", level, status)
void set_state(int level, int status) {
  g_level = constrain(level, 0, 2);
  g_status = status;
  g_lastMsg = millis();
}

void led4(bool r, bool g, bool b) {          // LED4 is active LOW
  digitalWrite(LED4_R, r ? LOW : HIGH);
  digitalWrite(LED4_G, g ? LOW : HIGH);
  digitalWrite(LED4_B, b ? LOW : HIGH);
}

unsigned int currentTone = 0;
void setTone(unsigned int freq) {
  if (freq == currentTone) return;
  if (freq == 0) noTone(SPEAKER_PIN);
  else tone(SPEAKER_PIN, freq);
  currentTone = freq;
}

int matrixState = -1;
void setMatrix(bool on) {
  if ((int)on == matrixState) return;
  if (on) matrix.loadFrame(MATRIX_FULL);
  else matrix.clear();
  matrixState = on;
}

void setup() {
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  pinMode(SPEAKER_PIN, OUTPUT);
  led4(false, false, false);

  matrix.begin();
  matrix.clear();

  Bridge.begin();
  Bridge.provide("set_state", set_state);

  Serial1.begin(GPS_BAUD);
}

void loop() {
  unsigned long now = millis();
  gpsPoll(now);
  unsigned long last = g_lastMsg;
  bool linkOk = (last != 0) && (now - last < LINK_TIMEOUT_MS);
  bool blink = (now / 500) % 2;

  if (!linkOk) {                      // Python app not talking to us
    led4(blink, blink, blink);
    setTone(0);
    setMatrix(false);
  } else if (g_level == 2) {          // ALARM: steady 988 Hz, same tone as beep.wav
    bool phase = (now / 200) % 2;
    led4(phase, false, false);
    setTone(ALARM_HZ);
    setMatrix(phase);
  } else if (g_level == 1) {          // WARNING: light only, no sound
    led4(true, true, false);
    setTone(0);
    setMatrix(false);
  } else {                            // OK - show status
    setTone(0);
    setMatrix(false);
    switch (g_status) {
      case ST_OK:          led4(false, true, false); break;
      case ST_CALIBRATING: led4(false, blink, blink); break;
      case ST_NO_CAMERA:   led4(false, false, blink); break;
      case ST_NO_FACE:     led4(false, false, true); break;
      case ST_LOW_LIGHT:   led4(true, false, true); break;
      default:             led4(false, false, false); break;
    }
  }
  delay(2);
}
