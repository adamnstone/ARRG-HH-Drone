// Bench test for the v3 headband PCB (not the headband firmware, see firmware/esp32s3/headband.ino): buzz motor M3
// (ULN2803A I3 <- GPIO 14). Open this folder in the Arduino IDE, or from the repo root:
//   arduino-cli compile --upload -p /dev/ttyACM0 --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,\
//     PartitionScheme=huge_app,FlashSize=4M,PSRAM=opi,CPUFreq=240,UploadSpeed=921600" firmware/bench/m3_buzz
// All eight ULN inputs (GPIO 12 13 14 15 16 17 18 21) are driven LOW first so none floats; only GPIO 14 is driven,
// at the firmware's duty cap 190/255 (~3 V average across a 3 V motor after the ULN's ~1 V drop), 200 Hz LEDC.
// Nothing moves on boot (opening the USB port resets the board). Commands from the Serial Monitor:
//   c = 10 s continuous     b = 3 pulses of 0.4 s
// Detect LED (GPIO 2) is on while the motor runs and blinks once a second while idle.
//
// Arduino IDE settings this was built and flashed with (Tools menu, esp32 core 3.3.7):
//   Board: "ESP32S3 Dev Module"            Port: /dev/ttyACM0 (the S3's native USB, "USB JTAG/serial debug unit")
//   USB CDC On Boot: Enabled               USB Mode: Hardware CDC and JTAG
//   Flash Size: 4MB (32Mb)                 Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS)
//   PSRAM: OPI PSRAM                       CPU Frequency: 240MHz (WiFi)       Upload Speed: 921600
// Serial Monitor: any baud (native USB ignores it); the line ending setting does not matter.
// Wiring on the v3 PCB: GPIO 14 -> ULN2803A pin 3 (I3); O3 (pin 16) -> M3 connector pin 2; M3 pin 1 = +5 V
// (PowerBoost 5V); ULN pin 10 (COM) = +5 V, pin 9 = GND.
#include <Arduino.h>

static const int PIN_M3 = 14;
static const int MOTOR_PINS[8] = {12, 13, 14, 15, 16, 17, 18, 21};   // M1 .. M8 (ULN I1 .. I8)
static const int PIN_LED = 2;
static const uint32_t DUTY = 190;      // of 255; 255 = full 5 V, more than a 3 V coin motor wants

static void buzz(int n) {
    for (int i = 0; i < n; i++) {
        Serial.printf("# m3 buzz %d/%d on (GPIO %d, duty %lu/255)\n", i + 1, n, PIN_M3, (unsigned long)DUTY);
        digitalWrite(PIN_LED, HIGH);
        ledcWrite(PIN_M3, DUTY);
        delay(400);
        ledcWrite(PIN_M3, 0);
        digitalWrite(PIN_LED, LOW);
        Serial.println("# m3 buzz off");
        delay(300);
    }
}

static void continuous(uint32_t ms) {
    Serial.printf("# m3 continuous on (GPIO %d, duty %lu/255, %lu ms)\n", PIN_M3, (unsigned long)DUTY, (unsigned long)ms);
    digitalWrite(PIN_LED, HIGH);
    uint32_t t0 = millis();
    ledcWrite(PIN_M3, DUTY);
    while (millis() - t0 < ms) delay(10);
    ledcWrite(PIN_M3, 0);
    digitalWrite(PIN_LED, LOW);
    Serial.printf("# m3 off after %lu ms\n", (unsigned long)(millis() - t0));
}

void setup() {
    for (int p : MOTOR_PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
    pinMode(PIN_LED, OUTPUT); digitalWrite(PIN_LED, LOW);
    ledcAttach(PIN_M3, 200, 8);        // Arduino-ESP32 3.x API: pin, frequency (Hz), resolution (bits)
    ledcWrite(PIN_M3, 0);
    Serial.begin(115200);
}

void loop() {
    static uint32_t t_banner = 0;
    if (millis() - t_banner >= 1000) {                  // so a monitor that just opened the port sees it is ready
        t_banner = millis();
        Serial.println("# m3_buzz ready: 'c' = 10 s continuous, 'b' = 3 pulses");
    }
    while (Serial.available() > 0) {
        int ch = Serial.read();
        if (ch == 'b') buzz(3);
        else if (ch == 'c') continuous(10000);
    }
    digitalWrite(PIN_LED, (millis() / 1000) % 2 ? HIGH : LOW);
    delay(10);
}
