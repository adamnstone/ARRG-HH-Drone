// Pin map of the headband (ESP32-S3-DevKitC-1, GPIO numbers). Single source of truth: headband.ino includes this
// file and README "Device bring-up" copies the tables below. Change a pin here and nowhere else.
//
// ---- Microphones: Adafruit PDM MEMS breakouts (ST MP34DT01-M, 3.3 V), I2S0 PDM RX, 48 kHz, PDM clock 3.072 MHz.
// One shared clock, three data lines; each line carries a stereo pair selected by the breakout's SEL strap.
// The full map has five positions; this build populates FOUR (B, L, R, T): FRONT is not fitted.
//
//   mic        position          CLK      DAT               SEL    PDM slot        DMA word   ring / WAV channel
//   ---------  ----------------  -------  ----------------  -----  --------------  --------   ------------------
//   F  front   forehead          GPIO 4   GPIO 5  (DIN0)    GND    line 0, left    0          -- not populated on
//                                                                                                the 4-mic build;
//                                                                                                slot 0 ignored
//   B  back    back of the head  GPIO 4   GPIO 5  (DIN0)    3V3    line 0, right   1          0
//   L  left    left temple       GPIO 4   GPIO 6  (DIN1)    GND    line 1, left    2          1
//   R  right   right temple      GPIO 4   GPIO 6  (DIN1)    3V3    line 1, right   3          2
//   T  top     crown             GPIO 4   GPIO 7  (DIN2)    GND    line 2, left    4          3
//   all mics: 3V -> 3V3, GND -> GND. Line 2's right slot (DMA word 5) has no mic.
//   Channel order B, L, R, T = droneears/config.py MIC_NAMES = the order of Spectrum S[] passed to the core.
//   DMA word = assumed position inside the 6-word DMA frame (see CH_SLOT in headband.ino; verify by tap test).
//   Bench board of Oct 2026 (built for the 3-mic firmware): mics on DIN0 SEL GND, DIN0 SEL 3V3 and DIN1, motors on
//   M1-M3. Under this map the DIN0 SEL GND mic is the unpopulated F position (read, ignored): move it to DIN2 (T) or
//   wire the missing positions before trusting bearing / elevation; detection works with any subset.
//
// ---- Motors: 8 coin motors on a ULN2803A (low side, active-high inputs), one LEDC channel each, 200 Hz, 8-bit.
//   motor  ULN  GPIO  bearing        motor  ULN  GPIO  bearing
//   F      M1   12      0 (ahead)    B      M5   16    180 (behind)
//   FR     M2   13     45            BL     M6   17   -135
//   R      M3   14     90 (right)    L      M7   18    -90 (left)
//   BR     M4   15    135            FL     M8   21    -45
//   Order F, FR, R, BR, B, BL, L, FL = droneears/config.py MOTOR_ANGLES = haptic_weights() out[].
//   Duty cap MOTOR_DUTY_MAX / 255. HARD LIMIT: at most MOTOR_MAX_ON motors energised at any instant (the PowerBoost
//   supplies 1 A).
//
// ---- IMU: BNO085 on I2C (address 0x4A, pull-ups on the breakout): SCL GPIO 8, SDA GPIO 9, INT GPIO 10 (active
//   low), RST GPIO 11 (active low). Not used yet (USE_IMU 0 in the sketch).
// ---- Detect LED: GPIO 2, high = on.
// ---- Battery sense: GPIO 1 = ADC1 channel 0, through a 1:2 divider (V_batt = 2 x V_adc); 4.20 V full, 3.30 V empty.
//
// ---- Reserved, do not use: GPIO 0, 3, 45, 46 (strapping), 19/20 (native USB D-/D+), 43/44 (UART0 TX/RX, the USB-UART
//   bridge), 26-37 (flash / octal PSRAM). Free spares: 3, 47, 48 (the PowerBoost LBO pin is not wired).
#pragma once

// microphones (I2S0 PDM RX)
#define PIN_PDM_CLK    4      // -> CLK of all mics
#define PIN_PDM_DIN0   5      // <- F (SEL GND, left slot; not populated) + B (SEL 3V3, right slot)
#define PIN_PDM_DIN1   6      // <- L (SEL GND, left slot) + R (SEL 3V3, right slot)
#define PIN_PDM_DIN2   7      // <- T (SEL GND, left slot)

// motors (ULN2803A inputs), order F, FR, R, BR, B, BL, L, FL
#define PIN_MOT_F      12     // M1,    0 deg
#define PIN_MOT_FR     13     // M2,   45 deg
#define PIN_MOT_R      14     // M3,   90 deg
#define PIN_MOT_BR     15     // M4,  135 deg
#define PIN_MOT_B      16     // M5,  180 deg
#define PIN_MOT_BL     17     // M6, -135 deg
#define PIN_MOT_L      18     // M7,  -90 deg
#define PIN_MOT_FL     21     // M8,  -45 deg
#define MOTOR_PWM_HZ   200
#define MOTOR_PWM_BITS 8
#define MOTOR_DUTY_MAX 190    // of 255
#define MOTOR_MAX_ON   3      // never more than this many motors energised at once (1 A supply)

// IMU (BNO085, I2C)
#define PIN_IMU_SCL    8
#define PIN_IMU_SDA    9
#define PIN_IMU_INT    10     // active low
#define PIN_IMU_RST    11     // active low
#define IMU_I2C_ADDR   0x4A

// detect LED
#define PIN_LED_DETECT 2      // high = on

// battery sense (ADC1 channel 0)
#define PIN_BATT_SENSE 1
#define BATT_DIVIDER   2.0f   // V_batt = BATT_DIVIDER x V_adc
#define BATT_FULL_V    4.20f
#define BATT_EMPTY_V   3.30f
