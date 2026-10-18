// Headband acoustic drone detector -- ESP32-S3 firmware (Arduino-ESP32 3.x core, ESP-IDF 5.1-5.5 underneath)
//
// Hardware: the team's headband board (ESP32-S3; full pin map and wiring tables in pins.h, the single source of
// truth): PDM MEMS mics (ST MP34DT01-M class, 3.3 V, PDM clock 1-3.25 MHz) on one clock and three data lines, eight
// coin motors on a ULN2803A, BNO085 IMU (not used yet), detect LED, battery sense, powered from a PowerBoost (1 A).
// The DSP has FOUR mic channels, B (back), L (left), R (right) in the band plane and T on the crown above it, so the
// elevation is measured directly ("overhead" is no longer inferred from the planar fit). The headset calibration ('C',
// the dashboard's Calibrate tab) says which mics are connected: the DSP uses exactly those (detection from their mean,
// bearing from their pairs), and per frame the bearing keeps only the mics whose pairs are reliable (a weak, noisy or
// dead mic drops out on the fly; dsp_core.h robust_direction). Four reliable mics: azimuth + elevation; three: B, L,
// R the azimuth (elevation too when the drone is clearly above), other threes a mirror pair the upper-hemisphere
// prior may resolve; fewer: detection only. Channel 0 holds the back mic, or a front mic instead ('C...,F': it then
// sits at FRONT_POS, the mirror image of B, and every rule above holds with F for B). Any header MK1..MK5 can feed any
// channel. Mic geometry: droneears/config.py MIC_POS / FRONT_POS (x right / y forward / z up, placeholders to be
// measured on the band). Motors F, FR, R, BR, B, BL, L, FL at
// 0, 45, 90, 135, 180, -135, -90, -45 deg; duty capped at MOTOR_DUTY_MAX, AT MOST MOTOR_MAX_ON (3) ENERGISED AT ONCE.
//
//   function            | GPIO | use
//   --------------------+------+--------------------------------------------------------------------------------
//   PDM CLK             |   4  | clock to every mic (3.072 MHz)
//   PDM DIN0            |   5  | MK2 (SEL 3V3, DMA word 0; default B) + MK1 (SEL GND, word 1; unused by default)
//   PDM DIN1            |   6  | MK4 (SEL 3V3, word 2; default R) + MK3 (SEL GND, word 3; default L)
//   PDM DIN2            |   7  | MK5 (SEL GND, word 5; default T); word 4 (its SEL 3V3 phase) has no header
//   motors F .. FL      | 12 13 14 15 16 17 18 21 | ULN M1 .. M8; LEDC 200 Hz 8-bit (all eight of the S3's channels)
//   detect LED          |   2  | high = on (DevKitC header pin 40, R3 330 ohm): the patterns below; 'Q1' = always off
//   battery sense       |   1  | ADC1 ch0, V_batt = 2 x V_adc, read once per second -> batt_v in T and S lines
//   IMU BNO085          | 8-11 | pins defined, USE_IMU 0: no driver yet
//
// Detect LED (alert_task, sampled every ~20 ms): the FIRST row that applies wins.
//   state               | when                                                     | pattern
//   --------------------+----------------------------------------------------------+---------------------------------
//   stealth             | 'Q1' (dashboard Stealth button; NVS, survives a reboot)  | off, whatever else is going on
//                       | (the DevKitC's RGB LED is always off; its power LED and the PowerBoost's are hard-wired)
//   fault               | a task failed to start (tasks_failed), or no audio chunk | 10 Hz blink (50 ms on / 50 off)
//                       | read for LED_NO_AUDIO_MS (1 s; 3 s after boot)           |
//   calibration capture | a 'W' capture is being recorded (DSP paused: clap now)   | 2 Hz blink (250 ms on / 250 off)
//   overhead            | DETECTED and overhead                                    | ~8 Hz flicker (60 ms on / 60 off)
//   DETECTED            | the detector's persistence latch                         | solid on
//   manual haptics      | 'h' active (dashboard Bench > Motors)                     | long-short every 1 s (400 on,
//                       |                                                          | 150 off, 100 on)
//   motor test          | 'M' (blocks alert_task for 6.4 s: nothing above shows)   | long with each motor (500 ms),
//                       |                                                          | a 100 ms blip in the gap
//   evidence building   | not DETECTED, hits >= PERSIST_ON / 2 (35 of 94 frames)   | double blip (2 x 40 ms) every 1 s
//   low battery         | 0.5 V < batt_v < LED_BATT_LOW_V (3.5 V); 0.5: no cell    | triple blip (3 x 40 ms) every 5 s
//   idle                | listening, nothing found                                 | 30 ms heartbeat every 3 s
//
//   SEL low  = "left"  slot: mic drives DAT on one CLK phase, high-Z on the other (IDF i2s_types.h:
//   SEL high = "right" slot: data on the other phase   I2S_PDM_SLOT_LEFT = SEL pulled down, _RIGHT = SEL pulled up).
//   Two mics with opposite SEL tri-state alternately on one DIN line, so each shared line carries a stereo pair.
//   Keep the CLK wire short (it fans out to four mics at 3 MHz, T on a longer run over the crown); route it away from
//   the motor wires. TODO(bench): with four loads, check the CLK edges at the farthest mic; add a 33-47 ohm series
//   resistor at GPIO 4 if it rings.
//
// Capture: one I2S port (I2S0 -- the only S3 port with PDM RX) in PDM RX mode with three data lines and one clock, so
// all channels are sampled by the same peripheral from the same clock edge and are sample-aligned by construction.
// (The old INMP441 design needed a second port slaved to the first plus a clap-calibrated port offset; that is gone.)
//
// Status: the 3-mic / 3-motor version of this sketch ran on the bench board (Oct 2026): every task up, Wi-Fi AP up,
// DSP real-time at 47 frames/s (15.3 of the 21.3 ms budget: spectra 4.6, log-mel 0.5, detector 3.1, bearing 7.1 ms;
// 0 DMA overflows), 39 KB internal heap free. With four mics and six full-length GCC-PHAT IFFTs the board measured
// spectra 6.2 + mel 0.6 + det 3.4 + bearing 14.3 = 24.6 ms per frame, over the 21.3 ms budget (~36 DMA overflows/s).
// The bearing now evaluates each pair's correlation only at the +-~25 physical lags over the masked bins (no IFFT)
// and runs the robust solver: estimated ~0.6 ms per frame with the drone's harmonic mask, ~2.2 ms with the full band
// mask (no drone), ~3 ms worst case (operation counts calibrated on the old measurement); total ~11-13 ms.
// TODO(bench): read the "# dsp ... total=" line ('S') and dma_ovf= to confirm. The 4th Spectrum, the 6-word chunk buffer and the 12-descriptor DMA ring take ~24 KB more internal RAM
// than the 3-mic build (the new bearing state is ~8 KB smaller than the old IFFT scratch): check free_heap= in the
// boot / 'S' lines with Wi-Fi up (keep > ~15 KB, or drop DMA_DESC_NUM to 8).
// Verify first on the bench, in this order:
//   1. Slot / SEL mapping: `python scripts/telemetry.py --levels` polls the board's 'L' command, which reports the
//      RMS / peak / DC of all six DMA words of lines 0-2 (not just the four the DSP reads). Tap each mic in turn
//      with a fingernail: exactly one word's peak must jump per mic, and the words with no mic stay near the
//      floor. The DMA word order (left slot before right within a line) is an assumption; the IDF guide says the
//      S3 receives the right slot first, so a swap inside a line is the likely surprise (then CH_SLOT = {0, 3, 2, 5}).
//      Set the words of the mics at the head's back / left / right / top with the calibration command 'C' (the
//      dashboard's Calibrate tab can find them by itself from three claps: front, right, left, using the 'W'
//      capture); it is kept in NVS, no reflash. CH_SLOT_DEFAULT below is only what a fresh board starts with.
//      Also confirm the chunk period printed at boot is ~21333 us (validates the 6-slot DMA frame assumption; ~25600
//      us would mean 5 slots).
//   2. DC offset / gain of the PDM path vs the training data. Gain does not matter (the classifier normalises each
//      window to its max), but a large DC offset would leak into the low bins; the S3 PDM RX has no hardware
//      high-pass filter (SOC_I2S_SUPPORTS_PDM_RX_HP_FILTER is not set for the S3), so check the mean of each channel.
//   3. That the 3.072 MHz clock gives clean audio: it is above the Adafruit guide's "1-3 MHz" but inside the
//      MP34DT01-M's 3.25 MHz max, and it comes from a fractional divider of the 160 MHz PLL (the S3 has no APLL).
//      Record a few seconds with scripts/record_device.py (below) and listen / look at the spectrum for tones or crackle.
//   Board checks: `python scripts/telemetry.py --motor-test` sends 'M': the board buzzes F, FR, R, BR, B, BL, L, FL
//   in turn (0.5 s each; hap_* in the status line show which one is meant to be on). Where each output actually sits
//   on the head (and which are fitted) is the motor calibration 'Y' (angles, NVS; the Calibrate tab maps them to the
//   band's holes); the detector pans between the fitted motors at those angles. Overhead plays a rotating triplet of
//   fitted motors (k, k+3, k+6 with all eight), never more than MOTOR_MAX_ON.
//   Per motor: the dashboard's Bench > Motors toggles send 'h' (each motor on / off, a strength slider = PWM duty,
//   steady or pulsed): find the lowest duty that still starts each motor; the detector pans with fractional
//   weights (a source at 22.5 deg gives F and FR 50 % each = duty 95). batt_v in the T / S lines matches a multimeter
//   on the cell to ~0.05 V (if not, fix BATT_DIVIDER in pins.h); TODO(bench): measure the supply current with the
//   overhead triplet at full duty plus Wi-Fi TX, it must stay under the PowerBoost's 1 A.
// Everything from the DSP core down is the same code that passes scripts/compare_core.py on a laptop.
//
// Host tools (serial protocol: see "serial protocol" below). `python scripts/telemetry.py` shows the live detector /
// bearing / elevation / haptic / battery state and logs every frame's T line to CSV. `python scripts/record_device.py
// --seconds 60 --out data/team/...` switches the board to raw mode ('R') and writes the four mic channels (B, L, R, T,
// exactly what the DSP sees) to a 48 kHz 16-bit 4-channel WAV, then returns it to telemetry mode. Raw streaming is
// ~384 KB/s, so it needs the S3's native USB port (USB CDC, "USB CDC On Boot: Enabled"); over the UART bridge at
// 921600 baud most chunks are dropped (the chunk-index gaps in the packets tell the host). TODO(bench): 384 KB/s is
// near what TinyUSB CDC full-speed sustains on some hosts; if dropped= in 'S' or the host's gap count climbs over
// USB, record fewer seconds at a time.
//
// Wi-Fi (USE_WIFI 1): the SAME protocol, byte for byte, over TCP, so the headband can be worn and walked around
// while a laptop shows it live. Default: the board is a soft access point "HEADBAND" (WPA2, password WIFI_AP_PASS,
// default "dronedetect"; change it in wifi_config.h), at 192.168.4.1. Join that network from the laptop, then
// connect to headband.local:5000 (mDNS) or 192.168.4.1:5000. To put the board on an existing network instead,
// copy wifi_config.h.example to wifi_config.h (gitignored) and set WIFI_STA_SSID / WIFI_STA_PASS: it joins that
// network (15 s timeout, then falls back to the AP) and is reachable as headband.local:5000 or at the IP printed in
// the "# wifi ..." line. Host side: every tool takes --tcp [HOST[:PORT]] (default headband.local:5000, falling back
// to 192.168.4.1:5000): `python scripts/telemetry.py --tcp`, `python scripts/record_device.py --tcp --seconds 60`,
// `python scripts/dashboard.py --tcp` (with no flag the dashboard tries USB, then headband.local:5000, then demo).
// One TCP client at a time; a new connection replaces the old one. On connect the board sends "# headband fw ...",
// "# wifi ...", "# pdm chunk_us=..." (once measured) and the H header; the client then sends 'T' / 'R' / 'S' exactly
// as over USB. USB and Wi-Fi each have their own mode (USB can stay in 'T' while Wi-Fi records raw, or vice versa).
// Raw audio over Wi-Fi is best effort: a stalled client loses whole packets (counted as wifi_dropped in 'S', and
// visible as chunk-index gaps on the host), never the DSP's time. USB stays the reliable path for the holdout
// recording. Power: Wi-Fi adds roughly 100 mA (more in TX bursts) on top of the mics/motors; wear it with a USB
// power bank, not a laptop cable, when walking around.
// Wi-Fi bench checks (in addition to 1-3 above):
//   4. DSP timing with Wi-Fi on: `# pdm chunk_us=` must still read ~21333 and the T-line frame counter must advance
//      at 46.9/s (scripts/telemetry.py --tcp shows fr/s) while a client streams raw; dropped= in 'S' must stay 0
//      (that counter is the dsp_task -> telemetry_task queue, i.e. the DSP side, not Wi-Fi).
//   5. Raw throughput over Wi-Fi: `record_device.py --tcp --seconds 60` and read the summary's dropped-chunk count
//      and the board's wifi_dropped (send 'S' over USB). ~384 KB/s is needed; AP mode at 1-3 m may manage it,
//      a crowded 2.4 GHz band will not. Compare to the same recording over USB.
//
// Libraries: ESP-DSP optional (replace dsp::fft_* with dsps_fft2r_fc32 for speed); TensorFlow Lite Micro via the
// "TensorFlowLite_ESP32" Arduino library or Espressif's esp-tflite-micro component (set USE_TFLM to 1). Wi-Fi uses
// only the core's WiFi / ESPmDNS libraries (Arduino-ESP32 3.x: WiFiServer / WiFiClient are NetworkServer /
// NetworkClient typedefs).
#define USE_TFLM 0
#define USE_WIFI 1          // 0: USB serial only, no Wi-Fi code compiled in (everything else unchanged)

// Core placement. Arduino-ESP32 runs the Wi-Fi driver task and the LwIP TCP/IP task on core 0 at high priority
// (~23 / ~18, far above anything here), so with Wi-Fi on, a dsp_task on core 0 would be preempted for every Wi-Fi
// burst. DSP is the one hard real-time task (the 64 ms DMA ring must be drained every chunk), so it moves to core 1,
// where it shares the core only with Arduino's idle loopTask. Everything that is not real-time goes to core 0 with
// Wi-Fi: telemetry_task (USB + TCP output), net_task (Wi-Fi bring-up, TCP accept), alert_task and ml_task.
// Caveat: TFLM inference (ml_task, every 500 ms) now shares core 0 with Wi-Fi; Wi-Fi bursts stretch an inference
// and an inference delays telemetry_task (lower priority) by its run time. Neither is real-time: ml only gates the
// detector with a 0.5 s cadence, and late T lines are just late. With USE_WIFI 0 the old layout (DSP on core 0,
// the rest on core 1) is equally fine; swap the pair on the bench if needed.
// TODO(bench): confirm CONFIG_ESP_WIFI_TASK_CORE_ID / CONFIG_LWIP_TCPIP_TASK_AFFINITY are core 0 in the installed
// core's sdkconfig, and that the TinyUSB "usbd" task (unpinned, high priority, native USB CDC) does not disturb DSP.
#define DSP_CORE 1
#define IO_CORE  0

#include <Arduino.h>
#include <atomic>
#include <driver/i2s_pdm.h>
#include <esp_heap_caps.h>  // the bulk buffers go to PSRAM (big_alloc below)
#include <Preferences.h>    // NVS: the headset calibration (mic map, motor angles) survives a reboot
#include "pins.h"           // every GPIO, and the wiring tables
#include "src/dsp_core.h"   // src/ holds symlinks to ../core: Arduino only compiles files inside the sketch folder
struct Shared;              // the Arduino preprocessor puts its auto-prototypes (sh_snapshot) before the definition
#if USE_TFLM
#include "model_data.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#endif
#if USE_WIFI
#include <WiFi.h>
#include <ESPmDNS.h>
#include <lwip/sockets.h>    // send / recv with MSG_DONTWAIT on the client's fd (see "TCP sink" below)
#include <errno.h>
#if __has_include("wifi_config.h")
#include "wifi_config.h"     // optional, gitignored: WIFI_AP_PASS, WIFI_STA_SSID / WIFI_STA_PASS (see .example)
#endif
#endif

// ---------------------------------------------------------------- Wi-Fi settings (overridable in wifi_config.h)
#ifndef WIFI_AP_SSID
#define WIFI_AP_SSID "HEADBAND"
#endif
#ifndef WIFI_AP_PASS
#define WIFI_AP_PASS "dronedetect"          // same default as wifi_config.h.example; WPA2 needs >= 8 characters
#endif
#ifndef WIFI_HOSTNAME
#define WIFI_HOSTNAME "headband"            // mDNS: headband.local
#endif
#ifndef WIFI_TCP_PORT
#define WIFI_TCP_PORT 5000
#endif
#ifndef WIFI_STA_TIMEOUT_MS
#define WIFI_STA_TIMEOUT_MS 15000           // station join timeout before falling back to the AP
#endif
#ifndef WIFI_STALL_MS
#define WIFI_STALL_MS 3000                  // a client that accepts no bytes for this long is disconnected
#endif
static_assert(sizeof(WIFI_AP_PASS) - 1 >= 8 && sizeof(WIFI_AP_PASS) - 1 <= 63, "WIFI_AP_PASS: WPA2 needs 8-63 chars");

// ---------------------------------------------------------------- pins: see pins.h
static_assert(N_MICS == 4 && N_MOTORS == 8, "this sketch reads 4 mics (B, L, R, T) and drives 8 motors (F .. FL)");
// IMU (BNO085 on PIN_IMU_SCL / PIN_IMU_SDA, INT / RST active low, I2C address IMU_I2C_ADDR): not used yet.
// TODO(imu): head-yaw compensation -- read the rotation vector and subtract head yaw from theta between DSP frames, so
// the haptic cue stays world-fixed while the wearer turns. Needs a BNO08x driver (e.g. SparkFun / Adafruit BNO08x).
#define USE_IMU 0

// ---------------------------------------------------------------- memory
// The S3 has 512 KB of SRAM; the Arduino core gives static data ~320 KB and Wi-Fi + lwIP + mDNS need ~70 KB of heap
// on top of the task stacks and the 24.5 KB I2S DMA ring. With every buffer static the first build left no heap at
// all (the Wi-Fi tasks and even telemetry_task failed to start, silently). So the bulk buffers that are only
// copied, not computed on -- the int16 sample rings, the raw packet pool and the TCP carry buffer (~57 KB) -- are
// allocated once at boot in the module's 8 MB octal PSRAM (flash.sh builds with PSRAM=opi; the core's sdkconfig
// also sends mallocs above 4 KB and the Wi-Fi/lwIP buffers there). The DSP-hot data (the four spectra, 32.8 KB,
// heap but internal; FFT tables, detector and bearing state, the chunk buffer, static) stays in internal RAM:
// streaming the spectra from PSRAM cost ~3 ms of the 21.3 ms frame. If PSRAM is missing or fails to initialise,
// big_alloc falls back to internal RAM and the boot line says so ("psram_bufs=0/3"); Wi-Fi will then most likely
// be out of memory again.
static int big_allocs_psram = 0, big_allocs_total = 0;
static void* big_alloc(size_t n) {
    big_allocs_total++;
    void* p = heap_caps_calloc(1, n, MALLOC_CAP_SPIRAM);
    if (p) { big_allocs_psram++; return p; }
    p = heap_caps_calloc(1, n, MALLOC_CAP_8BIT);
    if (!p) abort();                                     // nothing sensible to do without the buffers (console shows it)
    return p;
}

// ---------------------------------------------------------------- audio capture (PDM RX, I2S0, 3 data lines)
// Clock: sample rate FS = 48 kHz, PDM clock = FS x 64 = 3.072 MHz with I2S_PDM_DSR_8S (driver: bclk = rate x
// I2S_LL_PDM_BCK_FACTOR(64) x (DSR_16S ? 2 : 1)). DSR_16S would need FS x 128 = 6.144 MHz, above the MP34DT01-M's
// 3.25 MHz maximum, so it is not allowed. PDM RX on the S3 is 16-bit only (PDM -> PCM filter output).
//
// Slots: the S3 PDM RX maps (line n, slot) to TDM channel bits: line0 right = bit0, line0 left = bit1, line1 right =
// bit2, line1 left = bit3, line2 right = bit4, line2 left = bit5 (I2S_PDM_RX_LINEx_SLOT_*). The pin map uses line0
// both slots (MK1, MK2), line1 both (MK3, MK4) and line2 left (MK5). We enable all six slots of lines 0-2 (mask 0x3F,
// contiguous; line2 right is the one extra) so the DMA frame is 6 words and its layout does not depend on how the
// hardware treats a disabled channel in the middle of the mask. Words without a mic are read (and reported by 'L')
// and ignored.
// DMA layout, VERIFIED on the v3 board (2026-10-17, tap test with MK1..MK4 fitted): the RIGHT (SEL 3V3) slot comes
// first on each line, as the IDF guide says ("the right slot first in stereo mode"), lines in order:
//     word   0                 1                2                 3                4                 5
//     frame  line0 right (MK2) line0 left (MK1) line1 right (MK4) line1 left (MK3) line2 right (none) line2 left (MK5)
// The dashboard's slot-order switch ("right first", the default) is this; "left first" is the old assumption.
static const int PDM_SLOTS = 6;                       // int16 words per DMA frame (one per enabled slot)
// Channel map: ring channel (0 = B, 1 = L, 2 = R, 3 = T, the order the DSP expects) -> word index inside one DMA
// frame, or -1 = no mic there (the channel reads zeros and the DSP leaves it out). CH_SLOT_DEFAULT is the PCB map
// (MK2 back, MK3 left, MK4 right, MK5 top, right slot first); the headset calibration replaces it at runtime ('C'
// command, the dashboard's Calibrate tab; kept in NVS, see "calibration" below), so a remounted mic, a front mic in
// place of the back one, or a mic that is not connected needs no reflash. Rings, spectra and raw 'R' packets all go
// through CH_SLOT (channel 0's raw column is the front mic when the calibration says front).
static const int CH_SLOT_DEFAULT[N_MICS] = {
    0,   // B: MK2, line 0, SEL = 3V3 (the right slot, first)
    3,   // L: MK3, line 1, SEL = GND
    2,   // R: MK4, line 1, SEL = 3V3
    5,   // T: MK5, line 2, SEL = GND
};
static_assert(sizeof(CH_SLOT_DEFAULT) / sizeof(CH_SLOT_DEFAULT[0]) == N_MICS, "one CH_SLOT entry per mic");
static int CH_SLOT[N_MICS] = { 0, 3, 2, 5 };          // the live map: dsp_task only (set at start, and again
                                                      // between chunks when the calibration changes)
static i2s_chan_handle_t rx;
static const int CHUNK = HOP;                         // samples per channel per i2s_channel_read (21.3 ms)
// DMA: one descriptor holds at most 4092 bytes. The driver sizes a descriptor as dma_frame_num x 2 slots x 2 bytes
// (PDM RX always counts 2 active slots, even with extra lines enabled), so dma_frame_num = 768 gives 3072-byte
// descriptors = 256 six-slot frames (12 B each) = 5.3 ms. 12 descriptors = 36864 bytes = 64 ms of ring, 3 chunks of
// slack (8 descriptors would be only 42.7 ms with six slots, but would save 12 KB of internal heap).
// TODO(bench): the "2 slots" sizing rule is from the driver source for 1-2 lines; with 3 lines confirm the
// descriptor size (the boot chunk_us check still validates the frame size itself).
// i2s_channel_read is a byte stream across descriptors, so the frame alignment of the reads is what matters, and every
// read below is a whole number of 12-byte frames.
static const int DMA_DESC_NUM = 12, DMA_FRAME_NUM = 768;
static int16_t buf[CHUNK * PDM_SLOTS];                // 12 KB: one chunk, all six slots interleaved (internal RAM)
static int16_t (*ring)[NFFT] = nullptr;               // [N_MICS][NFFT] int16, PSRAM: last NFFT samples (B, L, R, T)
static int ring_pos = 0;
static uint32_t chunks_captured = 0;                  // dsp_task only: successful chunk reads since boot
static std::atomic<uint32_t> audio_ms{0};             // millis() of the last successful chunk read (the LED's fault)
static std::atomic<uint32_t> dma_ovf{0};              // I2S RX DMA overflows (the DSP fell behind the audio)
static bool IRAM_ATTR on_rx_ovf(i2s_chan_handle_t, i2s_event_data_t*, void*) {
    dma_ovf.fetch_add(1, std::memory_order_relaxed);
    return false;
}
// bench profile: micros spent per frame in each DSP stage (dsp_task adds, the 'S' handler averages and resets)
static std::atomic<uint32_t> prof_n{0}, prof_spec{0}, prof_mel{0}, prof_det{0}, prof_bear{0}, prof_wait{0};
// Per-word levels of the last chunk, full scale 1.0 (dsp_task only; the 'L' command reports max-holds of them):
static float lvl_rms[PDM_SLOTS], lvl_peak[PDM_SLOTS], lvl_dc[PDM_SLOTS];

static void i2s_setup() {
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);   // PDM RX exists on I2S0 only
    cc.dma_desc_num = DMA_DESC_NUM; cc.dma_frame_num = DMA_FRAME_NUM;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));
    i2s_pdm_rx_config_t pc = {
        .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(FS),                                 // DSR_8S, 160 MHz PLL source
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = { .clk = (gpio_num_t)PIN_PDM_CLK, .invert_flags = { .clk_inv = false } },
    };
    pc.clk_cfg.dn_sample_mode = I2S_PDM_DSR_8S;       // explicit: PDM clock = FS x 64 = 3.072 MHz (see above)
    // Stereo + a mask above I2S_PDM_SLOT_BOTH keeps the mask as given (i2s_hal_pdm_set_rx_slot).
    pc.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)(I2S_PDM_RX_LINE0_SLOT_RIGHT | I2S_PDM_RX_LINE0_SLOT_LEFT |
                                                  I2S_PDM_RX_LINE1_SLOT_RIGHT | I2S_PDM_RX_LINE1_SLOT_LEFT |
                                                  I2S_PDM_RX_LINE2_SLOT_RIGHT | I2S_PDM_RX_LINE2_SLOT_LEFT);
    // dins[] is an anonymous union with din; set it outside the initialiser. Only lines whose mask bits are set are
    // routed by the driver, but mark the rest unused rather than leaving them at 0 (= GPIO 0, a strapping pin).
    static_assert(SOC_I2S_PDM_MAX_RX_LINES >= 3, "three PDM RX data lines");
    for (int i = 0; i < SOC_I2S_PDM_MAX_RX_LINES; i++) pc.gpio_cfg.dins[i] = I2S_GPIO_UNUSED;
    pc.gpio_cfg.dins[0] = (gpio_num_t)PIN_PDM_DIN0;   // words 0, 1: (F) + B
    pc.gpio_cfg.dins[1] = (gpio_num_t)PIN_PDM_DIN1;   // words 2, 3: L + R
    pc.gpio_cfg.dins[2] = (gpio_num_t)PIN_PDM_DIN2;   // words 4, 5: T + (none). Net: dins = {5, 6, 7, -1}
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx, &pc));
    i2s_event_callbacks_t cbs = {};
    cbs.on_recv_q_ovf = on_rx_ovf;                    // counts DMA ring overflows (dma_ovf= in 'S')
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(rx, &cbs, NULL));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

// Reads one chunk (CHUNK frames of PDM_SLOTS int16 words) and appends B, L, R, T to the rings (zeros for a channel
// mapped to -1).
static bool capture_chunk() {
    size_t n = 0;
    if (i2s_channel_read(rx, buf, sizeof(buf), &n, portMAX_DELAY) != ESP_OK || n != sizeof(buf)) return false;
    for (int i = 0; i < CHUNK; i++) {
        int p = (ring_pos + i) % NFFT;
        const int16_t* f = &buf[i * PDM_SLOTS];
        for (int c = 0; c < N_MICS; c++) { int w = CH_SLOT[c]; ring[c][p] = w >= 0 ? f[w] : 0; }
    }
    ring_pos = (ring_pos + CHUNK) % NFFT;
    chunks_captured++;
    audio_ms.store(millis(), std::memory_order_relaxed);
    {   // per-word RMS / peak / DC of this chunk, every word of the DMA frame (tap test; ~4k MACs, negligible)
        int64_t sq[PDM_SLOTS] = {0}; int32_t sm[PDM_SLOTS] = {0}; int pk[PDM_SLOTS] = {0};
        for (int i = 0; i < CHUNK; i++) {
            const int16_t* f = &buf[i * PDM_SLOTS];
            for (int w = 0; w < PDM_SLOTS; w++) {
                int v = f[w]; sq[w] += (int64_t)v * v; sm[w] += v;
                int a = v < 0 ? -v : v; if (a > pk[w]) pk[w] = a;
            }
        }
        for (int w = 0; w < PDM_SLOTS; w++) {
            lvl_rms[w] = sqrtf((float)sq[w] / (float)CHUNK) / 32768.0f;
            lvl_peak[w] = (float)pk[w] / 32768.0f;
            lvl_dc[w] = (float)sm[w] / (float)CHUNK / 32768.0f;
        }
    }
    return true;
}

static void ring_to_frame(int ch, float* out) {       // oldest sample first, scaled to +/-1 (no per-sample modulo)
    const int16_t* r = ring[ch];
    int p = ring_pos;
    for (int i = 0; i < NFFT; i++) { out[i] = (float)r[p] * (1.0f / 32768.0f); if (++p == NFFT) p = 0; }
}

// ---------------------------------------------------------------- shared state between tasks
// Frame fields are written by dsp_task (DSP_CORE), ml by ml_task, hap by alert_task; alert_task and telemetry_task
// (IO_CORE) read them. Every access goes through the spinlock, so a reader always gets one frame's consistent set; the
// critical sections are a few field copies.
struct Shared {
    uint32_t frame = 0, t_ms = 0;      // DSP frame counter, millis() when the frame finished
    float f0 = 0, score = 0, strength = 0, flat = 0, ml = 0, theta = 0, conf = 0;
    float elev = 0;                    // degrees above the band plane (0 horizon, 90 straight up)
    int hits = 0;
    bool detected = false, overhead = false;
    bool theta_ok = false;             // theta known (else empty in T lines, and the motors play the pulsed ring)
    bool elev_ok = false;              // this frame's elevation known (else empty in T lines)
    int bstat = 0;                     // bearing status: 0 none, 1 ambiguous, 2 constrained / azimuth-only, 3 full 3D
    int mused = 0;                     // mics in this frame's direction fit, bitmask B 1, L 2, R 4, T 8
    float hap[N_MOTORS] = {0, 0, 0, 0, 0, 0, 0, 0};   // weights being applied (F, FR, R, BR, B, BL, L, FL)
    float batt_v = 0;                  // battery volts (alert_task, once per second); 0 until the first reading
    // tap test: per-DMA-word max-hold of chunk RMS and peak since the last 'L' command, and the latest DC (dsp_task
    // writes, the 'L' handler reads and clears)
    float lvl_rms[PDM_SLOTS] = {0}, lvl_peak[PDM_SLOTS] = {0}, lvl_dc[PDM_SLOTS] = {0};
};
static Shared sh;
static std::atomic<bool> motor_test_req{false};        // 'M' command -> alert_task buzzes each motor in turn
// Manual haptics ('h' command, the dashboard's Bench toggles): telemetry_task writes, alert_task reads, both under
// sh_mux. While on, alert_task drives each motor at its own duty instead of the detector's weights; it lapses back to
// the detector HAP_MANUAL_MS after the last 'h', so a host that goes away never leaves a motor running.
struct ManualHap { bool on; uint8_t duty[N_MOTORS]; uint8_t hz; uint32_t t_ms; };
static ManualHap man_hap = { false, { 0 }, 0, 0 };
static const uint32_t HAP_MANUAL_MS = 3000;
static uint8_t tasks_failed = 0;                      // setup(): bit per xTaskCreate that failed (bit0 tele, 1 alert, 2 dsp, 3 ml, 4 net)
static portMUX_TYPE sh_mux = portMUX_INITIALIZER_UNLOCKED;
static Shared sh_snapshot() { portENTER_CRITICAL(&sh_mux); Shared s = sh; portEXIT_CRITICAL(&sh_mux); return s; }
static dsp::ClsWindow cls_win;

// ---------------------------------------------------------------- calibration (headset wiring, kept in NVS)
// Which DMA word feeds each DSP channel (mic map) and where each motor output sits on the head (angle in degrees,
// 0 front, +90 right; >= MOT_UNFITTED = not fitted). Set by 'C' and 'Y' (the dashboard's Calibrate tab), saved in NVS
// (Preferences namespace "headband"), loaded in setup(); with nothing saved the compiled defaults apply
// (CH_SLOT_DEFAULT, MOTOR_ANGLES). telemetry_task writes cal_mic / cal_mot under cal_mux; dsp_task adopts a changed mic
// map between chunks (cal_mic_ver) and clears the rings, so no frame mixes two wirings; alert_task copies the motor
// angles every pass. Saving is a flash write: the DSP can lose a chunk while it runs (dma_ovf in 'S'), harmless
// during calibration.
static const float MOT_UNFITTED = 900.0f;
static const uint8_t CAL_NVS_VERSION = 1;
static portMUX_TYPE cal_mux = portMUX_INITIALIZER_UNLOCKED;
static int cal_mic[N_MICS] = { 0, 3, 2, 5 };         // configured mic map B (or F), L, R, T (cal_mux)
static bool cal_front = false;                         // channel 0's mic is at the front ('C...,F'), not the back (cal_mux)
static float cal_mot[N_MOTORS];                        // configured motor angles, MOT_UNFITTED = not fitted (cal_mux)
static std::atomic<uint32_t> cal_mic_ver{0};           // bumped by every accepted 'C'
static bool cal_from_nvs = false;                      // the K line's src=nvs|default (setup / telemetry_task only)
static const char CH_NAME[N_MICS] = { 'B', 'L', 'R', 'T' };
static char ch_name(int c, bool front) { return c == 0 && front ? 'F' : CH_NAME[c]; }

static void cal_save() {
    int8_t mic[N_MICS]; float mot[N_MOTORS];
    portENTER_CRITICAL(&cal_mux);
    for (int c = 0; c < N_MICS; c++) mic[c] = (int8_t)cal_mic[c];
    for (int m = 0; m < N_MOTORS; m++) mot[m] = cal_mot[m];
    bool front = cal_front;
    portEXIT_CRITICAL(&cal_mux);
    Preferences p;
    if (p.begin("headband", false)) {
        p.putBytes("cal_mic", mic, sizeof(mic));
        p.putUChar("cal_front", front ? 1 : 0);
        p.putBytes("cal_mot", mot, sizeof(mot));
        p.putUChar("cal_ver", CAL_NVS_VERSION);
        p.end();
    }
    cal_from_nvs = true;
}

// setup(), before any task: defaults, then NVS if a valid record is there. (Opened read-write so the namespace
// exists afterwards: a read-only open of a missing namespace logs an error line into the telemetry stream.) The mic
// record is 4 bytes (channel 0, L, R, T words) plus "cal_front" (channel 0 at the front; absent = back); a 5-byte
// record written by the interim front-mic firmware (B, L, R, T, F) loads its first 4 bytes.
static void cal_load() {
    for (int c = 0; c < N_MICS; c++) cal_mic[c] = CH_SLOT_DEFAULT[c];
    for (int m = 0; m < N_MOTORS; m++) cal_mot[m] = MOTOR_ANGLES[m];
    Preferences p;
    if (p.begin("headband", false)) {
        int8_t mic[N_MICS + 1]; float mot[N_MOTORS];      // + 1: room for the interim 5-byte record
        size_t nm = p.getUChar("cal_ver", 0) == CAL_NVS_VERSION ? p.getBytes("cal_mic", mic, sizeof(mic)) : 0;
        bool ok = (nm == N_MICS || nm == N_MICS + 1) && p.getBytes("cal_mot", mot, sizeof(mot)) == sizeof(mot);
        for (int c = 0; ok && c < N_MICS; c++) ok = mic[c] >= -1 && mic[c] < PDM_SLOTS;
        for (int c = 0; ok && c < N_MICS; c++)
            for (int d = c + 1; ok && d < N_MICS; d++) ok = mic[c] < 0 || mic[c] != mic[d];
        for (int m = 0; ok && m < N_MOTORS; m++) ok = isfinite(mot[m]) && (mot[m] >= MOT_UNFITTED || fabsf(mot[m]) <= 180.0f);
        if (ok) {
            for (int c = 0; c < N_MICS; c++) cal_mic[c] = mic[c];
            for (int m = 0; m < N_MOTORS; m++) cal_mot[m] = mot[m];
            cal_front = p.getUChar("cal_front", 0) == 1;
            cal_from_nvs = true;
        }
        p.end();
    }
}

// The DSP's view of a mic map (channel 0, L, R, T words): each channel's word, and the layout over the channels that
// have a mic (dsp::mic_layout at MIC_POS, channel 0 at FRONT_POS when front): the bearing measures only their pairs.
static void dsp_map(const int* mic, bool front, int* slot, dsp::MicLayout& lay) {
    bool act[N_MICS]; float pos[N_MICS][3];
    memcpy(pos, MIC_POS, sizeof(pos));
    if (front) memcpy(pos[0], FRONT_POS, sizeof(pos[0]));
    for (int c = 0; c < N_MICS; c++) { slot[c] = mic[c]; act[c] = mic[c] >= 0; }
    dsp::mic_layout(act, pos, lay);
}

// "B=w0 L=w3 R=w2 T=w5" (F for channel 0 when front; '-' for -1) and "M1=0.0 M2=45.0 M3=- ..." for the # cal lines.
static int fmt_cal_mics(char* o, size_t n, const int* mic, bool front) {
    int k = 0;
    for (int c = 0; c < N_MICS && k >= 0 && (size_t)k < n; c++)
        k += mic[c] >= 0 ? snprintf(o + k, n - (size_t)k, "%s%c=w%d", c ? " " : "", ch_name(c, front), mic[c])
                         : snprintf(o + k, n - (size_t)k, "%s%c=-", c ? " " : "", ch_name(c, front));
    return k;
}
static int fmt_cal_motors(char* o, size_t n, const float* mot) {
    int k = 0;
    for (int m = 0; m < N_MOTORS && k >= 0 && (size_t)k < n; m++)
        k += mot[m] < MOT_UNFITTED ? snprintf(o + k, n - (size_t)k, "%sM%d=%.1f", m ? " " : "", m + 1, mot[m])
                                   : snprintf(o + k, n - (size_t)k, "%sM%d=-", m ? " " : "", m + 1);
    return k;
}
static SemaphoreHandle_t cls_mutex;

// ---------------------------------------------------------------- classifier (IO_CORE)
#if USE_TFLM
static tflite::MicroInterpreter* interp = nullptr;
static uint8_t tensor_arena[60 * 1024];
static void ml_setup() {
    static tflite::MicroMutableOpResolver<6> res;
    res.AddConv2D(); res.AddMaxPool2D(); res.AddMean(); res.AddFullyConnected(); res.AddSoftmax(); res.AddReshape();
    const tflite::Model* model = tflite::GetModel(drone_cls_int8_tflite);
    static tflite::MicroInterpreter static_interp(model, res, tensor_arena, sizeof(tensor_arena));
    interp = &static_interp;
    interp->AllocateTensors();
}
static float ml_infer() {
    int8_t* in = interp->input(0)->data.int8;
    xSemaphoreTake(cls_mutex, portMAX_DELAY);
    cls_win.export_int8(in, MODEL_IN_SCALE, MODEL_IN_ZERO);
    xSemaphoreGive(cls_mutex);
    if (interp->Invoke() != kTfLiteOk) return 0;
    int8_t* out = interp->output(0)->data.int8;
    return (out[1] - MODEL_OUT_ZERO) * MODEL_OUT_SCALE;    // p(drone)
}
#else
static void ml_setup() {}
static float ml_infer() { return 1.0f; }                    // no model compiled in: harmonic path only
#endif

// ---------------------------------------------------------------- haptics / alert / battery (IO_CORE)
// Eight motors F, FR, R, BR, B, BL, L, FL (pins.h), one LEDC channel each: 8 channels = all of the S3's LEDC, which
// is why there is no buzzer. ledcAttach() (Arduino-ESP32 3.x) picks the channel; all share 200 Hz / 8 bit.
static const uint8_t MOT_PIN[N_MOTORS] = { PIN_MOT_F, PIN_MOT_FR, PIN_MOT_R, PIN_MOT_BR,
                                           PIN_MOT_B, PIN_MOT_BL, PIN_MOT_L, PIN_MOT_FL };   // = haptic_weights() order
static const char* const MOT_NAME[N_MOTORS] = { "F", "FR", "R", "BR", "B", "BL", "L", "FL" };

static void haptics_setup() {
    for (uint8_t p : MOT_PIN) {
        bool ok = ledcAttach(p, MOTOR_PWM_HZ, MOTOR_PWM_BITS);
        (void)ok;                                     // TODO(bench): a false here means the LEDC channels ran out
        ledcWrite(p, 0);
    }
    pinMode(PIN_LED_DETECT, OUTPUT); digitalWrite(PIN_LED_DETECT, LOW);
    // The DevKitC's own RGB LED (a WS2812 on GPIO 48 on v1.0 boards, 38 on v1.1; neither pin is wired on the headband
    // PCB) is never used, but it keeps the last colour it latched (e.g. a factory demo's) until power-off: send it
    // black once, then hold its data line low. The DevKitC's red power LED and the PowerBoost's LEDs (blue power, red
    // low battery, charge) are hard-wired; no GPIO turns them off (tape over them for stealth).
    for (uint8_t p : { (uint8_t)48, (uint8_t)38 }) {
        rgbLedWrite(p, 0, 0, 0); rmtDeinit(p);
        pinMode(p, OUTPUT); digitalWrite(p, LOW);
    }
    analogSetPinAttenuation(PIN_BATT_SENSE, ADC_11db);  // full range ~0-3.1 V; V_adc <= 2.1 V at 4.2 V battery
}

// Power limit: never more than MOTOR_MAX_ON (3) motors energised at once (the PowerBoost supplies 1 A).
// w[] in, w[] out: keeps the MOTOR_MAX_ON largest non-zero weights, zeroes the rest. The panner only ever returns
// 2 non-zero weights, so this is a backstop that guarantees the limit whatever the core or a host asks for.
static void limit_motors(float w[N_MOTORS]) {
    for (;;) {
        int on = 0, weakest = -1;
        for (int m = 0; m < N_MOTORS; m++)
            if (w[m] > 0.0f) { on++; if (weakest < 0 || w[m] < w[weakest]) weakest = m; }
        if (on <= MOTOR_MAX_ON) return;
        w[weakest] = 0.0f;
    }
}

// Overhead: haptic_weights_at() returns all ones (fitted motors) as a marker. Driving them all would exceed the
// supply, so the firmware plays a spinning ring instead: the fitted motors in clockwise order from the front (by their
// calibrated angle), and every OVERHEAD_STEP_MS motors k, k+step, k+2*step of that list (step = round(n/3), so three
// motors spread around the head) at full weight, advancing k by one, so the triplet rotates. With all eight fitted at
// the default angles this is motors k, k+3, k+6 as before; with 3 or fewer fitted, all of them, steady.
static const uint32_t OVERHEAD_STEP_MS = 100;
static_assert(MOTOR_MAX_ON >= 3, "the overhead triplet needs 3 motors");
static void overhead_triplet(uint32_t k, const float* ang, const bool* fit, float w[N_MOTORS]) {
    int ord[N_MOTORS], n = 0;
    for (int m = 0; m < N_MOTORS; m++) {
        if (!fit[m]) continue;
        float key = fmodf(ang[m] + 360.0f, 360.0f);
        int j = n++;
        while (j > 0 && fmodf(ang[ord[j - 1]] + 360.0f, 360.0f) > key) { ord[j] = ord[j - 1]; j--; }
        ord[j] = m;
    }
    for (int m = 0; m < N_MOTORS; m++) w[m] = 0.0f;
    if (n <= 3) { for (int j = 0; j < n; j++) w[ord[j]] = 1.0f; return; }
    int step = (2 * n + 3) / 6;                                    // round(n / 3): 8 -> 3, 4 -> 1, 5..7 -> 2
    for (int j = 0; j < 3; j++) w[ord[(k + (uint32_t)(j * step)) % (uint32_t)n]] = 1.0f;
}

// Battery: mean of 8 calibrated ADC1 reads (analogReadMilliVolts: eFuse-calibrated mV at the pin, 11 dB
// attenuation, ~0-3.1 V; a full cell gives 2.1 V at the pin). ADC1 is usable with Wi-Fi on (ADC2 is not).
static float read_batt_v() {
    uint32_t mv = 0;
    for (int i = 0; i < 8; i++) mv += analogReadMilliVolts(PIN_BATT_SENSE);
    return BATT_DIVIDER * (float)mv / 8.0f / 1000.0f;
}

// Detect LED: the pattern table at the top of the file. Stealth ('Q') is kept in NVS next to the calibration
// (namespace "headband", key "stealth"); telemetry_task writes it, alert_task reads it.
static const uint32_t LED_NO_AUDIO_MS = 1000;          // no chunk read for this long: fault blink
static const float LED_BATT_LOW_V = 3.5f;              // low-battery blips below this (BATT_EMPTY_V is 3.3)
static std::atomic<bool> led_stealth{false};           // 'Q1': the LED stays off
static std::atomic<bool> led_capture{false};           // dsp_task: a 'W' capture is being recorded

static void led_load() {                               // setup(), before the tasks
    Preferences p;
    if (p.begin("headband", false)) { led_stealth.store(p.getBool("stealth", false)); p.end(); }
}
static void led_save(bool on) {
    Preferences p;
    if (p.begin("headband", false)) { p.putBool("stealth", on); p.end(); }
}
static void led_set(bool on) {
    digitalWrite(PIN_LED_DETECT, on && !led_stealth.load(std::memory_order_relaxed) ? HIGH : LOW);
}

// n blips of on_ms, gap_ms apart, at the start of every period_ms
static bool led_blips(uint32_t now, uint32_t period_ms, uint32_t n, uint32_t on_ms, uint32_t gap_ms) {
    uint32_t t = now % period_ms, step = on_ms + gap_ms;
    return t < n * step && t % step < on_ms;
}

// The LED level for this alert_task pass: the first row of the table that applies. t_start = alert_task's start (the
// no-audio grace at boot); manual = 'h' haptics active.
static bool led_level(uint32_t now, uint32_t t_start, const Shared& s, bool manual) {
    uint32_t a = audio_ms.load(std::memory_order_relaxed);         // signed: dsp_task may have stored a later millis()
    bool no_audio = a ? (int32_t)(now - a) > (int32_t)LED_NO_AUDIO_MS : now - t_start > 3 * LED_NO_AUDIO_MS;
    if (tasks_failed || no_audio) return now % 100 < 50;                         // fault: 10 Hz
    if (led_capture.load(std::memory_order_relaxed)) return now % 500 < 250;     // 'W' recording: 2 Hz
    if (s.detected && s.overhead) return now % 120 < 60;                         // overhead: ~8 Hz flicker
    if (s.detected) return true;                                                 // DETECTED: solid
    if (manual) { uint32_t t = now % 1000; return t < 400 || (t >= 550 && t < 650); }   // long-short
    if (s.hits >= PERSIST_ON / 2) return led_blips(now, 1000, 2, 40, 120);       // evidence building: double blip
    if (s.batt_v > 0.5f && s.batt_v < LED_BATT_LOW_V) return led_blips(now, 5000, 3, 40, 120);   // low battery
    return now % 3000 < 30;                                                      // idle heartbeat
}

static void alert_task(void*) {
    uint32_t t0 = millis(), t_ovh = 0, k_ovh = 0;
    uint32_t t_batt = millis() - 1000;                              // first reading on the first pass
    bool was = false;
    for (;;) {
        if (millis() - t_batt >= 1000) {                            // once per second (outside the spinlock)
            t_batt = millis();
            float v = read_batt_v();
            portENTER_CRITICAL(&sh_mux); sh.batt_v = v; portEXIT_CRITICAL(&sh_mux);
        }
        if (motor_test_req.exchange(false)) {                   // 'M': each motor in turn, 0.5 s on / 0.3 s off
            for (int m = 0; m < N_MOTORS; m++) {
                portENTER_CRITICAL(&sh_mux);
                for (int k = 0; k < N_MOTORS; k++) sh.hap[k] = k == m ? 1.0f : 0.0f;   // hap_* shows which one is on
                portEXIT_CRITICAL(&sh_mux);
                ledcWrite(MOT_PIN[m], MOTOR_DUTY_MAX); led_set(true); vTaskDelay(pdMS_TO_TICKS(500));   // LED: long,
                ledcWrite(MOT_PIN[m], 0); led_set(false);
                portENTER_CRITICAL(&sh_mux); sh.hap[m] = 0.0f; portEXIT_CRITICAL(&sh_mux);
                vTaskDelay(pdMS_TO_TICKS(100)); led_set(true);                                          // short
                vTaskDelay(pdMS_TO_TICKS(100)); led_set(false);
                vTaskDelay(pdMS_TO_TICKS(100));
            }
            continue;
        }
        portENTER_CRITICAL(&sh_mux);                                // millis() inside: ordered with the 'h' handler
        uint32_t now = millis();
        if (man_hap.on && now - man_hap.t_ms > HAP_MANUAL_MS) man_hap.on = false;   // host went quiet: detector again
        ManualHap mh = man_hap;
        portEXIT_CRITICAL(&sh_mux);
        Shared s = sh_snapshot();
        bool onset = s.detected && !was;
        was = s.detected;
        float ang[N_MOTORS]; bool fit[N_MOTORS];                   // the calibrated motor angles ('Y')
        portENTER_CRITICAL(&cal_mux);
        for (int m = 0; m < N_MOTORS; m++) ang[m] = cal_mot[m];
        portEXIT_CRITICAL(&cal_mux);
        for (int m = 0; m < N_MOTORS; m++) fit[m] = ang[m] < MOT_UNFITTED;
        // Pan between the two fitted motors bracketing theta at their calibrated angles (defaults 45 deg apart:
        // ahead -> F; 20 deg right -> F + FR; left -> L; back-left -> BL), weight sum 1, so at most 2 motors run (one
        // fitted motor: it alone, for any bearing). Overhead (elevation above OVERHEAD_ELEV_ON for OVERHEAD_FRAMES
        // frames) -> the spinning triplet above; no usable azimuth (theta_ok false: fewer than three reliable mics,
        // or mirror candidates that do not share an azimuth, for BEARING_HOLD_FRAMES detected frames) -> the same
        // triplet, pulsed like a bearing alert (no direction to pan to). Manual: hap_* report each motor's duty as a share of MOTOR_DUTY_MAX
        // (manual 'h' and the 'M' test drive any output, fitted or not). Either way limit_motors() keeps at most
        // MOTOR_MAX_ON on.
        float w[N_MOTORS] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (mh.on) for (int m = 0; m < N_MOTORS; m++) w[m] = (float)mh.duty[m] / MOTOR_DUTY_MAX;
        else if (s.detected) {
            bool ring = s.overhead || !s.theta_ok;
            dsp::haptic_weights_at(s.theta, ring, ang, fit, w);
            if (ring) {
                if (now - t_ovh >= OVERHEAD_STEP_MS) { k_ovh++; t_ovh = now; }
                overhead_triplet(k_ovh, ang, fit, w);
            }
        }
        limit_motors(w);
        // sh.hap = the weights being applied now (after the overhead triplet and the 3-motor limit, before the
        // on/off pulse gate below, so the T line does not flicker with the pulse rate).
        portENTER_CRITICAL(&sh_mux);
        for (int m = 0; m < N_MOTORS; m++) sh.hap[m] = w[m];
        portEXIT_CRITICAL(&sh_mux);
        if (mh.on) {                                                // steady, or the detector's 50 % gate at mh.hz
            bool on = mh.hz == 0 || fmodf((now - t0) / 1000.0f * mh.hz, 1.0f) < 0.5f;
            for (int m = 0; m < N_MOTORS; m++)                      // duty already <= MOTOR_DUTY_MAX
                ledcWrite(MOT_PIN[m], on && w[m] > 0.0f ? mh.duty[m] : 0);
        } else if (s.detected) {
            // pulse rate rises with harmonic strength; the overhead ring spins continuously instead (no gate)
            float pulse_hz = 1.0f + 4.0f * fminf(1.0f, fmaxf(0.0f, (s.strength - HARM_THRESH_DB) / 20.0f));
            bool on = s.overhead || onset || fmodf((now - t0) / 1000.0f * pulse_hz, 1.0f) < 0.5f;
            for (int m = 0; m < N_MOTORS; m++)                      // duty cap MOTOR_DUTY_MAX: ~3 V after the ULN drop
                ledcWrite(MOT_PIN[m], on ? (uint32_t)(MOTOR_DUTY_MAX * fminf(1.0f, fmaxf(0.0f, w[m])) + 0.5f) : 0);
        } else {
            for (int m = 0; m < N_MOTORS; m++) ledcWrite(MOT_PIN[m], 0);
        }
        led_set(led_level(now, t0, s, mh.on));                      // detect LED (the table at the top)
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

static void ml_task(void*) {
    for (;;) {
        if (cls_win.full()) {
            float p = ml_infer();
            portENTER_CRITICAL(&sh_mux); sh.ml = p; portEXIT_CRITICAL(&sh_mux);
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

// ---------------------------------------------------------------- raw audio queue (dsp_task -> telemetry_task)
// While raw mode is on, dsp_task copies every captured chunk into a packet slot; telemetry_task (IO_CORE) writes the
// packets to every sink in raw mode (USB and/or the TCP client). Single producer / single consumer ring of RAW_SLOTS
// packets with atomic head/tail: the producer never waits (full ring -> chunk dropped, raw_dropped++), so neither
// USB nor Wi-Fi can ever stall the DSP.
// Packet = 12-byte header + HOP x n_ch int16, all little endian (the channel count is explicit in the header):
//   [0] 0xA5  [1] 0x5A  [2] uint8 n_ch  [3] uint8 flags (= 0)  [4..5] uint16 n_frames (= HOP)
//   [6..9] uint32 chunk_index  [10..11] uint16 reserved (= 0)  [12..] samples, frame by frame
// 'R': n_ch = N_MICS = 4, B,L,R,T,B,L,R,T,... through CH_SLOT (8204 B; ~384 KB/s, more than the ~256 KB/s the S3's
// USB CDC delivered on the bench, so a long 'R' recording drops chunks over USB).
// chunk_index counts every captured chunk (raw mode or not), so a gap of k means k chunks were dropped.
// 'W' (calibration) uses the same packet format with n_ch = PDM_SLOTS = 6 but its own buffer: see below.
// (The 3-mic firmware used an 8-byte header without n_ch: A5 5A, uint16 n_frames, uint32 chunk_index.)
static const int RAW_SLOTS = 4;
static const size_t RAW_HDR = 12;
static const size_t RAW_BYTES = RAW_HDR + (size_t)HOP * N_MICS * sizeof(int16_t);      // 8204: an 'R' packet
static const size_t RAW_BYTES_MAX = RAW_HDR + (size_t)HOP * PDM_SLOTS * sizeof(int16_t); // 12300: a 'W' packet
static_assert(CHUNK == HOP && HOP <= 0xFFFF, "raw packet n_frames is a uint16 equal to HOP");
static_assert(PDM_SLOTS <= 0xFF && RAW_BYTES == 8204 && RAW_BYTES_MAX == 12300, "raw packet: 12-byte header + samples");
static uint8_t (*raw_pool)[RAW_BYTES_MAX] = nullptr;                          // [RAW_SLOTS][RAW_BYTES_MAX], 49 KB, PSRAM
static size_t raw_len[RAW_SLOTS];                                             // bytes in each slot (published by raw_head)
static std::atomic<uint32_t> raw_head{0}, raw_tail{0};                        // free-running; slot = index % RAW_SLOTS
static std::atomic<bool> raw_on{false};                                       // telemetry_task: any sink in 'R'
static std::atomic<uint32_t> raw_dropped{0};
static std::atomic<uint32_t> pdm_chunk_us{0};                                 // boot measurement, printed by telemetry_task

// dsp_task only. pdm: one chunk as read from the DMA (CHUNK frames x PDM_SLOTS words); idx: its chunk index.
static void raw_push(const int16_t* pdm, uint32_t idx) {
    uint32_t h = raw_head.load(std::memory_order_relaxed);
    if (h - raw_tail.load(std::memory_order_acquire) >= (uint32_t)RAW_SLOTS) {
        raw_dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    uint8_t* p = raw_pool[h % RAW_SLOTS];
    const int nch = N_MICS;
    p[0] = 0xA5; p[1] = 0x5A;
    p[2] = (uint8_t)nch; p[3] = 0;                                          // n_ch, flags
    p[4] = (uint8_t)(CHUNK & 0xFF); p[5] = (uint8_t)((CHUNK >> 8) & 0xFF);  // n_frames
    p[6] = (uint8_t)(idx & 0xFF); p[7] = (uint8_t)((idx >> 8) & 0xFF);
    p[8] = (uint8_t)((idx >> 16) & 0xFF); p[9] = (uint8_t)((idx >> 24) & 0xFF);
    p[10] = 0; p[11] = 0;                                                   // reserved
    uint8_t* o = p + RAW_HDR;
    for (int i = 0; i < CHUNK; i++) {
        const int16_t* f = &pdm[i * PDM_SLOTS];
        for (int c = 0; c < N_MICS; c++) {                  // same CH_SLOT mapping as the rings: B, L, R, T
            int w = CH_SLOT[c];
            uint16_t v = (uint16_t)(w >= 0 ? f[w] : 0);
            *o++ = (uint8_t)(v & 0xFF); *o++ = (uint8_t)(v >> 8);
        }
    }
    raw_len[h % RAW_SLOTS] = RAW_HDR + (size_t)CHUNK * nch * sizeof(int16_t);
    raw_head.store(h + 1, std::memory_order_release);
}

// ---------------------------------------------------------------- calibration capture ('W<n>')
// The Calibrate tab needs a few seconds of all six DMA words with nothing missing, while the user claps. Six words in
// real time is 576 KB/s, over twice what the S3's USB CDC delivered on the bench (~256 KB/s), so 'W' does not stream
// live: dsp_task records the next n chunks into PSRAM (and skips the DSP meanwhile, so DMA overflows cannot punch
// holes in the capture), and telemetry_task sends them as fast as the link takes them, starting at once. Nothing is
// dropped; a 3 s capture just finishes arriving ~7 s after the 'W'. Packets as 'R' (12-byte header) with n_ch = 6,
// the words w0..w5 unmapped, chunk_index 0 .. n-1. 'T' ends it (also mid-transfer); a new 'W' restarts it.
static const int WCAP_MAX_CHUNKS = 375;                                       // 8 s
static const int WCAP_DEFAULT_CHUNKS = 141;                                   // 3 s
static const size_t WCAP_CHUNK_BYTES = (size_t)CHUNK * PDM_SLOTS * sizeof(int16_t);   // 12288
static int16_t* wcap = nullptr;                // [WCAP_MAX_CHUNKS][CHUNK * PDM_SLOTS], 4.6 MB PSRAM, on the first 'W'
static uint8_t* wpkt = nullptr;                // [RAW_BYTES_MAX], PSRAM: the 'W' packet being sent
static std::atomic<uint32_t> wcap_req{0};      // telemetry_task -> dsp_task: (generation << 16) | chunks wanted (0: none)
static std::atomic<uint32_t> wcap_prog{0};     // dsp_task -> telemetry_task: (generation << 16) | chunks recorded
static uint16_t wcap_gen = 0;                  // telemetry_task: generation of the latest 'W' (never 0 once used)

// ---------------------------------------------------------------- DSP (DSP_CORE)
static dsp::Spectrum* S = nullptr;                    // [N_MICS] B, L, R, T, 32.8 KB, internal heap (read by GCC-PHAT)
static float frame_buf[NFFT], mono[NBINS], lm[NMEL];
static TaskHandle_t dsp_handle = nullptr;             // for the stack high-water mark in 'S'
static dsp::Detector det; static dsp::BearingEstimator bear;
static dsp::MicLayout lay;                            // dsp_task: the connected channels and their bearing solve
static int n_on = 0;                                  // dsp_task: channels with a mic
static dsp::FrameEvidence ev; static dsp::Bearing b;

// Never touches Serial: everything it reports goes through sh / the raw ring / pdm_chunk_us.
static void dsp_task(void*) {
    uint32_t cap_gen = 0, cap_have = 0;                      // calibration capture: generation in progress, chunks kept
    // prime the ring with two chunks so the first frame is full
    capture_chunk(); capture_chunk();
    // Bench check: with 6 int16 slots per frame a chunk (12288 bytes) takes HOP / FS = 21333 us. A different value
    // means the DMA frame is not 6 slots (~25600 us -> 5 slots, ~32000 us -> 4) and PDM_SLOTS / CH_SLOT must be
    // revisited.
    // Drain the DMA backlog first (<= 64 ms = 3 chunks) so the timed reads block on fresh data.
    for (int i = 0; i < 4; i++) capture_chunk();
    uint32_t t0 = micros();
    for (int i = 0; i < 16; i++) capture_chunk();
    uint32_t us = (micros() - t0) / 16;
    pdm_chunk_us.store(us ? us : 1, std::memory_order_release);   // telemetry_task prints "# pdm chunk_us=..."
    uint32_t mic_ver = cal_mic_ver.load(std::memory_order_acquire) - 1;   // adopt the map on the first pass
    for (;;) {
        uint32_t v = cal_mic_ver.load(std::memory_order_acquire);
        if (v != mic_ver) {                                  // a new mic map ('C'): adopt it, start the rings afresh
            mic_ver = v;
            int mic[N_MICS];
            portENTER_CRITICAL(&cal_mux);
            for (int c = 0; c < N_MICS; c++) mic[c] = cal_mic[c];
            bool front = cal_front;
            portEXIT_CRITICAL(&cal_mux);
            dsp_map(mic, front, CH_SLOT, lay);
            bear.set_layout(lay);
            n_on = 0;
            for (int c = 0; c < N_MICS; c++) n_on += lay.active[c] ? 1 : 0;
            memset(ring, 0, N_MICS * sizeof(int16_t[NFFT]));
        }
        uint32_t t0 = micros();
        if (!capture_chunk()) { vTaskDelay(1); continue; }   // yield so IDLE0 (task watchdog) still runs on a stuck port
        uint32_t t1 = micros();
        if (raw_on.load(std::memory_order_relaxed)) raw_push(buf, chunks_captured - 1);
        uint32_t creq = wcap_req.load(std::memory_order_acquire);
        if (creq & 0xFFFF) {                                 // calibration capture: keep the chunk as read, skip the DSP
            uint32_t g = creq >> 16, want = creq & 0xFFFF;
            if (g != cap_gen) { cap_gen = g; cap_have = 0; } // a new 'W': start at chunk 0
            if (cap_have < want) {
                memcpy(wcap + (size_t)cap_have * CHUNK * PDM_SLOTS, buf, WCAP_CHUNK_BYTES);
                cap_have++;
                wcap_prog.store((g << 16) | cap_have, std::memory_order_release);
                led_capture.store(true, std::memory_order_relaxed);
                continue;                                    // no DSP until the capture is complete
            }
        }
        led_capture.store(false, std::memory_order_relaxed);
        for (int c = 0; c < N_MICS; c++)                    // only the channels with a mic (the bearing reads no other)
            if (lay.active[c]) { ring_to_frame(c, frame_buf); dsp::spectrum_from_frame(frame_buf, S[c]); }
        uint32_t t2 = micros();
        for (int k = 0; k < NBINS; k++) {                    // mono = mean of the connected channels' power spectra
            float acc = 0;
            for (int c = 0; c < N_MICS; c++) if (lay.active[c]) acc += dsp::power_at(S[c], k);
            mono[k] = n_on ? acc / (float)n_on : 0.0f;
        }
        dsp::logmel_frame(mono, lm);
        xSemaphoreTake(cls_mutex, portMAX_DELAY); cls_win.push(lm); xSemaphoreGive(cls_mutex);
        uint32_t t3 = micros();
        portENTER_CRITICAL(&sh_mux); float ml = sh.ml; portEXIT_CRITICAL(&sh_mux);
        det.set_ml(ml);
        det.feed(mono, USE_TFLM != 0, ev);
        uint32_t t4 = micros();
        bear.feed(S, ev.mask, ev.detected, b);              // S[0..3] = B, L, R, T (connected ones only)
        uint32_t t5 = micros();
        prof_wait.fetch_add(t1 - t0, std::memory_order_relaxed); prof_spec.fetch_add(t2 - t1, std::memory_order_relaxed);
        prof_mel.fetch_add(t3 - t2, std::memory_order_relaxed);  prof_det.fetch_add(t4 - t3, std::memory_order_relaxed);
        prof_bear.fetch_add(t5 - t4, std::memory_order_relaxed); prof_n.fetch_add(1, std::memory_order_relaxed);
        uint32_t now = millis();
        portENTER_CRITICAL(&sh_mux);
        sh.t_ms = now; sh.f0 = ev.f0; sh.score = ev.score; sh.strength = ev.strength; sh.flat = ev.flat;
        sh.hits = ev.hits; sh.detected = ev.detected; sh.theta = b.theta; sh.conf = b.conf;
        sh.elev = b.elev; sh.overhead = b.overhead; sh.theta_ok = b.theta_ok; sh.elev_ok = b.elev_ok;
        sh.bstat = b.bstat; sh.mused = b.mused;
        for (int w = 0; w < PDM_SLOTS; w++) {                 // max-hold since the last 'L'
            if (lvl_rms[w] > sh.lvl_rms[w]) sh.lvl_rms[w] = lvl_rms[w];
            if (lvl_peak[w] > sh.lvl_peak[w]) sh.lvl_peak[w] = lvl_peak[w];
            sh.lvl_dc[w] = lvl_dc[w];
        }
        sh.frame++;
        portEXIT_CRITICAL(&sh_mux);
    }
}

// ---------------------------------------------------------------- serial protocol (telemetry_task, IO_CORE)
// telemetry_task is the ONLY writer to Serial and to the TCP client (setup() and the other tasks print nothing;
// net_task only accepts connections and hands them over).
// The protocol is the same on both transports; each "sink" (USB Serial, the TCP client) has its own mode.
// Host -> device, single ASCII bytes (CR/LF and anything else ignored):
//   'T' telemetry mode (default): reprints the H header line, then one T line per DSP frame
//   'R' raw mode: binary packets only (see raw audio queue above), no ASCII at all
//   'W<n>\n' calibration capture: records the next n chunks (1-375, default 141 = 3 s; the DSP pauses meanwhile) of
//       all six DMA words into PSRAM and sends them as raw packets (n_ch = 6, unmapped, chunk_index 0 .. n-1) as fast
//       as the link allows, with nothing dropped (see "calibration capture"); then idle in 'W' until 'T'. No ASCII.
//   'S' one "# status ..." line and one "# wifi ..." line, mode unchanged (ignored in raw mode, where no ASCII may
//       be sent)
//   'L' one "# levels ..." line: per DMA word (w0..w5 = DIN0 SEL 3V3 (MK2), DIN0 SEL GND (MK1), DIN1 SEL 3V3 (MK4),
//       DIN1 SEL GND (MK3), DIN2 SEL 3V3 (no header), DIN2 SEL GND (MK5): the right slot comes first)
//       the max-hold since the previous 'L' of the chunk RMS and peak (dBFS) and the latest DC (full scale 1), then
//       which word each DSP channel reads (tap test; telemetry mode only)
//   'M' motor test: alert_task buzzes F, FR, R, BR, B, BL, L, FL in turn (0.5 s on, 0.3 s off) and hap_* shows
//       which one; one
//       "# motor test ..." line (telemetry mode only)
//   'h<F>,<FR>,<R>,<BR>,<B>,<BL>,<L>,<FL>[,<hz>]\n' manual haptics, the one multi-byte command (dashboard Bench
//       toggles): each motor at its own PWM duty 0-255 (clamped to MOTOR_DUTY_MAX; at most MOTOR_MAX_ON of them
//       non-zero, the weakest are dropped), steady (hz 0 or absent) or gated 50 % on at hz (1-20, like the
//       detector's pulse); hap_* = duty / MOTOR_DUTY_MAX. Overrides the detector until 'A', or until
//       HAP_MANUAL_MS (3 s) pass without another 'h': a host keeps it on by repeating it. One "# haptics manual ..."
//       line when the state changes (telemetry mode only; repeats of the same state are silent).
//       e.g. "h0,0,0,0,0,0,190,0" = L alone.
//   'A' haptics back to the detector now (ends 'h'); one "# haptics auto ..." line (telemetry mode only)
//   'C<b>,<l>,<r>,<t>[,F|,B]\n' headset calibration, mics: the DMA word (0-5, or -1 = not connected) of the channel 0,
//       left, right and top mic; channel 0 is the back mic, or with ",F" a front mic instead (at FRONT_POS; ",B" or
//       nothing: back); no word twice. The DSP uses
//       exactly the connected ones: the mono mix is their mean, the bearing comes from their pairs (and, per frame,
//       only the reliable ones: see bstat / mused in the T line). Adopted by the DSP between chunks (rings cleared)
//       and saved to NVS. Reply "# cal mics B=w0 L=- R=w2 T=w5 saved" ("F=w0 ..." when front; '-' for -1), or
//       "# cal: bad 'C...', want ..."
//       (telemetry mode only; the setting applies in either mode).
//   'Y<m1>,...,<m8>\n' headset calibration, motors: the angle in degrees (decimals allowed; 0 front, +90 right,
//       -360..360, normalised to (-180, 180]) of outputs M1..M8 (F, FR, R, BR, B, BL, L, FL in pins.h); >= 900 = not
//       fitted (the detector never drives it; 'h' and 'M' still do). Saved to NVS. Reply "# cal motors M1=0.0 ...
//       M3=- ... saved", or "# cal: bad 'Y...', want ...".
//   'K' one "# cal mics ... motors ... src=nvs|default" line: the current calibration (telemetry mode only); also sent
//       at boot (after the # boot line) and to a TCP client on connect.
//   'Q<0|1>\n' stealth mode: Q1 keeps the detect LED off whatever happens (the motors are unaffected), Q0 brings back
//       the patterns (the "Detect LED" table at the top). Saved to NVS (key "stealth"). Reply "# led stealth=on|off
//       saved", or "# led: bad 'Q...', want Q0 or Q1"; 'Q\n' alone replies "# led stealth=on|off" (telemetry mode
//       only; the setting applies in either mode).
//   h, C, Y, W and Q are the multi-byte commands: digits, ',', '-' and '.' up to the newline (CR or LF); any other
//   byte drops the partial command unapplied (and is then handled as a command itself).
// Device -> host in telemetry mode: lines starting with '#' are comments; then
//   H,frame,t_ms,f0,score,strength,flat,hits,ml,detected,theta,conf,elev,overhead,batt_v,hap_F,hap_FR,hap_R,hap_BR,
//     hap_B,hap_BL,hap_L,hap_FL,bstat,mused                                  (one line; wrapped here)
//   T,<frame>,<t_ms>,<f0 %.1f>,<score %.2f>,<strength %.2f>,<flat %.3f>,<hits>,<ml %.3f>,<detected 0/1>,
//     <theta %.1f>,<conf %.3f>,<elev %.1f>,<overhead 0/1>,<batt_v %.2f>,<hap_F %.2f>,<hap_FR %.2f>,<hap_R %.2f>,
//     <hap_BR %.2f>,<hap_B %.2f>,<hap_BL %.2f>,<hap_L %.2f>,<hap_FL %.2f>,<bstat %d>,<mused %d>
//   theta = the smoothed azimuth, EMPTY when unknown (no usable azimuth yet, or none for BEARING_HOLD_FRAMES (~1 s)
//   detected frames); elev = this frame's elevation in degrees (0 horizon, 90 overhead), EMPTY when unknown;
//   conf = the robust solver's confidence 0..1 (0 without a direction; the theta EMA takes frames >= BEARING_MIN_CONF);
//   bstat = this frame's bearing status: 0 unreliable / no bearing (fewer than three reliable mics), 1 ambiguous (a
//   mirror pair the upper-hemisphere prior cannot separate; elev / theta_raw only where both candidates share them),
//   2 constrained 3D (one candidate picked by that prior) or azimuth-only (B, L, R: azimuth, elevation unknown),
//   3 full 3D (four mics); mused = the mics in this frame's fit, bitmask B 1, L 2, R 4, T 8 (15 = all four,
//   7 = B L R, 14 = L R T, 0 = none); hap_* = the motor weights being applied (after the
//   overhead triplet and the 3-motor limit, so during the overhead chase they show the current triplet; before the
//   on/off pulse gate).
//   t_ms = millis() when the frame finished; hap_* = weights alert_task is applying (updated every 20 ms, so they
//   can trail the frame's detected/overhead by one alert period); batt_v = battery volts, refreshed once per
//   second (0.00 until the first reading). Hosts parse by header name, so the inserted / appended columns break
//   nothing.
// Comment lines:
//   # headband fw 1 fs=48000 nfft=2048 hop=1024                 boot (USB) / on connect (TCP)
//   # pdm chunk_us=<us> (expect 21333)                            once measured (USB); on connect (TCP)
//   # boot free_heap=<bytes> psram_free=<bytes> psram_bufs=<k>/<n> tasks_failed=0x<mask>   boot (USB), after fw
//     (psram_bufs: how many of the bulk buffers landed in PSRAM; tasks_failed: bit0 tele, 1 alert, 2 dsp, 3 ml, 4 net)
//   # status mode=T frame=<n> dropped=<n> wifi_dropped=<n> free_heap=<bytes> psram_free=<bytes>
//            dsp_stack_min=<bytes> batt_v=<V %.2f> pdm_us=<us>                                 on 'S'
//     (pdm_us = the boot chunk-period measurement, 0 until done; expect 21333; dsp_stack_min = dsp_task's stack
//     high-water mark, i.e. the least free stack it has ever had)
//   # wifi mode=<AP|STA|off|starting> ssid=<ssid> ip=<ip> port=5000 mdns=<headband.local|off> client=<ip|none>
//          [sta=failed]                                           boot (USB, once up) / on connect (TCP) / on 'S'
//   # wifi client <ip> connected | disconnected (<reason>)       USB only, when the TCP client changes
//   # dsp frames=<n> us/frame wait=<us> spec=<us> mel=<us> det=<us> bear=<us> total=<us> dma_ovf=<n> (budget 21333)
//          on 'S', after the wifi line: mean microseconds per frame in each DSP stage since the previous 'S' (wait =
//          blocked in i2s_channel_read); total must stay under the budget or the DMA ring overflows (dma_ovf counts)
//   # levels w0 rms=<dBFS> peak=<dBFS> dc=<+/-x.xxx> ... w5 ... | B=w<n> L=w<n> R=w<n> T=w<n>   on 'L'
//   # motor test: F=GPIO<n> FR=GPIO<n> ... FL=GPIO<n>, 0.5 s each ...                            on 'M'
//   # haptics manual F=<duty> FR=<duty> ... FL=<duty> pulse_hz=<hz> max=<MOTOR_DUTY_MAX> ...    on a new 'h' state
//   # haptics auto ...                                                                         on 'A'
//   # cal mics B=w1 L=w2 R=w3 T=w4 motors M1=0.0 M2=45.0 ... M8=-45.0 src=nvs|default   on 'K', boot, TCP connect
//   # cal mics ... saved | # cal motors ... saved | # cal: bad ...                        on 'C' / 'Y'
//   # led stealth=on|off                                       on 'Q' alone, boot (after the # cal line), TCP connect
//   # led stealth=on|off saved | # led: bad ...                                          on 'Q0' / 'Q1'
// dropped = chunks the DSP could not queue (raw ring full); wifi_dropped = raw packets / T lines not sent to a
// slow TCP client (see the TCP sink below).
static const char* TELEM_HEADER =
    "H,frame,t_ms,f0,score,strength,flat,hits,ml,detected,theta,conf,elev,overhead,batt_v,"
    "hap_F,hap_FR,hap_R,hap_BR,hap_B,hap_BL,hap_L,hap_FL,bstat,mused";

enum { SINK_SERIAL = 0, SINK_TCP = 1, N_SINKS = 2 };
struct Sink {
    bool active;            // USB: always; TCP: a client is connected
    char mode;              // 'T', 'R' or 'W'
    int wsent;              // 'W': capture chunks already sent to this sink
};
static Sink sinks[N_SINKS] = { { true, 'T', 0 }, { false, 'T', 0 } };
static char arg_cmd[N_SINKS];                                 // h, C, Y, W, Q: the multi-byte command being collected,
static char arg_buf[N_SINKS][96];                             // its argument up to the newline,
static int arg_len[N_SINKS] = { -1, -1 };                     // and its length (-1: not in one)
static std::atomic<uint32_t> wifi_dropped{0};
static char line[320];                                        // telemetry_task only: one formatted ASCII line

static bool is_raw(char mode) { return mode == 'R' || mode == 'W'; }

static void update_raw_on() {
    bool on = false, words = false;
    for (int s = 0; s < N_SINKS; s++) {
        on |= sinks[s].active && sinks[s].mode == 'R';
        words |= sinks[s].active && sinks[s].mode == 'W';
    }
    raw_on.store(on, std::memory_order_relaxed);              // no sink in 'R' -> the raw queue is not fed
    if (!words && (wcap_req.load(std::memory_order_relaxed) & 0xFFFF))   // nobody wants the capture any more:
        wcap_req.store((uint32_t)wcap_gen << 16, std::memory_order_release);   // stop it, dsp_task resumes the DSP
}

// ---------------------------------------------------------------- Wi-Fi + TCP sink (net_task, IO_CORE)
// net_task brings Wi-Fi up (AP, or STA with AP fallback), starts mDNS and the TCP server, then only accepts
// connections and hands each one to telemetry_task through tcp_pending (mutex). telemetry_task owns the adopted
// client: it reads commands with recv(MSG_DONTWAIT) and writes with send(MSG_DONTWAIT) on the client's fd.
// Why not WiFiClient::write: in 3.x (NetworkClient.cpp) it select()s up to 1 s per try with 10 retries, i.e. a
// stalled client could block telemetry_task for seconds. Drop policy instead: every write is one whole unit (a raw
// packet or a line). If the socket takes only part of it, the tail goes to tcp_carry and is finished on later
// passes; while a tail is pending every new unit for the TCP sink is dropped (wifi_dropped++), so framing is never
// torn and the task never waits. A client that accepts nothing for WIFI_STALL_MS is disconnected.
// TODO(bench): lwIP's per-socket send buffer (CONFIG_LWIP_TCP_SND_BUF_DEFAULT, ~5.7 KB in Arduino builds) is smaller
// than one raw packet (8204 B), so raw streaming relies on the carry path every packet; check the throughput
// (bench check 5) and raise the send buffer (setsockopt SO_SNDBUF is not honoured by lwIP) if it falls short.
#if USE_WIFI
struct WifiInfo { char mode[12]; char ssid[33]; char ip[16]; bool mdns; bool sta_failed; };
static WifiInfo wifi_info;                                    // written by net_task before wifi_ready (release)
static std::atomic<bool> wifi_ready{false};
static SemaphoreHandle_t tcp_mutex;
static WiFiClient tcp_pending;                                // guarded by tcp_mutex
static bool tcp_has_pending = false;                          // guarded by tcp_mutex
static WiFiClient tcp_client;                                 // telemetry_task only
static uint8_t* tcp_carry = nullptr;                          // [RAW_BYTES_MAX], PSRAM: unsent tail of the last unit (<= one raw packet)
static size_t tcp_carry_off = 0, tcp_carry_len = 0;
static uint32_t tcp_progress_ms = 0;                          // millis() of the last byte the client accepted
static char tcp_peer[16] = "none";

static void wifi_bringup() {
    WiFi.setHostname(WIFI_HOSTNAME);                          // before mode()/begin() so DHCP uses it
    bool sta = false, sta_failed = false;
#if defined(WIFI_STA_SSID) && defined(WIFI_STA_PASS)
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_STA_TIMEOUT_MS) vTaskDelay(pdMS_TO_TICKS(250));
    sta = WiFi.status() == WL_CONNECTED;
    if (!sta) { WiFi.disconnect(true); sta_failed = true; }  // radio off, then fall through to the AP
#endif
    WifiInfo& w = wifi_info;
    if (sta) {
        WiFi.setSleep(false);                                 // modem sleep adds latency and cuts raw throughput
        strlcpy(w.mode, "STA", sizeof(w.mode));
        strlcpy(w.ssid, WiFi.SSID().c_str(), sizeof(w.ssid));
        strlcpy(w.ip, WiFi.localIP().toString().c_str(), sizeof(w.ip));
    } else {
        WiFi.mode(WIFI_AP);
        bool ok = WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASS);   // channel 1, visible, max 4 stations (header defaults)
        strlcpy(w.mode, ok ? "AP" : "off", sizeof(w.mode));
        strlcpy(w.ssid, WIFI_AP_SSID, sizeof(w.ssid));
        strlcpy(w.ip, ok ? WiFi.softAPIP().toString().c_str() : "0.0.0.0", sizeof(w.ip));   // 192.168.4.1
    }
    w.sta_failed = sta_failed;
    w.mdns = MDNS.begin(WIFI_HOSTNAME);                       // headband.local
    if (w.mdns) MDNS.addService("headband", "tcp", WIFI_TCP_PORT);   // TODO(bench): `dns-sd -B _headband._tcp`
    wifi_ready.store(true, std::memory_order_release);
}

// Stack: ~6 KB (Wi-Fi/LwIP run in their own tasks; this one only does bring-up calls and accept()).
static void net_task(void*) {
    wifi_bringup();
    static WiFiServer server(WIFI_TCP_PORT, 1);               // one client at a time (listen backlog 1)
    server.setNoDelay(true);                                  // accepted sockets get TCP_NODELAY
    server.begin();                                           // non-blocking listen socket (NetworkServer.cpp)
    for (;;) {
        if (server.hasClient()) {
            WiFiClient c = server.accept();
            if (c) {
                c.setNoDelay(true);
                xSemaphoreTake(tcp_mutex, portMAX_DELAY);
                tcp_pending = c; tcp_has_pending = true;      // a newer connection replaces an unadopted one
                xSemaphoreGive(tcp_mutex);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
#endif

static void sink_write(int s, const uint8_t* p, size_t n);

static void sink_puts(int s, const char* text) { sink_write(s, (const uint8_t*)text, strlen(text)); }

static void sink_printf(int s, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n > 0) sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
}

static void print_wifi(int s) {
#if USE_WIFI
    if (!wifi_ready.load(std::memory_order_acquire)) {
        sink_printf(s, "# wifi mode=starting ssid=%s port=%d\n", WIFI_AP_SSID, WIFI_TCP_PORT);
        return;
    }
    const WifiInfo& w = wifi_info;
    sink_printf(s, "# wifi mode=%s ssid=%s ip=%s port=%d mdns=%s client=%s%s\n", w.mode, w.ssid, w.ip, WIFI_TCP_PORT,
                w.mdns ? WIFI_HOSTNAME ".local" : "off", sinks[SINK_TCP].active ? tcp_peer : "none",
                w.sta_failed ? " sta=failed" : "");
#else
    sink_puts(s, "# wifi mode=off (USE_WIFI 0)\n");
#endif
}

#if USE_WIFI
static void tcp_close(const char* why) {
    if (!sinks[SINK_TCP].active) return;
    tcp_client.stop();
    sinks[SINK_TCP].active = false; sinks[SINK_TCP].mode = 'T';
    tcp_carry_off = tcp_carry_len = 0;
    update_raw_on();
    if (sinks[SINK_SERIAL].mode == 'T') {          // own buffer: a caller may be in the middle of using line[]
        char m[96];
        snprintf(m, sizeof(m), "# wifi client %s disconnected (%s)\n", tcp_peer, why);
        sink_puts(SINK_SERIAL, m);
    }
}

static bool would_block(int r) { return r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK); }

// Pushes as much of the pending tail as the socket takes right now. false: the connection is gone.
static bool tcp_flush_carry() {
    while (tcp_carry_off < tcp_carry_len) {
        int r = send(tcp_client.fd(), tcp_carry + tcp_carry_off, tcp_carry_len - tcp_carry_off, MSG_DONTWAIT);
        if (r > 0) { tcp_carry_off += (size_t)r; tcp_progress_ms = millis(); continue; }
        if (r == 0 || would_block(r)) {
            if (millis() - tcp_progress_ms > WIFI_STALL_MS) return false;    // stalled client: give up on it
            return true;
        }
        return false;                                         // ECONNRESET, EPIPE, ...
    }
    tcp_carry_off = tcp_carry_len = 0;
    return true;
}

static void tcp_write(const uint8_t* p, size_t n) {
    if (!tcp_flush_carry()) { tcp_close("send failed or stalled"); return; }
    if (tcp_carry_len) { wifi_dropped.fetch_add(1, std::memory_order_relaxed); return; }   // busy: drop this unit
    int r = send(tcp_client.fd(), p, n, MSG_DONTWAIT);
    if (r < 0 && !would_block(r)) { tcp_close("send failed"); return; }
    size_t sent = r > 0 ? (size_t)r : 0;
    if (sent) tcp_progress_ms = millis();
    if (sent < n) {
        if (n - sent > RAW_BYTES_MAX) { wifi_dropped.fetch_add(1, std::memory_order_relaxed); return; }  // n/a
        if (sent == 0) tcp_progress_ms = millis();            // the stall clock starts with the first refused unit
        memcpy(tcp_carry, p + sent, n - sent);
        tcp_carry_off = 0; tcp_carry_len = n - sent;
    }
}
#endif

static void sink_write(int s, const uint8_t* p, size_t n) {
    if (!sinks[s].active || n == 0) return;
    if (s == SINK_SERIAL) { Serial.write(p, n); return; }
#if USE_WIFI
    tcp_write(p, n);
#endif
}

// 'W' packets must arrive whole: wait for room instead of dropping. USB: write until all bytes are taken (Serial.write
// can return short when the host is slow), giving up after 2 s without progress; TCP: only when no tail is pending.
static bool sink_ready(int s) {
#if USE_WIFI
    if (s == SINK_TCP) {
        if (!tcp_flush_carry()) { tcp_close("send failed or stalled"); return false; }
        return tcp_carry_len == 0;
    }
#endif
    return true;
}
static void sink_write_whole(int s, const uint8_t* p, size_t n) {
    if (s != SINK_SERIAL) { sink_write(s, p, n); return; }   // TCP: sink_ready() said a whole unit fits the carry path
    size_t off = 0; uint32_t t = millis();
    while (off < n) {
        size_t w = Serial.write(p + off, n - off);
        if (w) { off += w; t = millis(); }
        else if (millis() - t > 2000) return;                  // host gone: the remainder is lost, the next header resyncs
        else vTaskDelay(1);
    }
}
static void send_wcap_packet(int s, int k) {                 // capture chunk k as a 6-word raw packet
    uint8_t* p = wpkt;
    p[0] = 0xA5; p[1] = 0x5A; p[2] = (uint8_t)PDM_SLOTS; p[3] = 0;
    p[4] = (uint8_t)(CHUNK & 0xFF); p[5] = (uint8_t)((CHUNK >> 8) & 0xFF);
    p[6] = (uint8_t)(k & 0xFF); p[7] = (uint8_t)((k >> 8) & 0xFF); p[8] = 0; p[9] = 0; p[10] = 0; p[11] = 0;
    memcpy(p + RAW_HDR, wcap + (size_t)k * CHUNK * PDM_SLOTS, WCAP_CHUNK_BYTES);
    sink_write_whole(s, p, RAW_BYTES_MAX);
}

static void print_pdm(int s, uint32_t us) {
    sink_printf(s, "# pdm chunk_us=%lu (expect %d)\n", (unsigned long)us, (int)(1000000LL * HOP / FS));
}

static void enter_telemetry(int s) {
    sinks[s].mode = 'T';
    update_raw_on();
    sink_printf(s, "%s\n", TELEM_HEADER);                     // "\n" like every other line (println would add "\r\n")
}

// "h<F>,<FR>,<R>,<BR>,<B>,<BL>,<L>,<FL>[,<hz>]": manual haptics (see the protocol above). Applied in either mode;
// the reply only in 'T'.
static void manual_haptics(int s, const char* arg) {
    int d[N_MOTORS + 1] = { 0 };
    int got = sscanf(arg, "%d,%d,%d,%d,%d,%d,%d,%d,%d", &d[0], &d[1], &d[2], &d[3], &d[4], &d[5], &d[6], &d[7], &d[8]);
    if (got < N_MOTORS) {
        if (sinks[s].mode == 'T') sink_printf(s, "# haptics: bad 'h%s', want h<F>,<FR>,<R>,<BR>,<B>,<BL>,<L>,<FL>[,<hz>]\n", arg);
        return;
    }
    ManualHap m = { true, { 0 }, 0, 0 };
    float w[N_MOTORS];
    for (int k = 0; k < N_MOTORS; k++) w[k] = (float)constrain(d[k], 0, MOTOR_DUTY_MAX);   // never above the 3 V cap
    limit_motors(w);                                                      // never more than MOTOR_MAX_ON at once
    for (int k = 0; k < N_MOTORS; k++) m.duty[k] = (uint8_t)w[k];
    m.hz = (uint8_t)constrain(d[N_MOTORS], 0, 20);
    portENTER_CRITICAL(&sh_mux);
    bool changed = !man_hap.on || memcmp(man_hap.duty, m.duty, sizeof(m.duty)) != 0 || man_hap.hz != m.hz;
    m.t_ms = millis();
    man_hap = m;
    portEXIT_CRITICAL(&sh_mux);
    if (changed && sinks[s].mode == 'T') {                    // keepalive repeats of the same state stay quiet
        int n = snprintf(line, sizeof(line), "# haptics manual");
        for (int k = 0; k < N_MOTORS && n > 0 && (size_t)n < sizeof(line); k++)
            n += snprintf(line + n, sizeof(line) - (size_t)n, " %s=%d", MOT_NAME[k], m.duty[k]);
        if (n > 0 && (size_t)n < sizeof(line))
            n += snprintf(line + n, sizeof(line) - (size_t)n, " pulse_hz=%d max=%d max_on=%d (detector again %lu ms "
                          "after the last h)\n", m.hz, MOTOR_DUTY_MAX, MOTOR_MAX_ON, (unsigned long)HAP_MANUAL_MS);
        if (n > 0) sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    }
}

// "# cal mics ... motors ... src=..." (the K line).
static void print_cal(int s) {
    int mic[N_MICS]; float mot[N_MOTORS];
    portENTER_CRITICAL(&cal_mux);
    for (int c = 0; c < N_MICS; c++) mic[c] = cal_mic[c];
    for (int m = 0; m < N_MOTORS; m++) mot[m] = cal_mot[m];
    bool front = cal_front;
    portEXIT_CRITICAL(&cal_mux);
    int n = snprintf(line, sizeof(line), "# cal mics ");
    n += fmt_cal_mics(line + n, sizeof(line) - (size_t)n, mic, front);
    if ((size_t)n < sizeof(line)) n += snprintf(line + n, sizeof(line) - (size_t)n, " motors ");
    if ((size_t)n < sizeof(line)) n += fmt_cal_motors(line + n, sizeof(line) - (size_t)n, mot);
    if ((size_t)n < sizeof(line)) n += snprintf(line + n, sizeof(line) - (size_t)n, " src=%s\n", cal_from_nvs ? "nvs" : "default");
    sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
}

// "C<b>,<l>,<r>,<t>[,F|,B]": the mic map (see the protocol above). Applied in either mode; the reply only in 'T'.
static void cal_mics_cmd(int s, const char* arg) {
    int w[N_MICS]; char side = 'B', tail;
    int got = sscanf(arg, "%d,%d,%d,%d,%c%c", &w[0], &w[1], &w[2], &w[3], &side, &tail);   // more after: rejected
    bool ok = got == N_MICS || (got == N_MICS + 1 && (side == 'F' || side == 'B'));
    bool front = side == 'F';
    for (int c = 0; ok && c < N_MICS; c++) ok = w[c] >= -1 && w[c] < PDM_SLOTS;
    for (int c = 0; ok && c < N_MICS; c++)
        for (int d = c + 1; ok && d < N_MICS; d++) ok = w[c] < 0 || w[c] != w[d];
    if (!ok) {
        if (sinks[s].mode == 'T')
            sink_printf(s, "# cal: bad 'C%s', want C<b>,<l>,<r>,<t>[,F] with words 0-5 or -1, none twice\n", arg);
        return;
    }
    portENTER_CRITICAL(&cal_mux);
    for (int c = 0; c < N_MICS; c++) cal_mic[c] = w[c];
    cal_front = front;
    portEXIT_CRITICAL(&cal_mux);
    cal_mic_ver.fetch_add(1, std::memory_order_release);      // dsp_task adopts it before its next chunk
    cal_save();
    if (sinks[s].mode == 'T') {
        int n = snprintf(line, sizeof(line), "# cal mics ");
        n += fmt_cal_mics(line + n, sizeof(line) - (size_t)n, w, front);
        if ((size_t)n < sizeof(line)) n += snprintf(line + n, sizeof(line) - (size_t)n, " saved\n");
        sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    }
}

// "Y<m1>,...,<m8>": the motor angles (see the protocol above). Applied in either mode; the reply only in 'T'.
static void cal_motors_cmd(int s, const char* arg) {
    float a[N_MOTORS];
    const char* q = arg;
    bool ok = true;
    for (int m = 0; ok && m < N_MOTORS; m++) {
        char* end = nullptr;
        a[m] = strtof(q, &end);
        ok = end != q && isfinite(a[m]) && (a[m] >= MOT_UNFITTED || fabsf(a[m]) <= 360.0f);
        if (!ok) break;
        if (a[m] >= MOT_UNFITTED) a[m] = MOT_UNFITTED;
        else {                                                // normalise to (-180, 180]
            a[m] = fmodf(a[m], 360.0f);
            if (a[m] > 180.0f) a[m] -= 360.0f;
            if (a[m] <= -180.0f) a[m] += 360.0f;
            a[m] += 0.0f;                                     // -0.0 -> 0.0 (fmodf(-360, 360) is -0)
        }
        q = end;
        if (m < N_MOTORS - 1) { ok = *q == ','; q++; }
        else ok = *q == 0;
    }
    if (!ok) {
        if (sinks[s].mode == 'T')
            sink_printf(s, "# cal: bad 'Y%s', want Y<m1>,...,<m8> in degrees (>= 900 = not fitted)\n", arg);
        return;
    }
    portENTER_CRITICAL(&cal_mux);
    for (int m = 0; m < N_MOTORS; m++) cal_mot[m] = a[m];
    portEXIT_CRITICAL(&cal_mux);
    cal_save();
    if (sinks[s].mode == 'T') {
        int n = snprintf(line, sizeof(line), "# cal motors ");
        n += fmt_cal_motors(line + n, sizeof(line) - (size_t)n, a);
        if ((size_t)n < sizeof(line)) n += snprintf(line + n, sizeof(line) - (size_t)n, " saved\n");
        sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    }
}

// "W<n>": record the next n chunks (1..WCAP_MAX_CHUNKS; empty = WCAP_DEFAULT_CHUNKS) of all six DMA words and send
// them to this sink (see "calibration capture"). Another sink already in 'W' follows the new capture from chunk 0.
static void start_capture(int s, const char* arg) {
    int n = arg[0] ? atoi(arg) : WCAP_DEFAULT_CHUNKS;
    if (n < 1 || n > WCAP_MAX_CHUNKS || strchr(arg, '-') || strchr(arg, '.') || strchr(arg, ',')) {
        if (sinks[s].mode == 'T') sink_printf(s, "# cal: bad 'W%s', want W<chunks> 1-%d\n", arg, WCAP_MAX_CHUNKS);
        return;
    }
    if (!wcap) wcap = (int16_t*)heap_caps_malloc((size_t)WCAP_MAX_CHUNKS * WCAP_CHUNK_BYTES, MALLOC_CAP_SPIRAM);
    if (!wcap || !wpkt) {
        if (sinks[s].mode == 'T') sink_puts(s, "# cal: no PSRAM for the calibration capture\n");
        return;
    }
    wcap_gen = wcap_gen == 0xFFFF ? 1 : (uint16_t)(wcap_gen + 1);
    for (int k = 0; k < N_SINKS; k++) if (k == s || sinks[k].mode == 'W') { sinks[k].mode = 'W'; sinks[k].wsent = 0; }
    wcap_req.store(((uint32_t)wcap_gen << 16) | (uint32_t)n, std::memory_order_release);
    update_raw_on();
}

// "# led stealth=on|off[ saved]" (the Q line).
static void print_led(int s, bool saved) {
    sink_printf(s, "# led stealth=%s%s\n", led_stealth.load(std::memory_order_relaxed) ? "on" : "off",
                saved ? " saved" : "");
}

// "Q<0|1>": stealth mode (see the protocol above); "Q" alone reports it. Applied in either mode; the reply only in 'T'.
static void led_cmd(int s, const char* arg) {
    if (arg[0] && !((arg[0] == '0' || arg[0] == '1') && arg[1] == 0)) {
        if (sinks[s].mode == 'T') sink_printf(s, "# led: bad 'Q%s', want Q0 or Q1\n", arg);
        return;
    }
    if (arg[0]) {
        bool on = arg[0] == '1';
        led_stealth.store(on, std::memory_order_relaxed);   // alert_task turns the LED off within 20 ms
        led_save(on);
    }
    if (sinks[s].mode == 'T') print_led(s, arg[0] != 0);
}

static void run_arg_cmd(int s, char cmd, const char* arg) {
    if (cmd == 'h') manual_haptics(s, arg);
    else if (cmd == 'C') cal_mics_cmd(s, arg);
    else if (cmd == 'Y') cal_motors_cmd(s, arg);
    else if (cmd == 'W') start_capture(s, arg);
    else if (cmd == 'Q') led_cmd(s, arg);
}

static void handle_cmd(int s, int c) {
    if (arg_len[s] >= 0) {                                    // inside h / C / Y / W / Q: its argument to the newline
        if (c == '\n' || c == '\r') { arg_buf[s][arg_len[s]] = 0; arg_len[s] = -1; run_arg_cmd(s, arg_cmd[s], arg_buf[s]); return; }
        bool side = arg_cmd[s] == 'C' && (c == 'F' || c == 'B');   // C...,F / C...,B: channel 0 at the front / back
        if (((c >= '0' && c <= '9') || c == ',' || c == '-' || c == '.' || side) && arg_len[s] < (int)sizeof(arg_buf[s]) - 1) {
            arg_buf[s][arg_len[s]++] = (char)c;
            return;
        }
        arg_len[s] = -1;                                      // anything else drops it unapplied; c is handled below
    }
    if (c == 'h' || c == 'C' || c == 'Y' || c == 'W' || c == 'Q') { arg_cmd[s] = (char)c; arg_len[s] = 0; return; }
    if (c == 'A') {
        portENTER_CRITICAL(&sh_mux); man_hap.on = false; portEXIT_CRITICAL(&sh_mux);
        if (sinks[s].mode == 'T') sink_puts(s, "# haptics auto (the detector drives the motors)\n");
        return;
    }
    if (c == 'T') enter_telemetry(s);
    else if (c == 'R') { sinks[s].mode = 'R'; update_raw_on(); }
    else if (c == 'K' && sinks[s].mode == 'T') print_cal(s);
    else if (c == 'S' && sinks[s].mode == 'T') {
        Shared sn = sh_snapshot();
        sink_printf(s, "# status mode=%c frame=%lu dropped=%lu wifi_dropped=%lu free_heap=%lu psram_free=%lu "
                       "dsp_stack_min=%lu batt_v=%.2f pdm_us=%lu\n",
                    sinks[s].mode, (unsigned long)sn.frame, (unsigned long)raw_dropped.load(std::memory_order_relaxed),
                    (unsigned long)wifi_dropped.load(std::memory_order_relaxed), (unsigned long)ESP.getFreeHeap(),
                    (unsigned long)ESP.getFreePsram(),
                    (unsigned long)(dsp_handle ? uxTaskGetStackHighWaterMark(dsp_handle) : 0),
                    sn.batt_v, (unsigned long)pdm_chunk_us.load(std::memory_order_acquire));
        print_wifi(s);
        uint32_t n = prof_n.exchange(0), tw = prof_wait.exchange(0), ts = prof_spec.exchange(0),
                 tm = prof_mel.exchange(0), td = prof_det.exchange(0), tb = prof_bear.exchange(0);
        if (n == 0) n = 1;
        sink_printf(s, "# dsp frames=%lu us/frame wait=%lu spec=%lu mel=%lu det=%lu bear=%lu total=%lu dma_ovf=%lu "
                       "(budget %d)\n", (unsigned long)n, (unsigned long)(tw / n), (unsigned long)(ts / n),
                    (unsigned long)(tm / n), (unsigned long)(td / n), (unsigned long)(tb / n),
                    (unsigned long)((ts + tm + td + tb) / n), (unsigned long)dma_ovf.load(std::memory_order_relaxed),
                    (int)(1000000LL * HOP / FS));
    } else if (c == 'L' && sinks[s].mode == 'T') {
        float rms[PDM_SLOTS], pk[PDM_SLOTS], dc[PDM_SLOTS];
        portENTER_CRITICAL(&sh_mux);
        for (int w = 0; w < PDM_SLOTS; w++) {
            rms[w] = sh.lvl_rms[w]; pk[w] = sh.lvl_peak[w]; dc[w] = sh.lvl_dc[w];
            sh.lvl_rms[w] = sh.lvl_peak[w] = 0.0f;                              // restart the max-hold
        }
        portEXIT_CRITICAL(&sh_mux);
        int n = snprintf(line, sizeof(line), "# levels");
        for (int w = 0; w < PDM_SLOTS && n > 0 && (size_t)n < sizeof(line); w++)
            n += snprintf(line + n, sizeof(line) - (size_t)n, " w%d rms=%.1f peak=%.1f dc=%+.3f", w,
                          20.0f * log10f(fmaxf(rms[w], 1e-6f)), 20.0f * log10f(fmaxf(pk[w], 1e-6f)), dc[w]);
        if (n > 0 && (size_t)n < sizeof(line)) {              // which word each DSP channel reads (the calibration)
            int mic[N_MICS];
            portENTER_CRITICAL(&cal_mux);
            for (int k = 0; k < N_MICS; k++) mic[k] = cal_mic[k];
            bool front = cal_front;
            portEXIT_CRITICAL(&cal_mux);
            n += snprintf(line + n, sizeof(line) - (size_t)n, " | ");
            if ((size_t)n < sizeof(line)) n += fmt_cal_mics(line + n, sizeof(line) - (size_t)n, mic, front);
            if ((size_t)n < sizeof(line)) n += snprintf(line + n, sizeof(line) - (size_t)n, "\n");
        }
        if (n > 0) sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    } else if (c == 'M' && sinks[s].mode == 'T') {
        motor_test_req.store(true, std::memory_order_relaxed);
        int n = snprintf(line, sizeof(line), "# motor test:");
        for (int k = 0; k < N_MOTORS && n > 0 && (size_t)n < sizeof(line); k++)
            n += snprintf(line + n, sizeof(line) - (size_t)n, " %s=GPIO%d", MOT_NAME[k], (int)MOT_PIN[k]);
        if (n > 0 && (size_t)n < sizeof(line))
            n += snprintf(line + n, sizeof(line) - (size_t)n, ", 0.5 s each in that order; hap_* show which one is on\n");
        if (n > 0) sink_write(s, (const uint8_t*)line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    }
}

#if USE_WIFI
// Takes over a connection accepted by net_task (replacing the current one) and greets it like a USB boot.
static void tcp_adopt() {
    WiFiClient c; bool got = false;
    xSemaphoreTake(tcp_mutex, portMAX_DELAY);
    if (tcp_has_pending) { c = tcp_pending; tcp_pending = WiFiClient(); tcp_has_pending = false; got = true; }
    xSemaphoreGive(tcp_mutex);                                // (assignment drops the reference; stop() would close)
    if (!got) return;
    tcp_close("replaced by a new connection");
    tcp_client = c;
    strlcpy(tcp_peer, tcp_client.remoteIP().toString().c_str(), sizeof(tcp_peer));
    sinks[SINK_TCP].active = true; sinks[SINK_TCP].mode = 'T'; arg_len[SINK_TCP] = -1;
    tcp_carry_off = tcp_carry_len = 0; tcp_progress_ms = millis();
    update_raw_on();
    if (sinks[SINK_SERIAL].mode == 'T') sink_printf(SINK_SERIAL, "# wifi client %s connected\n", tcp_peer);
    sink_printf(SINK_TCP, "# headband fw 1 fs=%d nfft=%d hop=%d\n", (int)FS, (int)NFFT, (int)HOP);
    print_wifi(SINK_TCP);
    uint32_t us = pdm_chunk_us.load(std::memory_order_acquire);
    if (us) print_pdm(SINK_TCP, us);
    print_cal(SINK_TCP);
    print_led(SINK_TCP, false);
    enter_telemetry(SINK_TCP);
}

// Commands from the TCP client; also notices an orderly close (recv() == 0).
static void tcp_poll_cmds() {
    uint8_t b[16];
    while (sinks[SINK_TCP].active) {
        int r = recv(tcp_client.fd(), b, sizeof(b), MSG_DONTWAIT);
        if (r > 0) { for (int i = 0; i < r && sinks[SINK_TCP].active; i++) handle_cmd(SINK_TCP, b[i]); continue; }
        if (r == 0) tcp_close("closed by host");
        else if (!would_block(r)) tcp_close("recv failed");
        return;
    }
}
#endif

static void telemetry_task(void*) {
    bool pdm_printed = false;
#if USE_WIFI
    bool wifi_printed = false;
#endif
    uint32_t last = 0;
    sink_printf(SINK_SERIAL, "# headband fw 1 fs=%d nfft=%d hop=%d\n", (int)FS, (int)NFFT, (int)HOP);
    vTaskDelay(pdMS_TO_TICKS(50));                            // let setup() finish creating the other tasks
    sink_printf(SINK_SERIAL, "# boot free_heap=%lu psram_free=%lu psram_bufs=%d/%d tasks_failed=0x%02x\n",
                (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getFreePsram(), big_allocs_psram, big_allocs_total,
                (unsigned)tasks_failed);
    print_cal(SINK_SERIAL);
    print_led(SINK_SERIAL, false);
    enter_telemetry(SINK_SERIAL);
    for (;;) {
        // commands: non-blocking
        while (Serial.available() > 0) handle_cmd(SINK_SERIAL, Serial.read());
#if USE_WIFI
        tcp_adopt();
        if (sinks[SINK_TCP].active) tcp_poll_cmds();
        if (sinks[SINK_TCP].active && !tcp_flush_carry()) tcp_close("send failed or stalled");
        if (!wifi_printed && wifi_ready.load(std::memory_order_acquire) && sinks[SINK_SERIAL].mode == 'T') {
            print_wifi(SINK_SERIAL);                          // boot: "# wifi mode=AP ssid=HEADBAND ip=192.168.4.1 ..."
            wifi_printed = true;
        }
#endif
        // raw packets: one write per packet per sink in raw mode; sinks in telemetry mode never see them
        uint32_t t = raw_tail.load(std::memory_order_relaxed);
        while (t != raw_head.load(std::memory_order_acquire)) {
            for (int s = 0; s < N_SINKS; s++)
                if (sinks[s].active && sinks[s].mode == 'R') sink_write(s, raw_pool[t % RAW_SLOTS], raw_len[t % RAW_SLOTS]);
            raw_tail.store(++t, std::memory_order_release);
            if (is_raw(sinks[SINK_SERIAL].mode) && Serial.available() > 0) break;   // stay responsive to 'T'
        }
        {   // calibration capture: what dsp_task has recorded so far, two packets per sink per pass (~48 ms each on USB)
            uint32_t prog = wcap_prog.load(std::memory_order_acquire);
            int have = (prog >> 16) == wcap_gen ? (int)(prog & 0xFFFF) : 0;
            for (int s = 0; s < N_SINKS; s++)
                for (int k = 0; k < 2 && sinks[s].active && sinks[s].mode == 'W' && sinks[s].wsent < have; k++) {
                    if (!sink_ready(s)) break;
                    send_wcap_packet(s, sinks[s].wsent++);
                }
        }
        bool any_raw = false, any_tel = false;
        for (int s = 0; s < N_SINKS; s++) {
            any_raw |= sinks[s].active && is_raw(sinks[s].mode);
            any_tel |= sinks[s].active && sinks[s].mode == 'T';
        }
        if (sinks[SINK_SERIAL].mode == 'T') {
            uint32_t us = pdm_chunk_us.load(std::memory_order_acquire);
            if (us && !pdm_printed) { print_pdm(SINK_SERIAL, us); pdm_printed = true; }
        }
        if (any_tel) {
            Shared s = sh_snapshot();
            if (s.frame != last) {
                last = s.frame;
                static_assert(N_MOTORS == 8, "T line prints hap_F .. hap_FL");
                char th[12] = "", el[12] = "";                 // empty when unknown
                if (s.theta_ok) snprintf(th, sizeof(th), "%.1f", s.theta);
                if (s.elev_ok) snprintf(el, sizeof(el), "%.1f", s.elev);
                int n = snprintf(line, sizeof(line),
                                 "T,%lu,%lu,%.1f,%.2f,%.2f,%.3f,%d,%.3f,%d,%s,%.3f,%s,%d,%.2f,"
                                 "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d\n",
                                 (unsigned long)s.frame, (unsigned long)s.t_ms, s.f0, s.score, s.strength, s.flat,
                                 s.hits, s.ml, s.detected ? 1 : 0, th, s.conf, el, s.overhead ? 1 : 0,
                                 s.batt_v, s.hap[0], s.hap[1], s.hap[2], s.hap[3], s.hap[4], s.hap[5], s.hap[6],
                                 s.hap[7], s.bstat, s.mused);
                if (n > 0 && (size_t)n < sizeof(line))
                    for (int k = 0; k < N_SINKS; k++)
                        if (sinks[k].active && sinks[k].mode == 'T') sink_write(k, (const uint8_t*)line, (size_t)n);
            }
        }
        // raw: a packet every 21.3 ms (8204 B 'R' ~384 KB/s, 12300 B 'W' ~576 KB/s), 4 slots (85 ms) of slack;
        // telemetry: a frame every 21.3 ms
        vTaskDelay(any_raw ? 1 : pdMS_TO_TICKS(5));
    }
}

void setup() {
    Serial.begin(921600);                                    // baud ignored on native USB CDC; matters on the UART bridge
    cls_mutex = xSemaphoreCreateMutex();
    ring = (int16_t (*)[NFFT])big_alloc(N_MICS * sizeof(int16_t[NFFT]));  // see "memory" above: PSRAM when present
    S = (dsp::Spectrum*)heap_caps_calloc(N_MICS, sizeof(dsp::Spectrum), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!S) abort();
    raw_pool = (uint8_t (*)[RAW_BYTES_MAX])big_alloc((size_t)RAW_SLOTS * RAW_BYTES_MAX);
#if USE_WIFI
    tcp_carry = (uint8_t*)big_alloc(RAW_BYTES_MAX);
#endif
    wpkt = (uint8_t*)heap_caps_malloc(RAW_BYTES_MAX, MALLOC_CAP_SPIRAM);   // 'W' staging packet (the capture: on demand)
#if USE_WIFI
#endif
    cal_load();                                              // mic map + motor angles: NVS, else the compiled defaults
    led_load();                                              // stealth mode ('Q'): NVS, else off
    dsp::fft_init(); dsp::mel_init();
    // i2s_setup() runs here, in Arduino's loopTask (core 1 = ARDUINO_RUNNING_CORE), so the I2S DMA interrupt is
    // allocated on core 1, the same core as dsp_task with DSP_CORE 1. TODO(bench): if DSP_CORE is swapped to 0,
    // consider creating the channel from a core-0 task so the ISR follows it.
    haptics_setup(); ml_setup(); i2s_setup();
    // telemetry_task first, so that whatever else fails to start is reported (tasks_failed in the boot line).
    // 8 KB: it formats into line[] and owns the TCP client (lwIP send/recv are shallow).
    if (xTaskCreatePinnedToCore(telemetry_task, "tele", 8 * 1024, NULL, 1, NULL, IO_CORE) != pdPASS) tasks_failed |= 1;
    if (xTaskCreatePinnedToCore(alert_task, "alert", 4 * 1024, NULL, 3, NULL, IO_CORE) != pdPASS) tasks_failed |= 2;
    // 12 KB: the DSP works in static / PSRAM buffers; measured high-water mark ~1 KB used (dsp_stack_min in 'S')
    if (xTaskCreatePinnedToCore(dsp_task, "dsp", 12 * 1024, NULL, 5, &dsp_handle, DSP_CORE) != pdPASS) tasks_failed |= 4;
    // ml_task only runs the classifier; without TFLM it is a 500 ms idle loop and needs no 16 KB stack
    if (xTaskCreatePinnedToCore(ml_task, "ml", USE_TFLM ? 16 * 1024 : 3 * 1024, NULL, 2, NULL, IO_CORE) != pdPASS) tasks_failed |= 8;
#if USE_WIFI
    tcp_mutex = xSemaphoreCreateMutex();
    if (xTaskCreatePinnedToCore(net_task, "net", 6 * 1024, NULL, 1, NULL, IO_CORE) != pdPASS) tasks_failed |= 16;
#endif
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }
