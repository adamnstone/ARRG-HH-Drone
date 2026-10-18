// Bench test for the v3 headband PCB (not the headband firmware, see firmware/esp32s3/headband.ino): live level of
// every PDM mic slot, for the tap test. Open this folder in the Arduino IDE, or from the repo root:
//   arduino-cli compile --upload -p /dev/ttyACM0 --fqbn "esp32:esp32:esp32s3:CDCOnBoot=cdc,USBMode=hwcdc,\
//     PartitionScheme=huge_app,FlashSize=4M,PSRAM=opi,CPUFreq=240,UploadSpeed=921600" firmware/bench/mic_levels
// Arduino IDE: Board "ESP32S3 Dev Module", USB CDC On Boot: Enabled, USB Mode: Hardware CDC and JTAG, Flash Size 4MB,
// Partition Scheme Huge APP, PSRAM OPI PSRAM; Serial Monitor at any baud.
//
// Capture is the headband firmware's: I2S0 PDM RX, 48 kHz, PDM clock 3.072 MHz on GPIO 4 (through a 33 ohm resistor
// per mic on v3), three data lines DIN0 GPIO 5, DIN1 GPIO 6, DIN2 GPIO 7, all six slots enabled, so one DMA frame is six
// int16 words. Word -> mic: the right slot (SEL 3V3) comes first on each line, as the IDF guide says (verified on the
// v3 board by tapping with MK1..MK4 fitted, 2026-10-17):
//   w0 MK2 (DIN0 SEL 3V3)  w1 MK1 (DIN0 SEL GND)  w2 MK4 (DIN1 SEL 3V3)  w3 MK3 (DIN1 SEL GND)
//   w4 none (DIN2 SEL 3V3) w5 MK5 (DIN2 SEL GND)
// Output, four times a second, one line (the firmware's 'L' format, so scripts/telemetry.py's levels_parse reads it):
//   # levels w0 rms=<dBFS> peak=<dBFS> dc=<+/-x.xxx> zc=<Hz> w1 ... w5 ...
// rms = mean RMS over the last 0.25 s, peak = max |sample|, dc = mean (full scale 1), zc = zero-crossing rate / 2 of
// the AC part, a rough pitch (a whistle reads its frequency; broadband noise reads high). An empty phase on a line with
// one mic usually mirrors that mic (same numbers in both words of the line); a dc far from 0 means a stuck line or two
// mics on the same SEL. At boot: "# pdm chunk_us=" should be ~21333 (another value means the DMA frame is not 6 words).
// The motor outputs are all held LOW.
#include <Arduino.h>
#include <atomic>
#include <driver/i2s_pdm.h>

static const int FS = 48000, CHUNK = 1024, PDM_SLOTS = 6;
static const int PIN_PDM_CLK = 4, PIN_PDM_DIN0 = 5, PIN_PDM_DIN1 = 6, PIN_PDM_DIN2 = 7;
static const int MOTOR_PINS[8] = {12, 13, 14, 15, 16, 17, 18, 21};
static const char* const WORD_NAME[PDM_SLOTS] = {"MK2 (DIN0 SEL 3V3)", "MK1 (DIN0 SEL GND)",
                                                 "MK4 (DIN1 SEL 3V3)", "MK3 (DIN1 SEL GND)",
                                                 "none (DIN2 SEL 3V3)", "MK5 (DIN2 SEL GND)"};
static const int DMA_DESC_NUM = 12, DMA_FRAME_NUM = 768;     // as headband.ino: 64 ms of DMA ring with 6 slots

static i2s_chan_handle_t rx;
static int16_t buf[CHUNK * PDM_SLOTS];                       // one chunk (21.3 ms), all six words interleaved
static std::atomic<uint32_t> dma_ovf{0};                     // I2S DMA ring overflows (the loop fell behind)
static bool IRAM_ATTR on_rx_ovf(i2s_chan_handle_t, i2s_event_data_t*, void*) {
    dma_ovf.fetch_add(1, std::memory_order_relaxed);
    return false;
}

static void i2s_setup() {
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);   // PDM RX exists on I2S0 only
    cc.dma_desc_num = DMA_DESC_NUM; cc.dma_frame_num = DMA_FRAME_NUM;
    ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));
    i2s_pdm_rx_config_t pc = {
        .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(FS),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = { .clk = (gpio_num_t)PIN_PDM_CLK, .invert_flags = { .clk_inv = false } },
    };
    pc.clk_cfg.dn_sample_mode = I2S_PDM_DSR_8S;               // PDM clock = FS x 64 = 3.072 MHz (MP34DT01-M max 3.25)
    pc.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)(I2S_PDM_RX_LINE0_SLOT_RIGHT | I2S_PDM_RX_LINE0_SLOT_LEFT |
                                                  I2S_PDM_RX_LINE1_SLOT_RIGHT | I2S_PDM_RX_LINE1_SLOT_LEFT |
                                                  I2S_PDM_RX_LINE2_SLOT_RIGHT | I2S_PDM_RX_LINE2_SLOT_LEFT);
    for (int i = 0; i < SOC_I2S_PDM_MAX_RX_LINES; i++) pc.gpio_cfg.dins[i] = I2S_GPIO_UNUSED;
    pc.gpio_cfg.dins[0] = (gpio_num_t)PIN_PDM_DIN0;
    pc.gpio_cfg.dins[1] = (gpio_num_t)PIN_PDM_DIN1;
    pc.gpio_cfg.dins[2] = (gpio_num_t)PIN_PDM_DIN2;
    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx, &pc));
    i2s_event_callbacks_t cbs = {};
    cbs.on_recv_q_ovf = on_rx_ovf;
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(rx, &cbs, NULL));
    ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

static bool read_chunk() {
    size_t n = 0;
    return i2s_channel_read(rx, buf, sizeof(buf), &n, portMAX_DELAY) == ESP_OK && n == sizeof(buf);
}

// per-word accumulators over one report interval
static double s_sum[PDM_SLOTS], s_sq[PDM_SLOTS];
static int s_pk[PDM_SLOTS];
static uint32_t s_zc[PDM_SLOTS], s_n = 0;
static float dc_prev[PDM_SLOTS];                              // last interval's mean, for the zero crossings
static int16_t last_ac_sign[PDM_SLOTS];

static void legend() {
    Serial.print("# legend:");
    for (int w = 0; w < PDM_SLOTS; w++) Serial.printf(" w%d=%s%s", w, WORD_NAME[w], w + 1 < PDM_SLOTS ? "," : "\n");
}

void setup() {
    for (int p : MOTOR_PINS) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }   // no ULN input floats
    Serial.begin(115200);
    delay(300);
    i2s_setup();
    for (int i = 0; i < 4; i++) read_chunk();                 // drain the DMA backlog, then time 16 chunks
    uint32_t t0 = micros();
    for (int i = 0; i < 16; i++) read_chunk();
    uint32_t us = (micros() - t0) / 16;
    Serial.printf("# mic_levels bench sketch: 6 PDM words, 48 kHz\n# pdm chunk_us=%lu (expect %d; ~25600 = 5 words, "
                  "~32000 = 4)\n", (unsigned long)us, (int)(1000000LL * CHUNK / FS));
    legend();
}

void loop() {
    if (!read_chunk()) { delay(1); return; }
    for (int i = 0; i < CHUNK; i++) {
        const int16_t* f = &buf[i * PDM_SLOTS];
        for (int w = 0; w < PDM_SLOTS; w++) {
            int v = f[w];
            s_sum[w] += v; s_sq[w] += (double)v * v;
            int a = v < 0 ? -v : v; if (a > s_pk[w]) s_pk[w] = a;
            int16_t sg = (float)v / 32768.0f >= dc_prev[w] ? 1 : -1;
            if (sg != last_ac_sign[w]) { s_zc[w]++; last_ac_sign[w] = sg; }
        }
    }
    s_n += CHUNK;
    static uint32_t t_rep = 0, n_rep = 0;
    if (millis() - t_rep < 250) return;
    t_rep = millis();
    static char line[512];
    int len = snprintf(line, sizeof(line), "# levels");
    for (int w = 0; w < PDM_SLOTS; w++) {
        double mean = s_sum[w] / s_n, var = s_sq[w] / s_n - mean * mean;
        float rms = sqrt(var > 0 ? var : 0) / 32768.0, pk = s_pk[w] / 32768.0f, dc = mean / 32768.0;
        float zc_hz = 0.5f * (float)s_zc[w] * FS / (float)s_n;
        len += snprintf(line + len, sizeof(line) - len, " w%d rms=%.1f peak=%.1f dc=%+.3f zc=%.0f", w,
                        20.0f * log10f(fmaxf(rms, 1e-6f)), 20.0f * log10f(fmaxf(pk, 1e-6f)), dc, zc_hz);
        dc_prev[w] = dc;
        s_sum[w] = s_sq[w] = 0; s_pk[w] = 0; s_zc[w] = 0;
    }
    s_n = 0;
    snprintf(line + len, sizeof(line) - len, " dma_ovf=%lu\n", (unsigned long)dma_ovf.load());
    Serial.print(line);
    if (++n_rep % 40 == 0) legend();                          // every 10 s, for the Serial Monitor
}
