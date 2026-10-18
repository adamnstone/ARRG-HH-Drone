// Portable DSP core. See dsp_core.h. Kept deliberately plain C++ so it runs identically on a laptop and the ESP32-S3.
#include "dsp_core.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#ifdef ESP_PLATFORM
// On the ESP32-S3 the FFT goes through Espressif's esp-dsp (bundled with Arduino-ESP32 3.x; the S3 build uses its
// assembly kernel): a radix-2 2048-point complex FFT drops from ~3.3 ms to well under 0.5 ms, which is the
// difference between the DSP keeping up with the audio (21.3 ms per frame, six FFTs) and not. The portable radix-2
// below is still what the host harness (test_core / compare_core.py) runs; results differ only by float rounding.
#include "esp_dsp.h"
#endif

namespace dsp {

// ================================================================ FFT (radix-2, in place)
static float s_cos[NFFT / 2], s_sin[NFFT / 2];
static uint16_t s_rev[NFFT];
static float s_hann[NFFT];
static bool s_fft_ready = false;
#ifdef ESP_PLATFORM
static float s_cplx[2 * NFFT] __attribute__((aligned(16)));   // interleaved re, im for dsps_fft2r_fc32 (in place)
#endif

void fft_init() {
    if (s_fft_ready) return;
#ifdef ESP_PLATFORM
    dsps_fft2r_init_fc32(NULL, NFFT);                  // library twiddle table (CONFIG_DSP_MAX_FFT_SIZE >= NFFT)
#endif
    int bits = 0; while ((1 << bits) < NFFT) bits++;
    for (int i = 0; i < NFFT; i++) {
        unsigned r = 0, v = (unsigned)i;
        for (int b = 0; b < bits; b++) { r = (r << 1) | (v & 1); v >>= 1; }
        s_rev[i] = (uint16_t)r;
    }
    for (int k = 0; k < NFFT / 2; k++) {
        double a = -2.0 * M_PI * k / NFFT;
        s_cos[k] = (float)cos(a); s_sin[k] = (float)sin(a);
    }
    for (int n = 0; n < NFFT; n++) s_hann[n] = 0.5f - 0.5f * (float)cos(2.0 * M_PI * n / NFFT);
    s_fft_ready = true;
}

static void fft_core(float* re, float* im, bool inverse) {
#ifdef ESP_PLATFORM
    // inverse = conj(FFT(conj(x))) / N, so one forward kernel serves both directions
    float* c = s_cplx;
    if (inverse) for (int i = 0; i < NFFT; i++) { c[2 * i] = re[i]; c[2 * i + 1] = -im[i]; }
    else         for (int i = 0; i < NFFT; i++) { c[2 * i] = re[i]; c[2 * i + 1] =  im[i]; }
    dsps_fft2r_fc32(c, NFFT);
    dsps_bit_rev_fc32(c, NFFT);
    if (inverse) { float s = 1.0f / NFFT; for (int i = 0; i < NFFT; i++) { re[i] = c[2 * i] * s; im[i] = -c[2 * i + 1] * s; } }
    else         for (int i = 0; i < NFFT; i++) { re[i] = c[2 * i]; im[i] = c[2 * i + 1]; }
    return;
#endif
    for (int i = 0; i < NFFT; i++) {
        int j = s_rev[i];
        if (j > i) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= NFFT; len <<= 1) {
        int half = len >> 1, step = NFFT / len;
        for (int i = 0; i < NFFT; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = s_sin[k * step];
                if (inverse) wi = -wi;
                int a = i + k, b = a + half;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
    if (inverse) { float s = 1.0f / NFFT; for (int i = 0; i < NFFT; i++) { re[i] *= s; im[i] *= s; } }
}
void fft_forward(float* re, float* im) { fft_init(); fft_core(re, im, false); }
void fft_inverse(float* re, float* im) { fft_init(); fft_core(re, im, true); }

static float g_re[NFFT], g_im[NFFT];   // scratch (single-threaded use)

void spectrum_from_frame(const float* x, Spectrum& s) {
    fft_init();
    for (int n = 0; n < NFFT; n++) { g_re[n] = x[n] * s_hann[n]; g_im[n] = 0.0f; }
    fft_core(g_re, g_im, false);
    for (int k = 0; k < NBINS; k++) { s.re[k] = g_re[k]; s.im[k] = g_im[k]; }
}

// ================================================================ mel
// Sparse triangular filterbank. Filter m is non-zero only on bins s_mel_lo[m]..s_mel_hi[m]; those weights are stored
// contiguously at s_mel_wv[s_mel_off[m]]. A bin lies inside at most two triangles, so 2*NBINS floats (8 KB) hold
// everything (the dense NMEL x NBINS table was 160 KB). Weights and summation order are identical to the dense form.
static float s_mel_wv[2 * NBINS];
static int   s_mel_lo[NMEL], s_mel_hi[NMEL], s_mel_off[NMEL];
static bool  s_mel_ready = false;

static double hz2mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel2hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }
static double mel_tri(double f, double lo, double mid, double hi) {
    double up = (f - lo) / (mid - lo), down = (hi - f) / (hi - mid);
    double w = up < down ? up : down; return w < 0 ? 0 : w;
}

void mel_init() {
    if (s_mel_ready) return;
    double m0 = hz2mel(MEL_FMIN), m1 = hz2mel(MEL_FMAX);
    double edges[NMEL + 2];
    for (int i = 0; i < NMEL + 2; i++) edges[i] = mel2hz(m0 + (m1 - m0) * i / (NMEL + 1));
    int off = 0;
    for (int m = 0; m < NMEL; m++) {
        double lo = edges[m], mid = edges[m + 1], hi = edges[m + 2], sum = 0;
        s_mel_lo[m] = NBINS; s_mel_hi[m] = -1;
        for (int k = 0; k < NBINS; k++) {                 // pass 1: support and normaliser (sum of the double weights)
            double w = mel_tri((double)k * FS / NFFT, lo, mid, hi); sum += w;
            if (w > 0) { if (k < s_mel_lo[m]) s_mel_lo[m] = k; if (k > s_mel_hi[m]) s_mel_hi[m] = k; }
        }
        s_mel_off[m] = off;
        for (int k = s_mel_lo[m]; k <= s_mel_hi[m]; k++) { // pass 2: float weight / sum, rounded exactly as the dense table was
            float w = (float)mel_tri((double)k * FS / NFFT, lo, mid, hi);
            s_mel_wv[off++] = sum > 0 ? (float)(w / sum) : w;
        }
    }
    s_mel_ready = true;
}

void logmel_frame(const float* power, float* out) {
    mel_init();
    // float accumulation, like features.py's float32 matmul (and the ESP32-S3 has no double FPU)
    for (int m = 0; m < NMEL; m++) {
        const float* w = s_mel_wv + s_mel_off[m];
        float e = 0;
        for (int k = s_mel_lo[m]; k <= s_mel_hi[m]; k++) e += power[k] * w[k - s_mel_lo[m]];
        out[m] = 10.0f * log10f(e + 1e-10f);
    }
}

ClsWindow::ClsWindow() { memset(rows, 0, sizeof(rows)); }
void ClsWindow::push(const float* row) {
    memcpy(rows[head], row, sizeof(float) * NMEL);
    head = (head + 1) % CLS_WINDOW_FRAMES;
    if (count < CLS_WINDOW_FRAMES) count++;
}
void ClsWindow::export_float(float* out) const {
    float mx = -1e30f;
    for (int i = 0; i < CLS_WINDOW_FRAMES; i++) for (int m = 0; m < NMEL; m++) if (rows[i][m] > mx) mx = rows[i][m];
    int start = full() ? head : 0;
    for (int i = 0; i < CLS_WINDOW_FRAMES; i++) {
        const float* r = rows[(start + i) % CLS_WINDOW_FRAMES];
        for (int m = 0; m < NMEL; m++) {
            float v = r[m] - mx; if (v < CLS_DB_FLOOR) v = CLS_DB_FLOOR; if (v > 0) v = 0;
            out[i * NMEL + m] = (v - CLS_DB_FLOOR) / (-CLS_DB_FLOOR);
        }
    }
}
void ClsWindow::export_int8(int8_t* out, float scale, int zero) const {
    static float tmp[CLS_WINDOW_FRAMES * NMEL];
    export_float(tmp);
    for (int i = 0; i < CLS_WINDOW_FRAMES * NMEL; i++) {
        long q = lroundf(tmp[i] / scale) + zero;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        out[i] = (int8_t)q;
    }
}

// ================================================================ detector helpers
float spectral_flatness(const float* power) {
    int lo = (int)(DET_BAND_LO / BIN_HZ), hi = (int)(DET_BAND_HI / BIN_HZ) + 1;
    float lg = 0, mean = 0; int n = 0;                   // float: ~1e-6 relative, far inside the 1e-3 tolerance
    for (int k = lo; k < hi; k++) { float s = power[k] + 1e-12f; lg += logf(s); mean += s; n++; }
    return expf(lg / n) / (mean / n);
}

// Flatness of the prominence spectrum (dB minus its smooth baseline): geometric / arithmetic mean of 10^(S/10), with
// the geometric mean taken as exp(mean(S) * ln10/10). A smooth microphone response cancels between S and its baseline.
float local_flatness(const float* sdb) {
    const float k = 0.230258509f;                        // ln(10) / 10
    float s = 0, a = 0; const int n = LOCAL_HI_BIN - LOCAL_LO_BIN + 1;
    for (int b = LOCAL_LO_BIN; b <= LOCAL_HI_BIN; b++) { s += sdb[b]; a += expf(k * sdb[b]); }
    return expf(k * s / n) / (a / n);
}

// Neumaier-compensated float add: the sliding window sum below runs ~2 x NBINS adds/subtracts per pass, which in
// plain float would drift by ~0.1 dB; with the compensation term it stays within ~1e-4 dB of the exact sum, at
// float speed (the ESP32-S3 emulates double in software: the original per-bin double window sum cost ~45 ms per
// frame, a double sliding sum still ~10 ms, this ~0.1 ms).
static inline void kadd(float& s, float& c, float v) {
    float t = s + v;
    if (fabsf(s) >= fabsf(v)) c += (s - t) + v; else c += (v - t) + s;
    s = t;
}
static void movavg(const float* x, float* y, int w) {   // clamped-edge moving average, w odd; O(NBINS) sliding sum
    int h = w / 2;
    float s = 0, c = 0, inv = 1.0f / (float)w;
    for (int j = -h; j <= h; j++) kadd(s, c, x[j < 0 ? 0 : j]);
    y[0] = (s + c) * inv;
    for (int i = 1; i < NBINS; i++) {
        int add = i + h, rem = i - h - 1;
        kadd(s, c, x[add >= NBINS ? NBINS - 1 : add]);
        kadd(s, c, -x[rem < 0 ? 0 : rem]);
        y[i] = (s + c) * inv;
    }
}
void spectral_baseline(const float* pdb, float* base, float* tmp) {
    movavg(pdb, base, BASELINE_BINS);
    for (int it = 0; it < 2; it++) {
        for (int k = 0; k < NBINS; k++) tmp[k] = pdb[k] < base[k] ? pdb[k] : base[k];
        movavg(tmp, base, BASELINE_BINS);
    }
}

static int first_harmonic(float f0) {               // first harmonic number >= HARM_FMIN
    int h0 = (int)ceilf(HARM_FMIN / f0);
    return h0 < 1 ? 1 : h0;
}
void harmonic_search(const float* sdb, float prev_f0, float* f0_out, float* score_out, int* n_harm_out) {
    static float S[NBINS];
    static float sc[N_F0_CANDIDATES];
    for (int k = 0; k < NBINS; k++) S[k] = sdb[k] > 0 ? sdb[k] : 0;
    int best = 0;
    for (int c = 0; c < N_F0_CANDIDATES; c++) {
        float f0 = F0_MIN + c * F0_STEP, acc = 0;
        int h0 = first_harmonic(f0);
        for (int h = h0; h < h0 + N_HARMONICS; h++) {
            float pos = f0 * h / BIN_HZ; int b = (int)floorf(pos); float w = pos - b;
            acc += S[b] * (1 - w) + S[b + 1] * w;
        }
        sc[c] = acc / N_HARMONICS;
        if (sc[c] > sc[best]) best = c;
    }
    if (prev_f0 > 0 && F0_HYST_DB > 0) {                 // track continuity: prefer a near-equal candidate close to prev_f0
        int bl = -1;
        for (int c = 0; c < N_F0_CANDIDATES; c++) {
            float f0 = F0_MIN + c * F0_STEP;
            if (fabsf(f0 - prev_f0) / prev_f0 < F0_MAX_JUMP && (bl < 0 || sc[c] > sc[bl])) bl = c;
        }
        if (bl >= 0 && sc[bl] >= sc[best] - F0_HYST_DB) best = bl;
    }
    float bestf = F0_MIN + best * F0_STEP;
    int h0 = first_harmonic(bestf), nh = 0;
    for (int h = h0; h < h0 + 8; h++) {
        int b = (int)lroundf(bestf * h / BIN_HZ); if (b > NBINS - 1) b = NBINS - 1;
        if (S[b] > 6.0f) nh++;
    }
    *f0_out = bestf; *score_out = sc[best]; *n_harm_out = nh;
}

void harmonic_mask(float f0, float* mask) {
    memset(mask, 0, sizeof(float) * NBINS);
    for (int k = 1; k <= BEARING_MASK_HARMONICS; k++) {
        float f = k * f0; if (f < BEARING_BAND_LO || f > BEARING_BAND_HI) continue;
        int b = (int)lroundf(f / BIN_HZ);
        for (int j = b - BEARING_MASK_HALFWIDTH; j <= b + BEARING_MASK_HALFWIDTH; j++) if (j >= 0 && j < NBINS) mask[j] = 1;
    }
}
void band_mask(float* mask) {
    memset(mask, 0, sizeof(float) * NBINS);
    for (int k = (int)(BEARING_BAND_LO / BIN_HZ); k <= (int)(BEARING_BAND_HI / BIN_HZ) && k < NBINS; k++) mask[k] = 1;
}

static int cmpf(const void* a, const void* b) { float x = *(const float*)a, y = *(const float*)b; return (x > y) - (x < y); }
static float percentile(const float* v, int n, float p) {   // numpy 'linear' interpolation
    static float tmp[256];
    memcpy(tmp, v, sizeof(float) * (size_t)n); qsort(tmp, (size_t)n, sizeof(float), cmpf);
    float pos = p / 100.0f * (n - 1); int lo = (int)floorf(pos); float w = pos - lo;
    if (lo >= n - 1) return tmp[n - 1];
    return tmp[lo] * (1 - w) + tmp[lo + 1] * w;
}

// ================================================================ Detector
Detector::Detector() {
    memset(f0_hist_, 0, sizeof(f0_hist_)); memset(hist_, 0, sizeof(hist_)); memset(scores_, 0, sizeof(scores_));
    memset(g_ring_, 0, sizeof(g_ring_)); memset(prev_sb_, 0, sizeof(prev_sb_));
}

// DSP gate (detector.py gate_features / DroneDetector._gate): push this frame's six inputs, return the gate score
// = sum of GATE_W_* times the window means (level: its window standard deviation). Uses sdb_ / savg_ of this frame and
// prev_f0_ (still the previous frame's f0 here).
float Detector::gate_update(const float* power, float f0) {
    const int lo = LOCAL_LO_BIN, hi = LOCAL_HI_BIN, nb = hi - lo + 1;
    float dlogf0 = prev_f0_ > 0 ? fabsf(logf(f0) - logf(prev_f0_)) : 0.0f;
    float flux = 0;
    if (have_prev_) { for (int i = 0; i < nb; i++) flux += fabsf(sdb_[lo + i] - prev_sb_[i]); flux /= (float)nb; }
    for (int i = 0; i < nb; i++) prev_sb_[i] = sdb_[lo + i];
    have_prev_ = true;
    float ff = 0.5f * f0, fund = FUND_MISSING_DB;            // prominence at the shaft rate f0 / 2
    if (ff >= FUND_BAND_LO && ff <= FUND_BAND_HI) {
        int b = (int)lroundf(ff / BIN_HZ); if (b < 1) b = 1; if (b > NBINS - 2) b = NBINS - 2;
        fund = savg_[b - 1]; if (savg_[b] > fund) fund = savg_[b]; if (savg_[b + 1] > fund) fund = savg_[b + 1];
    }
    float r = f0 / BIN_HZ; int inharm = 0;                   // proud local maxima not within 1 bin of any k * f0
    for (int b = lo; b <= hi; b++) {
        float c = savg_[b];
        if (c > GATE_PROUD_DB && c >= savg_[b - 1] && c >= savg_[b + 1]) {
            float k0 = roundf((float)b / r); if (k0 < 1) k0 = 1;   // the only harmonic that can be within 1 bin
            if (fabsf((float)b - roundf(k0 * r)) > 1.0f) inharm++;
        }
    }
    float e = 0; for (int b = lo; b <= hi; b++) e += power[b];
    float level = 10.0f * log10f(e + 1e-20f);
    float lfpk = savg_[LFPK_LO_BIN]; for (int b = LFPK_LO_BIN + 1; b <= LFPK_HI_BIN; b++) if (savg_[b] > lfpk) lfpk = savg_[b];
    const float v[6] = {dlogf0, flux, fund, (float)inharm, lfpk, level};
    for (int i = 0; i < 6; i++) g_ring_[i][g_head_] = v[i];
    g_head_ = (g_head_ + 1) % GATE_FRAMES; if (g_count_ < GATE_FRAMES) g_count_++;
    float m[6];                                              // the ring fills from index 0: the first g_count_ are valid
    for (int i = 0; i < 6; i++) { float s = 0; for (int j = 0; j < g_count_; j++) s += g_ring_[i][j]; m[i] = s / g_count_; }
    float var = 0; for (int j = 0; j < g_count_; j++) { float d = g_ring_[5][j] - m[5]; var += d * d; }
    float level_std = sqrtf(var / g_count_);
    return GATE_W_DLOGF0 * m[0] + GATE_W_FLUX * m[1] + GATE_W_FUND_HALF * m[2] + GATE_W_INHARM * m[3]
         + GATE_W_LF_PEAK * m[4] + GATE_W_LEVEL_STD * level_std;
}

void Detector::feed(const float* power, bool use_ml, FrameEvidence& ev) {
    for (int k = 0; k < NBINS; k++) pdb_[k] = 10.0f * log10f(power[k] + 1e-12f);
    spectral_baseline(pdb_, base_, tmp_);
    for (int k = 0; k < NBINS; k++) sdb_[k] = pdb_[k] - base_[k];
    if (!savg_init_) { for (int k = 0; k < NBINS; k++) savg_[k] = sdb_[k]; savg_init_ = true; }
    else for (int k = 0; k < NBINS; k++) savg_[k] = PROM_EMA * sdb_[k] + (1 - PROM_EMA) * savg_[k];
    float f0, score; int nh;
    harmonic_search(savg_, prev_f0_, &f0, &score, &nh);
    float flat = spectral_flatness(power);
    float flat_local = local_flatness(sdb_);
    f0_hist_[fp_] = f0; fp_ = (fp_ + 1) % F0_HIST_FRAMES; if (f0_count_ < F0_HIST_FRAMES) f0_count_++;
    float steady_frac = 0;
    if (f0_count_ >= F0_STEADY_MIN) {               // fraction of the last ~1 s of f0 near its median
        static float srt[F0_HIST_FRAMES];           // ring fills from index 0, so the first f0_count_ are valid
        memcpy(srt, f0_hist_, sizeof(float) * (size_t)f0_count_);
        qsort(srt, (size_t)f0_count_, sizeof(float), cmpf);
        float med = srt[(f0_count_ - 1) / 2];       // lower median: an exact element
        float tol = F0_MAX_JUMP * med;
        int cnt = 0;
        for (int i = 0; i < f0_count_; i++) if (fabsf(f0_hist_[i] - med) <= tol) cnt++;
        steady_frac = (float)cnt / (float)f0_count_;
    }
    bool steady = steady_frac >= F0_STEADY_FRAC;
#if FLAT_MODE_LOCAL
    bool tonal_ok = flat_local > FLAT_LOCAL_MIN && flat_local < FLAT_LOCAL_MAX;
#else
    bool tonal_ok = flat > FLAT_MIN && flat < FLAT_THRESH;
#endif
    bool ok = score > HARM_THRESH_DB && tonal_ok && steady && nh >= MIN_HARMONICS;
    float gate = gate_update(power, f0);
    bool gate_ok = GATE_ON ? gate >= GATE_THRESH : true;
    prev_f0_ = f0;
    hist_[hp_] = ok ? 1 : 0; scores_[hp_] = score; hp_ = (hp_ + 1) % PERSIST_FRAMES;
    int hits = 0; for (int i = 0; i < PERSIST_FRAMES; i++) hits += hist_[i];
    bool ml_ok = use_ml ? (ml_prob_ > ML_THRESH || ml_age_ < ML_HOLD_FRAMES) : true;
    if (ml_age_ < ML_HOLD_FRAMES) ml_age_++;
    if (!detected_) { if (hits >= PERSIST_ON && ml_ok && gate_ok) { detected_ = true; release_ = 0; } }
    else { if (hits <= PERSIST_OFF) { if (++release_ >= RELEASE_FRAMES) detected_ = false; } else release_ = 0; }
    ev.f0 = f0; ev.score = score; ev.flat = flat; ev.flat_local = flat_local; ev.n_harm = nh; ev.f0_steady = steady_frac;
    ev.frame_ok = ok; ev.gate = gate; ev.gate_ok = gate_ok;
    ev.hits = hits; ev.detected = detected_; ev.strength = percentile(scores_, PERSIST_FRAMES, 50);
    if (ok) harmonic_mask(f0, ev.mask); else band_mask(ev.mask);
}

// ================================================================ bearing
// Rachel's robust solver (sig_proc_tri_robust.py estimate_direction_details) on one DSP frame; mirrors
// droneears/bearing.py line for line (float here, float64 there). See that file's docstring for the method.
const char* const BEARING_STATUS_NAME[5] = {"unreliable", "ambiguous", "azimuth_only", "constrained_3d", "full_3d"};

// e^{+j 2 pi k / NFFT} for the bearing bins k = BRG_LO_BIN..BRG_HI_BIN (1 KB; the FFT's own sine tables are not
// linked on the ESP32, where esp-dsp does the FFT, and would cost 8 KB of internal RAM)
static float s_bwc[BRG_NB], s_bws[BRG_NB];
static bool s_bw_ready = false;
static void bearing_tables_init() {
    if (s_bw_ready) return;
    for (int t = 0; t < BRG_NB; t++) {
        double a = 2.0 * M_PI * (double)(BRG_LO_BIN + t) / NFFT;
        s_bwc[t] = (float)cos(a); s_bws[t] = (float)sin(a);
    }
    s_bw_ready = true;
}

// rx[L + n] = sum_t xr_t cos(2 pi b_t n / N) - xi_t sin(2 pi b_t n / N) for n = -L..L (= N/2 x the inverse FFT of the
// spectrum xr + j xi on bins b, exactly, at those integer lags); the same for (yr, yi) -> ry when Y. The phasor
// e^{j 2 pi b n / N} advances by one complex multiply per lag (28 steps at most: ~1e-6 rounding drift); one phasor
// serves both curves and both signs of the lag, and two bins run side by side (two independent multiply chains).
// Cost: nb x (L + 1) x (4 rotation + 2 or 4 accumulate) multiply-adds.
template <bool Y>
static void lag_curves_t(const int16_t* b, const float* xr, const float* xi, const float* yr, const float* yi, int nb,
                         int L, float* rx, float* ry) {
    float ax[BRG_MAX_LAG + 1], bx[BRG_MAX_LAG + 1], ay[BRG_MAX_LAG + 1], by[BRG_MAX_LAG + 1];
    for (int n = 0; n <= L; n++) ax[n] = bx[n] = ay[n] = by[n] = 0.0f;
    int t = 0;
    for (; t + 1 < nb; t += 2) {
        const float wc0 = s_bwc[b[t] - BRG_LO_BIN], ws0 = s_bws[b[t] - BRG_LO_BIN];
        const float wc1 = s_bwc[b[t + 1] - BRG_LO_BIN], ws1 = s_bws[b[t + 1] - BRG_LO_BIN];
        const float pr0 = xr[t], pi0 = xi[t], pr1 = xr[t + 1], pi1 = xi[t + 1];
        const float qr0 = Y ? yr[t] : 0.0f, qi0 = Y ? yi[t] : 0.0f, qr1 = Y ? yr[t + 1] : 0.0f, qi1 = Y ? yi[t + 1] : 0.0f;
        float zr0 = 1.0f, zi0 = 0.0f, zr1 = 1.0f, zi1 = 0.0f;
        for (int n = 0; n <= L; n++) {
            ax[n] += pr0 * zr0 + pr1 * zr1; bx[n] += pi0 * zi0 + pi1 * zi1;
            if (Y) { ay[n] += qr0 * zr0 + qr1 * zr1; by[n] += qi0 * zi0 + qi1 * zi1; }
            float u0 = zr0 * wc0 - zi0 * ws0, u1 = zr1 * wc1 - zi1 * ws1;
            zi0 = zr0 * ws0 + zi0 * wc0; zi1 = zr1 * ws1 + zi1 * wc1;
            zr0 = u0; zr1 = u1;
        }
    }
    for (; t < nb; t++) {
        const float wc = s_bwc[b[t] - BRG_LO_BIN], ws = s_bws[b[t] - BRG_LO_BIN];
        const float pr = xr[t], pi = xi[t], qr = Y ? yr[t] : 0.0f, qi = Y ? yi[t] : 0.0f;
        float zr = 1.0f, zi = 0.0f;
        for (int n = 0; n <= L; n++) {
            ax[n] += pr * zr; bx[n] += pi * zi;
            if (Y) { ay[n] += qr * zr; by[n] += qi * zi; }
            float u = zr * wc - zi * ws; zi = zr * ws + zi * wc; zr = u;
        }
    }
    for (int n = 0; n <= L; n++) {
        rx[L + n] = ax[n] - bx[n]; rx[L - n] = ax[n] + bx[n];
        if (Y) { ry[L + n] = ay[n] - by[n]; ry[L - n] = ay[n] + by[n]; }
    }
}
static void lag_curves(const int16_t* b, const float* xr, const float* xi, const float* yr, const float* yi, int nb,
                       int L, float* rx, float* ry) {
    if (ry) lag_curves_t<true>(b, xr, xi, yr, yi, nb, L, rx, ry);
    else lag_curves_t<false>(b, xr, xi, nullptr, nullptr, nb, L, rx, nullptr);
}

static float interp_lag(const float* v, int L, float x) {   // np.interp on the lags -L..L
    int i = (int)floorf(x); if (i < -L) i = -L; if (i > L - 1) i = L - 1;
    float f = x - (float)i;
    return v[i + L] + (v[i + 1 + L] - v[i + L]) * f;
}

// her _ambiguous_correlation_peak: competing local maxima inside the physical interval [-bound, bound] (samples;
// the interpolated endpoints count): the runner-up >= BEARING_ALIAS_RATIO x the best (> 0)
static bool ambiguous_peak(const float* v, int L, float bound) {
    float b[2 * BRG_MAX_LAG + 3]; int nb = 0;
    b[nb++] = interp_lag(v, L, -bound);
    for (int n = -L; n <= L; n++) if (fabsf((float)n) < bound) b[nb++] = v[n + L];
    b[nb++] = interp_lag(v, L, bound);
    float best = -INFINITY, run = -INFINITY; int np = 0;
    for (int i = 0; i < nb; i++) {
        bool pk = i == 0 ? b[0] > b[1] : i == nb - 1 ? b[nb - 1] > b[nb - 2] : (b[i] >= b[i - 1] && b[i] > b[i + 1]);
        if (!pk) continue;
        np++;
        if (b[i] > best) { run = best; best = b[i]; } else if (b[i] > run) run = b[i];
    }
    return np >= 2 && best > 0 && run >= BEARING_ALIAS_RATIO * best;
}

// one pair's masked cross spectrum (single-threaded scratch, ~2.9 KB)
static int16_t s_gb[BRG_NB], s_sb[BRG_NB];
static float s_pr[BRG_NB], s_pi[BRG_NB], s_er[BRG_NB], s_ei[BRG_NB], s_mg[BRG_NB];

float gcc_phat(const Spectrum& Xi, const Spectrum& Xj, const float* mask, float tau_max_s, float* peak_out,
               bool* alias_out) {
    bearing_tables_init();
    if (peak_out) *peak_out = 0.0f;
    if (alias_out) *alias_out = false;
    int nb = 0; float band_peak = 0.0f;
    for (int k = BRG_LO_BIN; k <= BRG_HI_BIN; k++) {     // G = conj(Xi) Xj x mask: peaks at +D when x_j = x_i delayed by D
        if (!(mask[k] > 0.0f)) continue;
        float gr = (Xi.re[k] * Xj.re[k] + Xi.im[k] * Xj.im[k]) * mask[k];
        float gi = (Xi.re[k] * Xj.im[k] - Xi.im[k] * Xj.re[k]) * mask[k];
        float mg = sqrtf(gr * gr + gi * gi);
        s_gb[nb] = (int16_t)k; s_pr[nb] = gr; s_pi[nb] = gi; s_mg[nb] = mg; nb++;
        if (mg > band_peak) band_peak = mg;
    }
    if (nb == 0) return NAN;
    const float thr = fmaxf(band_peak * 1e-10f, 1e-20f), ibp = 1.0f / band_peak;
    int na = 0;                                          // keep the bins with energy: PHAT (pr, pi) and energy (er, ei)
    for (int t = 0; t < nb; t++) {
        float mg = s_mg[t];
        if (!(mg > thr)) continue;
        float gr = s_pr[t], gi = s_pi[t], im = 1.0f / mg;    // one divide per bin (the S3 FPU divides in ~tens of cycles)
        s_gb[na] = s_gb[t]; s_mg[na] = mg;
        s_pr[na] = gr * im; s_pi[na] = gi * im; s_er[na] = gr * ibp; s_ei[na] = gi * ibp;
        na++;
    }
    if (na == 0) return NAN;
    int L = (int)ceilf(tau_max_s * FS) + 1;
    if (L > BRG_MAX_LAG) L = BRG_MAX_LAG;
    const float bound = tau_max_s * FS;
    float r[2 * BRG_MAX_LAG + 1], re[2 * BRG_MAX_LAG + 1];
    lag_curves(s_gb, s_pr, s_pi, s_er, s_ei, na, L, r, re);
    const float inv = 1.0f / (float)na;
    for (int i = 0; i <= 2 * L; i++) r[i] *= inv;        // 1 at a lag where every bin is in phase
    bool alias = ambiguous_peak(r, L, bound);
    if (!alias) {                                        // PHAT over the well-supported bins only (her leakage check)
        const float sthr = fmaxf(band_peak * 1e-6f, 1e-20f);
        int ns = 0;
        for (int t = 0; t < na; t++)
            if (s_mg[t] > sthr) { s_sb[ns] = s_gb[t]; s_er[ns] = s_pr[t]; s_ei[ns] = s_pi[t]; ns++; }
        if (ns < na) {
            float rs[2 * BRG_MAX_LAG + 1];
            if (!ambiguous_peak(re, L, bound)) {         // the energy curve first: s_er / s_ei are reused below
                lag_curves(s_sb, s_er, s_ei, nullptr, nullptr, ns, L, rs, nullptr);
                alias = ambiguous_peak(rs, L, bound);
            } else alias = true;
        } else alias = ambiguous_peak(re, L, bound);
    }
    if (alias) { if (alias_out) *alias_out = true; return NAN; }
    int k = 0;
    for (int i = 1; i <= 2 * L; i++) if (r[i] > r[k]) k = i;
    float frac = 0.0f, val = r[k];
    if (k > 0 && k < 2 * L) {
        float y0 = r[k - 1], y1 = r[k], y2 = r[k + 1], den = y0 - 2 * y1 + y2;
        if (den < -1e-12f) {
            frac = 0.5f * (y0 - y2) / den; if (frac > 0.5f) frac = 0.5f; if (frac < -0.5f) frac = -0.5f;
            val = y1 - 0.25f * (y0 - y2) * frac;
        }
    }
    if (peak_out) *peak_out = val;
    float tau = ((float)(k - L) + frac) / (float)FS;
    if (tau > tau_max_s) tau = tau_max_s;
    if (tau < -tau_max_s) tau = -tau_max_s;
    return tau;
}

void least_squares_direction(const float tau[N_PAIRS], float u[3], float* conf) {
    least_squares_direction(A_PINV, tau, u, conf);
}

void least_squares_direction(const float pinv[3][N_PAIRS], const float tau[N_PAIRS], float u[3], float* conf) {
    float n2 = 0;
    for (int r = 0; r < 3; r++) {
        float acc = 0;
        for (int p = 0; p < N_PAIRS; p++) acc += pinv[r][p] * tau[p];
        u[r] = acc; n2 += acc * acc;
    }
    float n = sqrtf(n2);
    if (n > 0) for (int r = 0; r < 3; r++) u[r] /= n;
    if (conf) *conf = n;
}

float elevation_deg(const float u[3]) {
    float n = sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    float s = u[2] / (n > 1e-12f ? n : 1e-12f);
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    return asinf(s) * 180.0f / (float)M_PI;
}

// Symmetric 3x3 eigen-decomposition (cyclic Jacobi): w ascending, V[:][k] the k-th unit eigenvector. T = double for
// mic_layout (once per calibration), float for the per-frame fit (the S3 has no double FPU).
template <typename T>
static void sym3_eigen(T a[3][3], T w[3], T V[3][3]) {
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) V[i][j] = i == j ? (T)1 : (T)0;
    const T tol = sizeof(T) == sizeof(double) ? (T)1e-30 : (T)1e-14;
    for (int sweep = 0; sweep < (sizeof(T) == sizeof(double) ? 50 : 8); sweep++) {
        T off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        T on = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (off <= tol * on || off == (T)0) break;
        for (int p = 0; p < 2; p++)
            for (int q = p + 1; q < 3; q++) {
                if (a[p][q] == (T)0) continue;
                T th = (a[q][q] - a[p][p]) / ((T)2 * a[p][q]);
                T t = (th >= 0 ? (T)1 : (T)-1) / (std::fabs(th) + std::sqrt(th * th + (T)1));
                T c = (T)1 / std::sqrt(t * t + (T)1), s = t * c;
                for (int k = 0; k < 3; k++) { T x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
                for (int k = 0; k < 3; k++) { T x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
                for (int k = 0; k < 3; k++) { T x = V[k][p], y = V[k][q]; V[k][p] = c * x - s * y; V[k][q] = s * x + c * y; }
            }
    }
    for (int k = 0; k < 3; k++) w[k] = a[k][k];
    for (int i = 1; i < 3; i++)                                   // ascending, columns of V along
        for (int j = i; j > 0 && w[j - 1] > w[j]; j--) {
            T t = w[j]; w[j] = w[j - 1]; w[j - 1] = t;
            for (int k = 0; k < 3; k++) { t = V[k][j]; V[k][j] = V[k][j - 1]; V[k][j - 1] = t; }
        }
}

void mic_layout(const bool active[N_MICS], const float pos[N_MICS][3], MicLayout& L) {
    double A[N_PAIRS][3];
    for (int c = 0; c < N_MICS; c++) {
        L.active[c] = active[c];
        for (int r = 0; r < 3; r++) L.pos[c][r] = pos[c][r];
    }
    for (int p = 0; p < N_PAIRS; p++) {
        int i = PAIR_I[p], j = PAIR_J[p];
        bool on = active[i] && active[j];
        double n2 = 0;
        for (int r = 0; r < 3; r++) {
            double d = on ? ((double)pos[i][r] - (double)pos[j][r]) : 0.0;
            A[p][r] = d / C_SOUND; n2 += d * d;
        }
        L.tau_max[p] = (float)(sqrt(n2) / C_SOUND);
    }
    double M[3][3], w[3], V[3][3];                                // M = A^T A
    for (int r = 0; r < 3; r++)
        for (int s = 0; s < 3; s++) {
            double acc = 0;
            for (int p = 0; p < N_PAIRS; p++) acc += A[p][r] * A[p][s];
            M[r][s] = acc;
        }
    sym3_eigen<double>(M, w, V);
    double wmax = w[2] > 1e-30 ? w[2] : 1e-30;
    bool keep[3]; L.rank = 0;
    for (int k = 0; k < 3; k++) { keep[k] = w[k] > LAYOUT_RCOND * wmax; L.rank += keep[k]; }
    for (int r = 0; r < 3; r++)                                   // pinv = V_k diag(1/w_k) V_k^T A^T over the kept axes
        for (int p = 0; p < N_PAIRS; p++) {
            double acc = 0;
            for (int k = 0; k < 3; k++) {
                if (!keep[k]) continue;
                double vta = V[0][k] * A[p][0] + V[1][k] * A[p][1] + V[2][k] * A[p][2];
                acc += V[r][k] * vta / w[k];
            }
            L.pinv[r][p] = (float)acc;
        }
    double nz = L.rank == 2 ? fabs(V[2][0]) : 0.0;                // V[:][0]: the unobservable axis (the plane's normal)
    L.level = L.rank == 2 && nz > LAYOUT_LEVEL_Z;
    bool vertical = L.rank == 2 && nz < 1.0 - LAYOUT_LEVEL_Z;
    bool prior = L.rank == 2 && BEARING_UPPER_PRIOR && !vertical;
    L.azimuth_ok = L.rank == 3 || L.level || prior;
    L.elevation_ok = L.rank == 3 || vertical || prior;
}

// her _weighted_geometry through the 3x3 normal equations: minimum-norm fit over the singular values above
// sigma_max / BEARING_MAX_COND. Returns the rank; p, V (columns ascending), sig (ascending).
static int wls(const float G[N_PAIRS][3], const float d[N_PAIRS], const float w[N_PAIRS], float p[3], float V[3][3],
               float sig[3]) {
    float M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, bv[3] = {0, 0, 0};
    bool any = false;
    p[0] = p[1] = p[2] = 0.0f;
    for (int q = 0; q < N_PAIRS; q++) {
        if (!(w[q] > 0.0f)) continue;
        any = true;
        for (int r = 0; r < 3; r++) {
            bv[r] += G[q][r] * (w[q] * d[q]);
            for (int s = 0; s < 3; s++) M[r][s] += G[q][r] * w[q] * G[q][s];
        }
    }
    if (!any) {
        for (int i = 0; i < 3; i++) { sig[i] = 0; for (int j = 0; j < 3; j++) V[i][j] = i == j ? 1.0f : 0.0f; }
        return 0;
    }
    float lam[3];
    sym3_eigen<float>(M, lam, V);
    for (int k = 0; k < 3; k++) sig[k] = sqrtf(fmaxf(lam[k], 0.0f));
    if (sig[2] <= 1e-15f) return 0;
    const float th = fmaxf(sig[2] / BEARING_MAX_COND, 1e-15f);
    int rank = 0;
    for (int k = 0; k < 3; k++) {
        if (!(sig[k] > th)) continue;
        rank++;
        float c = (V[0][k] * bv[0] + V[1][k] * bv[1] + V[2][k] * bv[2]) / lam[k];
        for (int r = 0; r < 3; r++) p[r] += V[r][k] * c;
    }
    return rank;
}

static float wrms(const float G[N_PAIRS][3], const float d[N_PAIRS], const float w[N_PAIRS], const float u[3]) {
    float num = 0, den = 0;
    for (int q = 0; q < N_PAIRS; q++) {
        if (!(w[q] > 0.0f)) continue;
        float r = G[q][0] * u[0] + G[q][1] * u[1] + G[q][2] * u[2] - d[q];
        num += w[q] * r * r; den += w[q];
    }
    return den > 0 ? sqrtf(num / den) : 0.0f;
}

static float azimuth_of(const float u[3]) { return atan2f(u[0], u[1]) * 180.0f / (float)M_PI; }
static bool has_azimuth(const float u[3]) { return sqrtf(u[0] * u[0] + u[1] * u[1]) >= BEARING_MIN_HORIZ; }

// her _hemisphere_intervals + branch rule for the candidates p + h n (0) and p - h n (1) and a prior normal: the branch
// the prior selects within the 3-tolerance delay-error bound, or -1
static int prior_branch(const float p[3], const float n[3], const float sig[3], float sumw, const float prior[3]) {
    if (!(sig[1] > 0.0f)) return -1;
    float eps = 3.0f * BEARING_SCALE_M * sqrtf(sumw) / sig[1];
    if (!isfinite(eps)) return -1;
    float r = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    float rl = fmaxf(0.0f, r - eps);
    if (rl > 1.0f) return -1;
    float rh = fminf(1.0f, r + eps);
    float hl = sqrtf(fmaxf(0.0f, 1 - rh * rh)), hh = sqrtf(fmaxf(0.0f, 1 - rl * rl));
    float a = prior[0] * n[0] + prior[1] * n[1] + prior[2] * n[2];
    float center = prior[0] * p[0] + prior[1] * p[1] + prior[2] * p[2];
    float ip[3] = {prior[0] - a * n[0], prior[1] - a * n[1], prior[2] - a * n[2]};
    float spread = eps * sqrtf(ip[0] * ip[0] + ip[1] * ip[1] + ip[2] * ip[2]);
    float lower[2], upper[2];
    for (int k = 0; k < 2; k++) {
        float s = k == 0 ? 1.0f : -1.0f, lo = s * a * hl, hi = s * a * hh;
        lower[k] = center - spread + fminf(lo, hi);
        upper[k] = center + spread + fmaxf(lo, hi);
    }
    bool ok0 = lower[0] > 1e-9f && upper[1] < -1e-9f, ok1 = lower[1] > 1e-9f && upper[0] < -1e-9f;
    return ok0 == ok1 ? -1 : ok0 ? 0 : 1;
}

static float wrap_diff(float a, float b) { float d = fmodf(a - b + 540.0f, 360.0f) - 180.0f; return fabsf(d); }

struct Fit {                          // one run of her fit (robust_direction): what Bearing reports of it
    int status, rank, mused;
    float support, conf, w[N_PAIRS];
    float u[3]; bool u_ok;
    float raw; bool raw_ok;
    float elev; bool elev_ok;
    float hz[2]; bool hz_ok;           // the horizontal vector the smoothed theta takes
};

// her estimate_direction_details from the initial pair weights w0 on: Huber, residual outliers, status
static void fit(const float G[N_PAIRS][3], const float d[N_PAIRS], const float w0[N_PAIRS], const float peaks[N_PAIRS],
                const float coh[N_PAIRS], Fit& f) {
    float w[N_PAIRS], p[3], V[3][3], sig[3];
    for (int q = 0; q < N_PAIRS; q++) w[q] = w0[q];
    for (int it = 0; it < BEARING_HUBER_ITERS; it++) {           // Huber reweighting
        if (wls(G, d, w, p, V, sig) < 2) break;
        float mx = 0;
        for (int q = 0; q < N_PAIRS; q++) {
            float res = fabsf(G[q][0] * p[0] + G[q][1] * p[1] + G[q][2] * p[2] - d[q]);
            float upd = w0[q] * (res > BEARING_SCALE_M ? BEARING_SCALE_M / fmaxf(res, 1e-30f) : 1.0f);
            mx = fmaxf(mx, fabsf(upd - w[q]));
            w[q] = upd;
        }
        if (mx < 1e-6f) break;
    }
    wls(G, d, w, p, V, sig);
    for (int q = 0; q < N_PAIRS; q++)                            // residual outliers out, refit
        if (fabsf(G[q][0] * p[0] + G[q][1] * p[1] + G[q][2] * p[2] - d[q]) > 3.0f * BEARING_SCALE_M) w[q] = 0.0f;
    int rank = wls(G, d, w, p, V, sig);
    int mused = 0, nrel = 0; float sumw = 0, pkmin = 1.0f, cohw = 0;
    for (int q = 0; q < N_PAIRS; q++) {
        f.w[q] = w[q];
        if (!(w[q] > 0.0f)) continue;
        mused |= (1 << PAIR_I[q]) | (1 << PAIR_J[q]);
        nrel++; sumw += w[q]; cohw += coh[q] * w[q];
        pkmin = fminf(pkmin, fminf(1.0f, fmaxf(0.0f, peaks[q])));
    }
    float support = 0.0f;
    if (nrel) support = pkmin * (cohw / sumw) * sqrtf((float)nrel / (float)N_PAIRS) * expf(-wrms(G, d, w, p) / BEARING_SCALE_M);
    f.status = BS_UNRELIABLE; f.rank = rank; f.mused = mused; f.support = support; f.conf = 0.0f;
    f.u_ok = f.raw_ok = f.elev_ok = f.hz_ok = false;
    f.u[0] = f.u[1] = f.u[2] = 0.0f; f.raw = f.elev = 0.0f; f.hz[0] = f.hz[1] = 0.0f;
    if (rank >= 2) {
        float m = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        if (rank == 3) {
            if (m >= BEARING_MIN_MAG) {
                float u[3] = {p[0] / m, p[1] / m, p[2] / m};
                float urms = wrms(G, d, w, u);
                if (urms <= 3.0f * BEARING_SCALE_M) {
                    float cond = sig[0] > 1e-15f ? sig[2] / sig[0] : INFINITY;
                    float c = support * expf(-urms / BEARING_SCALE_M) * fminf(m, 1.0f / m) * fminf(1.0f, 3.0f / cond);
                    f.status = BS_FULL_3D; f.conf = fminf(1.0f, fmaxf(0.0f, c));
                    for (int r = 0; r < 3; r++) f.u[r] = u[r];
                    f.u_ok = true;
                }
            }
        } else {
            // the unit sphere meets p + t n in two mirror candidates; a projection outside the sphere by no more than the
            // delay-error bound (a source in the mic plane) gives the in-plane direction p / |p| (her rule: unreliable)
            float n[3] = {V[0][0], V[1][0], V[2][0]};
            float eps = sig[1] > 0.0f ? 3.0f * BEARING_SCALE_M * sqrtf(sumw) / sig[1] : INFINITY;
            if (m <= 1.0f + fminf(eps, 1.0f)) {
                float c0[3], c1[3];
                if (m > 1.0f) { for (int r = 0; r < 3; r++) c0[r] = c1[r] = p[r] / m; }
                else {
                    float h = sqrtf(fmaxf(0.0f, 1.0f - m * m));
                    for (int r = 0; r < 3; r++) { c0[r] = p[r] + h * n[r]; c1[r] = p[r] - h * n[r]; }
                }
                if (fmaxf(wrms(G, d, w, c0), wrms(G, d, w, c1)) <= 3.0f * BEARING_SCALE_M) {
                    f.conf = fminf(1.0f, fmaxf(0.0f, support));
                    static const float UP[3] = {0.0f, 0.0f, 1.0f};
                    int br = BEARING_UPPER_PRIOR ? prior_branch(p, n, sig, sumw, UP) : -1;
                    float hz[3] = {0.5f * (c0[0] + c1[0]), 0.5f * (c0[1] + c1[1]), 0.5f * (c0[2] + c1[2])};
                    if (br >= 0) {
                        const float* u = br == 0 ? c0 : c1;
                        f.status = BS_CONSTRAINED_3D;
                        for (int r = 0; r < 3; r++) f.u[r] = u[r];
                        f.u_ok = true;
                    } else if (fabsf(n[2]) > LAYOUT_LEVEL_Z && has_azimuth(hz)) {   // level mirror plane: azimuth
                        f.status = BS_AZIMUTH_ONLY;
                        f.raw = azimuth_of(hz); f.raw_ok = true;
                        f.hz[0] = hz[0]; f.hz[1] = hz[1]; f.hz_ok = true;
                    } else {
                        // what both candidates share within the delay-error bound: the candidates as far out of the
                        // plane as |p| - eps allows must agree (bearing.py robust_direction)
                        f.status = BS_AMBIGUOUS;
                        float rl = fmaxf(0.0f, m - eps), hmax = sqrtf(fmaxf(0.0f, 1.0f - rl * rl));
                        float w0[3], w1[3];
                        for (int r = 0; r < 3; r++) { w0[r] = p[r] + hmax * n[r]; w1[r] = p[r] - hmax * n[r]; }
                        float l0 = sqrtf(w0[0] * w0[0] + w0[1] * w0[1] + w0[2] * w0[2]);
                        float l1 = sqrtf(w1[0] * w1[0] + w1[1] * w1[1] + w1[2] * w1[2]);
                        float u0[3] = {w0[0] / l0, w0[1] / l0, w0[2] / l0}, u1[3] = {w1[0] / l1, w1[1] / l1, w1[2] / l1};
                        if (has_azimuth(u0) && has_azimuth(u1) && wrap_diff(azimuth_of(u0), azimuth_of(u1)) <= BEARING_SHARE_DEG) {
                            f.raw = azimuth_of(hz); f.raw_ok = true;
                            f.hz[0] = hz[0]; f.hz[1] = hz[1]; f.hz_ok = true;
                        }
                        if (fabsf(elevation_deg(w0) - elevation_deg(w1)) <= BEARING_SHARE_DEG) {
                            f.elev = 0.5f * (elevation_deg(c0) + elevation_deg(c1)); f.elev_ok = true;
                        }
                        if (!f.raw_ok && !f.elev_ok) f.conf = 0.0f;              // nothing shared: no direction
                    }
                }
            }
        }
    }
    if (f.u_ok) {
        if (has_azimuth(f.u)) { f.raw = azimuth_of(f.u); f.raw_ok = true; }
        f.elev = elevation_deg(f.u); f.elev_ok = true;
        f.hz[0] = f.u[0]; f.hz[1] = f.u[1]; f.hz_ok = true;
    }
    if (f.status == BS_UNRELIABLE) f.conf = 0.0f;
}

void robust_direction(const float taus[N_PAIRS], const float peaks[N_PAIRS], const float coh[N_PAIRS],
                      const bool on[N_PAIRS], const float pos[N_MICS][3], Bearing& out, float horiz[2], bool* horiz_ok) {
    float G[N_PAIRS][3], d[N_PAIRS], w0[N_PAIRS];
    for (int q = 0; q < N_PAIRS; q++) {
        int i = PAIR_I[q], j = PAIR_J[q];
        for (int r = 0; r < 3; r++) G[q][r] = pos[i][r] - pos[j][r];
        bool valid = on[q] && !isnan(taus[q]) && coh[q] >= BEARING_MIN_PAIR_COH;
        w0[q] = valid ? coh[q] * coh[q] * fminf(1.0f, fmaxf(0.0f, peaks[q])) : 0.0f;
        d[q] = isnan(taus[q]) ? 0.0f : C_SOUND * taus[q];
    }
    Fit f;
    fit(G, d, w0, peaks, coh, f);
    out.dropped = -1;
    int nm = 0;
    for (int c = 0; c < N_MICS; c++) nm += (f.mused >> c) & 1;
    if (f.status == BS_UNRELIABLE && f.rank >= 2 && nm == N_MICS) {   // leave one mic out (bearing.py robust_direction)
        Fit r; bool have = false; float best = 0.0f;
        for (int k = 0; k < N_MICS; k++) {
            float wk[N_PAIRS];
            for (int q = 0; q < N_PAIRS; q++) wk[q] = (PAIR_I[q] == k || PAIR_J[q] == k) ? 0.0f : w0[q];
            fit(G, d, wk, peaks, coh, r);
            if (r.status != BS_UNRELIABLE && (!have || r.support > best)) { f = r; best = r.support; have = true; out.dropped = k; }
        }
    }
    out.status = f.status; out.bstat = bstat_of(f.status); out.rank = f.rank; out.mused = f.mused;
    out.support = f.support; out.conf = f.conf;
    for (int q = 0; q < N_PAIRS; q++) out.weights[q] = f.w[q];
    for (int r = 0; r < 3; r++) out.u[r] = f.u[r];
    out.u_ok = f.u_ok; out.theta_raw = f.raw; out.raw_ok = f.raw_ok; out.elev = f.elev; out.elev_ok = f.elev_ok;
    horiz[0] = f.hz[0]; horiz[1] = f.hz[1]; *horiz_ok = f.hz_ok;
}

BearingEstimator::BearingEstimator() { bearing_tables_init(); coh_reset(); }

void BearingEstimator::set_layout(const MicLayout& L) { lay_ = L; has_lay_ = true; coh_reset(); }

void BearingEstimator::coh_reset() {
    memset(cP_, 0, sizeof(cP_)); memset(cSr_, 0, sizeof(cSr_)); memset(cSi_, 0, sizeof(cSi_));
    sw_ = sw2_ = swc_ = 0.0f; cn_ = 0;
}

// her bias-corrected magnitude-squared coherence of pair p over the masked bins, from the EMA'd spectra
float BearingEstimator::coh_pair(int p, int i, int j, const int16_t* bins, int nb) const {
    float den = sw2_ + 2.0f * COH_RHO1 * COH_RHO1 * swc_;
    float keff = den > 0.0f ? sw_ * sw_ / den : 0.0f;
    if (cn_ < BEARING_COH_MIN_FRAMES || keff <= 1.0f + 1e-6f || nb == 0) return 0.0f;
    float mi = 0, mj = 0;
    for (int t = 0; t < nb; t++) { int b = bins[t] - BRG_LO_BIN; mi = fmaxf(mi, cP_[i][b]); mj = fmaxf(mj, cP_[j][b]); }
    const float ti = fmaxf(mi * 1e-6f, 1e-20f), tj = fmaxf(mj * 1e-6f, 1e-20f);
    float num = 0, wsum = 0; bool any = false;
    for (int t = 0; t < nb; t++) {
        int b = bins[t] - BRG_LO_BIN;
        float Pi = cP_[i][b], Pj = cP_[j][b];
        if (!(Pi > ti && Pj > tj)) continue;
        float prod = Pi * Pj, sr = cSr_[p][b], si = cSi_[p][b];
        float msc = (sr * sr + si * si) / prod; if (msc > 1.0f) msc = 1.0f; if (msc < 0.0f) msc = 0.0f;
        float wt = sqrtf(prod);
        num += msc * wt; wsum += wt; any = true;
    }
    if (!any) return 0.0f;
    float bias = 1.0f / keff, c = (num / wsum - bias) / (1.0f - bias);
    return c < 0.0f ? 0.0f : c > 1.0f ? 1.0f : c;
}

void BearingEstimator::feed(const Spectrum* S, const float* mask, bool update, Bearing& out) {
    static const bool ALL[N_MICS] = {true, true, true, true};
    static_assert(N_MICS == 4, "ALL[] lists four mics");
    const bool* act = has_lay_ ? lay_.active : ALL;
    const float* tm = has_lay_ ? lay_.tau_max : TAU_MAX;
    const float (*pos)[3] = has_lay_ ? lay_.pos : MIC_POS;
    // pair coherence state: EMA of the auto and cross spectra over the bearing band (connected mics only)
    const float a = BEARING_COH_EMA, a1 = 1.0f - BEARING_COH_EMA;
    for (int c = 0; c < N_MICS; c++) {
        if (!act[c]) continue;
        for (int t = 0; t < BRG_NB; t++) cP_[c][t] = a1 * cP_[c][t] + a * power_at(S[c], BRG_LO_BIN + t);
    }
    for (int q = 0; q < N_PAIRS; q++) {
        const Spectrum& Xi = S[PAIR_I[q]]; const Spectrum& Xj = S[PAIR_J[q]];
        if (!act[PAIR_I[q]] || !act[PAIR_J[q]]) continue;
        for (int t = 0; t < BRG_NB; t++) {
            int k = BRG_LO_BIN + t;
            cSr_[q][t] = a1 * cSr_[q][t] + a * (Xi.re[k] * Xj.re[k] + Xi.im[k] * Xj.im[k]);
            cSi_[q][t] = a1 * cSi_[q][t] + a * (Xi.re[k] * Xj.im[k] - Xi.im[k] * Xj.re[k]);
        }
    }
    swc_ = (cn_ >= 1 ? a * a * a1 : 0.0f) + a1 * a1 * swc_;
    sw_ = a + a1 * sw_;
    sw2_ = a * a + a1 * a1 * sw2_;
    cn_++;
    int16_t bins[BRG_NB]; int nb = 0;
    for (int k = BRG_LO_BIN; k <= BRG_HI_BIN; k++) if (mask[k] > 0.0f) bins[nb++] = (int16_t)k;
    bool on[N_PAIRS];
    for (int q = 0; q < N_PAIRS; q++) {
        int i = PAIR_I[q], j = PAIR_J[q];
        on[q] = act[i] && act[j] && tm[q] > 0.0f;
        out.taus[q] = NAN; out.peaks[q] = 0.0f; out.coh[q] = 0.0f;
        if (!on[q]) continue;
        out.taus[q] = gcc_phat(S[i], S[j], mask, tm[q], &out.peaks[q], nullptr);
        out.coh[q] = coh_pair(q, i, j, bins, nb);
    }
    float hz[2]; bool hz_ok;
    robust_direction(out.taus, out.peaks, out.coh, on, pos, out, hz, &hz_ok);
    // smoothed direction: EMA of the horizontal vector (cos(elev) long) while detected and confident
    bool took = false;
    if (update && hz_ok && out.conf >= BEARING_MIN_CONF) {
        if (stale_ >= BEARING_HOLD_FRAMES) v_[0] = v_[1] = v_[2] = 0.0f;   // a stale EMA restarts from this frame
        float z = out.u_ok ? out.u[2] : out.elev_ok ? sinf(out.elev * (float)M_PI / 180.0f) : v_[2];
        const float in[3] = {hz[0], hz[1], z};
        for (int r = 0; r < 3; r++) v_[r] = (1 - BEARING_EMA) * v_[r] + BEARING_EMA * in[r];
        took = true;
    }
    if (took) stale_ = 0; else if (update) stale_++;
    float n = sqrtf(v_[0] * v_[0] + v_[1] * v_[1]);
    if (n > 1e-9f && stale_ < BEARING_HOLD_FRAMES) { out.theta = atan2f(v_[0], v_[1]) * 180.0f / (float)M_PI; out.theta_ok = true; }
    else { out.theta = out.theta_raw; out.theta_ok = out.raw_ok; }
    // overhead state machine (bearing.py BearingEstimator._update_overhead): frames without an elevation wait
    if (!update) {
        overhead_ = false; low_ = high_ = 0;
    } else if (!out.elev_ok) {
    } else if (!overhead_) {
        low_ = out.elev > OVERHEAD_ELEV_ON ? low_ + 1 : 0;
        if (low_ >= OVERHEAD_FRAMES) { overhead_ = true; low_ = high_ = 0; }
    } else {
        high_ = out.elev < OVERHEAD_ELEV_OFF ? high_ + 1 : 0;
        if (high_ >= OVERHEAD_FRAMES) { overhead_ = false; low_ = high_ = 0; }
    }
    out.overhead = overhead_;
}

void haptic_weights(float theta, bool overhead, float out[N_MOTORS]) {
    for (int m = 0; m < N_MOTORS; m++) out[m] = overhead ? 1.0f : 0.0f;
    if (overhead) return;
    // walk the motors in angle order (stable insertion sort of the indices, as numpy argsort kind="stable")
    int ord[N_MOTORS];
    for (int m = 0; m < N_MOTORS; m++) {
        int k = m;
        while (k > 0 && MOTOR_ANGLES[ord[k - 1]] > MOTOR_ANGLES[m]) { ord[k] = ord[k - 1]; k--; }
        ord[k] = m;
    }
    for (int k = 0; k < N_MOTORS; k++) {
        int i1 = ord[k], i2 = ord[(k + 1) % N_MOTORS];
        float a1 = MOTOR_ANGLES[i1], a2 = MOTOR_ANGLES[i2];
        float span = fmodf(a2 - a1 + 720.0f, 360.0f);
        float d = fmodf(theta - a1 + 720.0f, 360.0f);
        if (d <= span + 1e-6f) {
            float t = span > 0 ? d / span : 0;
            out[i1] = 1 - t; out[i2] = t; return;
        }
    }
}

void haptic_weights_at(float theta, bool overhead, const float* angles, const bool* fitted, float out[N_MOTORS]) {
    for (int m = 0; m < N_MOTORS; m++) out[m] = (overhead && fitted[m]) ? 1.0f : 0.0f;
    if (overhead) return;
    int ord[N_MOTORS], n = 0;                            // fitted motors in angle order (stable insertion sort)
    for (int m = 0; m < N_MOTORS; m++) {
        if (!fitted[m]) continue;
        int k = n++;
        while (k > 0 && angles[ord[k - 1]] > angles[m]) { ord[k] = ord[k - 1]; k--; }
        ord[k] = m;
    }
    if (n == 0) return;
    if (n == 1) { out[ord[0]] = 1.0f; return; }          // a single motor is the whole cue, whatever the bearing
    for (int k = 0; k < n; k++) {
        int i1 = ord[k], i2 = ord[(k + 1) % n];
        float a1 = angles[i1], a2 = angles[i2];
        float span = fmodf(a2 - a1 + 720.0f, 360.0f);
        float d = fmodf(theta - a1 + 720.0f, 360.0f);
        if (d <= span + 1e-6f) {
            float t = span > 0 ? d / span : 0;
            out[i1] = 1 - t; out[i2] = t; return;
        }
    }
}

} // namespace dsp
