// Portable DSP core for the headband drone detector.
// Mirrors droneears/{features,detector,bearing}.py. No heap after init, no dependencies beyond <math.h>.
// Compiles on Linux/macOS (test harness) and on the ESP32-S3 (Arduino / ESP-IDF).
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "dsp_config.h"

namespace dsp {

// ---------------------------------------------------------------- FFT
// In-place radix-2 complex FFT, N = NFFT. Replace with esp-dsp (dsps_fft2r_fc32) on the ESP32 for ~5x speed.
void fft_init();
void fft_forward(float* re, float* im);     // length NFFT each
void fft_inverse(float* re, float* im);     // scaled by 1/N

// ---------------------------------------------------------------- per-channel spectrum
struct Spectrum {                       // 8.2 KB; the power spectrum is |X|^2 = power_at(s, k), not stored
    float re[NBINS];
    float im[NBINS];
};
static inline float power_at(const Spectrum& s, int k) { return s.re[k] * s.re[k] + s.im[k] * s.im[k]; }
// x: NFFT samples (float, any scale). Applies the periodic Hann window and fills s.
void spectrum_from_frame(const float* x, Spectrum& s);

// ---------------------------------------------------------------- mel / classifier features
void mel_init();
void logmel_frame(const float* power, float* out /*NMEL*/);          // dB, un-normalised
struct ClsWindow {                                                    // ring of the last CLS_WINDOW_FRAMES log-mel rows
    float rows[CLS_WINDOW_FRAMES][NMEL];
    int head = 0;
    int count = 0;
    ClsWindow();                                                      // rows zero-filled: export_* read every row
    void push(const float* row);
    bool full() const { return count >= CLS_WINDOW_FRAMES; }
    // Normalised window (time-ordered), values in [0,1], then quantised to int8 with the model's input quant params.
    void export_int8(int8_t* out /*CLS_WINDOW_FRAMES*NMEL*/, float scale, int zero) const;
    void export_float(float* out /*CLS_WINDOW_FRAMES*NMEL*/) const;
};

// ---------------------------------------------------------------- harmonic detector
struct FrameEvidence {
    float f0;          // Hz
    float score;       // mean dB prominence over N_HARMONICS comb harmonics from the first >= HARM_FMIN
    float flat;        // spectral flatness in DET_BAND (0 tonal .. 1 white)
    float flat_local;  // flatness of the prominence spectrum (smooth shape divided out): the tonality test if FLAT_MODE_LOCAL
    int   n_harm;      // harmonics (of 8, from the first >= HARM_FMIN) > 6 dB proud
    float f0_steady;   // fraction of the last F0_HIST_FRAMES f0 within F0_MAX_JUMP of their (lower) median
    bool  frame_ok;
    int   hits;        // ok frames in the last PERSIST_FRAMES
    bool  detected;
    float strength;    // median score over the persistence window
    float gate;        // DSP gate score over the last GATE_FRAMES (onset needs >= GATE_THRESH when GATE_ON)
    bool  gate_ok;
    float mask[NBINS]; // bearing mask for this frame (harmonic comb if frame_ok, else band)
};

class Detector {
public:
    Detector();
    // classifier update (every ~0.5 s); detection onset needs the latest p, or any p in the last ML_HOLD_FRAMES, > ML_THRESH
    void set_ml(float p) { ml_prob_ = p; if (p > ML_THRESH) ml_age_ = 0; }
    // power: NBINS power spectrum of the mono sum. use_ml gates detection on ml_prob as well.
    void feed(const float* power, bool use_ml, FrameEvidence& ev);
    bool detected() const { return detected_; }
private:
    float prev_f0_ = 0;
    float f0_hist_[F0_HIST_FRAMES]; int fp_ = 0; int f0_count_ = 0;
    uint8_t hist_[PERSIST_FRAMES]; int hp_ = 0;
    float scores_[PERSIST_FRAMES];
    bool detected_ = false;
    int release_ = 0;
    float ml_prob_ = 0;
    int ml_age_ = ML_HOLD_FRAMES;                  // frames since the last classifier update above ML_THRESH
    float base_[NBINS], pdb_[NBINS], sdb_[NBINS], tmp_[NBINS];
    float savg_[NBINS]; bool savg_init_ = false;   // EMA (PROM_EMA) of the prominence spectrum the comb searches
    // DSP gate: per-frame inputs (dlogf0, flux, fund_half, inharm_peaks, lf_peak, level dB) for the last GATE_FRAMES
    float g_ring_[6][GATE_FRAMES]; int g_head_ = 0; int g_count_ = 0;
    float prev_sb_[LOCAL_HI_BIN - LOCAL_LO_BIN + 1]; bool have_prev_ = false;   // last frame's prominence, for flux
    float gate_update(const float* power, float f0);
};

// ---------------------------------------------------------------- bearing
// The connected mics (bearing.py mic_layout): active[c] says channel c (B, L, R, T) carries a mic, pos[c] where it
// sits (MIC_POS). tau_max is zero for the pairs missing a mic (the robust solver measures only the others); pinv is
// the plain least-squares solve over the connected pairs (kept for reference / tests; the per-frame solver weights
// the pairs itself). rank 3: the full 3D direction is observable. rank 2 (three mics): two mirror candidates; level
// (B, L, R): the azimuth always, the elevation from the upper-hemisphere prior when the drone is clearly above;
// a vertical plane (L, R, T): the elevation (shared by both candidates), no azimuth. azimuth_ok / elevation_ok say
// whether frames CAN carry them. Fewer than three mics: detection only.
struct MicLayout {
    bool  active[N_MICS];
    float pos[N_MICS][3];
    float tau_max[N_PAIRS];
    float pinv[3][N_PAIRS];
    int   rank;
    bool  level, azimuth_ok, elevation_ok;
};
void mic_layout(const bool active[N_MICS], const float pos[N_MICS][3], MicLayout& L);

// Per-frame status of the robust solver (bearing.py STATUS, same order); bstat is what the T line reports.
enum { BS_UNRELIABLE = 0, BS_AMBIGUOUS = 1, BS_AZIMUTH_ONLY = 2, BS_CONSTRAINED_3D = 3, BS_FULL_3D = 4 };
extern const char* const BEARING_STATUS_NAME[5];
// 0 unreliable / no bearing, 1 ambiguous, 2 constrained_3d or azimuth_only, 3 full_3d
static inline int bstat_of(int status) { return status == BS_FULL_3D ? 3 : status >= BS_AZIMUTH_ONLY ? 2 : status; }

struct Bearing {
    float theta;       // smoothed, degrees, 0 ahead, +90 right, +/-180 behind (azimuth of the EMA'd direction)
    bool  theta_ok;    //   false: no usable azimuth (none yet, or none for BEARING_HOLD_FRAMES detected frames and
                       //   none this frame): the T line leaves theta empty, the motors play the pulsed ring
    float theta_raw;   // this frame's azimuth (raw_ok false: none)
    bool  raw_ok;
    float elev;        // this frame's elevation, degrees: 0 in the head plane, +90 straight up (elev_ok false: unknown)
    bool  elev_ok;
    float conf;        // the solver's confidence (sig_proc_tri_robust: support x fit terms), 0..1; 0 without a direction
    float support;     // retained pair evidence (peak x coherence x pair count x fit), 0..1
    bool  overhead;    // drone above the head: elev > OVERHEAD_ELEV_ON for OVERHEAD_FRAMES known-elevation frames
    int   status;      // BS_* this frame
    int   bstat;       // bstat_of(status)
    int   mused;       // mics in this frame's fit, bitmask B 1, L 2, R 4, T 8
    int   rank;        // rank of the weighted pair geometry (0-3)
    int   dropped;     // the mic the leave-one-out step removed (0-3; -1 none)
    float taus[N_PAIRS];    // seconds, pairs (PAIR_I[p], PAIR_J[p]); NAN: no delay (pair off, silent or aliased)
    float peaks[N_PAIRS];   // PHAT correlation peak (1 = every masked bin in phase); 0 when no delay
    float coh[N_PAIRS];     // bias-corrected pair coherence over this frame's masked bins
    float weights[N_PAIRS]; // final fit weight (0: not used)
    float u[3];        // unit vector toward the source (x right, y forward, z up), valid when u_ok
    bool  u_ok;
};

class BearingEstimator {
public:
    BearingEstimator();
    // S: N_MICS spectra in MIC_NAMES order (B, L, R, T; only the connected ones are read); mask from the detector;
    // update (= detected) false keeps the smoothed value frozen and clears the overhead state.
    void feed(const Spectrum* S, const float* mask, bool update, Bearing& out);
    // only the connected mics from now on (the smoothed direction carries over, the coherence state restarts);
    // without it, all four at MIC_POS
    void set_layout(const MicLayout& L);
private:
    MicLayout lay_;
    bool has_lay_ = false;
    // pair coherence: EMA (BEARING_COH_EMA) of the auto and cross spectra over bins BRG_LO_BIN..BRG_HI_BIN (7.8 KB)
    float cP_[N_MICS][BRG_NB];
    float cSr_[N_PAIRS][BRG_NB], cSi_[N_PAIRS][BRG_NB];
    float sw_ = 0, sw2_ = 0, swc_ = 0;  // sum of the EMA weights, of their squares, of adjacent products
    int   cn_ = 0;                      // frames in the EMA
    float v_[3] = {0, 0, 0};           // EMA of the direction (horizontal part cos(elev) long)
    int   stale_ = 0;                  // detected frames since the EMA last took one
    bool overhead_ = false;
    int low_ = 0, high_ = 0;           // frames above OVERHEAD_ELEV_ON / below OVERHEAD_ELEV_OFF (unknown elev skipped)
    void coh_reset();
    float coh_pair(int p, int i, int j, const int16_t* bins, int nb) const;
};

// The per-frame decision (bearing.py robust_direction): from the pair delays (NAN = none), PHAT peaks, coherences
// and which pairs are on, fills status, bstat, mused, rank, dropped, weights, conf, support, u / u_ok, theta_raw /
// raw_ok, elev / elev_ok of out, and the horizontal vector the smoothed theta takes (horiz_ok false: none). When the
// fit over all four mics fails her consistency checks, each mic is left out in turn and the passing three-mic fit
// with the most support wins (a noisy mic that still passes the pair coherence test).
void robust_direction(const float taus[N_PAIRS], const float peaks[N_PAIRS], const float coh[N_PAIRS],
                      const bool on[N_PAIRS], const float pos[N_MICS][3], Bearing& out, float horiz[2], bool* horiz_ok);

// pan weights for the N_MOTORS motors at MOTOR_ANGLES (dsp_config.h; order F, FR, R, BR, B, BL, L, FL): the two
// motors bracketing theta share the weight linearly, sum = 1, at most 2 non-zero. overhead: all = 1 (a marker; the
// firmware decides how to drive it).
void haptic_weights(float theta_deg, bool overhead, float out[N_MOTORS]);
// The same panning rule over runtime motor angles (the headset calibration): only motors with fitted[m] take part;
// the two fitted motors bracketing theta (in angle order) share the weight linearly. One fitted motor gets 1.0 for any
// theta; none fitted -> all 0. overhead: every fitted motor = 1 (a marker). With MOTOR_ANGLES and all fitted the result
// equals haptic_weights().
void haptic_weights_at(float theta_deg, bool overhead, const float* angles_deg, const bool* fitted,
                       float out[N_MOTORS]);

// ---------------------------------------------------------------- helpers shared with tests
// Masked GCC-PHAT delay of j relative to i (seconds), evaluated at the integer lags within +-(ceil(tau_max fs) + 1)
// directly from the masked bins (no inverse FFT), parabolic peak, clipped to +-tau_max; NAN when no masked bin has
// energy or the correlation has a competing peak (period alias, *alias_out). *peak_out: the PHAT peak in [-1, 1].
float gcc_phat(const Spectrum& Xi, const Spectrum& Xj, const float* mask, float tau_max_s, float* peak_out,
               bool* alias_out);
void least_squares_direction(const float tau[N_PAIRS], float u[3], float* conf);
void least_squares_direction(const float pinv[3][N_PAIRS], const float tau[N_PAIRS], float u[3], float* conf);
float elevation_deg(const float u[3]);
float spectral_flatness(const float* power);
float local_flatness(const float* sdb);                // flatness of 10^(sdb/10), bins LOCAL_LO_BIN..LOCAL_HI_BIN
void spectral_baseline(const float* pdb, float* base, float* tmp);
// prev_f0 <= 0: no previous frame (plain argmax); else hysteresis toward candidates within F0_MAX_JUMP of prev_f0
void harmonic_search(const float* sdb, float prev_f0, float* f0, float* score, int* n_harm);
void harmonic_mask(float f0, float* mask);
void band_mask(float* mask);

} // namespace dsp
