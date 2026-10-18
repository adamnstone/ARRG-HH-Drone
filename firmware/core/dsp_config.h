// Auto-generated from droneears/config.py by scripts/export_c_headers.py. Do not edit by hand.
#pragma once
#define FS 48000
#define NFFT 2048
#define HOP 1024
#define NBINS 1025
#define BIN_HZ (48000.0f/2048.0f)
#define FRAMES_PER_SEC (48000.0f/1024.0f)
#define NMEL 40
#define MEL_FMIN 100.0f
#define MEL_FMAX 8000.0f
#define CLS_WINDOW_FRAMES 47
#define CLS_DB_FLOOR (-80.0f)
#define F0_MIN 120.0f
#define F0_MAX 400.0f
#define F0_STEP 2.0f
#define N_F0_CANDIDATES 141
#define N_HARMONICS 5
#define BASELINE_BINS 31
#define DET_BAND_LO 300.0f
#define DET_BAND_HI 3000.0f
#define HARM_FMIN 300.0f
#define PROM_EMA 0.5f
#define F0_HYST_DB 0.5f
#define HARM_THRESH_DB 5.5f
#define FLAT_THRESH 0.45f
#define FLAT_MIN 0.17f
#define F0_MAX_JUMP 0.06f
#define PERSIST_FRAMES 94
#define PERSIST_ON 70
#define PERSIST_OFF 20
#define RELEASE_FRAMES 93
#define F0_HIST_FRAMES 47
#define F0_STEADY_FRAC 0.35f
#define F0_STEADY_MIN 24
#define MIN_HARMONICS 3
#define ML_THRESH 0.7f
#define ML_HOLD_FRAMES 47
// tonality test: 1 = local flatness (prominence spectrum, bins LOCAL_LO_BIN..LOCAL_HI_BIN), 0 = global
#define FLAT_MODE_LOCAL 1
#define FLAT_LOCAL_MIN 0.3f
#define FLAT_LOCAL_MAX 1.0f
#define LOCAL_LO_BIN 13
#define LOCAL_HI_BIN 128
// DSP gate (replaces the classifier's veto on the board): onset needs sum(w * rolling feature) >= GATE_THRESH
#define GATE_ON 1
#define GATE_FRAMES 47
#define GATE_THRESH (-5.94529655f)
#define GATE_W_DLOGF0 (-1.142282f)
#define GATE_W_FLUX (-0.806491111f)
#define GATE_W_FUND_HALF (0.0489385064f)
#define GATE_W_INHARM (-0.367635477f)
#define GATE_W_LEVEL_STD (-0.24653551f)
#define GATE_W_LF_PEAK (0.092012103f)
#define GATE_PROUD_DB 6.0f
#define FUND_BAND_LO 100.0f
#define FUND_BAND_HI 400.0f
#define FUND_MISSING_DB (-40.0f)
#define LFPK_LO_BIN 2
#define LFPK_HI_BIN 12
#define C_SOUND 343.0f
#define BEARING_BAND_LO 200.0f
#define BEARING_BAND_HI 3000.0f
#define BEARING_MASK_HARMONICS 8
#define BEARING_MASK_HALFWIDTH 1
#define BEARING_EMA 0.1f
#define BEARING_MIN_CONF 0.2f
#define BEARING_HOLD_FRAMES 47
// robust per-frame solver (bearing.py robust_direction; Rachel's sig_proc_tri_robust.py)
#define BEARING_MIN_PAIR_COH 0.35f
#define BEARING_COH_EMA 0.25f
#define BEARING_COH_MIN_FRAMES 3
#define COH_RHO1 0.166666667f   // overlap correlation of adjacent (50 %) Hann frames
#define BEARING_DELAY_TOL 0.5f
#define BEARING_SCALE_M 0.00357291667f   // Huber scale, metres
#define BEARING_MAX_COND 20.0f
#define BEARING_HUBER_ITERS 8
#define BEARING_ALIAS_RATIO 0.9f
#define BEARING_MIN_MAG 0.05f
#define BEARING_MIN_HORIZ 0.05f
#define BEARING_UPPER_PRIOR 1
#define BEARING_SHARE_DEG 10.0f
// bins any bearing mask can set (the coherence state covers them); longest GCC lag over the pairs
#define BRG_LO_BIN 8
#define BRG_HI_BIN 129
#define BRG_NB 122
#define BRG_MAX_LAG 27
#define OVERHEAD_ELEV_ON 60.0f
#define OVERHEAD_ELEV_OFF 45.0f
#define OVERHEAD_FRAMES 23
#define N_MICS 4
#define N_PAIRS 6
#define N_MOTORS 8
// mic positions (m): x right, y forward, z up; order B, L, R, T
static const float MIC_POS[N_MICS][3] = {{0.0000f, -0.1000f, 0.0000f}, {-0.0900f, 0.0000f, 0.0000f}, {0.0900f, 0.0000f, 0.0000f}, {0.0000f, 0.0000f, 0.0900f}};
// mic pairs (i, j), tau_ij = arrival at j - arrival at i: (B,L) (B,R) (B,T) (L,R) (L,T) (R,T)
static const int PAIR_I[N_PAIRS] = {0, 0, 0, 1, 1, 2};
static const int PAIR_J[N_PAIRS] = {1, 2, 3, 2, 3, 3};
// max |tau| per pair, seconds
static const float TAU_MAX[N_PAIRS] = {3.922339372e-04f, 3.922339372e-04f, 3.922339372e-04f, 5.247813411e-04f, 3.710764449e-04f, 3.710764449e-04f};
// pseudo-inverse of A (tau = A u), 3 x N_PAIRS, row-major
static const float A_PINV[3][N_PAIRS] = {{4.763888889e+02f, -4.763888889e+02f, 0.000000000e+00f, -9.527777778e+02f, -4.763888889e+02f, 4.763888889e+02f}, {-1.286250000e+03f, -1.286250000e+03f, -8.575000000e+02f, 1.390787922e-13f, 4.287500000e+02f, 4.287500000e+02f}, {4.763888889e+02f, 4.763888889e+02f, -9.527777778e+02f, -9.716475099e-14f, -1.429166667e+03f, -1.429166667e+03f}};
// mic_layout tolerances
#define LAYOUT_RCOND 1e-06
#define LAYOUT_LEVEL_Z 0.9f
// channel 0 at the front of the head instead of the back (the headset calibration 'C...,F')
static const float FRONT_POS[3] = {0.0000f, 0.1000f, 0.0000f};
// haptic motor angles (deg, 0 ahead, +90 right); order F, FR, R, BR, B, BL, L, FL
static const float MOTOR_ANGLES[N_MOTORS] = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f, -135.0f, -90.0f, -45.0f};
