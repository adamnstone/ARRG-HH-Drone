"""Write firmware/core/dsp_config.h from droneears/config.py so the C core and the Python reference never drift."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from droneears import config as C
from droneears.bearing import A_PINV, TAU_MAX, BRG_LO, BRG_HI, COH_RHO1, SCALE_M, max_lag
from droneears.detector import LOCAL_LO, LOCAL_HI, LFPK_LO, LFPK_HI

GATE_C = {"dlogf0": "GATE_W_DLOGF0", "flux_db": "GATE_W_FLUX", "fund_half_db": "GATE_W_FUND_HALF",
          "inharm_peaks": "GATE_W_INHARM", "level_std": "GATE_W_LEVEL_STD", "lf_peak_db": "GATE_W_LF_PEAK"}
unknown = set(C.GATE_WEIGHTS) - set(GATE_C)
if unknown:
    sys.exit(f"config.GATE_WEIGHTS uses features the C core does not implement: {sorted(unknown)}")
if C.FLAT_MODE not in ("local", "global"):
    sys.exit(f"config.FLAT_MODE must be 'local' or 'global', not {C.FLAT_MODE!r}")

out = os.path.join(os.path.dirname(__file__), "..", "firmware", "core", "dsp_config.h")
L = ["// Auto-generated from droneears/config.py by scripts/export_c_headers.py. Do not edit by hand.",
     "#pragma once",
     f"#define FS {C.FS}", f"#define NFFT {C.NFFT}", f"#define HOP {C.HOP}", f"#define NBINS {C.NBINS}",
     f"#define BIN_HZ ({C.FS}.0f/{C.NFFT}.0f)", f"#define FRAMES_PER_SEC ({C.FS}.0f/{C.HOP}.0f)",
     f"#define NMEL {C.NMEL}", f"#define MEL_FMIN {C.MEL_FMIN}f", f"#define MEL_FMAX {C.MEL_FMAX}f",
     f"#define CLS_WINDOW_FRAMES {C.CLS_WINDOW_FRAMES}", f"#define CLS_DB_FLOOR ({C.CLS_DB_FLOOR}f)",
     f"#define F0_MIN {C.F0_MIN}f", f"#define F0_MAX {C.F0_MAX}f", f"#define F0_STEP {C.F0_STEP}f",
     f"#define N_F0_CANDIDATES {int(round((C.F0_MAX - C.F0_MIN) / C.F0_STEP)) + 1}",
     f"#define N_HARMONICS {C.N_HARMONICS}", f"#define BASELINE_BINS {C.BASELINE_BINS}",
     f"#define DET_BAND_LO {float(C.DET_BAND[0])}f", f"#define DET_BAND_HI {float(C.DET_BAND[1])}f",
     f"#define HARM_FMIN {float(C.HARM_FMIN)}f", f"#define PROM_EMA {float(C.PROM_EMA)}f",
     f"#define F0_HYST_DB {float(C.F0_HYST_DB)}f",
     f"#define HARM_THRESH_DB {float(C.HARM_THRESH_DB)}f", f"#define FLAT_THRESH {float(C.FLAT_THRESH)}f",
     f"#define FLAT_MIN {float(C.FLAT_MIN)}f",
     f"#define F0_MAX_JUMP {float(C.F0_MAX_JUMP)}f", f"#define PERSIST_FRAMES {C.PERSIST_FRAMES}",
     f"#define PERSIST_ON {C.PERSIST_ON}", f"#define PERSIST_OFF {C.PERSIST_OFF}",
     f"#define RELEASE_FRAMES {int(C.RELEASE_SEC * C.FRAMES_PER_SEC)}",
     f"#define F0_HIST_FRAMES {C.F0_HIST_FRAMES}", f"#define F0_STEADY_FRAC {float(C.F0_STEADY_FRAC)}f",
     f"#define F0_STEADY_MIN {int(C.F0_STEADY_MIN)}",
     f"#define MIN_HARMONICS {C.MIN_HARMONICS}", f"#define ML_THRESH {C.ML_THRESH}f", f"#define ML_HOLD_FRAMES {int(C.ML_HOLD_FRAMES)}",
     "// tonality test: 1 = local flatness (prominence spectrum, bins LOCAL_LO_BIN..LOCAL_HI_BIN), 0 = global",
     f"#define FLAT_MODE_LOCAL {1 if C.FLAT_MODE == 'local' else 0}",
     f"#define FLAT_LOCAL_MIN {float(C.FLAT_LOCAL_MIN)}f", f"#define FLAT_LOCAL_MAX {float(C.FLAT_LOCAL_MAX)}f",
     f"#define LOCAL_LO_BIN {LOCAL_LO}", f"#define LOCAL_HI_BIN {LOCAL_HI}",
     "// DSP gate (replaces the classifier's veto on the board): onset needs sum(w * rolling feature) >= GATE_THRESH",
     f"#define GATE_ON {1 if C.GATE_ON else 0}", f"#define GATE_FRAMES {int(C.GATE_FRAMES)}",
     f"#define GATE_THRESH ({C.GATE_THRESH:.9g}f)",
     *[f"#define {GATE_C[k]} ({C.GATE_WEIGHTS.get(k, 0.0):.9g}f)" for k in GATE_C],
     f"#define GATE_PROUD_DB {float(C.GATE_PROUD_DB)}f",
     f"#define FUND_BAND_LO {float(C.FUND_BAND[0])}f", f"#define FUND_BAND_HI {float(C.FUND_BAND[1])}f",
     f"#define FUND_MISSING_DB ({float(C.FUND_MISSING_DB)}f)",
     f"#define LFPK_LO_BIN {LFPK_LO}", f"#define LFPK_HI_BIN {LFPK_HI}",
     f"#define C_SOUND {C.C_SOUND}f",
     f"#define BEARING_BAND_LO {C.BEARING_BAND[0]}f", f"#define BEARING_BAND_HI {C.BEARING_BAND[1]}f",
     f"#define BEARING_MASK_HARMONICS {C.BEARING_MASK_HARMONICS}",
     f"#define BEARING_MASK_HALFWIDTH {C.BEARING_MASK_HALFWIDTH_BINS}",
     f"#define BEARING_EMA {C.BEARING_EMA}f", f"#define BEARING_MIN_CONF {C.BEARING_MIN_CONF}f",
     f"#define BEARING_HOLD_FRAMES {int(C.BEARING_HOLD_FRAMES)}",
     "// robust per-frame solver (bearing.py robust_direction; Rachel's sig_proc_tri_robust.py)",
     f"#define BEARING_MIN_PAIR_COH {float(C.BEARING_MIN_PAIR_COH)}f", f"#define BEARING_COH_EMA {float(C.BEARING_COH_EMA)}f",
     f"#define BEARING_COH_MIN_FRAMES {int(C.BEARING_COH_MIN_FRAMES)}",
     f"#define COH_RHO1 {COH_RHO1:.9g}f   // overlap correlation of adjacent (50 %) Hann frames",
     f"#define BEARING_DELAY_TOL {float(C.BEARING_DELAY_TOL)}f", f"#define BEARING_SCALE_M {SCALE_M:.9g}f   // Huber scale, metres",
     f"#define BEARING_MAX_COND {float(C.BEARING_MAX_COND)}f", f"#define BEARING_HUBER_ITERS {int(C.BEARING_HUBER_ITERS)}",
     f"#define BEARING_ALIAS_RATIO {float(C.BEARING_ALIAS_RATIO)}f", f"#define BEARING_MIN_MAG {float(C.BEARING_MIN_MAG)}f",
     f"#define BEARING_MIN_HORIZ {float(C.BEARING_MIN_HORIZ)}f", f"#define BEARING_UPPER_PRIOR {1 if C.BEARING_UPPER_PRIOR else 0}",
     f"#define BEARING_SHARE_DEG {float(C.BEARING_SHARE_DEG)}f",
     "// bins any bearing mask can set (the coherence state covers them); longest GCC lag over the pairs",
     f"#define BRG_LO_BIN {BRG_LO}", f"#define BRG_HI_BIN {BRG_HI}", f"#define BRG_NB {BRG_HI - BRG_LO + 1}",
     f"#define BRG_MAX_LAG {max(max_lag(t) for t in TAU_MAX)}",
     f"#define OVERHEAD_ELEV_ON {float(C.OVERHEAD_ELEV_ON)}f", f"#define OVERHEAD_ELEV_OFF {float(C.OVERHEAD_ELEV_OFF)}f",
     f"#define OVERHEAD_FRAMES {int(C.OVERHEAD_FRAMES)}",
     f"#define N_MICS {int(C.N_MICS)}", f"#define N_PAIRS {int(C.N_PAIRS)}", f"#define N_MOTORS {int(C.N_MOTORS)}",
     "// mic positions (m): x right, y forward, z up; order " + ", ".join(C.MIC_NAMES),
     "static const float MIC_POS[N_MICS][3] = {" + ", ".join("{" + ", ".join("%.4ff" % v for v in p) + "}" for p in C.MIC_POS) + "};",
     "// mic pairs (i, j), tau_ij = arrival at j - arrival at i: " + " ".join("(%s,%s)" % (C.MIC_NAMES[i], C.MIC_NAMES[j]) for i, j in C.PAIRS),
     "static const int PAIR_I[N_PAIRS] = {" + ", ".join(str(i) for i, _ in C.PAIRS) + "};",
     "static const int PAIR_J[N_PAIRS] = {" + ", ".join(str(j) for _, j in C.PAIRS) + "};",
     "// max |tau| per pair, seconds",
     "static const float TAU_MAX[N_PAIRS] = {" + ", ".join("%.9ef" % t for t in TAU_MAX) + "};",
     f"// pseudo-inverse of A (tau = A u), 3 x N_PAIRS, row-major",
     "static const float A_PINV[3][N_PAIRS] = {" + ", ".join("{" + ", ".join("%.9ef" % v for v in row) + "}" for row in A_PINV) + "};",
     "// mic_layout tolerances",
     f"#define LAYOUT_RCOND {C.LAYOUT_RCOND}", f"#define LAYOUT_LEVEL_Z {C.LAYOUT_LEVEL_Z}f",
     "// channel 0 at the front of the head instead of the back (the headset calibration 'C...,F')",
     "static const float FRONT_POS[3] = {" + ", ".join("%.4ff" % v for v in C.FRONT_POS) + "};",
     "// haptic motor angles (deg, 0 ahead, +90 right); order " + ", ".join(C.MOTOR_ANGLES),
     "static const float MOTOR_ANGLES[N_MOTORS] = {" + ", ".join("%.1ff" % v for v in C.MOTOR_ANGLES.values()) + "};",
     ]
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out)
