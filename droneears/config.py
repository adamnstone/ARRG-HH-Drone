"""Shared constants. Everything the ESP32 firmware and the Python reference must agree on lives here.

If you change a value here, regenerate firmware/core/dsp_config.h with scripts/export_c_headers.py.
"""
import numpy as np

# ---- capture
FS = 48000               # Hz, I2S sample rate on the ESP32-S3 and the rate all training audio is resampled to
NFFT = 2048              # FFT size: 42.7 ms frames, 23.44 Hz bins
HOP = 1024               # 50 % overlap: 21.3 ms per frame, 46.9 frames per second
NBINS = NFFT // 2 + 1    # 1025 rfft bins
BIN_HZ = FS / NFFT
FRAMES_PER_SEC = FS / HOP

# ---- classifier features (log-mel, computed from the same FFT as everything else)
NMEL = 40
MEL_FMIN = 100.0
MEL_FMAX = 8000.0
CLS_WINDOW_FRAMES = 47   # ~1.0 s of frames per classifier input
CLS_DB_FLOOR = -80.0     # log-mel is normalised to the window max, then clipped to [CLS_DB_FLOOR, 0]

# ---- harmonic detector
F0_MIN, F0_MAX, F0_STEP = 120.0, 400.0, 2.0
N_HARMONICS = 5
BASELINE_BINS = 31                 # width of the spectral-baseline smoother (~730 Hz)
HARM_FMIN = 300.0                  # comb uses the first N_HARMONICS harmonics at or above this (below ~300 Hz the
                                   #   prominence is dominated by wind / handling rumble on real recordings)
DET_BAND = (HARM_FMIN, 3000.0)     # band for spectral flatness: same lower edge as the comb, so a strong low tone or
                                   #   rumble below 300 Hz cannot make a broadband drone frame look tonal
PROM_EMA = 0.5                     # EMA weight of the new frame in the prominence spectrum the comb searches
F0_HYST_DB = 0.5                   # keep tracking prev f0 if a candidate within F0_MAX_JUMP of it scores within this of the best
HARM_THRESH_DB = 5.5               # mean dB prominence across the N_HARMONICS comb harmonics
FLAT_THRESH = 0.45                 # spectral flatness must be below this (1 = white noise)
FLAT_MIN = 0.17                    #   ... and above this: rotor noise always carries a broadband airflow floor (real
                                   #   clips: median 0.16-0.33 in DET_BAND); near-pure tones (sirens, alarms, horns, bells) don't
F0_MAX_JUMP = 0.06                 # f0 tolerance: hysteresis neighbourhood and the steadiness band around the median
PERSIST_FRAMES = 94                # window for the persistence counter (~2 s)
PERSIST_ON = 70                    # frames of 94 needed to declare DETECTED (~1.5 s of evidence)
PERSIST_OFF = 20                   # frames of 94 below which we start the release timer
F0_HIST_FRAMES = 47                # f0 must be steady over ~1 s:
F0_STEADY_FRAC = 0.35              #   at least this fraction of the last F0_HIST_FRAMES f0 estimates within F0_MAX_JUMP of
                                   #   their median (a dominant track exists; tolerant of the comb hopping on faint frames)
F0_STEADY_MIN = 24                 #   ... evaluated once this many f0 estimates exist (~0.5 s), over those available
MIN_HARMONICS = 3                  # of the 8 harmonics from the first >= HARM_FMIN, at least this many must be > 6 dB proud
RELEASE_SEC = 2.0
ML_THRESH = 0.7                    # classifier probability needed alongside the harmonic path
ML_HOLD_FRAMES = 47                # ... from the latest update or any update in the last ~1 s (classifier runs every ~0.5 s)

# ---- tonality test on LOCAL flatness (interp/README.md). "local": flatness of the prominence spectrum (the frame's dB
#      spectrum minus its smooth spectral baseline) in DET_BAND, which a smooth microphone / mounting response cancels out
#      of exactly. "global": the original test FLAT_MIN < flatness(power) < FLAT_THRESH, which swung 0.07-0.38 on the
#      same drone under a +-6 dB/octave tilt and rejected 97 % of the public DADS drone recordings.
FLAT_MODE = "local"
FLAT_LOCAL_MIN = 0.3
FLAT_LOCAL_MAX = 1.0

# ---- DSP gate: a linear rule on six rolling (~1 s) features that does the classifier's job of vetoing the rules' false
#      alarms, cheaply enough for the ESP32 (where the classifier is off, USE_TFLM 0). Onset needs gate >= GATE_THRESH
#      (and the classifier as well when it runs). Weights chosen on dev data by interp/part_c.py --invariant (tilt-
#      invariant features only); the threshold is the dev choice (-6.445) raised by 0.5 so that the team's own ambient
#      recording, which the rules alone fire on, is rejected with a margin. Held-out numbers in interp/README.md.
GATE_ON = True
GATE_FRAMES = 47                   # rolling window of the gate features (~1 s, same as the classifier's)
GATE_THRESH = -5.945296551446445
GATE_WEIGHTS = {                   # gate = sum(w * rolling mean of the feature)
    "dlogf0": -1.1422819973833758,       # |ln f0 - ln previous f0|: drones hold a steady pitch
    "flux_db": -0.806491110564291,       # mean |frame-to-frame change| of the prominence in DET_BAND: drones are stationary
    "fund_half_db": 0.048938506385496204,  # prominence at f0 / 2 in FUND_BAND (the shaft rate of 2-blade props), else -40
    "inharm_peaks": -0.3676354767516828,   # proud peaks in DET_BAND that are not within 1 bin of any k * f0
    "level_std": -0.24653550961466092,   # std over the window of the DET_BAND level in dB: drones are steady
    "lf_peak_db": 0.0920121029756167,    # sharpest prominence in LF_PEAK_BAND (the low harmonics the comb skips)
}
GATE_PROUD_DB = 6.0                # "proud" peak level for inharm_peaks (as n_harm)
FUND_BAND = (100.0, 400.0)
LF_PEAK_BAND = (40.0, 300.0)
FUND_MISSING_DB = -40.0            # fund_half_db when f0 / 2 is outside FUND_BAND

# ---- noise floor (minimum statistics)
FLOOR_SMOOTH = 0.3                 # EMA alpha on per-bin power before the min tracker
FLOOR_BLOCK_FRAMES = 23            # ~0.5 s blocks
FLOOR_N_BLOCKS = 10                # ~5 s memory
FLOOR_BIAS = 1.5                   # min statistics underestimate the mean; scale up

# ---- bearing
C_SOUND = 343.0                    # m/s
# PLACEHOLDER positions: measure them on the assembled band (a centimetre of error is a few degrees of bearing).
# Channel order B, L, R, T is the convention everywhere (MIC_NAMES, Spectrum arrays, WAV channels, raw packets).
# The physical I2S/DMA slot mapping is CH_SLOT[] in firmware/esp32s3/headband.ino, set at runtime by the headset
# calibration (the dashboard's Calibrate tab: which mics are connected and where); verify it by tapping each mic.
MIC_POS = np.array([[ 0.00, -0.10, 0.00],   # B (nape)  (x right, y forward, z up), metres
                    [-0.09,  0.00, 0.00],   # L
                    [ 0.09,  0.00, 0.00],   # R
                    [ 0.00,  0.00, 0.09]])  # T (crown): out of the B-L-R plane, so elevation is measured directly
MIC_NAMES = ["B", "L", "R", "T"]
# Channel 0 may hold a FRONT mic instead of the back one (the headset calibration: 'C...,F'); it then sits here, the
# mirror image of B, so every bearing rule (level plane, mirrors, the upper prior) is unchanged with F for B.
FRONT_POS = np.array([0.00, 0.10, 0.00])


def channel_positions(front=False):
    """MIC_POS for this headset: channel 0 at FRONT_POS when its mic is at the front, else at the back."""
    pos = MIC_POS.copy()
    if front:
        pos[0] = FRONT_POS
    return pos


LAYOUT_RCOND = 1e-6                # mic_layout: baseline directions with AtA eigenvalue below this x the largest are unobservable
LAYOUT_LEVEL_Z = 0.9               # a mic plane whose unobservable (normal) axis is this close to vertical is "level": its
                                   #   two mirror candidates share the azimuth (B, L, R: azimuth always known)
N_MICS = len(MIC_NAMES)
PAIRS = [(i, j) for i in range(N_MICS) for j in range(i + 1, N_MICS)]   # all 6 pairs in index order: (B,L) (B,R) (B,T) (L,R) (L,T) (R,T)
N_PAIRS = len(PAIRS)
BEARING_BAND = (200.0, 3000.0)
BEARING_MASK_HARMONICS = 8
BEARING_MASK_HALFWIDTH_BINS = 1
BEARING_EMA = 0.1                  # EMA weight of the new frame in the smoothed direction vector (theta)
BEARING_MIN_CONF = 0.2             # the EMA takes a frame only with conf >= this (conf: the robust solver's confidence below;
                                   #   0.2 is sig_proc_tri_robust's turn threshold)
BEARING_HOLD_FRAMES = 47           # detected frames without a usable azimuth after which the smoothed theta is stale (~1 s):
                                   #   theta is then unknown (motors: the pulsed ring) and the next azimuth restarts the EMA
# ---- robust per-frame solver (Rachel's sig_proc_tri_robust.py estimate_direction_details, ported to one DSP frame:
#      bearing.BearingEstimator.feed and dsp_core.cpp). Per pair: masked GCC-PHAT delay at integer lags + parabola, with
#      her period-alias test; pair coherence; a reliable pair has a delay and coherence >= BEARING_MIN_PAIR_COH and
#      weight coherence^2 * peak. Huber-weighted least squares over the reliable pairs, outliers > 3 tolerances dropped.
#      Status: full_3d (rank 3), constrained_3d / azimuth_only / ambiguous (rank 2: the two mirror candidates), unreliable.
BEARING_MIN_PAIR_COH = 0.35        # min_pair_coherence
BEARING_COH_EMA = 0.25             # coherence = magnitude-squared coherence of the EMA'd cross / auto spectra over consecutive
                                   #   frames (EMA weight of the new frame; ~6.7 effective segments, as her 7 Welch segments)
BEARING_COH_MIN_FRAMES = 3         # frames in the coherence EMA before any pair counts (her >= 3 segments)
BEARING_DELAY_TOL = 0.5            # delay_tolerance_samples: Huber scale; residuals above 3x this drop the pair
BEARING_MAX_COND = 20.0            # max_condition: singular values below sigma_max / this do not count toward the rank
BEARING_HUBER_ITERS = 8
BEARING_ALIAS_RATIO = 0.9          # a 2nd correlation peak inside the physical lag interval this close to the best: alias
BEARING_MIN_MAG = 0.05             # min_magnitude of the rank-3 least-squares vector
BEARING_MIN_HORIZ = 0.05           # a unit direction with a horizontal part below this has no azimuth (near-vertical)
BEARING_UPPER_PRIOR = True         # rank 2: assume the drone is at or above the head plane (hemisphere normal +z, her
                                   #   interval rule) to pick a mirror candidate; never applied to a rank-3 (full_3d) fit
BEARING_SHARE_DEG = 10.0           # ambiguous: the two mirror candidates "share" an azimuth / elevation within this
# overhead: elevation = asin(u_z) of the per-frame direction; hysteresis on it while DETECTED
OVERHEAD_ELEV_ON = 60.0            # deg: elevation above this (while DETECTED) is overhead evidence
OVERHEAD_ELEV_OFF = 45.0           # deg: elevation below this is evidence the drone is off to the side again
OVERHEAD_FRAMES = 23               # consecutive frames (~0.5 s) of evidence to declare overhead, and again to release

# ---- haptics: 8 motors 45 deg apart; this order is used everywhere (arrays, C, telemetry). Overhead = all 1.0 as a
# marker (the firmware time-multiplexes that into rotating triplets: the power rail allows only 3 motors at once)
MOTOR_ANGLES = {"F": 0.0, "FR": 45.0, "R": 90.0, "BR": 135.0, "B": 180.0, "BL": -135.0, "L": -90.0, "FL": -45.0}
N_MOTORS = len(MOTOR_ANGLES)
