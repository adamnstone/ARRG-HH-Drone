"""Adaptive noise floor, harmonic-comb search, and the detection state machine.

Everything is per-frame and stateful so the same logic runs unchanged on a stream (ESP32) or a file.
"""
import numpy as np
from . import config as C
from .features import spectral_flatness

F0_CANDIDATES = np.arange(C.F0_MIN, C.F0_MAX + 1e-6, C.F0_STEP)
# first harmonic number at or above HARM_FMIN, per candidate (the comb skips the rumble-dominated low band)
_H0 = np.maximum(1, np.ceil(C.HARM_FMIN / F0_CANDIDATES)).astype(int)
# bin positions (fractional) of each comb harmonic of each candidate: (n_cand, N_HARMONICS)
_HARM_BINS = (F0_CANDIDATES[:, None] * (_H0[:, None] + np.arange(C.N_HARMONICS)[None, :])) / C.BIN_HZ
_HB0 = np.floor(_HARM_BINS).astype(int)
_HBW = (_HARM_BINS - _HB0).astype(np.float32)
# nearest bins of the 8 harmonics h0 .. h0+7 used for n_harm: (n_cand, 8)
_HB8 = np.minimum((F0_CANDIDATES[:, None] * (_H0[:, None] + np.arange(8)[None, :]) / C.BIN_HZ).round().astype(int),
                  C.NBINS - 1)
_FREQ = np.arange(C.NBINS) * C.BIN_HZ


def _band_bins(lo, hi):
    """First and last bin whose centre lies in [lo, hi] Hz (inclusive)."""
    b = np.flatnonzero((_FREQ >= lo) & (_FREQ <= hi))
    return int(b[0]), int(b[-1])


LOCAL_LO, LOCAL_HI = _band_bins(*C.DET_BAND)          # bins of the local flatness and of the gate's DET_BAND features
LFPK_LO, LFPK_HI = _band_bins(*C.LF_PEAK_BAND)
# per-frame values kept in the gate's ring; the gate uses the window mean of each, except level_db, whose window
# standard deviation is the "level_std" feature
GATE_RING = ["dlogf0", "flux_db", "fund_half_db", "inharm_peaks", "lf_peak_db", "level_db"]
_LN10_10 = np.log(10.0) / 10.0


def local_flatness(S_db):
    """Spectral flatness of the prominence spectrum (dB minus the smooth baseline) in DET_BAND: geometric over
    arithmetic mean of 10^(S/10). A smooth change of microphone response shifts the dB spectrum and its baseline
    together, so it cancels; tonal frames still score low and a broadband floor high."""
    S = S_db[LOCAL_LO: LOCAL_HI + 1].astype(np.float64)
    return float(np.exp(_LN10_10 * S.mean()) / np.mean(np.exp(_LN10_10 * S)))


def gate_features(P, S_db, S_avg, f0, prev_f0, prev_S):
    """Per-frame inputs of the DSP gate (config.GATE_WEIGHTS), from what the detector already has this frame:
    power P, this frame's prominence S_db, the averaged prominence S_avg the comb searched, and f0."""
    lo, hi = LOCAL_LO, LOCAL_HI
    dlogf0 = 0.0 if prev_f0 is None else abs(float(np.log(f0)) - float(np.log(prev_f0)))
    flux = 0.0 if prev_S is None else float(np.mean(np.abs(S_db[lo: hi + 1] - prev_S)))
    ff = f0 / 2.0
    if C.FUND_BAND[0] <= ff <= C.FUND_BAND[1]:
        b = min(max(int(np.round(ff / C.BIN_HZ)), 1), C.NBINS - 2)
        fund = float(max(S_avg[b - 1], S_avg[b], S_avg[b + 1]))
    else:
        fund = C.FUND_MISSING_DB
    Sw = S_avg[lo - 1: hi + 2]
    c = Sw[1:-1]
    peak = (c > C.GATE_PROUD_DB) & (c >= Sw[:-2]) & (c >= Sw[2:])
    bins = np.arange(lo, hi + 1)
    r = f0 / C.BIN_HZ
    k0 = np.maximum(1.0, np.round(bins / r))                  # the only harmonic that can lie within 1 bin
    near = np.abs(bins - np.round(k0 * r)) <= 1
    inharm = float(np.count_nonzero(peak & ~near))
    level = float(10.0 * np.log10(float(P[lo: hi + 1].sum()) + 1e-20))
    lfpk = float(S_avg[LFPK_LO: LFPK_HI + 1].max())
    return dict(dlogf0=dlogf0, flux_db=flux, fund_half_db=fund, inharm_peaks=inharm, level_db=level, lf_peak_db=lfpk)


class NoiseFloor:
    """Minimum-statistics floor per FFT bin: EMA-smoothed power, min over the last ~5 s, times a bias."""

    def __init__(self):
        self.smooth = None
        self.block_min = None
        self.ring = []
        self.count = 0
        self.frozen = False

    def update(self, P):
        if self.smooth is None:
            self.smooth = P.copy()
            self.block_min = P.copy()
            self.ring = [P.copy()]
            return self.floor()
        self.smooth = C.FLOOR_SMOOTH * P + (1 - C.FLOOR_SMOOTH) * self.smooth
        if not self.frozen:
            self.block_min = np.minimum(self.block_min, self.smooth)
            self.count += 1
            if self.count >= C.FLOOR_BLOCK_FRAMES:
                self.ring.append(self.block_min)
                self.ring = self.ring[-C.FLOOR_N_BLOCKS:]
                self.block_min = self.smooth.copy()
                self.count = 0
        return self.floor()

    def floor(self):
        return np.minimum(np.min(np.stack(self.ring), axis=0), self.block_min) * C.FLOOR_BIAS + 1e-12


def db_above_floor(P, floor):
    return 10.0 * np.log10(P / floor + 1e-12)


def _movavg(x, w):
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")[: len(x)]


def spectral_baseline(P_db, w=C.BASELINE_BINS, iters=3):
    """Smooth background level per bin, ignoring narrow peaks: repeated moving average of min(spectrum, baseline).
    History-free, so it works on a clip where the drone is present from the first frame."""
    base = _movavg(P_db, w)
    for _ in range(iters - 1):
        base = _movavg(np.minimum(P_db, base), w)
    return base


def harmonic_search(S_db, prev_f0=None):
    """S_db: one frame, dB above the spectral baseline per bin. Returns (f0_hz, score_db, n_harm).

    score  = mean over N_HARMONICS comb harmonics (h0 .. h0+N-1, h0 = first harmonic >= HARM_FMIN) of max(0, prominence dB)
    n_harm = how many of the 8 harmonics h0 .. h0+7 of the winning f0 stand more than 6 dB proud of the baseline
    prev_f0: last frame's f0. If a candidate within F0_MAX_JUMP of it scores within F0_HYST_DB of the global best,
             the best such candidate wins (track continuity instead of hopping between near-equal comb fits).
    """
    S = np.maximum(S_db, 0.0)
    v = S[_HB0] * (1 - _HBW) + S[_HB0 + 1] * _HBW          # linear interpolation at each harmonic
    score = v.mean(axis=1)
    k = int(np.argmax(score))
    if prev_f0 is not None and C.F0_HYST_DB > 0:
        near = np.abs(F0_CANDIDATES - prev_f0) / prev_f0 < C.F0_MAX_JUMP
        kl = int(np.argmax(np.where(near, score, -1.0)))
        if near[kl] and score[kl] >= score[k] - C.F0_HYST_DB:
            k = kl
    f0 = float(F0_CANDIDATES[k])
    n_harm = int((S[_HB8[k]] > 6.0).sum())
    return f0, float(score[k]), n_harm


class DroneDetector:
    """Per-frame harmonic detector with persistence and hysteresis.

    feed(P) takes one power-spectrum frame and returns a dict with the frame's evidence and the current state.
    """

    def __init__(self):
        self.floor = NoiseFloor()
        self.hist = np.zeros(C.PERSIST_FRAMES, dtype=bool)
        self.hp = 0
        self.prev_f0 = None
        self.detected = False
        self.release_frames = 0
        self.ml_prob = 0.0
        self.ml_age = C.ML_HOLD_FRAMES                       # frames since the last classifier update above ML_THRESH
        self.scores = np.zeros(C.PERSIST_FRAMES, dtype=np.float32)
        self.f0_hist = np.zeros(C.F0_HIST_FRAMES, dtype=np.float32)
        self.fp = 0
        self.f0_count = 0
        self.S_avg = None                                    # EMA of the prominence spectrum (what the comb searches)
        self.g_ring = np.zeros((len(GATE_RING), C.GATE_FRAMES), np.float32)   # gate inputs, one column per frame
        self.g_head = 0
        self.g_count = 0
        self.prev_S = None                                   # last frame's prominence in DET_BAND (for flux)

    def set_ml(self, p):
        self.ml_prob = float(p)
        if self.ml_prob > C.ML_THRESH:
            self.ml_age = 0

    def feed(self, P, use_ml=False):
        floor = self.floor.update(P)
        P_db = 10.0 * np.log10(P + 1e-12)
        S_db = P_db - spectral_baseline(P_db)               # prominence above the local spectral background
        snr_db = db_above_floor(P, floor)                    # energy above the slow time floor (for strength)
        if self.S_avg is None:
            self.S_avg = S_db.copy()
        else:
            self.S_avg = C.PROM_EMA * S_db + (1 - C.PROM_EMA) * self.S_avg
        f0, score, n_harm = harmonic_search(self.S_avg, self.prev_f0)
        flat = float(spectral_flatness(P[None, :])[0])
        flat_local = local_flatness(S_db)
        self.f0_hist[self.fp] = f0; self.fp = (self.fp + 1) % C.F0_HIST_FRAMES
        self.f0_count = min(self.f0_count + 1, C.F0_HIST_FRAMES)
        steady_frac = 0.0
        if self.f0_count >= C.F0_STEADY_MIN:                 # steadiness: fraction of the last ~1 s near the median f0
            h = self.f0_hist[: self.f0_count]                # (ring fills from index 0, so the first f0_count are valid)
            med = np.sort(h)[(self.f0_count - 1) // 2]       # lower median: an exact element, float32 like the C core
            tol = np.float32(C.F0_MAX_JUMP) * med
            steady_frac = float(np.count_nonzero(np.abs(h - med) <= tol)) / self.f0_count
        steady = steady_frac >= C.F0_STEADY_FRAC
        if C.FLAT_MODE == "local":
            tonal_ok = C.FLAT_LOCAL_MIN < flat_local < C.FLAT_LOCAL_MAX
        else:
            tonal_ok = C.FLAT_MIN < flat < C.FLAT_THRESH
        frame_ok = (score > C.HARM_THRESH_DB) and tonal_ok and steady and (n_harm >= C.MIN_HARMONICS)
        gate, gate_ok = self._gate(P, S_db, f0)
        self.prev_f0 = f0
        self.hist[self.hp] = frame_ok
        self.scores[self.hp] = score
        self.hp = (self.hp + 1) % C.PERSIST_FRAMES
        hits = int(self.hist.sum())
        ml_ok = (self.ml_prob > C.ML_THRESH or self.ml_age < C.ML_HOLD_FRAMES) if use_ml else True
        self.ml_age = min(self.ml_age + 1, C.ML_HOLD_FRAMES)
        if not self.detected:
            if hits >= C.PERSIST_ON and ml_ok and gate_ok:
                self.detected = True
                self.release_frames = 0
        else:
            if hits <= C.PERSIST_OFF:
                self.release_frames += 1
                if self.release_frames >= int(C.RELEASE_SEC * C.FRAMES_PER_SEC):
                    self.detected = False
            else:
                self.release_frames = 0
        self.floor.frozen = self.detected
        return dict(f0=f0, score=score, flat=flat, flat_local=flat_local, frame_ok=frame_ok, hits=hits, n_harm=n_harm,
                    f0_steady=steady_frac, gate=gate, gate_ok=gate_ok, detected=self.detected,
                    strength=float(np.median(self.scores)), S_db=S_db, snr_db=snr_db)

    def _gate(self, P, S_db, f0):
        """Push this frame's gate features into the ring; return (gate score over the last GATE_FRAMES, gate open)."""
        g = gate_features(P, S_db, self.S_avg, f0, self.prev_f0, self.prev_S)
        self.prev_S = S_db[LOCAL_LO: LOCAL_HI + 1].copy()
        self.g_ring[:, self.g_head] = [g[k] for k in GATE_RING]
        self.g_head = (self.g_head + 1) % C.GATE_FRAMES
        self.g_count = min(self.g_count + 1, C.GATE_FRAMES)
        R = self.g_ring[:, : self.g_count].astype(np.float64)   # the ring fills from index 0: the first g_count are valid
        m = dict(zip(GATE_RING, R.mean(axis=1)))
        lv = R[GATE_RING.index("level_db")]
        m["level_std"] = float(np.sqrt(np.mean((lv - lv.mean()) ** 2)))   # two-pass: no cancellation in float
        gate = float(sum(w * m[k] for k, w in C.GATE_WEIGHTS.items()))
        return gate, (gate >= C.GATE_THRESH) if C.GATE_ON else True


def harmonic_mask(f0, nh=C.BEARING_MASK_HARMONICS, half=C.BEARING_MASK_HALFWIDTH_BINS, band=C.BEARING_BAND):
    """Bins near k*f0 within the bearing band. Used to make the bearing estimator listen only to the drone."""
    m = np.zeros(C.NBINS, dtype=np.float32)
    for k in range(1, nh + 1):
        f = k * f0
        if f < band[0] or f > band[1]:
            continue
        b = int(round(f / C.BIN_HZ))
        m[max(0, b - half): b + half + 1] = 1.0
    return m


def band_mask(band=C.BEARING_BAND):
    m = np.zeros(C.NBINS, dtype=np.float32)
    m[int(band[0] / C.BIN_HZ): int(band[1] / C.BIN_HZ) + 1] = 1.0
    return m
