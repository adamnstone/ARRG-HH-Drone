"""Candidate cheap features for replacing the classifier's veto with rules (Part C).

Everything here is computable on the ESP32 from what the DSP task already has each frame: the mono power
spectrum P, the detector's prominence spectrum (S_avg = what the comb searches, S_raw = this frame's), and the
detector's f0. Each feature is a handful of sums over fixed bin ranges or over the harmonics of f0, so the cost is
microseconds, not milliseconds. Window versions (rolling over the classifier's ~1 s) are built in window_features().
"""
import numpy as np
from droneears import config as C
from droneears.detector import F0_CANDIDATES, _H0

FREQ = np.arange(C.NBINS) * C.BIN_HZ
PROUD_DB = 6.0                                           # same "proud" level as the detector's n_harm


def _bins(lo, hi):
    return np.flatnonzero((FREQ >= lo) & (FREQ <= hi))


B_DET, B_WIDE, B_HF, B_LF = _bins(300, 3000), _bins(100, 8000), _bins(3000, 8000), _bins(50, 300)
B_LFPK = _bins(40, 300)


def _slope_w(b):
    x = np.log2(FREQ[b]); x = x - x.mean()
    return (x / (x ** 2).sum()).astype(np.float32)


W_SLOPE_DET, W_SLOPE_WIDE = _slope_w(B_DET), _slope_w(B_WIDE)

# ---- per-candidate harmonic tables (the detector's f0 is always one of F0_CANDIDATES)
NH_MAX = 64
_harm_bins = np.full((len(F0_CANDIDATES), NH_MAX), -1, int)    # nearest bin of harmonics h0, h0+1, ... up to 8 kHz
_harm_num = np.zeros((len(F0_CANDIDATES), NH_MAX), np.float32)
_mid_bins = np.full((len(F0_CANDIDATES), NH_MAX), -1, int)     # midpoints (h + 0.5) f0
_near = np.zeros((len(F0_CANDIDATES), C.NBINS), bool)          # within 1 bin of any k f0, k >= 1
for k, f0 in enumerate(F0_CANDIDATES):
    hs = np.arange(_H0[k], _H0[k] + NH_MAX)
    f = hs * f0
    ok = f <= 8000
    _harm_bins[k, ok] = np.round(f[ok] / C.BIN_HZ).astype(int)
    _harm_num[k] = hs
    fm = (hs + 0.5) * f0
    okm = fm <= 3000
    _mid_bins[k, okm] = np.round(fm[okm] / C.BIN_HZ).astype(int)
    for kk in range(1, int(8000 / f0) + 1):
        b = int(round(kk * f0 / C.BIN_HZ))
        _near[k, max(0, b - 1): b + 2] = True
_HARM_HZ = np.where(_harm_bins >= 0, _harm_bins * C.BIN_HZ, 0.0)


def f0_index(f0):
    return np.clip(np.round((np.asarray(f0) - C.F0_MIN) / C.F0_STEP).astype(int), 0, len(F0_CANDIDATES) - 1)


def _db(x):
    return 10.0 * np.log10(x + 1e-20)


def frame_features(P, S_avg, S_raw, f0):
    """Per-frame candidate features. P: (n, NBINS) power; S_avg / S_raw: (n, NBINS) prominence in dB; f0: (n,)."""
    n = len(P)
    P_db = _db(P)
    e_det = P[:, B_DET].sum(1)
    out = {}
    out["slope_det"] = P_db[:, B_DET] @ W_SLOPE_DET                   # dB / octave, 300-3000 Hz
    out["slope_wide"] = P_db[:, B_WIDE] @ W_SLOPE_WIDE                # dB / octave, 100-8000 Hz
    out["hf_db"] = _db(P[:, B_HF].sum(1)) - _db(e_det)                # 3-8 kHz energy relative to 300-3000 Hz
    out["lf_db"] = _db(P[:, B_LF].sum(1)) - _db(e_det)                # 50-300 Hz relative to 300-3000 Hz
    out["centroid"] = (P[:, B_WIDE] * FREQ[B_WIDE]).sum(1) / (P[:, B_WIDE].sum(1) + 1e-20)
    lp = np.log(P[:, B_HF] + 1e-20)
    out["flat_hf"] = np.exp(lp.mean(1)) / (P[:, B_HF].mean(1) + 1e-20)  # spectral flatness 3-8 kHz
    # flatness of the spectrum with its smooth shape divided out (this frame's prominence S_raw): a smooth mic /
    # mounting EQ cancels exactly, so unlike the rules' flatness this does not move with the microphone
    R = 10.0 ** (S_raw[:, B_DET] / 10.0)
    out["flat_local"] = np.exp(np.log(R + 1e-20).mean(1)) / (R.mean(1) + 1e-20)
    Ra = 10.0 ** (S_avg[:, B_DET] / 10.0)                             # same on the averaged prominence the comb uses
    out["flat_local_avg"] = np.exp(np.log(Ra + 1e-20).mean(1)) / (Ra.mean(1) + 1e-20)
    out["band_db"] = _db(e_det)                                       # absolute level: only for temporal stats
    out["lf_peak_db"] = S_avg[:, B_LFPK].max(1)                       # sharpest peak below 300 Hz (engine firing)

    k = f0_index(f0)
    hb = _harm_bins[k]                                                # (n, NH_MAX)
    valid = hb >= 0
    hbc = np.clip(hb, 1, C.NBINS - 2)
    rows = np.arange(n)[:, None]
    Sh = np.maximum(np.maximum(S_avg[rows, hbc - 1], S_avg[rows, hbc]), S_avg[rows, hbc + 1])   # max within +-1 bin
    hz = _HARM_HZ[k]
    in_det = valid & (hz >= 300) & (hz <= 3000)
    in_6k = valid & (hz >= 300) & (hz <= 6000)
    proud = (Sh > PROUD_DB)
    out["extent_frac"] = (proud & in_6k).sum(1) / np.maximum(1, in_6k.sum(1))   # proud harmonics up to 6 kHz
    out["top_harm_hz"] = np.where(proud & valid, hz, 0.0).max(1)                  # highest proud harmonic <= 8 kHz
    out["harm_peaks"] = (proud & in_det).sum(1).astype(np.float32)                # proud harmonics, 300-3000 Hz
    nd = np.maximum(1, in_det.sum(1))
    out["harm_mean_db"] = (np.maximum(Sh, 0) * in_det).sum(1) / nd
    mb = _mid_bins[k]; mv = mb >= 0
    Sm = np.maximum(S_avg[rows, np.clip(mb, 0, C.NBINS - 1)], 0) * mv
    out["mid_mean_db"] = Sm.sum(1) / np.maximum(1, mv.sum(1))                     # prominence between harmonics
    out["contrast_db"] = out["harm_mean_db"] - out["mid_mean_db"]
    # decay: least-squares slope of harmonic prominence vs harmonic number, 300-3000 Hz (dB per harmonic)
    x = _harm_num[k] * in_det; w = in_det.astype(np.float32)
    sw = w.sum(1); sx = x.sum(1); sy = (Sh * w).sum(1); sxx = (x * x).sum(1); sxy = (x * Sh * w).sum(1)
    den = sw * sxx - sx ** 2
    out["decay_db"] = np.where(den > 0, (sw * sxy - sx * sy) / np.where(den > 0, den, 1), 0.0)
    # proud local maxima in 300-3000 Hz that are NOT near a harmonic of f0 (inharmonic peaks)
    Sb = S_avg[:, B_DET[0] - 1: B_DET[-1] + 2]
    pk = (Sb[:, 1:-1] > PROUD_DB) & (Sb[:, 1:-1] >= Sb[:, :-2]) & (Sb[:, 1:-1] >= Sb[:, 2:])
    near = _near[k][:, B_DET]
    out["inharm_peaks"] = (pk & ~near).sum(1).astype(np.float32)
    out["harm_frac_peaks"] = (pk & near).sum(1) / np.maximum(1, pk.sum(1))       # share of proud peaks on the comb
    # the fundamental region the classifier relies on (Parts A/B: peaks in 200-400 Hz are sufficient and necessary):
    # prominence at f0 and f0/2 (the comb starts at >= 300 Hz and can lock an octave high) inside 100-400 Hz, as a
    # peak above the local shape, so broadband wind / rumble raises the shape but does not fake it
    fund = np.full((n, 2), -40.0, np.float32)
    for j, div in enumerate((1.0, 2.0)):
        ff = f0 / div
        b = np.clip(np.round(ff / C.BIN_HZ).astype(int), 1, C.NBINS - 2)
        v = np.maximum(np.maximum(S_avg[np.arange(n), b - 1], S_avg[np.arange(n), b]), S_avg[np.arange(n), b + 1])
        fund[:, j] = np.where((ff >= 100) & (ff <= 400), v, -40.0)
    out["fund_db"] = fund.max(1)                                       # best of f0, f0/2 in 100-400 Hz
    out["fund_f0_db"] = fund[:, 0]
    out["fund_half_db"] = fund[:, 1]
    # low comb: mean prominence of the harmonics of the fundamental candidate below 800 Hz (k * f0 / 2, k = 1..)
    lo = []
    for div in (1.0, 2.0):
        ff = f0 / div
        acc = np.zeros(n); cnt = np.zeros(n)
        for k in range(1, 9):
            fk = k * ff
            okk = (fk >= 100) & (fk <= 800)
            b = np.clip(np.round(fk / C.BIN_HZ).astype(int), 1, C.NBINS - 2)
            v = np.maximum(np.maximum(S_avg[np.arange(n), b - 1], S_avg[np.arange(n), b]), S_avg[np.arange(n), b + 1])
            acc += np.where(okk, np.maximum(v, 0), 0); cnt += okk
        lo.append(acc / np.maximum(cnt, 1))
    out["low_comb_db"] = np.maximum(lo[0], lo[1])
    # temporal: frame-to-frame change of this frame's prominence, and of f0
    d = np.abs(np.diff(S_raw[:, B_DET], axis=0)).mean(1)
    out["flux_db"] = np.concatenate([[0.0], d])
    out["dlogf0"] = np.concatenate([[0.0], np.abs(np.diff(np.log(np.maximum(f0, 1.0))))])
    return {kk: np.asarray(v, np.float32) for kk, v in out.items()}


def _roll_mean(x, w):
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    i = np.arange(len(x))
    lo = np.maximum(0, i + 1 - w)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def window_features(rec, w=C.CLS_WINDOW_FRAMES):
    """Rolling (last w frames, ~1 s) versions of the per-frame features, as the board would keep them in a ring.
    Means for spectral features, plus temporal stats: level wobble (std of band_db), f0 jumpiness, flux."""
    out = {}
    names = [k[2:] for k in rec if k.startswith("f:")]
    for k in names:
        if k == "band_db":
            continue
        out["w_" + k] = _roll_mean(rec["f:" + k], w)
    b = rec["f:band_db"].astype(np.float64)
    out["w_level_std"] = np.sqrt(np.maximum(0, _roll_mean(b ** 2, w) - _roll_mean(b, w) ** 2))
    out["w_f0_jump_frac"] = _roll_mean((rec["f:dlogf0"] > C.F0_MAX_JUMP).astype(np.float64), w)
    for k in ("score", "flat", "n_harm", "f0_steady"):
        out["w_" + k] = _roll_mean(rec[k], w)
    out["w_f0"] = _roll_mean(rec["f0"], w)
    return {k: v.astype(np.float32) for k, v in out.items()}
