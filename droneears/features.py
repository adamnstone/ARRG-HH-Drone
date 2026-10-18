"""Framing, FFT and log-mel features. Written to be mirrored line for line in firmware/core/dsp_core.cpp."""
import numpy as np
from . import config as C

_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(C.NFFT) / C.NFFT)   # periodic Hann, same as the C core


def frames(x, nfft=C.NFFT, hop=C.HOP):
    """Return (n_frames, nfft) windowed frames of a 1-D float signal."""
    x = np.asarray(x, dtype=np.float32)
    n = 1 + (len(x) - nfft) // hop if len(x) >= nfft else 0     # shorter than one frame -> no frames
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] * _HANN


def spectra(x):
    """Complex rfft spectra, shape (n_frames, NBINS)."""
    return np.fft.rfft(frames(x), axis=1)


def power(X):
    """Power spectrum from complex spectra."""
    return (X.real ** 2 + X.imag ** 2).astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(nmel=C.NMEL, fmin=C.MEL_FMIN, fmax=C.MEL_FMAX, nbins=C.NBINS, fs=C.FS):
    """Triangular HTK-style mel filterbank, shape (nmel, nbins). Rows sum to 1 (area-normalised)."""
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), nmel + 2))
    bins_hz = np.arange(nbins) * fs / ((nbins - 1) * 2)
    fb = np.zeros((nmel, nbins), dtype=np.float32)
    for m in range(nmel):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (bins_hz - lo) / (mid - lo)
        down = (hi - bins_hz) / (hi - mid)
        w = np.clip(np.minimum(up, down), 0, None)
        s = w.sum()
        fb[m] = w / s if s > 0 else w
    return fb


MELFB = mel_filterbank()


def logmel(P, fb=MELFB):
    """Log-mel in dB, shape (n_frames, NMEL). No normalisation here."""
    E = P @ fb.T
    return (10.0 * np.log10(E + 1e-10)).astype(np.float32)


def normalise_window(LM):
    """Classifier input normalisation: relative to the loudest bin in the window, clipped, scaled to [0, 1].

    Removes dependence on absolute gain (phone mic vs INMP441) and recording level.
    """
    LM = LM - LM.max()
    LM = np.clip(LM, C.CLS_DB_FLOOR, 0.0)
    return ((LM - C.CLS_DB_FLOOR) / (-C.CLS_DB_FLOOR)).astype(np.float32)


def classifier_windows(P, win=C.CLS_WINDOW_FRAMES, step=None):
    """Slice a power-spectrogram into normalised log-mel windows (n_win, win, NMEL)."""
    step = step or win // 2
    LM = logmel(P)
    out = []
    for s in range(0, LM.shape[0] - win + 1, step):
        out.append(normalise_window(LM[s:s + win]))
    return np.stack(out) if out else np.zeros((0, win, C.NMEL), np.float32)


def spectral_flatness(P, band=C.DET_BAND):
    """Geometric / arithmetic mean of power in a band, per frame. 1 = white, 0 = a single tone."""
    lo, hi = int(band[0] / C.BIN_HZ), int(band[1] / C.BIN_HZ) + 1
    S = P[:, lo:hi] + 1e-12
    return np.exp(np.mean(np.log(S), axis=1)) / np.mean(S, axis=1)
