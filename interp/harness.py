"""Shared harness: cached audio, the classifier, and a per-frame recorder that mirrors droneears.pipeline.Pipeline.

record(y) runs the rule-based detector exactly as Pipeline.step does (mean power of the mic channels, classifier
every ~0.5 s on the last 47 log-mel frames) and returns per-frame arrays: the detector's own evidence, the
classifier's latest p(drone), whether the classifier gate would be open (ml_ok), and the candidate features from
interp.features. replay() re-runs the detection state machine offline with any gate, so a candidate rule can be
scored without re-running the DSP.
"""
import hashlib, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from droneears import config as C
from droneears.audio import load
from droneears.detector import DroneDetector
from droneears.features import frames, logmel, normalise_window

CACHE = os.path.join(ROOT, "data", "cache")
OUT = os.path.join(ROOT, "interp", "out")
MODEL_INT8 = os.path.join(ROOT, "artifacts", "drone_cls_int8.tflite")
ESC50 = os.environ.get("ESC50_DIR", os.path.join(ROOT, "data", "ESC-50"))
PHONE_DRONE = sorted(os.path.join(ROOT, "data", "drone", f) for f in os.listdir(os.path.join(ROOT, "data", "drone")))
PHONE_AMBIENT = os.path.join(ROOT, "data", "non_drone", "ambient_adam.m4a")
ML_EVERY = int(round(C.FRAMES_PER_SEC / 2))              # Pipeline.ml_every
RELEASE_FRAMES = int(C.RELEASE_SEC * C.FRAMES_PER_SEC)


def load_cached(path, ch0=True):
    """Decode once (ffmpeg is the slow part) and cache as .npy. ch0=True keeps channel 0 only, as tests/ and the
    README numbers do for the phone recordings; ch0=False returns (channels, n)."""
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{ch0}".encode()).hexdigest()[:16]
    dst = os.path.join(CACHE, "audio", key + ".npy")
    if os.path.exists(dst):
        return np.load(dst)
    y = load(path, mono=False)
    y = y[0] if ch0 else y
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.save(dst, y.astype(np.float32))
    return y.astype(np.float32)


_CLF = None


def classifier():
    """The shipped int8 model as a callable (47, 40) normalised window -> p(drone). Loaded once per process."""
    global _CLF
    if _CLF is None:
        from droneears.classifier import load_tflite_classifier
        _CLF = load_tflite_classifier(MODEL_INT8)
    return _CLF


def power_frames(y):
    """(n,) or (ch, n) signal -> (frames, NBINS) mean power over channels, as Pipeline.step computes it."""
    y = np.atleast_2d(np.asarray(y, np.float32))
    X = np.fft.rfft(np.stack([frames(ch) for ch in y]), axis=2)
    return (X.real ** 2 + X.imag ** 2).mean(axis=0).astype(np.float32)


def record(y, clf=True, features=True):
    """Run the detector (rules + DSP gate as configured, classifier NOT gating) over a signal, with the classifier
    evaluated alongside.

    Returns a dict of per-frame arrays: frame_ok, hits, detected (rules only), f0, score, flat, n_harm, f0_steady,
    ml (latest p), ml_new (frames where the classifier ran), ml_ok (the fused gate as DroneDetector would see it),
    plus 'f:<name>' candidate features when features=True.
    """
    P = power_frames(y)
    n = len(P)
    LM = logmel(P) if clf else None
    model = classifier() if clf else None
    det = DroneDetector()
    keys = ["frame_ok", "hits", "detected", "f0", "score", "flat", "n_harm", "f0_steady", "ml", "ml_new", "ml_ok",
            "flat_local", "gate", "gate_ok"]
    out = {k: np.zeros(n, np.float32) for k in keys}
    S_avg = np.zeros((n, C.NBINS), np.float32) if features else None
    S_raw = np.zeros((n, C.NBINS), np.float32) if features else None
    for i in range(n):
        if model is not None and i >= C.CLS_WINDOW_FRAMES - 1 and i % ML_EVERY == 0:
            det.set_ml(model(normalise_window(LM[i - C.CLS_WINDOW_FRAMES + 1: i + 1])))
            out["ml_new"][i] = 1
        out["ml_ok"][i] = det.ml_prob > C.ML_THRESH or det.ml_age < C.ML_HOLD_FRAMES
        d = det.feed(P[i], use_ml=False)
        for k in ("frame_ok", "hits", "detected", "f0", "score", "flat", "n_harm", "f0_steady", "flat_local", "gate",
                  "gate_ok"):
            out[k][i] = d[k]
        out["ml"][i] = det.ml_prob
        if features:
            S_avg[i] = det.S_avg
            S_raw[i] = d["S_db"]
    if features:
        from .features import frame_features
        for k, v in frame_features(P, S_avg, S_raw, out["f0"]).items():
            out["f:" + k] = v.astype(np.float32)
    return out


def replay(frame_ok, gate=None):
    """Detection state machine of DroneDetector.feed, offline. frame_ok: (n,) bool per-frame evidence (hits are
    re-derived from it, so a frame-level rule can be tested too); gate: (n,) bool, needed at onset (None = rules
    only; rec['ml_ok'] = the shipped fused detector). Returns (n,) bool DETECTED."""
    ok = np.asarray(frame_ok, bool)
    c = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(len(ok))
    hits = c[idx + 1] - c[np.maximum(0, idx + 1 - C.PERSIST_FRAMES)]
    g = np.ones(len(ok), bool) if gate is None else np.asarray(gate, bool)
    out = np.zeros(len(ok), bool)
    detected, rel = False, 0
    for i in range(len(ok)):
        if not detected:
            if hits[i] >= C.PERSIST_ON and g[i]:
                detected, rel = True, 0
        elif hits[i] <= C.PERSIST_OFF:
            rel += 1
            if rel >= RELEASE_FRAMES:
                detected = False
        else:
            rel = 0
        out[i] = detected
    return out


def rules_frame_ok(rec, flat_min=C.FLAT_MIN, flat_max=C.FLAT_THRESH, harm_db=C.HARM_THRESH_DB,
                   min_harm=C.MIN_HARMONICS, steady_frac=C.F0_STEADY_FRAC, flat_key="flat"):
    """Re-derive DroneDetector's per-frame test from a record with different thresholds (exact: the f0 track,
    score, flatness, n_harm and steadiness do not depend on these thresholds). flat_key picks which flatness the
    tonality test uses ("flat" = the detector's, "f:flat_local" = with the smooth shape divided out)."""
    fl = rec[flat_key]
    return ((rec["score"] > harm_db) & (fl > flat_min) & (fl < flat_max)
            & (rec["f0_steady"] >= steady_frac) & (rec["n_harm"] >= min_harm))


def hits_of(frame_ok):
    ok = np.asarray(frame_ok, bool)
    c = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(ok))
    return c[i + 1] - c[np.maximum(0, i + 1 - C.PERSIST_FRAMES)]


def mix(sig, bg, snr_db, seed, lowpass_hz=None):
    """droneears.dataset.mix with a fixed seed (where in the background the drone lands)."""
    from droneears.dataset import mix as _mix
    return _mix(sig, bg, snr_db, np.random.default_rng(seed), lowpass_hz)


def esc50_rows(folds=(1, 2, 3, 4, 5)):
    import csv
    meta = list(csv.DictReader(open(os.path.join(ESC50, "meta", "esc50.csv"))))
    return sorted((r for r in meta if int(r["fold"]) in folds), key=lambda r: r["filename"])



def tilt(y, db_per_oct, pivot_hz=1000.0):
    """droneears.simulate.tilt (a different mic / mounting response), kept here for the analysis code."""
    from droneears.simulate import tilt as _tilt
    return _tilt(y, db_per_oct, pivot_hz)
