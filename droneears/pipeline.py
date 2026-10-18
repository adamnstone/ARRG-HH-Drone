"""Frame-by-frame pipeline: spectra -> detector -> (classifier) -> bearing -> haptics. Mirrors the firmware loop."""
import numpy as np
from . import config as C
from .features import frames, logmel, normalise_window
from .detector import DroneDetector, harmonic_mask, band_mask
from .bearing import BearingEstimator, haptic_weights

_BAND = band_mask()


class Pipeline:
    def __init__(self, classifier=None, use_ml=None, layout=None):
        """classifier: callable taking (win, NMEL) normalised log-mel -> p_drone, or None. layout: bearing.mic_layout()
        for a headset with only some mics connected (the mono mix and the bearing then use those only)."""
        self.det = DroneDetector()
        self.bear = BearingEstimator(layout=layout)
        self.layout = layout
        self.clf = classifier
        self.use_ml = (classifier is not None) if use_ml is None else use_ml
        self.lm_buf = []
        self.frame_i = 0
        self.ml_every = int(round(C.FRAMES_PER_SEC / 2))     # classifier twice per second

    def step(self, X):
        """X: (n_mics, NBINS) complex spectra of one frame (mono input: shape (1, NBINS)). Bearing and elevation run
        when n_mics == C.N_MICS (channels in C.MIC_NAMES order) and add theta (smoothed), theta_raw, elev (None when
        unknown), conf, overhead, taus, status (bearing.STATUS), bstat (0 unreliable / no bearing, 1 ambiguous,
        2 constrained_3d or azimuth_only, 3 full_3d), mused (bitmask of the mics in this frame's fit: B 1, L 2, R 4,
        T 8) and haptics."""
        P = X.real ** 2 + X.imag ** 2
        if self.layout is not None and X.shape[0] == C.N_MICS:            # mono power: mean over the connected channels
            act = self.layout["active"]
            P = P[act].mean(axis=0) if act.any() else np.zeros(P.shape[1])
        else:
            P = P.mean(axis=0)
        P = P.astype(np.float32)
        if self.clf is not None:
            self.lm_buf.append(logmel(P[None, :])[0])
            self.lm_buf = self.lm_buf[-C.CLS_WINDOW_FRAMES:]
            if len(self.lm_buf) == C.CLS_WINDOW_FRAMES and self.frame_i % self.ml_every == 0:
                self.det.set_ml(self.clf(normalise_window(np.stack(self.lm_buf))))
        d = self.det.feed(P, use_ml=self.use_ml)
        out = dict(d)
        out["ml"] = self.det.ml_prob
        if X.shape[0] == C.N_MICS:
            mask = harmonic_mask(d["f0"]) if d["frame_ok"] else _BAND
            b = self.bear.feed(X, mask, update=d["detected"])
            out.update(theta=b["theta"], theta_raw=b["theta_raw"], conf=b["conf"], elev=b["elev"],
                       overhead=b["overhead"], taus=b["taus"], status=b["status"], bstat=b["bstat"], mused=b["mused"])
            # detected: pan to theta; overhead, or no usable azimuth (theta None): every motor (the firmware's ring)
            out["haptics"] = haptic_weights(b["theta"], b["overhead"]) if d["detected"] else {k: 0.0 for k in C.MOTOR_ANGLES}
        self.frame_i += 1
        return out

    def run(self, y):
        """y: (n_mics, n) or (n,) float signal. Returns list of per-frame dicts."""
        y = np.atleast_2d(y)
        F = np.stack([frames(ch) for ch in y])              # (mics, frames, NFFT)
        X = np.fft.rfft(F, axis=2)                           # (mics, frames, NBINS)
        return [self.step(X[:, i, :]) for i in range(X.shape[1])]


def summarise(results):
    det = np.array([r["detected"] for r in results])
    out = dict(frames=len(results), detected_frac=float(det.mean()) if len(det) else 0.0,
               first_detect_s=None)
    if det.any():
        out["first_detect_s"] = float(np.argmax(det) * C.HOP / C.FS)
    if results and "theta" in results[0]:
        th = np.array([r["theta"] for r in results if r["detected"] and r["theta"] is not None])
        out["theta_median_when_detected"] = float(np.median(th)) if len(th) else None
        el = np.array([r["elev"] for r in results if r["detected"] and r["elev"] is not None])
        out["elev_median_when_detected"] = float(np.median(el)) if len(el) else None
    return out
