"""
Direction-of-arrival (DOA) for a 4-mic B/L/R/T headset.
Classical GCC-PHAT pair delays and a 3D least-squares fit. No ML.

What it does
  1. Pulls real drone clips (label 1) from the dataset.
  2. Simulates a plane wave from a chosen azimuth hitting four mics with
     fractional-sample delays, plus independent noise at a chosen SNR.
  3. Returns a 3D unit vector [right, forward, up] toward the drone, with
     azimuth/elevation available for display and horizontal turn instructions.
  4. Sweeps SNR and reports 3D angular accuracy and undefined estimates.

Dataset-free checks: python3 -m unittest discover -s tests -p test_triangulation.py

Conventions
  Azimuth 0 = straight ahead, +90 = right, -90 = left, +/-180 = directly behind.
  Mic coordinates are metres: x = right, y = forward, z = up.
  Geometry comes from droneears.config.MIC_POS in B/L/R/T channel order.
  The crown mic makes the geometry noncoplanar and resolves signed elevation.

Reuse on real hardware: estimate_direction(frames) takes an array of shape
(4, N) ordered [B, L, R, T], sampled simultaneously. It returns
(vector or None, confidence, grid, fit). The vector has unit length and is a
direction relative to the headset, not a position in metres: the far-field
model does not estimate distance. fs defaults to hardware.FS (48 kHz); pass the
actual recording rate explicitly for other sources. A vertical vector is valid even when its
horizontal azimuth is undefined.

Run:  python sig_proc_tri.py
      python sig_proc_tri.py --clips 300 --win 0.5 --snr 20 10 0 -5 -10 --demo-az 40
"""
import argparse
import io
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ------------------------------ geometry ------------------------------
from droneears import config as hardware

FS = hardware.FS  # public APIs default to the actual hardware sample rate
SIM_FS = 16000    # dataset experiment resampling rate; passed explicitly

MICS = hardware.MIC_POS.copy()
C = hardware.C_SOUND
PAIRS = list(hardware.PAIRS)

REPO = "geronimobasso/drone-audio-detection-samples"


# ------------------------------ helpers ------------------------------
def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def ang_err(a, b):
    return float(abs(wrap_deg(a - b)))


def unit_vector(az_deg, elevation_deg=0.0):
    """3D direction [right, forward, up] for headset azimuth and elevation."""
    az, el = np.deg2rad([az_deg, elevation_deg])
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def _normalized_direction(direction):
    if direction is None:
        return None
    u = np.asarray(direction, dtype=float)
    if u.shape != (3,) or not np.all(np.isfinite(u)):
        raise ValueError("direction must be a finite 3D vector or None")
    magnitude = float(np.linalg.norm(u))
    return u / magnitude if magnitude > 0 else None


def azimuth_from_direction(direction, min_horizontal=0.05):
    """Horizontal bearing in degrees, or None for missing/near-vertical vectors."""
    if not np.isfinite(min_horizontal) or not 0 < min_horizontal <= 1:
        raise ValueError("min_horizontal must be finite and in (0, 1]")
    u = _normalized_direction(direction)
    if u is None or np.linalg.norm(u[:2]) < min_horizontal:
        return None
    return float(wrap_deg(np.degrees(np.arctan2(u[0], u[1]))))


def elevation_from_direction(direction):
    """Signed angle above the headset plane; ±90 at zenith/nadir."""
    u = _normalized_direction(direction)
    if u is None:
        return None
    return float(np.degrees(np.arctan2(u[2], np.linalg.norm(u[:2]))))


def direction_error_deg(a, b):
    """3D angular separation; undefined when either vector is missing or zero."""
    a, b = _normalized_direction(a), _normalized_direction(b)
    if a is None or b is None:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def mic_delays(az_deg, mics=MICS, elevation_deg=0.0):
    """Arrival-time offsets from the coordinate origin for a far-field source.
    Negative = that mic hears the wavefront earlier."""
    return -(mics @ unit_vector(az_deg, elevation_deg)) / C


# ------------------------------ simulation ------------------------------
def simulate_array(sig, az_deg, snr_db, rng, mics=MICS, fs=FS, elevation_deg=0.0):
    """Fractional plane-wave delays and noise at SNR relative to source RMS."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 1 or sig.size < 4 or not np.all(np.isfinite(sig)):
        raise ValueError("sig must be a finite 1D signal with at least four samples")
    if not np.isfinite(fs) or fs <= 0 or np.isnan(snr_db):
        raise ValueError("positive finite fs and non-NaN SNR required")
    delays = mic_delays(az_deg, mics, elevation_deg)
    # A common arrival offset has no directional information. Remove it so
    # translating the coordinate origin cannot shift the whole clip away.
    delays -= delays.mean()
    n = len(sig)
    pad = max(64, int(np.ceil(np.max(np.abs(delays)) * fs)) + 2)
    x = np.concatenate([np.zeros(pad), sig, np.zeros(pad)])   # pad avoids wrap-around
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    out = np.stack([
        np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), n=len(x))[pad:pad + n]
        for tau in delays
    ])
    noise_std = float(np.sqrt(np.mean(sig ** 2))) * 10.0 ** (-snr_db / 20.0)
    return out + rng.normal(0.0, noise_std, out.shape)


# ------------------------------ estimation ------------------------------
def estimate_direction_srp(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0, step_deg=1.0):
    """Legacy horizontal-only SRP-PHAT comparison baseline.

    Returns (azimuth_deg, confidence, grid_deg, score_map).
    confidence is the peak of the normalised score map: ~1 = signals line up
    perfectly, ~0 = nothing coherent (noise only). Calibrate a threshold on
    recordings of your own mics with no drone present.
    """
    n = frames.shape[1]
    win = np.hanning(n)
    X = np.fft.rfft(frames * win, axis=1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= fmin) & (f <= fmax)
    f = f[band]
    X = X[:, band]

    grid = np.arange(-180.0, 180.0, step_deg)
    taus = np.stack([mic_delays(g, mics) for g in grid])      # (G, 4)

    score = np.zeros(len(grid))
    for i, j in PAIRS:
        cross = X[i] * np.conj(X[j])
        phat = cross / (np.abs(cross) + 1e-12)                # PHAT: keep phase only
        d = taus[:, i] - taus[:, j]                           # candidate lag per angle
        phase = np.exp(2j * np.pi * d[:, None] * f[None, :])  # (G, K), fractional lags
        score += np.real(np.sum(phase * phat[None, :], axis=1)) / len(f)
    score /= len(PAIRS)

    k = int(np.argmax(score))
    return float(grid[k]), float(score[k]), grid, score


@dataclass(frozen=True)
class DirectionEstimate:
    """Direction and the actual six pair delays used by the least-squares fit.

    Delays are arrival_i minus arrival_j, in PAIRS order; NaN means that pair
    had no usable spectral energy or had ambiguous correlation peaks.
    ambiguous_pairs identifies period aliases. vector=None means no 3D solution.
    """
    vector: Optional[np.ndarray]
    confidence: float
    grid: np.ndarray
    fit: np.ndarray
    delays: np.ndarray
    peaks: np.ndarray
    ambiguous_pairs: np.ndarray


def _ambiguous_correlation_peak(indices, values, physical_bound, ratio=0.9):
    """Detect competing positive lag peaks inside the physical interval.

    A pure high tone may have several near-equal period aliases. Geometric
    consistency alone cannot distinguish these, even with all six mic pairs.
    Include the constrained endpoints so an alias on a physical bound counts.
    """
    interior = indices[np.abs(indices) < physical_bound]
    locations = np.concatenate(([-physical_bound], interior, [physical_bound]))
    bounded = np.interp(locations, indices, values)
    peaks = list(bounded[1:-1][(bounded[1:-1] >= bounded[:-2]) &
                                 (bounded[1:-1] > bounded[2:])])
    if bounded[0] > bounded[1]:
        peaks.append(bounded[0])
    if bounded[-1] > bounded[-2]:
        peaks.append(bounded[-1])
    if len(peaks) < 2:
        return False
    runner_up, best = np.sort(peaks)[-2:]
    return bool(best > 0 and runner_up >= ratio * best)


def estimate_direction_details(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                               step_deg=1.0, min_magnitude=0.05, interp=32):
    """Fit a 3D direction and retain its pair-delay diagnostics.

    IFFT(X_i conj(X_j)) peaks at d_ij = arrival_i - arrival_j.
    Thus C*d_ij = (position_j-position_i) @ u. All six pairs constrain u.
    Confidence uses the weakest usable pair coherence (with a coverage penalty),
    pair-fit residual and vector magnitude;
    it is a heuristic, not a drone probability or a calibrated validity gate.
    """
    frames = np.asarray(frames, dtype=float)
    mics = np.asarray(mics, dtype=float)
    if frames.ndim != 2 or frames.shape[0] != 4 or frames.shape[1] < 4:
        raise ValueError("frames must have shape (4, N), N >= 4")
    if mics.shape != (4, 3) or not np.all(np.isfinite(mics)):
        raise ValueError("mics must be finite coordinates of shape (4, 3)")
    controls = [fs, fmin, fmax, step_deg, min_magnitude, interp]
    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(frames)):
        raise ValueError("finite frames and parameters required")
    if fs <= 0 or step_deg <= 0 or min_magnitude <= 0:
        raise ValueError("positive fs, step_deg and min_magnitude required")
    if not 0 <= fmin < fmax <= fs / 2 or interp < 2 or int(interp) != interp:
        raise ValueError("invalid frequency band or interpolation")
    interp = int(interp)
    geometry = np.array([mics[j] - mics[i] for i, j in PAIRS])
    if np.linalg.matrix_rank(geometry) < 3:
        raise ValueError("microphone geometry must span three dimensions")
    bounds = np.linalg.norm(geometry, axis=1) / C
    n = frames.shape[1]
    if np.max(bounds) * fs >= n - 1:
        raise ValueError("analysis window must exceed the maximum inter-mic delay")
    nfft = 1 << (2 * n - 1).bit_length()
    centered = frames - frames.mean(axis=1, keepdims=True)
    X = np.fft.rfft(centered * np.hanning(n), n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= fmin) & (f <= fmax) & (f > 0) & (f < fs / 2)
    if not np.any(band):
        raise ValueError("analysis window has no bins in requested band")
    lags, peaks, ambiguous = [], [], []
    for (i, j), bound in zip(PAIRS, bounds):
        cross = X[i] * np.conj(X[j])
        phat = np.zeros_like(cross)
        # Suppress numerical leakage; bins outside the analysis band must not
        # determine which in-band bins count as active.
        band_peak = float(np.max(np.abs(cross[band])))
        threshold = max(band_peak * 1e-10, 1e-20)
        active = band & (np.abs(cross) > threshold)
        count = int(np.count_nonzero(active))
        if count == 0:
            # argmax of a flat zero correlation would invent a lag at -bound.
            lags.append(float("nan"))
            peaks.append(0.0)
            ambiguous.append(False)
            continue
        phat[active] = cross[active] / np.abs(cross[active])
        cc = np.fft.irfft(phat, n=nfft * interp) * nfft * interp / (2 * count)
        # Include neighbours beyond the physical bound to interpolate peaks
        # exactly on the bound; clip the resulting lag back to that bound.
        limit = int(np.ceil(bound * fs * interp)) + 1
        indices = np.arange(-limit, limit + 1)
        values = cc[indices % len(cc)]
        aliased = _ambiguous_correlation_peak(indices, values, bound * fs * interp)
        # Short windows leak a pure tone into weak neighbouring bins. PHAT
        # promotes that leakage to unit magnitude and can hide a period alias.
        # Check ambiguity again using only bins with substantial spectral
        # support; keep the original full-band curve for fitting valid lags.
        supported = band & (np.abs(cross) > max(band_peak * 1e-6, 1e-20))
        supported_count = int(np.count_nonzero(supported))
        if not aliased and supported_count < count:
            supported_cc = np.fft.irfft(phat * supported, n=nfft * interp)
            supported_cc *= nfft * interp / (2 * supported_count)
            aliased = _ambiguous_correlation_peak(
                indices, supported_cc[indices % len(supported_cc)], bound * fs * interp)
        ambiguous.append(aliased)
        if aliased:
            lags.append(float("nan"))
            peaks.append(0.0)
            continue
        k = int(np.argmax(values))
        offset = 0.0
        if 0 < k < len(values) - 1:
            a, b, c = values[k-1:k+2]
            denom = a - 2*b + c
            if denom < -1e-15:
                offset = float(np.clip(0.5 * (a-c) / denom, -0.5, 0.5))
        lags.append(float(np.clip((indices[k] + offset) / (fs * interp), -bound, bound)))
        peaks.append(float(values[k]))
    lags, peaks, ambiguous = np.array(lags), np.array(peaks), np.array(ambiguous)
    valid = np.isfinite(lags)
    grid = np.arange(-180.0, 180.0, step_deg)
    if np.count_nonzero(valid) < 3 or np.linalg.matrix_rank(geometry[valid]) < 3:
        return DirectionEstimate(None, 0.0, grid, np.zeros_like(grid), lags, peaks, ambiguous)
    u = np.linalg.lstsq(geometry[valid], C * lags[valid], rcond=None)[0]
    magnitude = float(np.linalg.norm(u))
    fit = np.sin(np.deg2rad(grid)) * u[0] + np.cos(np.deg2rad(grid)) * u[1]
    if magnitude < min_magnitude or max(peaks) <= 0:
        return DirectionEstimate(None, 0.0, grid, fit, lags, peaks, ambiguous)
    residual = float(np.linalg.norm(geometry[valid] @ u - C * lags[valid]))
    # Three clean mics can otherwise conceal an incoherent fourth mic in the
    # mean pair score. Confidence is limited by the weakest usable baseline.
    coherence = np.clip(np.min(peaks[valid]), 0, 1) * np.count_nonzero(valid) / len(PAIRS)
    confidence = float(coherence * np.exp(-residual / 0.01) * min(magnitude, 1 / magnitude))
    return DirectionEstimate(u / magnitude, confidence, grid, fit, lags, peaks, ambiguous)


def estimate_direction(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                       step_deg=1.0, min_magnitude=0.05, interp=32):
    """Return (3D unit vector or None, confidence, grid, directional_fit).

    The vector is [right, forward, up], with unit length. It describes direction,
    not distance or position in metres. min_magnitude rejects a near-zero
    unnormalized fitted vector. Vertical sources remain valid here. The curve
    projects the raw fit horizontally;
    it is not an SRP score map. Use estimate_direction_details for pair delays.
    """
    result = estimate_direction_details(frames, fs, mics, fmin, fmax, step_deg,
                                        min_magnitude, interp)
    return result.vector, result.confidence, result.grid, result.fit


def estimate_direction_3d(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                          step_deg=1.0, min_magnitude=0.05, interp=32):
    """Compatibility name for estimate_direction; both return a 3D unit vector."""
    return estimate_direction(frames, fs, mics, fmin, fmax, step_deg, min_magnitude, interp)


def estimate_azimuth(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                     step_deg=1.0, min_horizontal=0.05, interp=32):
    """Optional horizontal-only API: (azimuth or None, confidence, grid, fit).

    Near-vertical directions have no azimuth. Use estimate_direction for their
    full 3D vector. Confidence still needs calibration before issuing a turn.
    """
    # Validate before running the FFT, including when the signal is undefined.
    azimuth_from_direction(None, min_horizontal)
    u, conf, grid, fit = estimate_direction(
        frames, fs, mics, fmin, fmax, step_deg, interp=interp)
    azimuth = azimuth_from_direction(u, min_horizontal)
    return azimuth, conf if azimuth is not None else 0.0, grid, fit


def turn_instruction(az_deg, deadband=10.0, *, confidence=1.0, min_confidence=0.2):
    """Format an azimuth only when it clears the chosen confidence threshold.

    The default is a demo threshold; calibrate it on the assembled hardware.
    Pass the estimator's confidence explicitly when generating instructions.
    """
    if (az_deg is None or not np.isfinite(az_deg) or not np.isfinite(confidence)
            or confidence < min_confidence):
        return "Azimuth unreliable or undefined - no turn instruction"
    az = float(wrap_deg(az_deg))
    side = "right" if az > 0 else "left"
    if abs(az) <= deadband:
        return "Drone ahead - hold course"
    if abs(az) >= 150.0:
        return f"Drone behind you - turn around (~{abs(az):.0f} deg, {side})"
    return f"Turn {side} {abs(az):.0f} deg"


# ------------------------------ data ------------------------------
def load_drone_clips(n, win_sec, fs, seed):
    """Random drone clips (label 1) from the dataset, cropped to win_sec, unit RMS."""
    if n < 1 or int(n) != n or not np.isfinite(win_sec) or win_sec <= 0:
        raise ValueError("positive integer clip count and positive finite window required")
    if not np.isfinite(fs) or fs <= 0 or round(fs * win_sec) < 4:
        raise ValueError("positive finite sample rate and at least four samples required")
    import soundfile as sf
    from datasets import Audio, load_dataset
    from scipy.signal import resample_poly

    ds = load_dataset(REPO, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    drone_idx = np.flatnonzero(np.array(ds["label"]) == 1)
    rng = np.random.default_rng(seed)

    clips, tried, n_target = [], 0, int(round(fs * win_sec))
    for i in rng.permutation(drone_idx):
        tried += 1
        if tried > 100 * n and len(clips) < n:
            raise RuntimeError("Too few drone clips are long enough; try a smaller --win.")
        data, sr = sf.read(io.BytesIO(ds[int(i)]["audio"]["bytes"]), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        win = int(round(sr * win_sec))
        if len(data) < win:                       # skip short clips (tiling would add periodicity)
            continue
        start = int(rng.integers(0, len(data) - win + 1))
        data = data[start:start + win]
        if sr != fs:
            g = math.gcd(int(sr), int(fs))
            data = resample_poly(data, fs // g, int(sr) // g)
        data = np.pad(data, (0, max(0, n_target - len(data))))[:n_target]
        rms = math.sqrt(float(np.mean(data ** 2)))
        if rms < 1e-4:                            # skip near-silent clips
            continue
        clips.append(data / rms)
        if len(clips) == n:
            break
    if len(clips) < n:
        raise RuntimeError(f"Only {len(clips)} usable drone clips found; requested {n}")
    return clips


# ------------------------------ experiment ------------------------------
def make_plots(snrs, med, p90, demo, path="doa_results.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    fig = plt.figure(figsize=(11, 4.3))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(snrs, med, "o-", label="median error")
    ax.plot(snrs, p90, "s--", label="90th percentile")
    ax.set_xlabel("SNR per mic (dB)")
    ax.set_ylabel("3D angular error (deg)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Accuracy vs. noise")

    true_vector, estimated_vector = demo
    spatial = fig.add_subplot(1, 2, 2, projection="3d")
    for vector, color, label in [(true_vector, "green", "True direction"),
                                  (estimated_vector, "red", "Estimated direction")]:
        if vector is not None:
            spatial.quiver(0, 0, 0, *vector, color=color, label=label)
    spatial.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                xlabel="Right (x)", ylabel="Forward (y)", zlabel="Up (z)")
    spatial.set_box_aspect((1, 1, 1))
    spatial.set_title("3D unit direction (0 dB SNR)")
    spatial.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--win", type=float, default=0.5, help="analysis window, seconds")
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 10, 0, -5, -10])
    ap.add_argument("--demo-az", type=float, default=40.0)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-confidence", type=float, default=0.2,
                    help="demo turn threshold in [0,1]; calibrate on hardware")
    args = ap.parse_args()

    if not np.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        ap.error("--min-confidence must be finite and in [0,1]")
    rng = np.random.default_rng(args.seed)
    clips = load_drone_clips(args.clips, args.win, SIM_FS, args.seed)
    print(f"Using {len(clips)} drone clips, {args.win}s window, {SIM_FS} Hz\n")

    # --- single demo ---
    est, conf, _, _ = estimate_direction(
        simulate_array(clips[0], args.demo_az, 0.0, rng, fs=SIM_FS, elevation_deg=args.elevation),
        fs=SIM_FS)
    true_vector = unit_vector(args.demo_az, args.elevation)
    azimuth = azimuth_from_direction(est)
    elevation = elevation_from_direction(est)
    vector_text = np.array2string(est, precision=4) if est is not None else "undefined"
    print(f"Demo @ 0 dB SNR: true azimuth {args.demo_az:.0f} deg, "
          f"elevation {args.elevation:.0f} deg\n"
          f"  direction [right, forward, up]: {vector_text}\n"
          f"  azimuth {azimuth} deg; elevation {elevation} deg (confidence {conf:.2f})\n"
          f"  -> {turn_instruction(azimuth, confidence=conf, min_confidence=args.min_confidence)}\n")
    demo = (true_vector, est)

    # --- SNR sweep ---
    print("3D angular error (degrees); statistics use defined direction estimates")
    print(f"{'SNR dB':>7} {'median':>8} {'mean':>8} {'p90':>8} {'<15deg':>8}")
    med, p90 = [], []
    for snr in args.snr:
        errs, undefined = [], 0
        for sig in clips:
            az = float(rng.uniform(-180, 180))
            estimated, _, _, _ = estimate_direction(
                simulate_array(sig, az, snr, rng, fs=SIM_FS, elevation_deg=args.elevation), fs=SIM_FS)
            if estimated is None:
                undefined += 1
                continue
            errs.append(direction_error_deg(estimated, unit_vector(az, args.elevation)))
        print(f"  Undefined 3D directions: {undefined}/{len(clips)}")
        errs = np.array(errs)
        if not len(errs):
            med.append(float("nan"))
            p90.append(float("nan"))
            continue
        med.append(np.median(errs))
        p90.append(np.percentile(errs, 90))
        print(f"{snr:7.0f} {med[-1]:8.1f} {errs.mean():8.1f} {p90[-1]:8.1f} "
              f"{100 * np.mean(errs < 15):7.0f}%")

    make_plots(args.snr, med, p90, demo)
    print("\nErrors are vs. simulated truth: free-field plane wave, static source, "
          "independent white noise, perfectly synced mics.\n"
          "Vectors give direction only; drone distance/position in metres is not estimated.")


if __name__ == "__main__":
    main()