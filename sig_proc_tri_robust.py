"""
Experimental robust direction-of-arrival for a 4-mic B/L/R/T helmet.
Copied from sig_proc_tri.py; the original solver is unchanged.
GCC-PHAT delays, multi-segment coherence, and robust weighted fitting. No ML.

Use estimate_direction_details for status, reliable pairs, and mirror candidates.
Only a full_3d or explicitly constrained_3d result has a vector. With exactly
three reliable mics both mirror candidates are kept (ambiguous) unless a prior
separates them: hemisphere_normal, or the opt-in assume_source_facing=True (the
side of their plane away from the excluded fourth mic, when delay uncertainty
permits; off by default: a dead or noisy mic says nothing about the source side).
An optional hemisphere_normal adds a constraint in microphone coordinates.
Neither assumption means automatic world-up or helmet-up.
This remains a free-field model: actual helmet scattering needs measurements.

What it does
  1. Pulls real drone clips (label 1) from the dataset.
  2. Simulates a plane wave from a chosen azimuth hitting four mics with
     fractional-sample delays, plus independent noise at a chosen SNR.
  3. Returns a 3D unit vector [right, forward, up] toward the drone, with
     azimuth/elevation available for display and horizontal turn instructions.
  4. Sweeps SNR and reports 3D angular accuracy and undefined estimates.

Dataset-free checks: python3 -m unittest discover -s tests -p test_triangulation_robust.py

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

Run:  python sig_proc_tri_robust.py
      python sig_proc_tri_robust.py --clips 100 --win 0.5 --snr 20 10 0 --demo-az 40
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
MIN_ANALYSIS_SECONDS = 0.01  # engineering floor; two cycles at the default 200 Hz band edge

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
    # Preserve ordinary normalization, but finite nonzero vectors can have a
    # norm that overflows or underflows. This matters for caller-scaled priors:
    # an overflowed norm used to turn a valid hemisphere normal into zero.
    largest = float(np.max(np.abs(u)))
    if largest == 0:
        return None
    with np.errstate(over='ignore', under='ignore'):
        magnitude = float(np.linalg.norm(u))
    if np.isfinite(magnitude) and magnitude > 0 and largest >= np.sqrt(np.finfo(float).tiny):
        return u / magnitude
    scaled = u / largest
    return scaled / np.linalg.norm(scaled)


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
class _PairEstimate:
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


def _pair_delay_estimate(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                         step_deg=1.0, min_magnitude=0.05, interp=32):
    """Original bounded GCC-PHAT observations, including period-alias rejection."""
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
        if not aliased:
            # PHAT can promote short-window tone leakage into an apparently
            # broadband, unique peak. Retain spectral energy in a second
            # ambiguity-only curve so the underlying period aliases survive.
            # Populate active bins only: out-of-band energy must not overflow
            # when divided by the in-band peak. The fitted lag stays PHAT-based.
            energy_cross = np.zeros_like(cross)
            energy_cross[active] = cross[active] / band_peak
            energy_cc = np.fft.irfft(energy_cross, n=nfft * interp)
            aliased = _ambiguous_correlation_peak(
                indices, energy_cc[indices % len(energy_cc)], bound * fs * interp)
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
    grid = np.arange(-180.0, 180.0, step_deg)
    return _PairEstimate(None, 0.0, grid, np.zeros_like(grid), lags, peaks, ambiguous)


@dataclass(frozen=True)
class DirectionEstimate(_PairEstimate):
    """A unique direction, an explicit constraint, or unresolved candidates.

    pair_coherence is a bias-corrected multi-segment magnitude-squared coherence
    heuristic in [0,1]. pair_weights also include peak quality and robust residual
    weights. geometry_rank is the effective weighted rank at max_condition, not
    merely the numerical rank. Low-information baselines cannot invent rank 3.
    support_confidence measures retained evidence, even when a vector is not
    identifiable. confidence is zero whenever vector is None; neither score is
    a calibrated probability. Candidates are hypotheses, not accepted fixes.
    For constrained outputs, constraint_source identifies the side assumption,
    constraint_normal is its primary normal, and excluded_mic is the fourth
    channel index when the source-facing plane assumption was used.
    """
    status: str
    candidates: np.ndarray
    pair_weights: np.ndarray
    pair_coherence: np.ndarray
    reliable_pairs: np.ndarray
    mic_reliability: np.ndarray
    geometry_rank: int
    geometry_condition: float
    support_confidence: float
    constraint_applied: bool
    reason: str
    constraint_normal: Optional[np.ndarray] = None
    excluded_mic: Optional[int] = None
    constraint_source: Optional[str] = None


def _segment_coherence(frames, fs, fmin, fmax):
    """Welch coherence, rather than the identically-one single-FFT ratio.

    At 2048 samples: seven 512-sample Hann windows with 50% overlap. The
    overlap correction estimates the effective average count (~6.7). Removing
    its 1/K noise bias is useful, but is not a statistical validity guarantee.
    Frequency averaging uses auto-power, never a maximum over coherence bins.
    Sub-10ms windows are deliberately unsupported: spectral smearing can hide
    tone aliases even when coherence is high. This engineering duration floor
    prevents known short-window failures; it is not a general alias guarantee.
    """
    n = frames.shape[1]
    if n < np.ceil(MIN_ANALYSIS_SECONDS*fs):
        return np.zeros(len(PAIRS))
    length = min(n, max(16, n // 4))
    hop = max(1, length // 2)
    starts = range(0, n-length+1, hop)
    count = len(starts)
    if count < 3:
        return np.zeros(len(PAIRS))
    window = np.hanning(length)
    frequency = np.fft.rfftfreq(length, 1/fs)
    band = (frequency >= fmin) & (frequency <= fmax) & (frequency > 0) & (frequency < fs/2)
    if not np.any(band):
        return np.zeros(len(PAIRS))
    segments = np.stack([frames[:, start:start+length] for start in starts])
    segments -= segments.mean(axis=2, keepdims=True)
    spectra = np.fft.rfft(segments*window, axis=2)[:, :, band]
    power = np.mean(np.abs(spectra)**2, axis=0)
    overlap = 0.0
    energy = float(np.sum(window**2))
    for lag in range(1, count):
        offset = lag*hop
        if offset >= length:
            break
        correlation = float(np.sum(window[:-offset]*window[offset:])) / energy
        overlap += 2*(1-lag/count)*correlation**2
    effective_count = count/(1+overlap)
    bias = 1/effective_count
    coherence = []
    for i, j in PAIRS:
        cross = np.mean(spectra[:, i]*np.conj(spectra[:, j]), axis=0)
        product = power[i]*power[j]
        supported = ((power[i] > max(float(power[i].max())*1e-6, 1e-20)) &
                     (power[j] > max(float(power[j].max())*1e-6, 1e-20)))
        if not np.any(supported):
            coherence.append(0.0)
            continue
        msc = np.clip(np.abs(cross[supported])**2/product[supported], 0, 1)
        average = float(np.average(msc, weights=np.sqrt(product[supported])))
        coherence.append(float(np.clip((average-bias)/(1-bias), 0, 1)))
    return np.array(coherence)


def _weighted_geometry(geometry, delays_m, weights, max_condition):
    """Truncated weighted SVD and minimum-norm unconstrained direction."""
    active = weights > 0
    if not np.any(active):
        return np.zeros(3), np.eye(3), 0, float('inf')
    root = np.sqrt(weights[active])
    matrix = geometry[active]*root[:, None]
    values = delays_m[active]*root
    left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    if not len(singular) or singular[0] <= 1e-15:
        return np.zeros(3), right, 0, float('inf')
    rank = int(np.count_nonzero(singular > max(singular[0]/max_condition, 1e-15)))
    condition = float(singular[0]/singular[2]) if len(singular) >= 3 and singular[2] > 1e-15 else float('inf')
    direction = right[:rank].T @ ((left[:, :rank].T @ values)/singular[:rank])
    return direction, right, rank, condition


def _source_facing_normal(mics, reliable):
    """Orient the plane of exactly three retained mics away from the fourth.

    This is a caller-authorized side assumption, not proof from signal quality
    that the excluded mic is farther from the source.
    """
    working = np.unique(np.asarray(PAIRS)[reliable])
    if len(working) != 3:
        return None
    excluded = int(next(mic for mic in range(4) if mic not in working))
    anchor, second, third = mics[working]
    span = float(np.max(np.linalg.norm(mics-anchor, axis=1)))
    cross = np.cross(second-anchor, third-anchor)
    area = float(np.linalg.norm(cross))
    if span <= 0 or area <= 1e-9*span**2:
        return None
    normal = cross/area
    side = float(normal @ (mics[excluded]-anchor))
    if abs(side) <= 1e-9*span:
        return None  # a coplanar fourth mic cannot orient the face
    if side > 0:
        normal = -normal
    return normal, excluded


def _hemisphere_intervals(projected, normal, hemisphere, geometry, weights, delay_error_m):
    """Bound both branches against a prior over the delay-error interval.

    For retained per-pair delay errors bounded by delay_error_m, weighted SVD
    bounds the minimum-norm projection error by
        epsilon = delay_error_m * sqrt(sum(weights)) / sigma_2.
    Bound both h @ projection and the signed null-space height over that ball.
    Their interval sum is conservative because it ignores their correlation.
    This is a sensitivity guard using the existing tolerance, not a calibrated
    statistical confidence interval. A boundary-touching branch stays ambiguous.
    """
    active = weights > 0
    singular = np.linalg.svd(geometry[active]*np.sqrt(weights[active, None]), compute_uv=False)
    if len(singular) < 2 or singular[1] <= 0:
        return None
    epsilon = delay_error_m*np.sqrt(np.sum(weights[active]))/singular[1]
    if not np.isfinite(epsilon):
        return None
    radius = float(np.linalg.norm(projected))
    radius_low = max(0.0, radius-epsilon)
    if radius_low > 1:
        return None
    radius_high = min(1.0, radius+epsilon)
    height_low = np.sqrt(max(0.0, 1-radius_high**2))
    height_high = np.sqrt(max(0.0, 1-radius_low**2))
    normal_part = float(hemisphere @ normal)
    in_plane = hemisphere-normal_part*normal
    center = float(hemisphere @ projected)
    spread = float(epsilon*np.linalg.norm(in_plane))
    coefficients = np.array([normal_part, -normal_part])
    ends = coefficients[:, None]*np.array([height_low, height_high])
    lower = center-spread+np.min(ends, axis=1)
    upper = center+spread+np.max(ends, axis=1)
    if not np.all(np.isfinite([lower, upper])):
        return None
    return lower, upper


def estimate_direction_details(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                               step_deg=1.0, min_magnitude=0.05, interp=32, *,
                               min_pair_coherence=0.35, max_condition=20.0,
                               delay_tolerance_samples=0.5, hemisphere_normal=None,
                               assume_source_facing=False):
    """Quality-weighted 3D fit with a source-facing three-mic fallback.

    Defaults are engineering starting points, not hardware-calibrated thresholds.
    Windows shorter than 10 ms return unreliable rather than a direction.
    Coherence requires multiple temporal segments. A Huber fit reduces isolated
    delay outliers; residuals above three tolerances are excluded. Effective
    weighted geometry must still span 3D. A rank-2 solution exposes two mirrored
    unit candidates instead of normalizing a projection into a false direction.

    assume_source_facing=True (opt-in) assumes the source lies on the side of
    the three retained microphones' plane away from the excluded fourth mic. It
    applies only when exactly three mics contribute reliable spatial constraints,
    the excluded mic is incoherent with every other mic (not merely dropped for a
    period alias or a residual outlier), and the geometry defines an oriented
    face. This is a side assumption, not an inference that a weak mic must be
    farther away: with T dead it picks the -z side, the wrong mirror for a source
    above the helmet. The default False leaves that ambiguity unresolved unless
    hemisphere_normal supplies a separating prior.

    hemisphere_normal is an optional caller-supplied assumption h @ u >= 0.
    It is in microphone coordinates; [0,0,1] means above the HELMET plane, not
    necessarily above the world horizon when the wearer tilts their head. Used
    alone, both candidates passing (or both failing) leaves it ambiguous. Branch
    selection must also remain unique within the existing three-tolerance delay
    error bound; a noisy candidate barely crossing the hemisphere is not enough.
    When both priors are present, the selected branch must satisfy both.
    """
    controls = [min_pair_coherence, max_condition, delay_tolerance_samples]
    if (not np.all(np.isfinite(controls)) or not 0 < min_pair_coherence < 1 or
            max_condition <= 1 or delay_tolerance_samples <= 0):
        raise ValueError('finite coherence in (0,1), condition >1 and positive delay tolerance required')
    if not isinstance(assume_source_facing, (bool, np.bool_)):
        raise ValueError('assume_source_facing must be boolean')
    hemisphere = None
    if hemisphere_normal is not None:
        hemisphere = _normalized_direction(hemisphere_normal)
        if hemisphere is None:
            raise ValueError('hemisphere_normal must be a finite nonzero 3D vector')
    observation = _pair_delay_estimate(frames, fs, mics, fmin, fmax, step_deg, min_magnitude, interp)
    frames, mics = np.asarray(frames, dtype=float), np.asarray(mics, dtype=float)
    coherence = _segment_coherence(frames, fs, fmin, fmax)
    mic_reliability = np.array([np.median([coherence[k] for k, pair in enumerate(PAIRS) if mic in pair])
                                for mic in range(4)])
    geometry = np.array([mics[j]-mics[i] for i, j in PAIRS])
    valid = np.isfinite(observation.delays) & (coherence >= min_pair_coherence)
    weights = np.where(valid, coherence**2*np.clip(observation.peaks, 0, 1), 0.0)
    # NaN observations must never enter weighted algebra, even at zero weight.
    delays_m = C*np.nan_to_num(observation.delays)
    scale = C*delay_tolerance_samples/fs
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('delay tolerance must produce a positive finite distance scale')
    original_weights = weights.copy()
    for _ in range(8):
        projected, _, rank, _ = _weighted_geometry(geometry, delays_m, weights, max_condition)
        if rank < 2:
            break
        residual = np.abs(geometry @ projected-delays_m)
        robust = np.ones(len(PAIRS))
        outside = residual > scale
        robust[outside] = scale/residual[outside]
        updated = original_weights*robust
        if np.max(np.abs(updated-weights)) < 1e-6:
            weights = updated
            break
        weights = updated
    projected, _, _, _ = _weighted_geometry(geometry, delays_m, weights, max_condition)
    residual = np.abs(geometry @ projected-delays_m)
    weights[residual > 3*scale] = 0.0
    projected, right, rank, condition = _weighted_geometry(geometry, delays_m, weights, max_condition)
    reliable = weights > 0
    support = 0.0
    if np.any(reliable):
        rms = float(np.sqrt(np.average((geometry[reliable] @ projected-delays_m[reliable])**2,
                                       weights=weights[reliable])))
        # WLS and its condition number are unchanged if every weight is scaled
        # down equally. Absolute GCC quality must therefore also enter the
        # confidence, rather than only the relative fit weights. Preserve the
        # conservative weakest-retained-baseline quality check.
        peak_quality = float(np.clip(np.min(observation.peaks[reliable]), 0, 1))
        support = float(peak_quality * np.average(coherence[reliable], weights=weights[reliable]) *
                        np.sqrt(np.count_nonzero(reliable)/len(PAIRS)) * np.exp(-rms/scale))
    fit = np.sin(np.deg2rad(observation.grid))*projected[0] + np.cos(np.deg2rad(observation.grid))*projected[1]
    common = dict(grid=observation.grid, fit=fit, delays=observation.delays, peaks=observation.peaks,
                  ambiguous_pairs=observation.ambiguous_pairs, pair_weights=weights, pair_coherence=coherence,
                  reliable_pairs=reliable, mic_reliability=mic_reliability, geometry_rank=rank,
                  geometry_condition=condition, support_confidence=support)

    def result(status, reason, vector=None, confidence=0.0, candidates=None, constrained=False,
               constraint_normal=None, excluded_mic=None, constraint_source=None):
        return DirectionEstimate(vector=vector, confidence=float(np.clip(confidence, 0, 1)), status=status,
                                 reason=reason, candidates=np.empty((0, 3)) if candidates is None else candidates,
                                 constraint_applied=constrained, constraint_normal=constraint_normal,
                                 excluded_mic=excluded_mic, constraint_source=constraint_source, **common)

    if rank < 2:
        if frames.shape[1] < np.ceil(MIN_ANALYSIS_SECONDS*fs):
            return result('unreliable', 'Analysis window is shorter than the supported 10 ms minimum')
        return result('unreliable', 'Fewer than two reliable spatial delay constraints')
    magnitude = float(np.linalg.norm(projected))
    if rank == 2:
        # The minimum-norm projection plus/minus the null-space component are
        # the two intersections with the unit sphere. Do not normalize an
        # impossible projection to manufacture an intersection.
        if magnitude > 1+1e-9:
            return result('unreliable', 'Rank-2 delay projection is outside the unit sphere')
        normal = right[2]
        height = float(np.sqrt(max(0.0, 1-magnitude**2)))
        candidates = np.stack([projected+height*normal, projected-height*normal])
        candidate_residuals = candidates @ geometry[reliable].T-delays_m[reliable]
        candidate_rms = np.sqrt(np.average(candidate_residuals**2, axis=1, weights=weights[reliable]))
        if np.any(candidate_rms > 3*scale):
            return result('unreliable', 'Rank-2 candidates do not fit the retained delay evidence')
        priors = []
        face = _source_facing_normal(mics, reliable) if assume_source_facing else None
        if face is not None:
            # Only a mic that is incoherent with every other mic can be read as shadowed;
            # pairs dropped for period aliases or residual outliers say nothing about side.
            incident = np.array([face[1] in pair for pair in PAIRS])
            if not np.all(coherence[incident] < min_pair_coherence):
                face = None
        if face is not None:
            priors.append(('source_facing', face[0]))
        if hemisphere is not None:
            priors.append(('hemisphere', hemisphere))
        if priors:
            intervals = [_hemisphere_intervals(projected, normal, prior, geometry, weights, 3*scale)
                         for _, prior in priors]
            if all(interval is not None for interval in intervals):
                lower = np.stack([interval[0] for interval in intervals])
                upper = np.stack([interval[1] for interval in intervals])
                # The chosen branch must satisfy every prior; at least one
                # prior must rule out its mirror. A prior admitting both is OK.
                allowed = np.all(lower > 1e-9, axis=0)
                excluded = np.any(upper < -1e-9, axis=0)
                branches = [branch for branch in range(2) if allowed[branch] and excluded[1-branch]]
                if len(branches) == 1:
                    branch = branches[0]
                    source = '_and_'.join(name for name, _ in priors)
                    return result('constrained_3d',
                                  'One branch selected by '+source+' assumption within the delay tolerance',
                                  vector=candidates[branch].copy(), confidence=support,
                                  candidates=candidates, constrained=True,
                                  constraint_normal=priors[0][1].copy(),
                                  excluded_mic=face[1] if face is not None else None,
                                  constraint_source=source)
            return result('ambiguous', 'Side assumptions cannot separate the candidates within the delay tolerance',
                          candidates=candidates)
        return result('ambiguous', 'Reliable geometry has a missing or poorly constrained third dimension',
                      candidates=candidates)
    if magnitude < min_magnitude:
        return result('unreliable', 'Fitted direction has negligible magnitude')
    vector = projected/magnitude
    if hemisphere is not None and float(vector @ hemisphere) < -1e-9:
        return result('unreliable', 'Full 3D estimate conflicts with the explicit hemisphere assumption')
    unit_rms = float(np.sqrt(np.average((geometry[reliable] @ vector-delays_m[reliable])**2,
                                        weights=weights[reliable])))
    if unit_rms > 3*scale:
        return result('unreliable', 'Reliable delays are inconsistent with a unit direction')
    confidence = support*np.exp(-unit_rms/scale)*min(magnitude, 1/magnitude)*min(1.0, 3.0/condition)
    return result('full_3d', 'Reliable weighted geometry spans three dimensions', vector=vector,
                  confidence=confidence, candidates=vector[None, :].copy())


def estimate_direction(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                       step_deg=1.0, min_magnitude=0.05, interp=32, **robust_options):
    """Return (3D unit vector or None, confidence, grid, directional_fit).

    The vector is [right, forward, up], with unit length. It describes direction,
    not distance or position in metres. min_magnitude rejects a near-zero
    unnormalized fitted vector. Vertical sources remain valid here. The curve
    projects the raw fit horizontally;
    it is not an SRP score map. Use estimate_direction_details for pair delays.
    """
    result = estimate_direction_details(frames, fs, mics, fmin, fmax, step_deg,
                                        min_magnitude, interp, **robust_options)
    return result.vector, result.confidence, result.grid, result.fit


def estimate_direction_3d(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                          step_deg=1.0, min_magnitude=0.05, interp=32, **robust_options):
    """Compatibility name for estimate_direction; both return a 3D unit vector."""
    return estimate_direction(frames, fs, mics, fmin, fmax, step_deg, min_magnitude, interp, **robust_options)


def estimate_azimuth(frames, fs=FS, mics=MICS, fmin=200.0, fmax=3000.0,
                     step_deg=1.0, min_horizontal=0.05, interp=32, **robust_options):
    """Optional horizontal-only API: (azimuth or None, confidence, grid, fit).

    Near-vertical directions have no azimuth. Use estimate_direction for their
    full 3D vector. Confidence still needs calibration before issuing a turn.
    """
    # Validate before running the FFT, including when the signal is undefined.
    azimuth_from_direction(None, min_horizontal)
    u, conf, grid, fit = estimate_direction(
        frames, fs, mics, fmin, fmax, step_deg, interp=interp, **robust_options)
    azimuth = azimuth_from_direction(u, min_horizontal)
    return azimuth, conf if azimuth is not None else 0.0, grid, fit


def turn_instruction(az_deg, deadband=10.0, *, confidence=1.0, min_confidence=0.2):
    """Format an azimuth only when it clears the chosen confidence threshold.

    The default is a demo threshold; calibrate it on the assembled hardware.
    Pass the estimator's confidence explicitly when generating instructions.
    """
    if (not np.all(np.isfinite([deadband, min_confidence])) or
            not 0 <= deadband <= 180 or not 0 <= min_confidence <= 1):
        raise ValueError('deadband must be finite in [0,180] and min_confidence in [0,1]')
    if (az_deg is None or not np.isfinite(az_deg) or not np.isfinite(confidence)
            or not 0 <= confidence <= 1
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
    ap.add_argument("--assume-above-helmet", action="store_true",
                    help="explicitly allow an upper-helmet-hemisphere constraint for rank-2 fallback")
    ap.add_argument("--source-facing-prior", action="store_true",
                    help="opt in to assuming a three-mic source is away from the excluded mic (off by default)")
    ap.add_argument("--no-source-facing-prior", action="store_true",
                    help="kept for old command lines; the prior is now off unless --source-facing-prior")
    args = ap.parse_args()

    if not np.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        ap.error("--min-confidence must be finite and in [0,1]")
    rng = np.random.default_rng(args.seed)
    clips = load_drone_clips(args.clips, args.win, SIM_FS, args.seed)
    print(f"Using {len(clips)} drone clips, {args.win}s window, {SIM_FS} Hz\n")

    # --- single demo ---
    options = dict(assume_source_facing=args.source_facing_prior and not args.no_source_facing_prior)
    if args.assume_above_helmet:
        options['hemisphere_normal'] = [0, 0, 1]
    details = estimate_direction_details(
        simulate_array(clips[0], args.demo_az, 0.0, rng, fs=SIM_FS, elevation_deg=args.elevation),
        fs=SIM_FS, **options)
    est, conf = details.vector, details.confidence
    true_vector = unit_vector(args.demo_az, args.elevation)
    azimuth = azimuth_from_direction(est)
    elevation = elevation_from_direction(est)
    vector_text = np.array2string(est, precision=4) if est is not None else "undefined"
    print(f"Demo @ 0 dB SNR: true azimuth {args.demo_az:.0f} deg, "
          f"elevation {args.elevation:.0f} deg\n"
          f"  direction [right, forward, up]: {vector_text}\n"
          f"  azimuth {azimuth} deg; elevation {elevation} deg (confidence {conf:.2f})\n"
          f"  status {details.status}; {details.reason}\n"
          f"  -> {turn_instruction(azimuth, confidence=conf, min_confidence=args.min_confidence)}\n")
    if details.status == 'ambiguous':
        print('  Ambiguous candidates [right, forward, up]:', details.candidates)
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
                simulate_array(sig, az, snr, rng, fs=SIM_FS, elevation_deg=args.elevation), fs=SIM_FS, **options)
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

    make_plots(args.snr, med, p90, demo, path="doa_robust_results.png")
    print("\nErrors are vs. simulated truth: free-field plane wave, static source, "
          "independent white noise, perfectly synced mics.\n"
          "Vectors give direction only; drone distance/position in metres is not estimated.")


if __name__ == "__main__":
    main()
