"""Direction of arrival from four synchronously sampled microphones (B, L, R in the head plane, T on the crown).

Per frame, Rachel's robust solver (sig_proc_tri_robust.py estimate_direction_details) ported to one DSP frame
(NFFT 2048, the per-channel spectra and the detector's harmonic bin mask), mirrored line for line by
firmware/core/dsp_core.cpp:

1. Pair delays. For each mic pair with both mics connected: masked GCC-PHAT, evaluated directly at the integer lags
   within the pair's physical bound (+-|p_i - p_j| / c, plus one lag) as a sum over the masked bins (no inverse FFT),
   peak + parabolic interpolation. Her period-alias test: a second local maximum inside the physical interval
   (endpoints count) >= BEARING_ALIAS_RATIO x the best, on the PHAT curve, on PHAT over the well-supported bins only,
   and on the energy-weighted cross-spectrum -> no delay. "peak" = the PHAT correlation (1 = every bin in phase).
2. Pair coherence: her bias-corrected magnitude-squared coherence, averaged over this frame's masked bins with weights
   sqrt(Pii Pjj), where the averaging runs over consecutive DSP frames (an EMA of the cross / auto spectra in the
   bearing band, ~6.7 effective segments as her 7 Welch sub-windows; one 2048-point frame has a single segment).
3. Reliable pair: delay present and coherence >= BEARING_MIN_PAIR_COH; weight coherence^2 * clip(peak, 0, 1). Huber
   reweighting of a weighted minimum-norm least-squares fit (rank: singular values > sigma_max / BEARING_MAX_COND),
   pairs with |residual| > 3 tolerances dropped, refit. The mics in the remaining pairs are this frame's `mused`: a
   weak, noisy or dead mic drops out on the fly (the "3 of 4").
4. Status. rank 3 (all four mics): full_3d, the unit vector p/|p|. rank 2 (three mics, or four whose reliable pairs
   span only a plane): the two mirror candidates p +- h n (n the unobservable axis, h = sqrt(1 - |p|^2)); the upper-
   hemisphere prior (BEARING_UPPER_PRIOR, normal +z: the drone is at or above the head plane) picks one through her
   delay-uncertainty interval rule -> constrained_3d. If it cannot: a level mirror plane (B, L, R) still gives the
   azimuth (azimuth_only, elevation unknown); otherwise ambiguous, with the azimuth and/or elevation only where both
   candidates share it (L, R, T: front/back mirror, elevation shared, no azimuth). Fewer than three mics / rank < 2:
   unreliable. The source-facing prior of her file is never applied (its premise fails for dead / noisy mics).

Conventions: x right, y forward, z up. theta = 0 deg dead ahead, +90 deg right, +/-180 deg behind;
elev = 0 deg in the head plane, +90 deg straight up.
tau_ij = (arrival time at mic j) - (arrival time at mic i), in seconds, for pair (i, j): c tau = (p_i - p_j) . u.
"""
import numpy as np
from . import config as C

A = np.array([(C.MIC_POS[i] - C.MIC_POS[j]) / C.C_SOUND for i, j in C.PAIRS])   # (N_PAIRS, 3): tau = A @ u
A_PINV = np.linalg.pinv(A)                                                         # (3, N_PAIRS)
TAU_MAX = [np.linalg.norm(C.MIC_POS[i] - C.MIC_POS[j]) / C.C_SOUND for i, j in C.PAIRS]   # seconds

STATUS = ["unreliable", "ambiguous", "azimuth_only", "constrained_3d", "full_3d"]   # index = C core status code
BSTAT = {"unreliable": 0, "ambiguous": 1, "azimuth_only": 2, "constrained_3d": 2, "full_3d": 3}


def _bearing_bins():
    """First / last bin any bearing mask (harmonic_mask, band_mask) can set: the coherence state covers these."""
    lo = int(C.BEARING_BAND[0] / C.BIN_HZ)
    hi = int(C.BEARING_BAND[1] / C.BIN_HZ)
    lo = min(lo, int(np.floor(C.BEARING_BAND[0] / C.BIN_HZ + 0.5)) - C.BEARING_MASK_HALFWIDTH_BINS)
    hi = max(hi, int(np.floor(C.BEARING_BAND[1] / C.BIN_HZ + 0.5)) + C.BEARING_MASK_HALFWIDTH_BINS)
    return max(lo, 1), min(hi, C.NBINS - 2)


BRG_LO, BRG_HI = _bearing_bins()
BRG_NB = BRG_HI - BRG_LO + 1
_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(C.NFFT) / C.NFFT)
COH_RHO1 = float(np.sum(_HANN[:-C.HOP] * _HANN[C.HOP:]) / np.sum(_HANN ** 2))     # window overlap of adjacent frames
_TWC = np.cos(2 * np.pi * np.arange(C.NFFT) / C.NFFT)                              # e^{+j 2 pi m / N}
_TWS = np.sin(2 * np.pi * np.arange(C.NFFT) / C.NFFT)
SCALE_M = C.C_SOUND * C.BEARING_DELAY_TOL / C.FS                                   # Huber scale, metres (~3.6 mm)


def max_lag(tau_max_s):
    return int(np.ceil(tau_max_s * C.FS)) + 1


def lag_curve(cr, ci, bins, L):
    """sum_k cr_k cos(2 pi k n / N) - ci_k sin(2 pi k n / N) for n = -L..L (= N/2 x irfft of the spectrum cr + j ci
    on those bins, exactly, at integer lags)."""
    n = np.arange(-L, L + 1)
    m = (np.asarray(bins)[:, None] * n[None, :]) % C.NFFT
    return np.asarray(cr) @ _TWC[m] - np.asarray(ci) @ _TWS[m]


def ambiguous_peak(values, bound):
    """Her _ambiguous_correlation_peak on integer lags -L..L (values[L] = lag 0): competing local maxima inside the
    physical interval [-bound, bound] (samples; the interpolated endpoints count)."""
    L = (len(values) - 1) // 2
    idx = np.arange(-L, L + 1)
    interior = idx[np.abs(idx) < bound]
    loc = np.concatenate(([-bound], interior, [bound]))
    b = np.interp(loc, idx, values)
    mid = b[1:-1]
    peaks = list(mid[(mid >= b[:-2]) & (mid > b[2:])])
    if b[0] > b[1]:
        peaks.append(b[0])
    if b[-1] > b[-2]:
        peaks.append(b[-1])
    if len(peaks) < 2:
        return False
    runner, best = np.sort(peaks)[-2:]
    return bool(best > 0 and runner >= C.BEARING_ALIAS_RATIO * best)


def gcc_phat(Xi, Xj, mask, tau_max, fs=C.FS, nfft=C.NFFT):
    """Delay of channel j relative to channel i (seconds) from one frame's spectra, via masked GCC-PHAT evaluated at
    the integer lags |n| <= ceil(tau_max fs) + 1, with parabolic interpolation and the period-alias test.

    Returns (tau, peak, alias): tau NaN (peak 0) when no masked bin carries energy or the peak is aliased; tau is
    clipped to +-tau_max; peak = the interpolated PHAT correlation peak in [-1, 1]."""
    bins = np.flatnonzero(mask[BRG_LO:BRG_HI + 1] > 0) + BRG_LO      # every bearing mask lies inside this range
    if not len(bins):
        return float("nan"), 0.0, False
    w = mask[bins]
    cross = np.conj(Xi[bins]) * Xj[bins] * w           # peaks at +D when x_j = x_i delayed by D
    mag = np.abs(cross)
    band_peak = float(mag.max())
    act = mag > max(band_peak * 1e-10, 1e-20)
    cnt = int(act.sum())
    if cnt == 0:
        return float("nan"), 0.0, False
    b, c, m = bins[act], cross[act], mag[act]
    ph = c / m
    L = max_lag(tau_max)
    bound = tau_max * fs
    r = lag_curve(ph.real, ph.imag, b, L) / cnt
    alias = ambiguous_peak(r, bound)
    if not alias:
        sup = m > max(band_peak * 1e-6, 1e-20)
        if sup.sum() < cnt:
            alias = ambiguous_peak(lag_curve(ph.real[sup], ph.imag[sup], b[sup], L), bound)
    if not alias:
        e = c / band_peak
        alias = ambiguous_peak(lag_curve(e.real, e.imag, b, L), bound)
    if alias:
        return float("nan"), 0.0, True
    k = int(np.argmax(r))
    frac, val = 0.0, float(r[k])
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        den = y0 - 2 * y1 + y2
        if den < -1e-12:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
            val = float(y1 - 0.25 * (y0 - y2) * frac)
    tau = float(np.clip((k - L + frac) / fs, -tau_max, tau_max))
    return tau, val, False


def solve_direction(taus, pinv=A_PINV):
    """Plain least-squares direction from the N_PAIRS pairwise delays. Returns (unit u (3,), |u|)."""
    u = pinv @ np.asarray(taus)
    n = np.linalg.norm(u)
    return u / n if n > 0 else u, float(n)


def mic_layout(active=None, pos=None):
    """The connected mics. active[c]: channel c (MIC_NAMES order) carries a mic; pos[c]: where it sits (default
    MIC_POS). Returns dict(active, pos, pinv (3, N_PAIRS) = plain least squares over the pairs with both mics (zero
    columns elsewhere), tau_max (0 for those pairs; the robust solver measures only the others), rank, level,
    azimuth_ok, elevation_ok: whether frames CAN carry them). rank 3: the full 3D direction. rank 2 (three mics):
    two mirror candidates; a level plane (B, L, R) always gives the azimuth, the upper-hemisphere prior the full
    direction when the drone is clearly above; a vertical plane (L, R, T) mirrors front / back, which the prior cannot
    separate: the elevation (shared by both candidates) but no azimuth. Fewer than three mics: detection only."""
    active = np.ones(C.N_MICS, bool) if active is None else np.asarray(active, bool)
    pos = C.MIC_POS if pos is None else np.asarray(pos, float)
    A_ = np.zeros((C.N_PAIRS, 3))
    tm = np.zeros(C.N_PAIRS)
    for p, (i, j) in enumerate(C.PAIRS):
        if active[i] and active[j]:
            A_[p] = (pos[i] - pos[j]) / C.C_SOUND
            tm[p] = np.linalg.norm(pos[i] - pos[j]) / C.C_SOUND
    w, V = np.linalg.eigh(A_.T @ A_)                                # ascending; V[:, k] the k-th axis
    keep = w > C.LAYOUT_RCOND * max(w[-1], 1e-30)
    rank = int(keep.sum())
    pinv = (V[:, keep] / w[keep]) @ V[:, keep].T @ A_.T            # = pinv(A), with the unobservable axes left out
    nz = abs(V[2, 0]) if rank == 2 else 0.0                         # vertical part of the plane's normal
    level = rank == 2 and nz > C.LAYOUT_LEVEL_Z
    vertical = rank == 2 and nz < 1 - C.LAYOUT_LEVEL_Z
    prior = rank == 2 and C.BEARING_UPPER_PRIOR and not vertical    # the +z prior can pick a mirror candidate
    return dict(active=active, pos=pos, pinv=pinv, tau_max=tm, rank=rank, level=bool(level),
                azimuth_ok=bool(rank == 3 or level or prior), elevation_ok=bool(rank == 3 or vertical or prior))


def bearing_deg(u):
    return float(np.degrees(np.arctan2(u[0], u[1])))


def elevation_deg(u):
    n = float(np.linalg.norm(u))
    return float(np.degrees(np.arcsin(np.clip(u[2] / max(n, 1e-12), -1.0, 1.0))))


def _azimuth(u):
    """Azimuth of a unit vector, None when it is near-vertical (horizontal part < BEARING_MIN_HORIZ)."""
    return bearing_deg(u) if np.hypot(u[0], u[1]) >= C.BEARING_MIN_HORIZ else None


def _wdiff(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


class _Coherence:
    """EMA (weight BEARING_COH_EMA) of the auto spectra and of the pair cross spectra conj(Xi) Xj over the bearing
    band, and the effective number of averaged frames (50 %-overlap Hann frames are correlated: COH_RHO1)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.P = np.zeros((C.N_MICS, BRG_NB))
        self.S = np.zeros((C.N_PAIRS, BRG_NB), complex)
        self.sw = self.sw2 = self.swc = 0.0
        self.n = 0

    def update(self, X, active):
        a = C.BEARING_COH_EMA
        Xb = X[:, BRG_LO:BRG_HI + 1]
        for c in range(C.N_MICS):
            if active[c]:
                self.P[c] = (1 - a) * self.P[c] + a * (Xb[c].real ** 2 + Xb[c].imag ** 2)
        for p, (i, j) in enumerate(C.PAIRS):
            if active[i] and active[j]:
                self.S[p] = (1 - a) * self.S[p] + a * (np.conj(Xb[i]) * Xb[j])
        self.swc = (a * a * (1 - a) if self.n >= 1 else 0.0) + (1 - a) ** 2 * self.swc
        self.sw = a + (1 - a) * self.sw
        self.sw2 = a * a + (1 - a) ** 2 * self.sw2
        self.n += 1

    def keff(self):
        d = self.sw2 + 2 * COH_RHO1 ** 2 * self.swc
        return self.sw ** 2 / d if d > 0 else 0.0

    def pair(self, p, i, j, bins):
        """Bias-corrected coherence of pair p over the given bins (absolute bin numbers)."""
        k = self.keff()
        if self.n < C.BEARING_COH_MIN_FRAMES or k <= 1.0 + 1e-6 or not len(bins):
            return 0.0
        b = bins - BRG_LO
        Pi, Pj = self.P[i, b], self.P[j, b]
        sup = (Pi > max(float(Pi.max()) * 1e-6, 1e-20)) & (Pj > max(float(Pj.max()) * 1e-6, 1e-20))
        if not sup.any():
            return 0.0
        prod = Pi[sup] * Pj[sup]
        S = self.S[p, b][sup]
        msc = np.clip((S.real ** 2 + S.imag ** 2) / prod, 0.0, 1.0)
        wt = np.sqrt(prod)
        avg = float(np.sum(msc * wt) / np.sum(wt))
        bias = 1.0 / k
        return float(np.clip((avg - bias) / (1.0 - bias), 0.0, 1.0))


def _wls(G, d, w):
    """Her _weighted_geometry via the 3x3 normal equations: minimum-norm fit over the singular values above
    sigma_max / BEARING_MAX_COND. Returns (p, V (columns: axes, ascending), rank, sigma (ascending))."""
    act = w > 0
    if not act.any():
        return np.zeros(3), np.eye(3), 0, np.zeros(3)
    M = (G[act].T * w[act]) @ G[act]
    bv = G[act].T @ (w[act] * d[act])
    lam, V = np.linalg.eigh(M)
    sig = np.sqrt(np.maximum(lam, 0.0))
    if sig[2] <= 1e-15:
        return np.zeros(3), V, 0, sig
    keep = sig > max(sig[2] / C.BEARING_MAX_COND, 1e-15)
    p = np.zeros(3)
    for k in range(3):
        if keep[k]:
            p += V[:, k] * (V[:, k] @ bv) / lam[k]
    return p, V, int(keep.sum()), sig


def _wrms(G, d, w, u):
    r = (G @ u - d)[w > 0]
    return float(np.sqrt(np.sum(w[w > 0] * r * r) / np.sum(w[w > 0])))


def _prior_branch(p, n, sig, w, prior):
    """Her _hemisphere_intervals + branch rule for candidates p + h n (branch 0) and p - h n (branch 1): the branch
    the prior selects within the 3-tolerance delay-error bound, or None."""
    if sig[1] <= 0:
        return None
    eps = 3 * SCALE_M * np.sqrt(np.sum(w[w > 0])) / sig[1]
    if not np.isfinite(eps):
        return None
    r = float(np.linalg.norm(p))
    rl = max(0.0, r - eps)
    if rl > 1:
        return None
    rh = min(1.0, r + eps)
    hl, hh = np.sqrt(max(0.0, 1 - rh * rh)), np.sqrt(max(0.0, 1 - rl * rl))
    a = float(prior @ n)
    center = float(prior @ p)
    spread = float(eps * np.linalg.norm(prior - a * n))
    lower, upper = [], []
    for s in (1.0, -1.0):
        lo_, hi_ = s * a * hl, s * a * hh
        lower.append(center - spread + min(lo_, hi_))
        upper.append(center + spread + max(lo_, hi_))
    ok = [k for k in (0, 1) if lower[k] > 1e-9 and upper[1 - k] < -1e-9]
    return ok[0] if len(ok) == 1 else None


_UP = np.array([0.0, 0.0, 1.0])


def robust_direction(taus, peaks, coh, active_pairs, pos=C.MIC_POS):
    """The per-frame decision (steps 3-4 above) from the pair delays (s, NaN = none), PHAT peaks and coherences.
    Returns dict(status, u (unit vector or None), cands (list of unit vectors), theta_raw, elev (None if unknown),
    horiz (the horizontal vector the smoothed theta takes, or None), conf, support, rank, mused, weights, dropped
    (the mic the leave-one-out step removed, or None)).

    Leave one mic out: a mic that is noisy but still coherent enough to pass the pair test skews all three of its
    pairs, which the Huber / residual-outlier passes cannot isolate (three bad pairs of six), so the four-mic fit
    fails its consistency checks. Then each mic in that fit is left out in turn (her fit on the other mics' pairs)
    and the passing subset with the most support wins: the "3 of 4" on a mic that does not show as incoherent."""
    G = np.array([pos[i] - pos[j] for i, j in C.PAIRS])
    taus, peaks, coh = np.asarray(taus, float), np.asarray(peaks, float), np.asarray(coh, float)
    valid = np.asarray(active_pairs, bool) & np.isfinite(taus) & (coh >= C.BEARING_MIN_PAIR_COH)
    w0 = np.where(valid, coh ** 2 * np.clip(peaks, 0.0, 1.0), 0.0)
    d = C.C_SOUND * np.nan_to_num(taus)
    out = _fit(G, d, w0, peaks, coh)
    out["dropped"] = None
    if out["status"] == "unreliable" and out["rank"] >= 2 and bin(out["mused"]).count("1") == C.N_MICS:
        best = None
        for k in range(C.N_MICS):
            wk = np.array([0.0 if k in pr else w0[q] for q, pr in enumerate(C.PAIRS)])
            r = _fit(G, d, wk, peaks, coh)
            if r["status"] != "unreliable" and (best is None or r["support"] > best["support"]):
                best, r["dropped"] = r, k
        if best is not None:
            out = best
    return out


def _fit(G, d, w0, peaks, coh):
    """Her estimate_direction_details from the initial pair weights on: Huber, residual outliers, status."""
    w = w0.copy()
    for _ in range(C.BEARING_HUBER_ITERS):
        p, _, rank, _ = _wls(G, d, w)
        if rank < 2:
            break
        res = np.abs(G @ p - d)
        rob = np.where(res > SCALE_M, SCALE_M / np.maximum(res, 1e-30), 1.0)
        upd = w0 * rob
        done = np.max(np.abs(upd - w)) < 1e-6
        w = upd
        if done:
            break
    p, _, _, _ = _wls(G, d, w)
    w = np.where(np.abs(G @ p - d) > 3 * SCALE_M, 0.0, w)
    p, V, rank, sig = _wls(G, d, w)
    rel = w > 0
    mused = 0
    for q, (i, j) in enumerate(C.PAIRS):
        if rel[q]:
            mused |= (1 << i) | (1 << j)
    support = 0.0
    if rel.any():
        rms = _wrms(G, d, w, p)
        support = float(np.clip(np.min(peaks[rel]), 0, 1) * np.sum(coh[rel] * w[rel]) / np.sum(w[rel]) *
                        np.sqrt(rel.sum() / C.N_PAIRS) * np.exp(-rms / SCALE_M))
    out = dict(status="unreliable", u=None, cands=[], theta_raw=None, elev=None, horiz=None, conf=0.0,
               support=support, rank=rank, mused=mused, weights=w)
    if rank < 2:
        return out
    m = float(np.linalg.norm(p))
    if rank == 3:
        if m < C.BEARING_MIN_MAG:
            return out
        u = p / m
        urms = _wrms(G, d, w, u)
        if urms > 3 * SCALE_M:
            return out
        cond = float(sig[2] / sig[0]) if sig[0] > 1e-15 else float("inf")
        conf = support * np.exp(-urms / SCALE_M) * min(m, 1 / m) * min(1.0, 3.0 / cond)
        out.update(status="full_3d", u=u, cands=[u], theta_raw=_azimuth(u), elev=elevation_deg(u),
                   horiz=u[:2].copy(), conf=float(np.clip(conf, 0, 1)))
        return out
    # rank 2: the two intersections of the unit sphere with the line p + t n (n: the unobservable axis). Her rule
    # rejects |p| > 1; a projection outside the sphere by no more than the delay-error bound (a source in the mic
    # plane, which is half of all B, L, R frames at low elevation) gives the in-plane direction p / |p| instead.
    n = V[:, 0]
    eps = 3 * SCALE_M * np.sqrt(np.sum(w[rel])) / sig[1] if sig[1] > 0 else float("inf")
    if m > 1 + min(eps, 1.0):
        return out
    if m > 1:
        cands = [p / m, p / m]
    else:
        h = np.sqrt(max(0.0, 1 - m * m))
        cands = [p + h * n, p - h * n]
    if max(_wrms(G, d, w, cands[0]), _wrms(G, d, w, cands[1])) > 3 * SCALE_M:
        return out
    out.update(cands=cands, conf=float(np.clip(support, 0, 1)))
    br = _prior_branch(p, n, sig, w, _UP) if C.BEARING_UPPER_PRIOR else None
    if br is not None:
        u = cands[br]
        out.update(status="constrained_3d", u=u, theta_raw=_azimuth(u), elev=elevation_deg(u), horiz=u[:2].copy())
        return out
    # no side selected: what the two candidates share
    hz = 0.5 * (cands[0] + cands[1])                      # = p (or p / |p|): their common in-plane part
    if abs(n[2]) > C.LAYOUT_LEVEL_Z and np.hypot(hz[0], hz[1]) >= C.BEARING_MIN_HORIZ:
        out.update(status="azimuth_only", theta_raw=bearing_deg(hz), horiz=hz[:2].copy())   # level mirror plane
        return out
    # Shared within the delay-error bound: the candidates as far out of the plane as |p| - eps allows must agree
    # (a source near the mic plane has |p| ~ 1 and nominal candidates that coincide, yet its out-of-plane part, and
    # with it the azimuth across a vertical plane, is uncertain by up to sqrt(1 - (|p| - eps)^2)).
    hmax = np.sqrt(max(0.0, 1 - max(0.0, m - eps) ** 2))
    wide = [p + hmax * n, p - hmax * n]
    az = [_azimuth(c / np.linalg.norm(c)) for c in wide]
    out.update(status="ambiguous")
    if az[0] is not None and az[1] is not None and _wdiff(az[0], az[1]) <= C.BEARING_SHARE_DEG:
        out.update(theta_raw=bearing_deg(hz), horiz=hz[:2].copy())
    if abs(elevation_deg(wide[0]) - elevation_deg(wide[1])) <= C.BEARING_SHARE_DEG:
        out.update(elev=0.5 * (elevation_deg(cands[0]) + elevation_deg(cands[1])))
    if out["theta_raw"] is None and out["elev"] is None:
        out["conf"] = 0.0                                 # nothing shared: no direction, no confidence (as hers)
    return out


class BearingEstimator:
    """Stateful: feed one frame of N_MICS spectra and a bin mask. Per frame the robust solver above; across frames the
    pair coherence state and an EMA of the direction (theta is the azimuth of the EMA'd vector, whose horizontal part
    is cos(elevation) long: near-overhead frames, whose azimuth is ill-conditioned, count little)."""

    def __init__(self, alpha=C.BEARING_EMA, layout=None):
        self.alpha = alpha
        self.layout = layout       # mic_layout(): only the connected mics; None = all four at MIC_POS
        self.coh = _Coherence()
        self.v = np.zeros(3)       # smoothed (non-unit) direction vector
        self.stale = 0             # detected frames since the EMA last took one
        self.last = None
        self.overhead = False      # drone above the head (elevation persistently high while detected)
        self._low = 0              # frames with elev > OVERHEAD_ELEV_ON (while not overhead; unknown elev skipped)
        self._high = 0             # frames with elev < OVERHEAD_ELEV_OFF (while overhead; unknown elev skipped)

    def _update_overhead(self, elev, update):
        if not update:                                   # not detected: no drone, no overhead
            self.overhead = False; self._low = self._high = 0
        elif elev is None:                               # no elevation this frame: the counts wait
            pass
        elif not self.overhead:
            self._low = self._low + 1 if elev > C.OVERHEAD_ELEV_ON else 0
            if self._low >= C.OVERHEAD_FRAMES:
                self.overhead = True; self._low = self._high = 0
        else:
            self._high = self._high + 1 if elev < C.OVERHEAD_ELEV_OFF else 0
            if self._high >= C.OVERHEAD_FRAMES:
                self.overhead = False; self._low = self._high = 0

    def feed(self, X, mask, update=True):
        """X: (N_MICS, NBINS) complex spectra in MIC_NAMES order (B, L, R, T). Returns dict(theta, theta_raw, conf,
        elev, overhead, status, bstat, mused, taus, peaks, coh, u). theta is smoothed; theta_raw / elev are this
        frame's (None when unknown); theta is None when no azimuth has reached the EMA for BEARING_HOLD_FRAMES
        detected frames (or ever) and this frame has none. update (= detected) gates the EMA (also on
        conf >= BEARING_MIN_CONF) and the overhead state (which needs a known elevation)."""
        L = self.layout
        act = np.ones(C.N_MICS, bool) if L is None else L["active"]
        tmax = TAU_MAX if L is None else L["tau_max"]
        pos = C.MIC_POS if L is None else L["pos"]
        self.coh.update(X, act)
        bins = np.flatnonzero(mask[BRG_LO:BRG_HI + 1] > 0) + BRG_LO
        taus, peaks, cohs, on = [], [], [], []
        for q, ((i, j), tm) in enumerate(zip(C.PAIRS, tmax)):
            pair_on = bool(act[i] and act[j] and tm > 0)
            t, pk = float("nan"), 0.0
            if pair_on:
                t, pk, _ = gcc_phat(X[i], X[j], mask, tm)
            taus.append(t); peaks.append(pk); on.append(pair_on)
            cohs.append(self.coh.pair(q, i, j, bins) if pair_on else 0.0)
        r = robust_direction(taus, peaks, cohs, on, pos)
        a = self.alpha
        took = False
        if update and r["horiz"] is not None and r["conf"] >= C.BEARING_MIN_CONF:
            if self.stale >= C.BEARING_HOLD_FRAMES:
                self.v[:] = 0.0                          # a stale EMA restarts from this frame
            z = r["u"][2] if r["u"] is not None else (np.sin(np.radians(r["elev"])) if r["elev"] is not None
                                                     else self.v[2])
            self.v = (1 - a) * self.v + a * np.array([r["horiz"][0], r["horiz"][1], z])
            took = True
        if took:
            self.stale = 0
        elif update:
            self.stale += 1
        n = np.linalg.norm(self.v[:2])
        theta = bearing_deg(self.v) if n > 1e-9 and self.stale < C.BEARING_HOLD_FRAMES else r["theta_raw"]
        self._update_overhead(r["elev"], update)
        self.last = dict(theta=theta, theta_raw=r["theta_raw"], conf=r["conf"], elev=r["elev"],
                         overhead=self.overhead, status=r["status"], bstat=BSTAT[r["status"]], mused=r["mused"],
                         taus=taus, peaks=peaks, coh=cohs, u=r["u"], cands=r["cands"], rank=r["rank"],
                         support=r["support"], dropped=r["dropped"], weights=r["weights"])
        return self.last


def haptic_weights(theta, overhead=False, motors=C.MOTOR_ANGLES):
    """Pan between the two motors bracketing theta (8 motors 45 deg apart; at most 2 non-zero). Returns {name: weight}
    in motors order, weights sum to 1. overhead=True, or theta None (detected, no usable azimuth): every motor at 1.0
    (the firmware plays its rotating ring)."""
    if overhead or theta is None:
        return {n: 1.0 for n in motors}
    names = list(motors)
    angs = np.array([motors[n] for n in names])
    w = {n: 0.0 for n in names}
    # walk the motors in angle order, find the bracketing pair on the circle
    order = [int(i) for i in np.argsort(angs, kind="stable")]
    for k in range(len(order)):
        i1, i2 = order[k], order[(k + 1) % len(order)]
        a1, a2 = angs[i1], angs[i2]
        span = (a2 - a1) % 360.0
        d = (theta - a1) % 360.0
        if d <= span + 1e-9:
            t = d / span if span > 0 else 0.0
            w[names[i1]] = 1.0 - t
            w[names[i2]] = t
            return w
    return w
