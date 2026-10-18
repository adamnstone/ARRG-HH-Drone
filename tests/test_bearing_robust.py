"""pytest: the robust per-frame bearing (droneears/bearing.py, Rachel's sig_proc_tri_robust.py ported to one DSP frame;
mirrored by firmware/core/dsp_core.cpp, which scripts/compare_core.py checks frame by frame).

Four reliable mics -> full_3d; three -> B, L, R azimuth (elevation too when the upper-hemisphere prior separates the
mirror pair), other threes a mirror pair the prior resolves or not; fewer -> unreliable. A dead, weak or noisy mic
drops out per frame (mused)."""
import functools, os, sys, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from droneears import config as C
from droneears import bearing as B
from droneears.simulate import synth_drone, arrivals, add_noise, unit_vector
from droneears.pipeline import Pipeline

BIT = {m: 1 << i for i, m in enumerate(C.MIC_NAMES)}          # B 1, L 2, R 4, T 8
ALL = 15


def mics(names):
    return sum(BIT[m] for m in names)


def wrap(a):
    return abs(((a + 180.0) % 360.0) - 180.0)


@functools.lru_cache(maxsize=None)
def run(theta, elev, active="1111", dead=None, weak=None, weak_db=0.0, noisy=None, noisy_db=0.0, snr=6.0, seed=4,
        front=False):
    """Pipeline over a synthetic drone at (theta, elev): per-channel noise snr dB below the drone; `active` = the
    connected mics (mic_layout); dead = a connected channel that reads zeros; weak = a channel hearing the drone
    weak_db quieter over the same noise; noisy = extra independent noise noisy_db above the drone on one channel;
    front = channel 0 is a front mic (at FRONT_POS) instead of the back one. Returns the detected frames."""
    x = synth_drone(4.0, f0=215.0, seed=seed)
    clean = arrivals(x, theta, elev, mic_pos=C.channel_positions(front))
    rng = np.random.default_rng(2)
    noise = add_noise(clean, snr, rng=rng) - clean
    if weak is not None:
        clean[weak] *= 10 ** (-weak_db / 20)
    y = clean + noise
    if noisy is not None:
        y[noisy] += rng.standard_normal(y.shape[1]) * np.sqrt(np.mean(x ** 2)) * 10 ** (noisy_db / 20)
    if dead is not None:
        y[dead] = 0.0
    layout = None
    if active != "1111" or front:
        layout = B.mic_layout(np.array([a == "1" for a in active]), C.channel_positions(front))
        y[~layout["active"]] = 0.0
    res = Pipeline(layout=layout).run(np.clip(0.1 * y, -1, 1).astype(np.float32))
    det = [r for r in res if r["detected"]]
    assert len(det) > 40, f"drone not detected ({len(det)} frames)"
    return det


def frac(det, cond):
    return float(np.mean([bool(cond(r)) for r in det]))


def err3d(r, theta, elev):
    """Angle (deg) between a frame's direction (theta_raw, elev) and the truth."""
    return float(np.degrees(np.arccos(np.clip(unit_vector(r["theta_raw"], r["elev"]) @ unit_vector(theta, elev), -1, 1))))


# ---------------------------------------------------------------- four mics
@pytest.mark.parametrize("theta,elev", [(0, 10), (60, 20), (150, 40), (-120, 0), (-45, 60)])
def test_four_mics_full_3d_direction(theta, elev):
    det = run(theta, elev)
    assert frac(det, lambda r: r["status"] == "full_3d" and r["bstat"] == 3 and r["mused"] == ALL) > 0.9
    th = [r["theta"] for r in det[-30:]]
    assert all(t is not None for t in th)
    assert wrap(np.median(th) - theta) < 4.0
    el = [r["elev"] for r in det if r["status"] == "full_3d"]
    assert abs(np.median(el) - elev) < 4.0
    full = [r for r in det if r["status"] == "full_3d"]
    raw = [wrap(r["theta_raw"] - theta) for r in full if r["theta_raw"] is not None]
    assert np.median(raw) < 4.0


# ---------------------------------------------------------------- a front mic in channel 0 instead of the back one
def test_channel_positions():
    assert np.allclose(C.channel_positions(False), C.MIC_POS)
    f = C.channel_positions(True)
    assert np.allclose(f[0], C.FRONT_POS) and np.allclose(f[1:], C.MIC_POS[1:]) and f[0][1] > 0


@pytest.mark.parametrize("theta,elev", [(60, 20), (150, 40), (-120, 0)])
def test_front_mic_four_mics_full_3d(theta, elev):
    det = run(theta, elev, front=True)
    assert frac(det, lambda r: r["status"] == "full_3d" and r["mused"] == ALL) > 0.9
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - theta) < 4.0
    assert abs(np.median([r["elev"] for r in det if r["status"] == "full_3d"]) - elev) < 4.0


def test_front_mic_level_plane_gives_azimuth():
    # F, L, R is the mirror image of B, L, R: a level plane, the azimuth always known
    det = run(60, 20, active="1110", front=True)
    assert frac(det, lambda r: r["mused"] == mics("BLR") and r["bstat"] == 2) > 0.95
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - 60) < 4.0


def test_front_mic_missing_never_guesses_front_or_back():
    det = run(30, 20, active="0111", front=True)
    assert all(r["theta_raw"] is None for r in det) and frac(det, lambda r: r["bstat"] == 1) > 0.9


# ---------------------------------------------------------------- one mic dropped: not connected, dead, or attenuated
LEVEL_T_GONE = [dict(active="1110"), dict(dead=3), dict(weak=3, weak_db=40.0)]
B_GONE = [dict(active="0111"), dict(dead=0), dict(weak=0, weak_db=40.0)]


@pytest.mark.parametrize("how", LEVEL_T_GONE, ids=["T-not-connected", "T-dead", "T-40dB"])
def test_blr_level_plane_gives_azimuth_only(how):
    # B, L, R: the mirror pair across the head plane shares the azimuth. A drone 20 deg up is too close to the plane
    # for the upper-hemisphere prior to separate the pair within the delay tolerance: azimuth, no elevation.
    det = run(60, 20, **how)
    assert frac(det, lambda r: r["mused"] == mics("BLR")) > 0.95
    assert frac(det, lambda r: r["status"] == "azimuth_only") > 0.8
    for r in det:
        assert r["status"] in ("azimuth_only", "constrained_3d") and r["bstat"] == 2
        assert r["theta_raw"] is not None                      # B, L, R: the azimuth is always known
        if r["status"] == "azimuth_only":
            assert r["elev"] is None
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - 60) < 4.0
    assert all(r["theta"] is not None for r in det)


def test_blr_upper_prior_resolves_a_clearly_raised_drone():
    det = run(60, 50, active="1110")
    assert frac(det, lambda r: r["status"] == "constrained_3d" and r["bstat"] == 2 and r["mused"] == mics("BLR")) > 0.9
    el = [r["elev"] for r in det if r["status"] == "constrained_3d"]
    assert abs(np.median(el) - 50) < 5.0
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - 60) < 4.0


def test_blr_without_the_prior_never_reports_an_elevation(monkeypatch):
    monkeypatch.setattr(C, "BEARING_UPPER_PRIOR", False)
    run.cache_clear()
    try:
        det = run(60, 50, active="1110")
        assert all(r["status"] == "azimuth_only" and r["elev"] is None for r in det)
        assert wrap(np.median([r["theta"] for r in det[-30:]]) - 60) < 4.0
    finally:
        run.cache_clear()


@pytest.mark.parametrize("how", B_GONE, ids=["B-not-connected", "B-dead", "B-40dB"])
def test_lrt_front_back_mirror_is_ambiguous(how):
    # L, R, T: the mirror flips front / back; the +z prior cannot separate it (its normal is horizontal). No
    # azimuth (60 deg and its mirror 120 deg are not shared), but both candidates share the elevation.
    det = run(60, 20, **how)
    assert frac(det, lambda r: r["mused"] == mics("LRT")) > 0.95
    for r in det:
        assert r["status"] == "ambiguous" and r["bstat"] == 1
        assert r["theta_raw"] is None and r["theta"] is None
        assert r["haptics"] == {k: 1.0 for k in C.MOTOR_ANGLES}   # detected, no azimuth: the (pulsed) ring
    el = [r["elev"] for r in det if r["elev"] is not None]
    assert len(el) > 0.9 * len(det) and abs(np.median(el) - 20) < 4.0


@pytest.mark.parametrize("theta,elev", [(90, 10), (90, 0), (30, 80)])
def test_lrt_never_guesses_front_or_back(theta, elev):
    # even a drone in the L-R-T plane (to the side) or near the zenith: within the delay-error bound its
    # front / back component could be ~0.4, so the mirror candidates do not share an azimuth; the elevation is known
    det = run(theta, elev, active="0111")
    for r in det:
        assert r["status"] == "ambiguous" and r["theta_raw"] is None and r["theta"] is None
    el = [r["elev"] for r in det if r["elev"] is not None]
    assert len(el) == len(det) and abs(np.median(el) - elev) < 4.0


def test_lrt_overhead_still_detected_from_the_shared_elevation():
    det = run(30, 80, active="0111")
    assert frac(det, lambda r: r["elev"] is not None and r["elev"] > C.OVERHEAD_ELEV_ON) > 0.8
    assert any(r["overhead"] for r in det)


@pytest.mark.parametrize("gone,keep", [(1, "BRT"), (2, "BLT")], ids=["L-dead", "R-dead"])
def test_tilted_plane_side_mics(gone, keep):
    # B, R, T or B, L, T: the mirror plane is tilted. 60 deg / 20 deg: the mirror candidate is above the head too
    # (R dead) or barely below it (L dead): ambiguous, or the prior picks the right side; never full_3d, never a
    # confidently wrong direction.
    det = run(60, 20, dead=gone)
    assert frac(det, lambda r: r["mused"] == mics(keep)) > 0.9
    for r in det:
        assert r["status"] in ("ambiguous", "constrained_3d", "unreliable")
        if r["status"] == "constrained_3d" and r["theta_raw"] is not None:
            assert err3d(r, 60, 20) < 15.0
        if r["theta_raw"] is not None:
            assert wrap(r["theta_raw"] - 60) < 15.0


@pytest.mark.parametrize("theta,gone,keep", [(-130, 2, "BLT"), (130, 1, "BRT")], ids=["R-dead", "L-dead"])
def test_tilted_plane_prior_resolves_a_drone_whose_mirror_is_below(theta, gone, keep):
    # the drone faces the plane of the three remaining mics from above; its mirror candidate is well below the head
    det = run(theta, 40, dead=gone)
    assert frac(det, lambda r: r["status"] == "constrained_3d" and r["mused"] == mics(keep)) > 0.9
    con = [r for r in det if r["status"] == "constrained_3d"]
    assert np.median([err3d(r, theta, 40) for r in con]) < 5.0
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - theta) < 5.0


def test_noisy_mic_dropped_by_the_leave_one_out_fit():
    # B carries independent noise 12 dB above the drone but stays coherent enough for the pair test: its three pairs
    # skew the four-mic fit, which then fails; leaving B out gives L, R, T.
    det = run(60, 20, noisy=0, noisy_db=12.0)
    assert any(r["mused"] == mics("LRT") for r in det)
    for r in det:
        assert r["mused"] in (ALL, mics("LRT"), 0)
        if r["status"] == "full_3d" and r["conf"] >= C.BEARING_MIN_CONF:     # what the smoothed theta takes
            assert wrap(r["theta_raw"] - 60) < 10.0
    assert wrap(np.median([r["theta"] for r in det[-30:]]) - 60) < 4.0


# ---------------------------------------------------------------- fewer than three mics
@pytest.mark.parametrize("active", ["1100", "0011", "0110"])
def test_fewer_than_three_mics_no_direction(active):
    det = run(60, 20, active=active)
    for r in det:
        assert r["status"] == "unreliable" and r["bstat"] == 0
        assert r["theta"] is None and r["theta_raw"] is None and r["elev"] is None
        assert r["conf"] == 0.0 and not r["overhead"]
        assert r["haptics"] == {k: 1.0 for k in C.MOTOR_ANGLES}


def test_two_dead_mics_no_direction():
    det = run(60, 20, dead=0, weak=3, weak_db=60.0)
    assert frac(det, lambda r: r["bstat"] == 0 and r["theta_raw"] is None and r["elev"] is None) > 0.95


# ---------------------------------------------------------------- building blocks
def test_integer_lag_gcc_equals_the_inverse_fft():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, C.NBINS)) + 1j * rng.standard_normal((2, C.NBINS))
    mask = np.zeros(C.NBINS, np.float32); mask[B.BRG_LO:B.BRG_HI + 1:3] = 1.0
    G = np.conj(X[0]) * X[1]
    ph = np.where(mask > 0, G / np.abs(G), 0)
    r = np.fft.irfft(ph, n=C.NFFT) * C.NFFT / 2
    L = 27
    bins = np.flatnonzero(mask)
    curve = B.lag_curve(ph[bins].real, ph[bins].imag, bins, L)
    assert np.allclose(curve, r[np.arange(-L, L + 1) % C.NFFT], atol=1e-9)


def test_pair_delay_recovers_a_fractional_delay():
    from droneears.features import frames
    from droneears.simulate import fractional_delay
    from droneears.detector import band_mask
    x = synth_drone(0.2, f0=215.0, seed=1)
    for d in [-20.3, -7.6, 0.0, 4.4, 18.9]:                     # samples; x_j = x_i delayed by d -> tau = +d
        Xa, Xb = np.fft.rfft(frames(x)[2]), np.fft.rfft(frames(fractional_delay(x, d / C.FS))[2])
        tau, peak, alias = B.gcc_phat(Xa, Xb, band_mask(), 25.2 / C.FS)
        assert not alias and abs(tau * C.FS - d) < 0.3 and peak > 0.8


def test_high_pure_tone_is_an_alias_not_a_delay():
    # a 2.5 kHz tone repeats every 19.2 samples, inside the +-25 sample L-R interval: competing peaks
    n = np.arange(C.NFFT)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / C.NFFT)
    a = np.sin(2 * np.pi * 2500 * n / C.FS) * w
    b = np.sin(2 * np.pi * 2500 * (n - 3.0) / C.FS) * w
    mask = np.zeros(C.NBINS, np.float32); mask[B.BRG_LO:B.BRG_HI + 1] = 1.0
    tau, peak, alias = B.gcc_phat(np.fft.rfft(a), np.fft.rfft(b), mask, B.TAU_MAX[3])
    assert alias and np.isnan(tau) and peak == 0.0


def test_mic_layout_flags():
    lay = lambda s: B.mic_layout(np.array([c == "1" for c in s]))
    assert lay("1111")["rank"] == 3 and lay("1111")["azimuth_ok"] and lay("1111")["elevation_ok"]
    blr = lay("1110")
    assert blr["rank"] == 2 and blr["level"] and blr["azimuth_ok"] and blr["elevation_ok"]
    lrt = lay("0111")
    assert lrt["rank"] == 2 and not lrt["level"] and not lrt["azimuth_ok"] and lrt["elevation_ok"]
    assert lay("1101")["azimuth_ok"] and lay("1011")["azimuth_ok"]
    for s in ["1100", "0011", "1000", "0000"]:
        assert not lay(s)["azimuth_ok"] and not lay(s)["elevation_ok"]


def test_status_codes_match_the_c_core():
    assert B.STATUS == ["unreliable", "ambiguous", "azimuth_only", "constrained_3d", "full_3d"]
    src = open(os.path.join(os.path.dirname(__file__), "..", "firmware", "core", "dsp_core.cpp")).read()
    assert '{"unreliable", "ambiguous", "azimuth_only", "constrained_3d", "full_3d"}' in src
    assert [B.BSTAT[s] for s in B.STATUS] == [0, 1, 2, 2, 3]


def test_robust_direction_from_exact_delays():
    # noiseless delays, unit quality: four mics -> the true vector; T removed -> B, L, R candidates mirrored in z
    for theta, elev in [(30, 10), (-100, 45), (170, 70)]:
        u = unit_vector(theta, elev)
        taus = B.A @ u
        r = B.robust_direction(taus, np.ones(6), np.ones(6), np.ones(6, bool))
        assert r["status"] == "full_3d" and r["mused"] == ALL and np.allclose(r["u"], u, atol=1e-6)
        on = np.array([True, True, False, True, False, False])         # pairs of B, L, R only
        r = B.robust_direction(np.where(on, taus, np.nan), np.ones(6), np.ones(6), on)
        assert r["mused"] == mics("BLR") and r["rank"] == 2
        assert wrap(r["theta_raw"] - theta) < 1e-3
        if r["status"] == "constrained_3d":
            assert np.allclose(r["u"], u, atol=1e-6)
        else:
            assert r["status"] == "azimuth_only" and r["elev"] is None
