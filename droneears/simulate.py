"""Synthetic signals for testing and augmentation: a physically-motivated drone model and multi-mic (C.MIC_POS) far-field arrival simulation."""
import numpy as np
from . import config as C

# synth_drone defaults (round 2): slower harmonic decay and a broadband airflow floor so the synthetic drone's
# spectral flatness and harmonic count overlap the real recordings in data/drone
SYN_N_HARM = 20
SYN_HARM_DECAY = 0.8
SYN_AERO_NOISE = 0.7
SYN_BLADE_AM = 0.3


def synth_drone(duration, fs=C.FS, f0=220.0, n_motors=4, rpm_spread=0.03, wander_hz=0.3, wander_depth=0.04,
                n_harm=SYN_N_HARM, harm_decay=SYN_HARM_DECAY, motor_noise=0.15, seed=0,
                aero_noise=SYN_AERO_NOISE, aero_hp_hz=100.0, blade_am=SYN_BLADE_AM, n_blades=2):
    """A quadcopter-like harmonic comb: n_motors combs with slightly different fundamentals that wander slowly,
    decaying harmonics, band-limited motor noise, and a broadband rotor/airflow floor. Output is unit-RMS float32.

    f0 is the blade-pass frequency. aero_noise: RMS of pink (-3 dB/octave, high-passed at aero_hp_hz) airflow noise
    relative to the harmonic comb; real recordings have a spectral flatness of ~0.1-0.35 in DET_BAND, a pure comb
    ~0.02. blade_am: depth of per-motor amplitude modulation at the shaft rate f0 / n_blades (blade-to-blade
    asymmetry), which puts weak sidebands between the harmonics. aero_noise=0, blade_am=0, n_harm=12,
    harm_decay=0.55 reproduce the original (pre-round-2) model sample for sample."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    y = np.zeros(n)
    for m in range(n_motors):
        f = f0 * (1 + rng.uniform(-rpm_spread, rpm_spread))
        # slow random wander of the fundamental
        ph = rng.uniform(0, 2 * np.pi)
        wander = 1 + wander_depth * np.sin(2 * np.pi * wander_hz * t + ph) * rng.uniform(0.5, 1.0)
        inst_f = f * wander
        phase = 2 * np.pi * np.cumsum(inst_f) / fs
        comb = np.zeros(n)
        for k in range(1, n_harm + 1):
            amp = harm_decay ** (k - 1) * rng.uniform(0.7, 1.0)
            comb += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        if blade_am > 0:
            comb *= 1 + blade_am * np.sin(phase / n_blades + rng.uniform(0, 2 * np.pi))
        y += comb
    # broadband motor/aero noise shaped toward 1-4 kHz
    noise = rng.standard_normal(n)
    from scipy.signal import butter, lfilter
    b, a = butter(2, [800 / (fs / 2), 5000 / (fs / 2)], btype="band")
    noise = lfilter(b, a, noise)
    y = y / (np.std(y) + 1e-9) + motor_noise * noise / (np.std(noise) + 1e-9)
    if aero_noise > 0:                                   # pink airflow floor (1/f power), high-passed
        W = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / fs)
        W *= np.where(fr >= aero_hp_hz, 1.0 / np.sqrt(np.maximum(fr, aero_hp_hz)), 0.0)
        a = np.fft.irfft(W, n=n)
        y = y / (np.std(y) + 1e-9) + aero_noise * a / (np.std(a) + 1e-9)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def fractional_delay(x, delay_s, fs=C.FS):
    """Delay x by a fractional number of samples using an FFT phase shift (zero-padded to avoid wrap)."""
    n = len(x)
    pad = int(fs * 0.05) + 64
    X = np.fft.rfft(np.concatenate([x, np.zeros(pad)]))
    f = np.fft.rfftfreq(n + pad, 1 / fs)
    y = np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_s), n=n + pad)
    return y[:n].astype(np.float32)


def unit_vector(theta_deg, elev_deg=0.0):
    """Unit vector toward a source at bearing theta (0 = ahead, +90 = right) and elevation elev (+90 = straight up)."""
    th, el = np.radians(theta_deg), np.radians(elev_deg)
    return np.array([np.sin(th) * np.cos(el), np.cos(th) * np.cos(el), np.sin(el)])


def arrivals(x, theta_deg, elev_deg=0.0, mic_pos=C.MIC_POS, fs=C.FS, c=C.C_SOUND):
    """Far-field (plane-wave) arrival of mono source x at each mic for a source at bearing theta, elevation elev.
    Returns (n_mics, n) array. Mics closer to the source receive the signal earlier."""
    u = unit_vector(theta_deg, elev_deg)[: mic_pos.shape[1]]   # unit vector toward the source
    out = []
    for p in mic_pos:
        delay = -(p @ u) / c                         # closer mic -> negative delay (earlier)
        out.append(fractional_delay(x, delay, fs))
    return np.stack(out)


def arrivals_moving(x, theta_fn, elev_fn=None, mic_pos=C.MIC_POS, fs=C.FS, c=C.C_SOUND, block=C.HOP * 4):
    """Piecewise-constant direction: theta_fn(t_seconds) -> degrees, elev_fn(t) -> degrees (None: 0). Neighbouring
    blocks overlap by 2*edge samples and are linearly cross-faded over exactly that overlap, so the weights sum to 1
    everywhere (no clicks, no gain ripple)."""
    n = len(x)
    out = np.zeros((len(mic_pos), n), np.float32)
    edge = block // 4
    for s in range(0, n, block):
        a, b = max(0, s - edge), min(n, s + block + edge)
        tc = (s + block / 2) / fs
        seg = arrivals(x[a:b], theta_fn(tc), 0.0 if elev_fn is None else elev_fn(tc), mic_pos, fs, c)
        w = np.ones(b - a, np.float32)
        if s > 0:                                   # overlap with the previous block: [s - edge, s + edge)
            r = min(b, s + edge) - a
            w[:r] = np.linspace(0, 1, r, endpoint=False, dtype=np.float32)
        if s + block < n:                           # overlap with the next block: [s + block - edge, b)
            r = b - (s + block - edge)
            w[-r:] = np.linspace(1, 0, r, endpoint=False, dtype=np.float32)
        out[:, a:b] += seg * w[None, :]
    return out


def add_noise(y, snr_db, rng=None, per_channel=True):
    """Add independent white noise per channel at a given SNR relative to channel RMS."""
    rng = rng or np.random.default_rng(0)
    sig = np.sqrt(np.mean(y ** 2, axis=-1, keepdims=True))
    noise = rng.standard_normal(y.shape).astype(np.float32)
    return y + noise * sig / (10 ** (snr_db / 20))


def tilt(y, db_per_oct, pivot_hz=1000.0, fs=C.FS):
    """Zero-phase spectral tilt, i.e. a different microphone / mounting response: gain in dB = db_per_oct *
    log2(f / pivot_hz), held flat below 50 Hz and above 12 kHz. Works on (n,) or (channels, n)."""
    y = np.asarray(y, np.float32)
    n = y.shape[-1]
    f = np.clip(np.fft.rfftfreq(n, 1 / fs), 50.0, 12000.0)
    g = 10 ** (db_per_oct * np.log2(f / pivot_hz) / 20)
    return np.fft.irfft(np.fft.rfft(y, axis=-1) * g, n=n, axis=-1).astype(np.float32)
