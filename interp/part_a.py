"""Part A: synthetic probes. What does the classifier call a drone? Built signals, one property varied at a time,
mixed into the phone ambient recording; p(drone) from the int8 classifier on single exact windows, and the rules'
response (fraction of frames passing the per-frame tests, and whether DETECTED is reached) on 4 s clips of the
same sound. Synthetic sounds can mislead (the model was partly trained on synth_drone), so Part B checks the
conclusions on real recordings.

The generator follows droneears.simulate.synth_drone (several motors with slightly different fundamentals that
wander, decaying harmonics, band-limited motor noise, pink airflow floor) but takes the list of partials explicitly
as multiples of f0, so harmonics can be dropped, restricted to one band, made inharmonic or replaced by random peaks.

Usage: python -m interp.part_a [--seeds 8] [--rule-seeds 3] [--workers 8] [--only f0,snr,...]
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from scipy.signal import butter, lfilter
from interp import harness as H
from droneears import config as C
from droneears.features import logmel, normalise_window
from droneears.detector import DroneDetector

WIN_N = (C.CLS_WINDOW_FRAMES - 1) * C.HOP + C.NFFT          # samples in exactly one classifier window (47 frames)
BASE = dict(f0=200.0, mults=None, n_harm=20, decay=0.8, amps=None, n_motors=4, rpm_spread=0.03, wander_depth=0.04,
            wander_hz=0.3, am_depth=0.0, am_hz=4.0, aero=0.7, motor_noise=0.15, inharm=0.0, snr=10.0, fill=1.0)


def gen(cond, seed, n):
    """Drone-like sound of n samples for condition dict cond (BASE overridden), unit RMS."""
    c = dict(BASE, **cond)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / C.FS
    mults = np.asarray(c["mults"] if c["mults"] is not None else np.arange(1, c["n_harm"] + 1), float)
    if c["inharm"] > 0:                                    # fixed per-partial mistuning, relative
        mults = mults * (1 + c["inharm"] * rng.standard_normal(len(mults)))
    amps = np.asarray(c["amps"] if c["amps"] is not None else c["decay"] ** (np.round(mults) - 1), float)
    y = np.zeros(n)
    for m in range(c["n_motors"]):
        f = c["f0"] * (1 + rng.uniform(-c["rpm_spread"], c["rpm_spread"]))
        wander = 1 + c["wander_depth"] * np.sin(2 * np.pi * c["wander_hz"] * t + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.5, 1.0)
        phase = 2 * np.pi * np.cumsum(f * wander) / C.FS
        for k, a in zip(mults, amps):
            if k * c["f0"] < C.FS / 2 - 500:
                y += a * rng.uniform(0.7, 1.0) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    y /= np.std(y) + 1e-9
    if c["motor_noise"] > 0:
        b, a = butter(2, [800 / (C.FS / 2), 5000 / (C.FS / 2)], btype="band")
        nz = lfilter(b, a, rng.standard_normal(n))
        y = y + c["motor_noise"] * nz / (np.std(nz) + 1e-9)
    if c["aero"] > 0:
        W = np.fft.rfft(rng.standard_normal(n)); fr = np.fft.rfftfreq(n, 1 / C.FS)
        W *= np.where(fr >= 100, 1 / np.sqrt(np.maximum(fr, 100)), 0)
        a = np.fft.irfft(W, n=n)
        y = y / (np.std(y) + 1e-9) + c["aero"] * a / (np.std(a) + 1e-9)
    if c["am_depth"] > 0:
        y = y * (1 + c["am_depth"] * np.sin(2 * np.pi * c["am_hz"] * t + rng.uniform(0, 2 * np.pi)))
    if c["fill"] < 1:                                      # sound present in one contiguous stretch of the clip
        L = int(c["fill"] * n); s = rng.integers(0, n - L + 1) if L < n else 0
        g = np.zeros(n); g[s:s + L] = 1
        r = min(480, L // 4) if L > 0 else 0               # 10 ms ramps
        if r > 0:
            g[s:s + r] = np.linspace(0, 1, r); g[s + L - r:s + L] = np.linspace(1, 0, r)
        y = y * g
    return y.astype(np.float32)


_AMB = None


def background(seed, n):
    global _AMB
    if _AMB is None:
        _AMB = H.load_cached(H.PHONE_AMBIENT)
    s = np.random.default_rng(seed + 7).integers(0, len(_AMB) - n)
    return _AMB[s:s + n]


def mixed(cond, seed, n):
    c = dict(BASE, **cond)
    x = gen(cond, seed, n)
    bg = background(seed, n)
    rb = np.sqrt(np.mean(bg ** 2)) + 1e-9
    if c["fill"] < 1:                                      # SNR refers to the sound while it is on
        on = np.abs(x) > 0
        rs = np.sqrt(np.mean(x[on] ** 2)) + 1e-9 if on.any() else 1.0
    else:
        rs = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x / rs * 10 ** (c["snr"] / 20) + bg / rb).astype(np.float32)


def eval_condition(args):
    name, value, cond, seeds, rule_seeds = args
    clf = H.classifier()
    ps = []
    for s in range(seeds):
        y = mixed(cond, 1000 + s, WIN_N)
        P = H.power_frames(y)
        assert len(P) == C.CLS_WINDOW_FRAMES
        ps.append(clf(normalise_window(logmel(P))))
    ok, det = [], []
    if dict(BASE, **cond)["fill"] >= 1:
        for s in range(rule_seeds):
            y = mixed(cond, 2000 + s, 4 * C.FS)
            d = DroneDetector(); hits = []
            for f in H.power_frames(y):
                r = d.feed(f); ok.append(r["frame_ok"]); hits.append(r["hits"])
            det.append(max(hits) >= C.PERSIST_ON)
    return dict(sweep=name, value=value, p_mean=float(np.mean(ps)), p_median=float(np.median(ps)),
                p_frac07=float(np.mean(np.array(ps) > C.ML_THRESH)),
                rules_frame_ok=float(np.mean(ok)) if ok else None, rules_detect=float(np.mean(det)) if det else None)


def sweeps():
    S = {}
    S["f0"] = [(v, dict(f0=v)) for v in [50, 70, 90, 110, 130, 150, 180, 210, 250, 300, 350, 400, 450, 500, 600, 700, 800]]
    S["n_harm"] = [(v, dict(n_harm=v)) for v in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]]
    # one octave band of harmonics only, flat amplitudes (f0 = 200 Hz)
    S["band_only"] = []
    for lo in [200, 400, 800, 1600, 3200, 6400]:
        m = [k for k in range(1, 80) if lo <= k * 200 < 2 * lo]
        S["band_only"].append((f"{lo}-{2 * lo}", dict(mults=m, amps=[1.0] * len(m))))
    S["drop_one"] = [(k, dict(mults=[j for j in range(1, 21) if j != k])) for k in range(1, 11)]
    S["structure"] = [("full 1-20", dict()),
                      ("no fundamental", dict(mults=list(range(2, 21)))),
                      ("no 1-2", dict(mults=list(range(3, 21)))),
                      ("no 1-3", dict(mults=list(range(4, 21)))),
                      ("odd only", dict(mults=list(range(1, 21, 2)))),
                      ("even only", dict(mults=list(range(2, 21, 2)))),
                      ("random peaks", dict(mults=list(np.random.default_rng(5).uniform(1, 20, 20)))),
                      ("no partials (noise only)", dict(mults=[], amps=[], n_harm=0))]
    S["inharm"] = [(v, dict(inharm=v / 100)) for v in [0, 0.5, 1, 2, 4, 8, 16]]
    S["snr"] = [(v, dict(snr=v)) for v in [-12, -9, -6, -3, 0, 3, 6, 9, 12, 18, 24]]
    S["aero"] = [(v, dict(aero=v)) for v in [0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
    S["wobble_depth"] = [(v, dict(wander_depth=v / 100)) for v in [0, 1, 2, 4, 8, 15, 25]]
    S["wobble_rate"] = [(v, dict(wander_hz=v, wander_depth=0.04)) for v in [0.1, 0.3, 1, 3, 10]]
    S["am_depth"] = [(v, dict(am_depth=v)) for v in [0, 0.25, 0.5, 0.75, 1.0]]
    S["fill"] = [(v, dict(fill=v)) for v in [0, 0.1, 0.25, 0.5, 0.75, 1.0]]
    S["n_motors"] = [(v, dict(n_motors=v)) for v in [1, 2, 4, 8]]
    S["rpm_spread"] = [(v, dict(rpm_spread=v / 100)) for v in [0, 1, 3, 6, 10]]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rule-seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", default=None)
    ap.add_argument("--rules", choices=["shipped", "current"], default="shipped",
                    help="rules column: the shipped detector (global flatness, no DSP gate; what interp/README.md "
                         "reports) or droneears/config.py as it stands")
    args = ap.parse_args()
    if args.rules == "shipped":
        C.FLAT_MODE, C.GATE_ON = "global", False
    S = sweeps()
    if args.only:
        S = {k: v for k, v in S.items() if k in args.only.split(",")}
    tasks = [(name, value, cond, args.seeds, args.rule_seeds) for name, lst in S.items() for value, cond in lst]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(eval_condition, tasks))
    out = {}
    for r in res:
        out.setdefault(r["sweep"], []).append(r)
    for name, lst in out.items():
        print(f"\n== {name}")
        for r in lst:
            rules = "" if r["rules_frame_ok"] is None else f"   rules: frame_ok {r['rules_frame_ok']:.2f} detect {r['rules_detect']:.2f}"
            print(f"  {str(r['value']):>24s}  p mean {r['p_mean']:.2f}  median {r['p_median']:.2f}  >0.7 {r['p_frac07']:.2f}{rules}")
    os.makedirs(H.OUT, exist_ok=True)
    json.dump(dict(base=BASE, seeds=args.seeds, rule_seeds=args.rule_seeds, sweeps=out),
              open(os.path.join(H.OUT, "part_a.json"), "w"), indent=1, default=str)
    print(f"\n{time.time() - t0:.0f} s; wrote", os.path.join(H.OUT, "part_a.json"))


if __name__ == "__main__":
    main()
