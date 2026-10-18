"""Build the C++ core, run it and the Python reference on the same N_MICS-channel (or mono) WAV, and compare frame
by frame.

Usage: python scripts/compare_core.py [--wav file.wav] [--theta 60] [--elev 20] [--drone clip.m4a] [--active 1110]
                                      [--front] [--weak-ch 1 --weak-db 30] [--noisy-ch 2 --noisy-db 6] [--seed 4]
  default: synthesises a drone at theta 60 deg, elev 20 deg with noise; --drone: a real (mono) recording placed at
  (--theta, --elev) through simulate.arrivals instead of the synthetic drone. --active (one 0/1 per channel B, L, R,
  T) runs both sides on that partial headset (bearing.mic_layout); the synthetic input then has those mics only
  (zeros elsewhere). --front: channel 0 is a front mic at FRONT_POS instead of the back one. --weak-ch / --weak-db: that mic hears the drone this many dB quieter over the same noise floor
  (a dead mic: a large --weak-db); --noisy-ch / --noisy-db: that mic gets extra independent noise this many dB above
  the drone. Both sides must agree on the detector, the bearing values and, per frame, the robust solver's bstat and
  mused (the mics in the fit).
"""
import argparse, atexit, os, shutil, subprocess, sys, io, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from droneears import config as C
from droneears.simulate import synth_drone, arrivals, add_noise
from droneears.audio import save_wav, load
from droneears.pipeline import Pipeline
from droneears.bearing import mic_layout, STATUS

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORE = os.path.join(ROOT, "firmware", "core")
ap = argparse.ArgumentParser(); ap.add_argument("--wav"); ap.add_argument("--drone")
ap.add_argument("--theta", type=float, default=60.0); ap.add_argument("--elev", type=float, default=20.0)
ap.add_argument("--snr", type=float, default=6.0); ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--active")
ap.add_argument("--front", action="store_true", help="channel 0 is a front mic (at FRONT_POS) instead of the back one")
ap.add_argument("--weak-ch", type=int); ap.add_argument("--weak-db", type=float, default=30.0)
ap.add_argument("--noisy-ch", type=int); ap.add_argument("--noisy-db", type=float, default=6.0)
args = ap.parse_args()
layout, core_args = None, []
if args.active or args.front:
    active = args.active or "1" * C.N_MICS
    if len(active) != C.N_MICS or set(active) - {"0", "1"}:
        ap.error(f"--active wants one 0/1 per mic ({''.join(C.MIC_NAMES)})")
    layout = mic_layout(np.array([ch == "1" for ch in active]), C.channel_positions(args.front))
    core_args = [active] + (["F"] if args.front else [])
    print(f"layout: active {active}{' (channel 0 at the front)' if args.front else ''}  rank {layout['rank']} "
          f"level {layout['level']} azimuth_ok {layout['azimuth_ok']} elevation_ok {layout['elevation_ok']}")

tmp = tempfile.mkdtemp(prefix="compare_core_")       # private build + input: concurrent runs cannot clobber each other
atexit.register(shutil.rmtree, tmp, True)
exe = os.path.join(tmp, "test_core")
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_c_headers.py")], check=True)
subprocess.run(["g++", "-O2", "-std=c++11", "-o", exe,
                os.path.join(CORE, "test_core.cpp"), os.path.join(CORE, "dsp_core.cpp")], check=True)

if args.wav:
    y = load(args.wav, mono=False)
    y = y[:C.N_MICS] if y.shape[0] >= C.N_MICS else y[:1]     # as test_core: N_MICS channels, else channel 0
else:
    if args.drone:
        x = load(args.drone)[: int(args.seconds * C.FS)]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    else:
        x = synth_drone(args.seconds, f0=215.0, seed=args.seed)
    clean = arrivals(x, args.theta, args.elev, mic_pos=C.channel_positions(args.front))
    rng = np.random.default_rng(2)
    noise = add_noise(clean, args.snr, rng=rng) - clean           # per-channel floor at the drone's level - snr
    if args.weak_ch is not None:
        clean[args.weak_ch] *= 10 ** (-args.weak_db / 20)
    y = clean + noise
    if args.noisy_ch is not None:
        y[args.noisy_ch] += rng.standard_normal(y.shape[1]) * np.sqrt(np.mean(x ** 2)) * 10 ** (args.noisy_db / 20)
    y = np.clip(y * 0.1, -1, 1)
    if layout is not None:
        y[~layout["active"]] = 0.0                    # unconnected channels read zeros, as on the headset
    print(f"input: theta {args.theta} elev {args.elev} snr {args.snr} dB"
          + (f", {C.MIC_NAMES[args.weak_ch]} {args.weak_db:g} dB weaker" if args.weak_ch is not None else "")
          + (f", {C.MIC_NAMES[args.noisy_ch]} + noise {args.noisy_db:g} dB" if args.noisy_ch is not None else ""))
wav = os.path.join(tmp, "compare_input.wav")
save_wav(wav, y)
y = load(wav, mono=False)            # read back the 16-bit version so both sides see identical samples

run = subprocess.run([exe, wav] + core_args, capture_output=True, text=True, check=True)
c = np.genfromtxt(io.StringIO(run.stdout), delimiter=",", names=True, dtype=None, encoding="utf-8")
res = Pipeline(layout=layout).run(y)
n = min(len(c), len(res))
nan = float("nan")
g = lambda k, d=None: np.array([nan if r.get(k, d) is None else r.get(k, d) for r in res], float)   # None -> NaN
py = dict(f0=g("f0"), score=g("score"), flat=g("flat"), hits=g("hits"), flat_local=g("flat_local"), gate=g("gate"),
          detected=g("detected"), theta_raw=g("theta_raw"), theta=g("theta"), elev=g("elev"), conf=g("conf", 0),
          overhead=g("overhead", 0), bstat=g("bstat", 0), mused=g("mused", 0))   # mono input: no bearing keys
print(f"frames: C {len(c)}  Python {len(res)}")
ok = True
for k in ["f0", "score", "flat", "flat_local", "gate", "hits", "detected", "conf", "theta_raw", "theta", "elev",
          "overhead", "bstat", "mused"]:
    a, b = np.asarray(c[k][:n], float), py[k][:n]
    known = ~np.isnan(a) & ~np.isnan(b)
    flag = np.isnan(a) != np.isnan(b)                 # one side knows the value, the other does not
    if k.startswith("theta"):
        d = np.abs(((a - b + 180) % 360) - 180)
    else:
        d = np.abs(a - b)
    d = np.where(known, d, 0.0)
    tol = dict(f0=2.01, score=0.05, flat=1e-3, flat_local=1e-3, gate=0.05, hits=0.5, detected=0.5, conf=0.02,
               theta_raw=1.5, theta=1.5, elev=1.5, overhead=0.5, bstat=0.5, mused=0.5)[k]
    frac = float(((d > tol) | flag).mean())
    extra = f"   known on one side only {flag.mean():6.1%}" if k in ("theta_raw", "theta", "elev") else ""
    print(f"  {k:10s} max|diff| {d.max():9.4f}   frames over tol {frac:6.1%}   (tol {tol}){extra}")
    if k in ("hits", "detected", "overhead", "bstat", "mused") and frac > 0.02: ok = False
    if k in ("f0", "score", "flat_local", "gate", "conf") and frac > 0.05: ok = False
    if k in ("theta_raw", "theta", "elev") and flag.mean() > 0.02: ok = False
    det = py["detected"][:n] > 0
    if k in ("theta", "elev") and (det & known).any() and np.median(d[det & known]) > 1.5: ok = False
det = py["detected"][:n] > 0
cs = [str(s) for s in c["status"][:n]]
pys = [r.get("status", "unreliable") for r in res[:n]]
for name, st, mu in (("C", cs, np.asarray(c["mused"][:n], int)), ("Python", pys, py["mused"][:n].astype(int))):
    sd = {s: sum(1 for i, x in enumerate(st) if x == s and det[i]) for s in STATUS}
    md = {}
    for i in np.flatnonzero(det):
        key = "".join(m for b_, m in enumerate(C.MIC_NAMES) if mu[i] >> b_ & 1) or "-"
        md[key] = md.get(key, 0) + 1
    print(f"  {name:6s} while detected: status {', '.join(f'{s} {v}' for s, v in sd.items() if v)};  mused "
          + ", ".join(f"{k_} {v}" for k_, v in sorted(md.items(), key=lambda t: -t[1])))
if det.any():
    med = lambda v: f"{np.nanmedian(v):.1f}" if np.isfinite(v).any() else "-"
    ct, pt = np.asarray(c["theta"][:n], float)[det], py["theta"][:n][det]
    ce, pe = np.asarray(c["elev"][:n], float)[det], py["elev"][:n][det]
    print(f"  C theta while detected: median {med(ct)} deg ({np.isfinite(ct).mean():.0%} known);  Python: {med(pt)} deg")
    print(f"  C elev  while detected: median {med(ce)} deg ({np.isfinite(ce).mean():.0%} known);  Python: {med(pe)} deg")
    print(f"  overhead frames while detected: C {c['overhead'][:n][det].mean():.0%}  Python {py['overhead'][:n][det].mean():.0%}")
bus = np.asarray(c["bear_us"][:n], float)
print(f"  C bearing on this host: {bus.mean():.1f} us/frame mean, {np.percentile(bus, 99):.1f} p99")
print("CORE MATCHES PYTHON" if ok else "CORE MISMATCH")
sys.exit(0 if ok else 1)
