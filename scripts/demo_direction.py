"""Direction-finding demo: a drone circles the head, rising to 75 deg elevation mid-scene (so the overhead state
shows) and back down; the 4-mic pipeline tracks azimuth and elevation. Writes artifacts/direction_demo.png.

Usage: python scripts/demo_direction.py [--drone path/to/drone.wav] [--snr 6] [--bg path/to/ambient.wav]
With no --drone, a synthetic quadcopter is used. With --bg, that recording is mixed in as ambient (else white noise).
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from droneears import config as C
from droneears.simulate import synth_drone, arrivals_moving, add_noise
from droneears.pipeline import Pipeline
from droneears.audio import load, rms_normalise
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--drone"); ap.add_argument("--bg"); ap.add_argument("--snr", type=float, default=6.0)
ap.add_argument("--seconds", type=float, default=16.0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "direction_demo.png"))
args = ap.parse_args()

dur = args.seconds
x = rms_normalise(load(args.drone)) if args.drone else synth_drone(dur, f0=240.0, seed=3)
if len(x) < dur * C.FS:
    x = np.tile(x, int(np.ceil(dur * C.FS / len(x))))
x = x[: int(dur * C.FS)]

# bearing path: starts ahead, sweeps clockwise through right, behind, left and back ahead
theta_fn = lambda t: ((360.0 * t / dur) + 180.0) % 360.0 - 180.0
# elevation: 10 deg, rising over the middle of the scene to 75 deg (held for ~20 % of it), then back down
def elev_fn(t):
    f = t / dur
    ramp = np.clip((0.25 - abs(f - 0.5)) / 0.15, 0.0, 1.0)   # 0 outside [0.25, 0.75], 1 over [0.4, 0.6]
    return 10.0 + 65.0 * float(ramp)


y = arrivals_moving(x, theta_fn, elev_fn)
if args.bg:
    bg = rms_normalise(load(args.bg))
    rng = np.random.default_rng(0)
    for ch in range(y.shape[0]):
        s0 = rng.integers(0, max(1, len(bg) - y.shape[1]))
        seg = bg[s0:s0 + y.shape[1]]
        if len(seg) < y.shape[1]:
            seg = np.tile(seg, int(np.ceil(y.shape[1] / len(seg))))[: y.shape[1]]
        y[ch] += seg * (np.sqrt(np.mean(y[ch] ** 2)) / 10 ** (args.snr / 20)) / (np.sqrt(np.mean(seg ** 2)) + 1e-9)
else:
    y = add_noise(y, args.snr, rng=np.random.default_rng(1))

res = Pipeline().run(y)
t = np.arange(len(res)) * C.HOP / C.FS + C.NFFT / C.FS / 2
true = np.array([theta_fn(tt) for tt in t]); true_el = np.array([elev_fn(tt) for tt in t])
est = np.array([r["theta"] for r in res], float); raw = np.array([r["theta_raw"] for r in res], float)   # None -> nan
el = np.array([r["elev"] for r in res], float); ov = np.array([r["overhead"] for r in res])
det = np.array([r["detected"] for r in res]); conf = np.array([r["conf"] for r in res])
err = ((est - true + 180) % 360) - 180
el_err = el - true_el
side = det & (true_el < C.OVERHEAD_ELEV_OFF)      # azimuth is only meaningful away from the zenith
if det.any():
    print(f"frames {len(res)}, detected {det.mean():.0%}, first detection at {t[np.argmax(det)]:.2f} s")
    print(f"bearing error while detected: median {np.nanmedian(np.abs(err[det])):.1f} deg, 90th pct {np.nanpercentile(np.abs(err[det]), 90):.1f} deg"
          f"  (true elev < {C.OVERHEAD_ELEV_OFF:.0f} deg: median {np.nanmedian(np.abs(err[side])) if side.any() else float('nan'):.1f} deg)")
    print(f"elevation error (per frame) while detected: median {np.nanmedian(np.abs(el_err[det])):.1f} deg, 90th pct {np.nanpercentile(np.abs(el_err[det]), 90):.1f} deg")
    hi = det & (true_el > C.OVERHEAD_ELEV_ON)
    print(f"overhead: {ov[hi].mean() if hi.any() else 0:.0%} of detected frames with true elev > {C.OVERHEAD_ELEV_ON:.0f} deg, "
          f"{ov[det & (true_el < C.OVERHEAD_ELEV_OFF)].mean() if side.any() else 0:.0%} of those below {C.OVERHEAD_ELEV_OFF:.0f} deg")
else:
    print(f"frames {len(res)}, detected 0%: no detection, so no bearing error to report")

fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
ax[0].plot(t, true, "k--", lw=1, label="true bearing")
ax[0].plot(t, raw, ".", ms=2, color="#b8541a", alpha=0.4, label="per-frame estimate")
ax[0].plot(t[det], est[det], "-", color="#0f6b6e", lw=2, label="smoothed, while DETECTED")
ax[0].set_ylabel("bearing (deg)"); ax[0].set_ylim(-190, 190); ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("Drone circling the head: 4-mic masked GCC-PHAT + least-squares 3D direction")
ax[3].plot(t, true_el, "k--", lw=1, label="true elevation")
ax[3].plot(t, el, ".", ms=2, color="#b8541a", alpha=0.5, label="per-frame elevation")
ax[3].fill_between(t, -10, 95, where=ov, color="#f3dccb", label="OVERHEAD")
ax[3].axhline(C.OVERHEAD_ELEV_ON, color="k", ls=":", lw=1); ax[3].axhline(C.OVERHEAD_ELEV_OFF, color="k", ls=":", lw=1)
ax[3].set_ylabel("elevation (deg)"); ax[3].set_ylim(-10, 95); ax[3].legend(loc="upper right", fontsize=8)
ax[1].plot(t, [r["score"] for r in res], color="#0f6b6e", label="harmonic score (dB)")
ax[1].axhline(C.HARM_THRESH_DB, color="k", ls=":", lw=1)
ax[1].fill_between(t, 0, 30, where=det, color="#dbeceb", label="DETECTED")
ax[1].set_ylabel("score"); ax[1].legend(loc="upper right", fontsize=8)
ax[2].plot(t, conf, color="#b8541a"); ax[2].axhline(C.BEARING_MIN_CONF, color="k", ls=":", lw=1)
ax[2].set_ylabel("|u| consistency"); ax[2].set_ylim(0, 1.2); ax[3].set_xlabel("time (s)")
plt.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True); plt.savefig(args.out, dpi=130)
print("wrote", args.out)
