"""Run the full pipeline over an audio file and print a timeline. Mono files get detection (+ classifier);
4-channel files (B, L, R, T, as config.MIC_NAMES) also get bearing and elevation.

Usage: python scripts/run_file.py path/to/audio.(wav|m4a|mp3) [--no-ml] [--plot out.png]
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from droneears import config as C
from droneears.audio import load
from droneears.pipeline import Pipeline, summarise
from droneears.classifier import load_tflite_classifier

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--no-ml", action="store_true"); ap.add_argument("--plot")
ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "drone_cls_int8.tflite"))
args = ap.parse_args()

y = load(args.path, mono=False)
if y.shape[0] not in (1, C.N_MICS):
    print(f"{y.shape[0]} channels: using channel 0 only (need 1 or {C.N_MICS})"); y = y[:1]
multi = y.shape[0] == C.N_MICS
clf = None if args.no_ml or not os.path.exists(args.model) else load_tflite_classifier(args.model)
pipe = Pipeline(classifier=clf)
res = pipe.run(y)
s = summarise(res)
print(f"{os.path.basename(args.path)}: {y.shape[1] / C.FS:.1f} s, {y.shape[0]} ch, classifier {'on' if clf else 'off'}")
print(f"detected {s['detected_frac']:.0%} of frames; first detection {s['first_detect_s']} s")
if multi and s["theta_median_when_detected"] is not None:
    print(f"median while detected: bearing {s['theta_median_when_detected']:.1f} deg, elevation {s['elev_median_when_detected']:.1f} deg")
# timeline, one line per ~0.5 s
step = int(C.FRAMES_PER_SEC / 2)
print("  t(s)   f0(Hz) score  flat  harm  hits  ml    state   " + ("theta  elev  conf  ovh" if multi else ""))
for i in range(0, len(res), step):
    r = res[i]
    line = f"{i * C.HOP / C.FS:6.1f}  {r['f0']:6.0f} {r['score']:5.1f}  {r['flat']:4.2f}  {r['n_harm']:3d}   {r['hits']:3d}  {r['ml']:.2f}  {'DRONE' if r['detected'] else '  -  ':7s}"
    if "theta" in r:                                 # theta / elev are None when the mics give none this frame
        fmt = lambda v, w: f"{v:{w}.1f}" if v is not None else f"{'-':>{w}}"
        line += f" {fmt(r['theta'], 6)} {fmt(r['elev'], 5)}  {r['conf']:.2f}  {'OVH' if r['overhead'] else ' - '}"
    print(line)
if args.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(len(res)) * C.HOP / C.FS
    fig, ax = plt.subplots(3 if multi else 1, 1, figsize=(10, 5), sharex=True, squeeze=False)
    ax[0, 0].plot(t, [r["score"] for r in res], label="harmonic score"); ax[0, 0].plot(t, [r["ml"] * 20 for r in res], label="ml x20")
    ax[0, 0].fill_between(t, 0, 30, where=[r["detected"] for r in res], alpha=0.2, label="DETECTED"); ax[0, 0].legend()
    if multi:
        ax[1, 0].plot(t, [r["theta"] for r in res]); ax[1, 0].set_ylabel("bearing")
        ax[2, 0].plot(t, [r["elev"] for r in res]); ax[2, 0].set_ylabel("elevation")
    plt.savefig(args.plot, dpi=120); print("wrote", args.plot)
