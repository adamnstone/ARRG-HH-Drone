"""Live detector on the Mac's microphone: harmonic path + classifier, printed once per frame-ish.
Direction and elevation need four synchronous channels, so they are off unless you pass --device with a
>=4-channel interface (channel order B, L, R, T).

    pip install sounddevice
    python scripts/live_mac.py              # default input device, mono
    python scripts/live_mac.py --list       # show devices
    python scripts/live_mac.py --device 3   # a USB interface with 4+ inputs -> bearing and elevation too
"""
import argparse, os, sys, time, queue, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sounddevice as sd
from droneears import config as C
from droneears.pipeline import Pipeline
from droneears.classifier import load_tflite_classifier

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int); ap.add_argument("--list", action="store_true"); ap.add_argument("--no-ml", action="store_true")
ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "artifacts", "drone_cls_int8.tflite"))
args = ap.parse_args()
if args.list:
    print(sd.query_devices()); sys.exit()

info = sd.query_devices(args.device, "input")
nch = C.N_MICS if info["max_input_channels"] >= C.N_MICS else 1
clf = None if args.no_ml or not os.path.exists(args.model) else load_tflite_classifier(args.model)
pipe = Pipeline(classifier=clf)
q = queue.Queue()
ring = np.zeros((nch, C.NFFT), np.float32)
pos = 0

def cb(indata, frames, t, status):
    q.put(indata.copy())

print(f"listening on '{info['name']}' at {C.FS} Hz, {nch} ch, classifier {'on' if clf else 'off'}. Ctrl-C to stop.")
with sd.InputStream(device=args.device, channels=nch, samplerate=C.FS, blocksize=C.HOP, dtype="float32", callback=cb):
    buf = np.zeros((nch, 0), np.float32); last_print = 0
    while True:
        block = q.get().T                                   # (nch, HOP)
        buf = np.concatenate([buf, block], axis=1)
        while buf.shape[1] >= C.NFFT:
            frame = buf[:, :C.NFFT]; buf = buf[:, C.HOP:]
            X = np.fft.rfft(frame * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(C.NFFT) / C.NFFT)), axis=1)
            r = pipe.step(X)
            now = time.time()
            if now - last_print > 0.25:
                last_print = now
                bar = "#" * int(min(30, max(0, r["score"])))
                line = f"\r f0 {r['f0']:4.0f} Hz  score {r['score']:5.1f} {bar:30s} ml {r['ml']:.2f}  {'DRONE' if r['detected'] else '  -  '}"
                if "theta" in r:                     # None when the mics give no bearing / elevation this frame
                    fmt = lambda v, w: f"{v:{w}.1f}" if v is not None else f"{'-':>{w}}"
                    line += f"  bearing {fmt(r['theta'], 6)}  elevation {fmt(r['elev'], 5)} ({r['conf']:.2f})"
                print(line, end="", flush=True)
