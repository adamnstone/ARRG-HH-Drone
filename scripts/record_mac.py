"""Record training / holdout audio from the laptop mic (or any CoreAudio input) straight into data/team/.

Same naming convention as scripts/record_device.py, so train.py and retune_holdout.py pick the files up:
    data/team/drone/mac_<timestamp>[_holdout].wav         (--drone)
    data/team/background/mac_<timestamp>[_holdout].wav    (default)

Usage:  python scripts/record_mac.py --seconds 60 --drone [--holdout] [--name mavic_25m] [--check]
        python scripts/record_mac.py --list-devices
Mono, 48 kHz, 16-bit WAV. --check runs the fused detector over the recording afterwards and prints the timeline summary.
macOS will ask for microphone permission for the terminal app the first time.
"""
import argparse, os, sys, time, datetime, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from droneears import config as C

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--drone", action="store_true", help="save under data/team/drone/ instead of data/team/background/")
ap.add_argument("--holdout", action="store_true", help="append _holdout so the file goes entirely to the test split")
ap.add_argument("--name", default="", help="optional label in the file name, e.g. mavic_25m_behind_building")
ap.add_argument("--out", help="explicit output path (overrides the data/team convention)")
ap.add_argument("--device", help="input device index or name substring (see --list-devices)")
ap.add_argument("--list-devices", action="store_true")
ap.add_argument("--check", action="store_true", help="run the fused detector on the recording afterwards")
args = ap.parse_args()

import sounddevice as sd, soundfile as sf
if args.list_devices:
    print(sd.query_devices()); sys.exit(0)

dev = None
if args.device is not None:
    dev = int(args.device) if args.device.isdigit() else next(
        (i for i, d in enumerate(sd.query_devices()) if args.device.lower() in d["name"].lower() and d["max_input_channels"] > 0), None)
    if dev is None:
        sys.exit(f"no input device matching {args.device!r}; see --list-devices")

if args.out:
    out = args.out
else:
    sub = "drone" if args.drone else "background"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"mac_{stamp}" + (f"_{args.name}" if args.name else "") + ("_holdout" if args.holdout else "") + ".wav"
    out = os.path.join(os.path.dirname(__file__), "..", "data", "team", sub, name)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

n = int(args.seconds * C.FS)
print(f"recording {args.seconds:.0f} s mono at {C.FS} Hz from {sd.query_devices(dev, 'input')['name']} -> {os.path.relpath(out)}")
print("Ctrl-C stops early and keeps what was recorded.")
buf = np.zeros((n, 1), np.float32); got = 0; t0 = time.time()

def cb(indata, frames, t, status):
    global got
    if status: print("  ", status, file=sys.stderr)
    k = min(frames, n - got)
    buf[got:got + k] = indata[:k]; got += k
    if got >= n: raise sd.CallbackStop()

try:
    with sd.InputStream(samplerate=C.FS, channels=1, dtype="float32", device=dev, callback=cb):
        last = -1
        while got < n:
            sd.sleep(200)
            s = int(got / C.FS)
            if s != last and got:
                last = s; seg = buf[max(0, got - C.FS):got, 0]
                rms = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9); pk = 20 * np.log10(np.max(np.abs(seg)) + 1e-9)
                print(f"\r  {s:4d} s  rms {rms:6.1f} dBFS  peak {pk:6.1f} dBFS   ", end="", flush=True)
except KeyboardInterrupt:
    print("\nstopped early")
x = buf[:got, 0]
if got == 0:
    sys.exit("nothing recorded")
sf.write(out, x, C.FS, subtype="PCM_16")
pk = float(np.max(np.abs(x)))
print(f"\nwrote {os.path.relpath(out)}: {got / C.FS:.1f} s, peak {20 * np.log10(pk + 1e-9):.1f} dBFS"
      + ("  WARNING: clipping" if pk > 0.89 else ""))

if args.check:
    from droneears.pipeline import Pipeline, summarise
    from droneears.classifier import load_tflite_classifier
    model = os.path.join(os.path.dirname(__file__), "..", "artifacts", "drone_cls_int8.tflite")
    clf = load_tflite_classifier(model) if os.path.exists(model) else None
    res = Pipeline(classifier=clf).run(x); s = summarise(res)
    ml = np.array([r["ml"] for r in res[70:]]) if len(res) > 70 else np.zeros(1)
    print(f"check (fused, classifier {'on' if clf else 'off'}): detected {s['detected_frac']:.0%} of frames, "
          f"first detection {s['first_detect_s']}, max hits {max(r['hits'] for r in res)}, median p(drone) {np.median(ml):.2f}")
    print("  (expected: a drone file detects; a background file shows 0 %. Full timeline: scripts/run_file.py <file>)")
