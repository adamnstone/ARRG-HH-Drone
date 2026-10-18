"""False-alarm rate of the detector on ESC-50 (no drones in it, so every DETECTED frame is a false alarm).

Runs the pipeline over every clip, mono, with the rule-based harmonic detector alone (--no-ml) or with the
classifier ANDed in (--model, default artifacts/drone_cls_int8.tflite), and reports:
  (a) fraction of frames in the DETECTED state, (b) fraction of clips with any detection,
  (c) per-category table (categories with a nonzero frame rate), sorted by frame false-alarm rate.

Usage: python scripts/eval_esc50.py [--no-ml | --model PATH] [--per-class N] [--folds 1,2,3,4,5] [--json out.json]
ESC-50 is expected at data/ESC-50 or $ESC50_DIR (same as droneears/dataset.py).
"""
import argparse, collections, csv, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from droneears import config as C
from droneears.audio import load
from droneears.pipeline import Pipeline

ROOT = os.path.join(os.path.dirname(__file__), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--esc50", default=os.environ.get("ESC50_DIR", os.path.join(ROOT, "data", "ESC-50")))
ap.add_argument("--no-ml", action="store_true", help="rule-based harmonic detector only")
ap.add_argument("--model", default=os.path.join(ROOT, "artifacts", "drone_cls_int8.tflite"))
ap.add_argument("--per-class", type=int, default=None, help="clips per category (default: all 40)")
ap.add_argument("--folds", default="1,2,3,4,5")
ap.add_argument("--seed", type=int, default=0, help="which clips are kept with --per-class")
ap.add_argument("--json", help="also write the summary and per-category numbers here")
args = ap.parse_args()

folds = {int(f) for f in args.folds.split(",") if f.strip()}
meta = list(csv.DictReader(open(os.path.join(args.esc50, "meta", "esc50.csv"))))
rows = sorted((r for r in meta if int(r["fold"]) in folds), key=lambda r: r["filename"])
if args.per_class:
    rng = np.random.default_rng(args.seed)
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rows = [r for cat in sorted(by_cat) for r in rng.permutation(by_cat[cat])[: args.per_class]]

if args.no_ml:
    clf = None
else:
    if not os.path.exists(args.model):
        sys.exit(f"classifier not found: {args.model} (pass --no-ml for the rule-based detector alone)")
    from droneears.classifier import load_tflite_classifier
    clf = load_tflite_classifier(args.model)

print(f"ESC-50 at {os.path.abspath(args.esc50)}: {len(rows)} clips, folds {sorted(folds)}, "
      f"classifier {'off (rule-based only)' if clf is None else 'on: ' + os.path.abspath(args.model)}")

per_cat = collections.defaultdict(lambda: dict(frames=0, det_frames=0, clips=0, det_clips=0))
tot_frames = tot_det = tot_clips = tot_det_clips = 0
t0 = time.time()
for i, r in enumerate(rows, 1):
    y = load(os.path.join(args.esc50, "audio", r["filename"]))
    res = Pipeline(classifier=clf).run(y)          # fresh state per clip, like scripts/run_file.py
    n_det = sum(1 for f in res if f["detected"])
    c = per_cat[r["category"]]
    c["frames"] += len(res); c["det_frames"] += n_det; c["clips"] += 1; c["det_clips"] += int(n_det > 0)
    tot_frames += len(res); tot_det += n_det; tot_clips += 1; tot_det_clips += int(n_det > 0)
    if i % 100 == 0 or i == len(rows):
        el = time.time() - t0
        print(f"  {i}/{len(rows)} clips, {tot_frames} frames, frame FA {tot_det / max(1, tot_frames):.2%}, "
              f"clip FA {tot_det_clips / tot_clips:.1%}  [{el:.0f} s, eta {el / i * (len(rows) - i):.0f} s]", flush=True)

frame_fa = tot_det / max(1, tot_frames)
clip_fa = tot_det_clips / max(1, tot_clips)
print(f"\nclassifier {'off' if clf is None else 'on'}: {tot_clips} clips, {tot_frames} frames ({time.time() - t0:.0f} s)")
print(f"(a) frames in DETECTED state: {tot_det} / {tot_frames} = {frame_fa:.2%}")
print(f"(b) clips with any detection: {tot_det_clips} / {tot_clips} = {clip_fa:.1%}")
table = sorted(per_cat.items(), key=lambda kv: (-kv[1]["det_frames"] / max(1, kv[1]["frames"]), kv[0]))
nonzero = [(k, v) for k, v in table if v["det_frames"] > 0]
print(f"(c) categories with false alarms ({len(nonzero)} of {len(per_cat)}), by frame rate:")
print(f"    {'category':18s} {'frame FA':>9s} {'det/frames':>13s} {'clips':>9s}")
for k, v in nonzero:
    print(f"    {k:18s} {v['det_frames'] / v['frames']:9.2%} {v['det_frames']:6d}/{v['frames']:<6d} {v['det_clips']:3d}/{v['clips']:<3d}")
if args.json:
    json.dump(dict(classifier=None if clf is None else os.path.abspath(args.model), folds=sorted(folds),
                   per_class=args.per_class, clips=tot_clips, frames=tot_frames, det_frames=tot_det,
                   det_clips=tot_det_clips, frame_fa=frame_fa, clip_fa=clip_fa,
                   per_category={k: v for k, v in table}), open(args.json, "w"), indent=1)
    print("wrote", args.json)
