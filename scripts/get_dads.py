"""Download shards of the public DADS drone-audio dataset (MIT licence) for training positives.

DADS = Drone Audio Detection Samples, https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples
180k clips (163k drone / 17k no-drone), 16 kHz mono WAV inside 39 parquet shards of ~175 MB each. Two shards are
plenty for this classifier; `droneears.dataset` reads whatever is in data/hf-drone/ (or $DADS_DIR).

Usage:  python scripts/get_dads.py [--shards 0,5,12,20,28,36] [--out data/hf-drone]
"""
import argparse, os, sys, urllib.request

BASE = "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/resolve/main/data"
N_SHARDS = 39

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="0,5,12,20,28,36", help="comma-separated shard indices (0-38); the shipped model used these six (0 = no-drone; highest drone shard = test)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "hf-drone"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for s in [int(v) for v in args.shards.split(",") if v.strip()]:
    name = f"train-{s:05d}-of-{N_SHARDS:05d}.parquet"
    dst = os.path.join(args.out, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"{name}: already present ({os.path.getsize(dst) / 1e6:.0f} MB)")
        continue
    url = f"{BASE}/{name}"
    print(f"{name}: downloading from {url}")
    tmp = dst + ".part"
    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0 and blocks % 200 == 0:
            sys.stdout.write(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB"); sys.stdout.flush()
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, dst)
    print(f"\r  done: {os.path.getsize(dst) / 1e6:.0f} MB")
print("DADS shards in", os.path.abspath(args.out))
