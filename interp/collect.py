"""Run interp.harness.record over every clip Part C needs and cache the per-frame records.

Groups (label 1 = drone, 0 = no drone):
  esc50      all 2000 ESC-50 clips (folds 1-4 = dev, fold 5 = test)                                  label 0
  ambient    data/non_drone/ambient_adam.m4a                                                           label 0
  phone      the 9 phone recordings in data/drone (channel 0, as tests/ do)                            label 1
  phonemix   each phone clip mixed into outdoor backgrounds at 0 / 6 / 12 dB SNR, plus low-passed       label 1
             (distance) versions; the background clip's ESC-50 fold decides dev / test
  dads       DADS shard 36 drone recordings (unseen by the classifier: its test shard), runs of
             consecutive 0.5 s chunks joined into 4-8 s clips                                          label 1
  dadsbg     DADS shard 0 no-drone clips from the classifier's test row groups (last 20 %)             label 0
  dadsdev    DADS shards 5, 12, 20, 28 drone recordings (the classifier's training shards): diverse drones
             for choosing gates (dev)                                                                  label 1
  confmix    the false-alarm-prone ESC-50 categories mixed into real backgrounds (phone ambient, wind,
             rain, train) at 0 / 6 / 12 dB: a gate must not just learn "clean studio clip"; split by the
             confuser clip's fold                                                                      label 0
  tilt       mic-response stress test: phone drones, DADS test drones and rule-firing ESC-50 clips with a
             spectral tilt of -6 / -3 / +3 / +6 dB per octave around 1 kHz (split "robust", reported only)

Records go to data/cache/records/<group>/<id>.npz; data/cache/records/index.json lists every job with its metadata.

Usage: python -m interp.collect [--groups esc50,phone,...] [--workers 8] [--limit N]
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from interp import harness as H
from droneears import config as C

REC = os.path.join(H.CACHE, "records")
BG_CATEGORIES = ["wind", "rain", "engine", "airplane", "train", "chirping_birds", "crickets"]   # + phone ambient
SNRS = [0, 6, 12]
LOWPASS = [2500, 4000]
CONFUSERS = ["chainsaw", "vacuum_cleaner", "toilet_flush", "engine", "washing_machine", "insects", "crow",
             "airplane", "hand_saw", "helicopter", "crickets", "siren"]
TILTS = [-6, -3, 3, 6]


def _job_signal(job):
    kind = job["kind"]
    if kind == "file":
        return H.load_cached(job["path"], ch0=True)
    if kind == "npy":
        return np.load(job["path"])
    if kind == "mix":
        sig = H.load_cached(job["path"], ch0=True)
        bg = H.load_cached(job["bg"], ch0=True)
        return H.mix(sig, bg, job["snr"], job["seed"], job.get("lowpass"))
    if kind == "tilt":
        src = np.load(job["path"]) if job["path"].endswith(".npy") else H.load_cached(job["path"], ch0=True)
        return H.tilt(src, job["tilt"])
    raise ValueError(kind)


def run_job(job, force=False):
    dst = os.path.join(REC, job["group"], job["id"] + ".npz")
    if os.path.exists(dst) and not force:
        return job["id"], "cached", 0.0
    t = time.time()
    y = _job_signal(job)
    rec = H.record(y)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, **{k.replace(":", "__"): v for k, v in rec.items()})
    return job["id"], len(rec["hits"]), time.time() - t


def load_record(job):
    z = np.load(os.path.join(REC, job["group"], job["id"] + ".npz"))
    return {k.replace("__", ":"): z[k] for k in z.files}


def dads_pos_jobs(shard, group, split, max_pos, min_chunks=8, max_chunks=16):
    """Drone recordings from one DADS shard: runs of consecutive 0.5 s chunks joined into 4-8 s clips."""
    import pyarrow.parquet as pq
    from droneears.dataset import DADS, _dads_id, _dads_decode, _join
    jobs = []
    out_dir = os.path.join(H.CACHE, "dads")
    os.makedirs(out_dir, exist_ok=True)
    p36 = os.path.join(DADS, f"train-{shard:05d}-of-00039.parquet")
    if os.path.exists(p36):
        t = pq.read_table(p36, columns=["audio"])
        rows = sorted((_dads_id(a.get("path")), a["bytes"]) for a in t.column("audio").to_pylist())
        runs, run = [], []
        for i, b in rows:
            if run and i != run[-1][0] + 1:
                if len(run) >= min_chunks:
                    runs.append(run)
                run = []
            run.append((i, b))
            if len(run) == max_chunks:
                runs.append(run); run = []
        if len(run) >= min_chunks:
            runs.append(run)
        rng = np.random.default_rng(0)
        for r in [runs[j] for j in sorted(rng.choice(len(runs), min(max_pos, len(runs)), replace=False))]:
            jid = f"dads{shard}_{r[0][0]}_{len(r)}"
            path = os.path.join(out_dir, jid + ".npy")
            if not os.path.exists(path):
                np.save(path, _join([_dads_decode(b) for _, b in r]))
            jobs.append(dict(id=jid, group=group, label=1, kind="npy", path=path, split=split, n_chunks=len(r),
                             shard=shard))
    return jobs


def dads_jobs(max_pos=150, max_neg=100):
    """Decode DADS shards into cached .npy clips and return jobs. Positives: shard 36; negatives: shard 0."""
    import pyarrow.parquet as pq
    from droneears.dataset import DADS, _dads_decode
    out_dir = os.path.join(H.CACHE, "dads")
    jobs = dads_pos_jobs(36, "dads", "test", max_pos)
    p0 = os.path.join(DADS, "train-00000-of-00039.parquet")
    if os.path.exists(p0) and not p0.endswith(".part"):
        pf = pq.ParquetFile(p0)
        groups = list(range(pf.metadata.num_row_groups))
        groups = groups[int(0.8 * len(groups)):]                      # the classifier's test row groups
        n = 0
        for g in groups:
            for j, a in enumerate(pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()):
                jid = f"dads0_{g}_{j}"
                path = os.path.join(out_dir, jid + ".npy")
                if not os.path.exists(path):
                    x = _dads_decode(a["bytes"])
                    sec = len(x) / C.FS
                    if sec < 3.0 or 4.9 <= sec <= 5.1:                # too short for the detector, or an ESC-50 clip
                        continue
                    np.save(path, x[: int(10 * C.FS)])
                jobs.append(dict(id=jid, group="dadsbg", label=0, kind="npy", path=path, split="test"))
                n += 1
                if n >= max_neg:
                    break
            if n >= max_neg:
                break
    return jobs


def build_jobs(groups):
    jobs = []
    rows = H.esc50_rows()
    if "esc50" in groups:
        for r in rows:
            jobs.append(dict(id=r["filename"][:-4], group="esc50", label=0, kind="file",
                             path=os.path.join(H.ESC50, "audio", r["filename"]), fold=int(r["fold"]),
                             category=r["category"], split="test" if r["fold"] == "5" else "dev"))
    if "ambient" in groups:
        jobs.append(dict(id="ambient_adam", group="ambient", label=0, kind="file", path=H.PHONE_AMBIENT, split="test"))
    if "phone" in groups:
        for p in H.PHONE_DRONE:
            jobs.append(dict(id=os.path.splitext(os.path.basename(p))[0].replace(" ", "_"), group="phone", label=1,
                             kind="file", path=p, split="dev"))
    if "phonemix" in groups:
        rng = np.random.default_rng(1)
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        for p in H.PHONE_DRONE:
            name = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            for cat in ["ambient"] + BG_CATEGORIES:
                for snr in SNRS:
                    if cat == "ambient":
                        bg, split, fold = H.PHONE_AMBIENT, "dev", None
                    else:
                        r = by_cat[cat][rng.integers(len(by_cat[cat]))]
                        bg, fold = os.path.join(H.ESC50, "audio", r["filename"]), int(r["fold"])
                        split = "test" if fold == 5 else "dev"
                    seed = int(rng.integers(1 << 30))
                    jobs.append(dict(id=f"{name}__{cat}__{snr}", group="phonemix", label=1, kind="mix", path=p, bg=bg,
                                     snr=snr, seed=seed, category=cat, split=split, bg_fold=fold))
            for lp in LOWPASS:
                seed = int(rng.integers(1 << 30))
                jobs.append(dict(id=f"{name}__ambient__12__lp{lp}", group="phonemix", label=1, kind="mix", path=p,
                                 bg=H.PHONE_AMBIENT, snr=12, seed=seed, lowpass=lp, category="ambient_lowpass",
                                 split="dev"))
    if "dads" in groups or "dadsbg" in groups:
        jobs += [j for j in dads_jobs() if j["group"] in groups]
    if "dadsdev" in groups:
        for shard in (5, 12, 20, 28):
            jobs += dads_pos_jobs(shard, "dadsdev", "dev", 50)
    if "confmix" in groups:
        rng = np.random.default_rng(2)
        bgs = [("ambient", H.PHONE_AMBIENT)]
        for cat in ("wind", "rain", "train"):
            r = [x for x in rows if x["category"] == cat]
            bgs += [(cat, os.path.join(H.ESC50, "audio", x["filename"])) for x in r[:5]]
        for r in rows:
            if r["category"] not in CONFUSERS:
                continue
            for snr in SNRS:
                cat, bg = bgs[rng.integers(len(bgs))]
                jobs.append(dict(id=f"{r['filename'][:-4]}__{cat}__{snr}", group="confmix", label=0, kind="mix",
                                 path=os.path.join(H.ESC50, "audio", r["filename"]), bg=bg, snr=snr,
                                 seed=int(rng.integers(1 << 30)), category=r["category"], bg_category=cat,
                                 fold=int(r["fold"]), split="test" if r["fold"] == "5" else "dev"))
    if "tilt" in groups:
        srcs = [(p, 1, "phone") for p in H.PHONE_DRONE]
        d36 = sorted(glob.glob(os.path.join(H.CACHE, "dads", "dads36_*.npy")))[:40]
        srcs += [(p, 1, "dads") for p in d36]
        fired = []
        for r in rows:                                   # ESC-50 clips the rules fire on at a relaxed floor
            z = os.path.join(REC, "esc50", r["filename"][:-4] + ".npz")
            if os.path.exists(z):
                rec = dict(np.load(z))
                if H.hits_of(H.rules_frame_ok(rec, flat_min=0.08)).max() >= C.PERSIST_ON:
                    fired.append(os.path.join(H.ESC50, "audio", r["filename"]))
        srcs += [(p, 0, "esc50") for p in fired]
        for p, lab, src in srcs:
            base = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            for tl in TILTS:
                jobs.append(dict(id=f"{base}__tilt{tl:+d}", group="tilt", label=lab, kind="tilt", path=p, tilt=tl,
                                 source=src, split="robust"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="esc50,ambient,phone,phonemix,dads,dadsbg,dadsdev,confmix,tilt")
    ap.add_argument("--force", action="store_true", help="recompute records that already exist")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="first N jobs per group (smoke test)")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    jobs = build_jobs(groups)
    if args.limit:
        seen = {}
        jobs = [j for j in jobs if seen.setdefault(j["group"], []).append(1) or len(seen[j["group"]]) <= args.limit]
    os.makedirs(REC, exist_ok=True)
    idx_path = os.path.join(REC, "index.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    for j in jobs:
        index[f"{j['group']}/{j['id']}"] = j
    json.dump(index, open(idx_path, "w"), indent=0)
    print(f"{len(jobs)} jobs: " + ", ".join(f"{g} {sum(j['group'] == g for j in jobs)}" for g in groups), flush=True)
    t0 = time.time(); done = 0; frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, args.force) for j in jobs]
        for f in as_completed(futs):
            jid, n, dt = f.result()
            done += 1
            frames += n if isinstance(n, int) else 0
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {frames} new frames  [{el:.0f} s, eta {el / done * (len(jobs) - done):.0f} s]",
                      flush=True)
    print("records in", REC)


if __name__ == "__main__":
    main()
