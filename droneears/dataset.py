"""Dataset assembly for the drone / background classifier.

Sources
  team/       your own recordings (data/team/drone/*, data/team/background/*, plus the committed
              data/drone/* and data/non_drone/*) -- real positives/negatives
  esc50       ESC-50 environmental sounds (CC BY-NC) -- backgrounds, including the confusers (engines, sirens, insects)
  dads        DADS, the public drone-audio dataset (MIT; scripts/get_dads.py) -- real positives from six drone
              recording campaigns plus urban/scene negatives (data/hf-drone/*.parquet)
  synthetic   physically-modelled quadcopter combs (droneears.simulate.synth_drone) -- positives for diversity

Every positive is also mixed with random backgrounds at random SNRs and optionally low-passed (distance) so the
model learns the drone comb under realistic ambient, not the clean recording.

Splits are by source file / fold / seed, never by window, so test windows never share audio with training windows.
"""
import csv, glob, io, os, numpy as np
from scipy.signal import butter, lfilter, resample_poly
from . import config as C
from .audio import load
from .features import frames, power, classifier_windows
from .simulate import synth_drone

ESC50 = os.environ.get("ESC50_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "ESC-50"))
TEAM = os.environ.get("TEAM_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "team"))
# Recordings already committed in the repo (data/drone = positives, data/non_drone = negatives) are used as team
# data too, so `python scripts/train.py` trains on real drone audio straight after a clone. (The public
# data/huggingface_itu_aris set is not picked up here: its files are under data/huggingface_itu_aris/audio/.)
REPO_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
LABEL_SUBDIRS = {"drone": ["drone"], "background": ["background", "ambient", "non_drone"]}
DADS = os.environ.get("DADS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "hf-drone"))
DADS_CHUNKS_PER_SEG = 5            # consecutive 0.5 s DADS chunks joined into one 2.5 s segment (3 classifier windows)
DADS_NEG_MAX_SEC = 10.0
WIN_SEC = C.CLS_WINDOW_FRAMES * C.HOP / C.FS + C.NFFT / C.FS   # audio needed per window (~1.04 s)


def esc50_clips(folds, per_fold=None, rng=None):
    meta = list(csv.DictReader(open(os.path.join(ESC50, "meta", "esc50.csv"))))
    rows = [r for r in meta if int(r["fold"]) in folds]
    if per_fold:
        rng = rng or np.random.default_rng(0)
        rng.shuffle(rows)
        rows = rows[: per_fold * len(folds)]
    return [(os.path.join(ESC50, "audio", r["filename"]), r["category"]) for r in rows]


def team_clips(label):
    """All real recordings for `label` ('drone' or 'background') from data/team/ and the committed data/drone,
    data/non_drone."""
    pats = ["*.wav", "*.m4a", "*.mp3", "*.aac", "*.flac", "*.ogg", "*.mp4", "*.WAV", "*.M4A"]
    files = []
    for root in (TEAM, REPO_DATA):
        for sub in LABEL_SUBDIRS[label]:
            for p in pats:
                files += glob.glob(os.path.join(root, sub, p))
    return sorted(set(files))


def _dads_shards():
    """Parquet shards present locally, sorted by shard index. Empty if pyarrow or the data is missing."""
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        return []
    return sorted(glob.glob(os.path.join(DADS, "train-*.parquet")))


def _dads_label(path):
    import pyarrow.parquet as pq
    lab = pq.read_table(path, columns=["label"]).column("label").to_pylist()
    return int(round(sum(lab) / max(1, len(lab))))      # shards are label-sorted; 0 = no-drone, 1 = drone


def _dads_decode(b):
    """WAV bytes (16 kHz mono in DADS) -> float32 at C.FS."""
    import soundfile as sf
    from math import gcd
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.FS:
        g = gcd(C.FS, sr)
        x = resample_poly(x, C.FS // g, sr // g)
    return x.astype(np.float32)


def _dads_id(name):
    try:
        return int(os.path.splitext(name or "")[0].rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _join(parts, fade=int(0.01 * C.FS)):
    """Concatenate chunks with a short equal-power crossfade so the joins do not click."""
    out = parts[0]
    w = np.sin(0.5 * np.pi * np.linspace(0, 1, fade, dtype=np.float32))     # 0 -> 1
    for nxt in parts[1:]:
        head, tail = nxt[:fade], out[-fade:]
        out = np.concatenate([out[:-fade], tail * np.sqrt(1 - w ** 2) + head * w, nxt[fade:]])
    return out.astype(np.float32)


def bandlimit16k(x):
    """Emulate a 16 kHz recording (nothing above 8 kHz), as DADS audio is. Applied to a random half of the other
    sources so that band-limiting cannot act as a label shortcut."""
    g = 3                                                   # C.FS / 16000
    return resample_poly(resample_poly(x, 1, g), g, 1)[: len(x)].astype(np.float32)


def dads_clips(split, n_pos, n_neg, rng, chunks=DADS_CHUNKS_PER_SEG):
    """Real positives / negatives from the public DADS dataset (download with scripts/get_dads.py).

    Drone rows are 0.5 s chunks whose consecutive numeric ids are adjacent slices of one recording, so `chunks`
    consecutive ids are concatenated into one continuous segment (no looping of tiny clips). Splits: for each label,
    the highest-numbered shard present is the test shard; with a single shard its rows (sorted by id) are split
    80/20 by position. No-drone clips of 4.9-5.1 s are skipped because DADS's negatives include ESC-50 (5.0 s
    clips), which is our held-out test set. Returns (positives, negatives) as lists of float32 arrays at C.FS.
    """
    shards = _dads_shards()
    if not shards or (n_pos <= 0 and n_neg <= 0):
        return [], []
    import pyarrow.parquet as pq
    by_label = {0: [], 1: []}
    for p in shards:
        by_label[_dads_label(p)].append(p)

    def pick(paths):
        if not paths:
            return [], None
        if len(paths) == 1:
            return paths, "rows"                       # one shard: split its rows by position
        return (paths[-1:] if split == "test" else paths[:-1]), "shards"

    pos, neg = [], []
    # ---- positives: 2.5 s segments of consecutive chunks
    paths, mode = pick(by_label[1])
    if n_pos > 0 and paths:
        segs = []
        for p in paths:
            t = pq.read_table(p, columns=["audio"])
            rows = sorted((_dads_id(a.get("path")), a["bytes"]) for a in t.column("audio").to_pylist())
            if mode == "rows":
                cut = int(0.8 * len(rows))
                rows = rows[:cut] if split == "train" else rows[cut:]
            run = []
            for i, b in rows:
                if run and i != run[-1][0] + 1:
                    run = []
                run.append((i, b))
                if len(run) == chunks:
                    segs.append([b_ for _, b_ in run]); run = []
        for k in rng.choice(len(segs), min(n_pos, len(segs)), replace=False):
            pos.append(_join([_dads_decode(b) for b in segs[k]]))
    # ---- negatives: whole clips, 1.1 s .. DADS_NEG_MAX_SEC, ESC-50-length clips excluded
    paths, mode = pick(by_label[0])
    if n_neg > 0 and paths:
        for p in paths:
            pf = pq.ParquetFile(p)
            groups = list(range(pf.metadata.num_row_groups))
            if mode == "rows":
                cut = int(0.8 * len(groups))
                groups = groups[:cut] if split == "train" else groups[cut:]
            rng.shuffle(groups)
            for g in groups:
                if len(neg) >= n_neg:
                    break
                for a in pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist():
                    x = _dads_decode(a["bytes"])
                    sec = len(x) / C.FS
                    if sec < 1.1 or 4.9 <= sec <= 5.1:
                        continue
                    neg.append(x[: int(DADS_NEG_MAX_SEC * C.FS)])
                    if len(neg) >= n_neg:
                        break
    return pos, neg


def windows_from_signal(x):
    P = power(np.fft.rfft(frames(x), axis=1))
    return classifier_windows(P)


def mix(sig, bg, snr_db, rng, lowpass_hz=None):
    """Mix signal into background at a given SNR (RMS based); optional low-pass to mimic distance."""
    n = len(sig)
    if len(bg) < n:
        bg = np.tile(bg, int(np.ceil(n / len(bg))))
    s0 = rng.integers(0, len(bg) - n + 1)
    bg = bg[s0:s0 + n]
    if lowpass_hz:
        b, a = butter(2, lowpass_hz / (C.FS / 2))
        sig = lfilter(b, a, sig).astype(np.float32)
    rs = np.sqrt(np.mean(sig ** 2)) + 1e-9
    rb = np.sqrt(np.mean(bg ** 2)) + 1e-9
    return (sig / rs * 10 ** (snr_db / 20) + bg / rb).astype(np.float32)


def build(split="train", n_synth=120, mixes_per_pos=3, esc_per_fold=None, seed=0, verbose=True,
          n_dads_pos=200, n_dads_neg=100):
    """Return (X, y, groups) with X (n, 47, 40) float32, y in {0 background, 1 drone}, groups = source tag."""
    rng = np.random.default_rng(seed + (0 if split == "train" else 1000))
    folds = [1, 2, 3, 4] if split == "train" else [5]
    Xs, ys, gs = [], [], []

    # ---- backgrounds: ESC-50
    bg_signals = []
    for path, cat in esc50_clips(folds, esc_per_fold, rng):
        x = load(path)
        if np.max(np.abs(x)) < 1e-4:
            continue
        if rng.random() < 0.5:
            x = bandlimit16k(x)
        bg_signals.append(x)
        W = windows_from_signal(x)
        Xs.append(W); ys.append(np.zeros(len(W), int)); gs += [f"esc50:{cat}"] * len(W)
    if verbose:
        print(f"[{split}] ESC-50 background windows: {sum(len(w) for w, l in zip(Xs, ys))}")

    # ---- backgrounds: team ambient (time split 70/30; files named *_holdout.* go entirely to test)
    for f in team_clips("background"):
        x = load(f)
        if "_holdout" in os.path.basename(f):
            if split == "train": continue
        else:
            cut = int(0.7 * len(x))
            x = x[:cut] if split == "train" else x[cut:]
        if rng.random() < 0.5:
            x = bandlimit16k(x)
        W = windows_from_signal(x)
        Xs.append(W); ys.append(np.zeros(len(W), int)); gs += ["team:background"] * len(W)
        bg_signals.append(x)

    # ---- DADS (public): negatives now, positives below
    dads_pos, dads_neg = dads_clips(split, n_dads_pos, n_dads_neg, rng)
    for x in dads_neg:
        W = windows_from_signal(x)
        Xs.append(W); ys.append(np.zeros(len(W), int)); gs += ["dads:background"] * len(W)
        bg_signals.append(x)
    if verbose and (dads_pos or dads_neg):
        print(f"[{split}] DADS: {len(dads_pos)} drone segments, {len(dads_neg)} no-drone clips")

    # ---- positives: team drone (time split 70/30), raw + mixed with backgrounds
    pos_signals = []
    for f in team_clips("drone"):
        x = load(f)
        if "_holdout" in os.path.basename(f):
            if split == "train": continue
        else:
            cut = int(0.7 * len(x))
            x = x[:cut] if split == "train" else x[cut:]
        if rng.random() < 0.5:
            x = bandlimit16k(x)
        pos_signals.append(("team:drone", x))
    for x in dads_pos:
        pos_signals.append(("dads:drone", x))

    # ---- positives: synthetic drones
    for i in range(n_synth):
        s = seed * 100000 + (i if split == "train" else 50000 + i)
        r = np.random.default_rng(s)
        x = synth_drone(3.0, f0=float(r.uniform(150, 360)), n_motors=int(r.integers(2, 5)),
                        harm_decay=float(r.uniform(0.4, 0.75)), motor_noise=float(r.uniform(0.05, 0.4)),
                        wander_depth=float(r.uniform(0.01, 0.06)), seed=s)
        if rng.random() < 0.5:
            x = bandlimit16k(x)
        pos_signals.append(("synthetic", x))

    n_pos_raw = 0
    for tag, x in pos_signals:
        W = windows_from_signal(x)
        Xs.append(W); ys.append(np.ones(len(W), int)); gs += [tag] * len(W); n_pos_raw += len(W)
        for k in range(mixes_per_pos):
            bg = bg_signals[rng.integers(len(bg_signals))]
            snr = float(rng.uniform(-6, 18))
            lp = float(rng.choice([None, 2500, 4000, 6000], p=[0.4, 0.2, 0.2, 0.2]) or 0) or None
            m = mix(x, bg, snr, rng, lp)
            W = windows_from_signal(m)
            Xs.append(W); ys.append(np.ones(len(W), int)); gs += [tag + ":mix"] * len(W)
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.array(gs)
    if verbose:
        print(f"[{split}] total windows {len(y)}: drone {int(y.sum())} (raw {n_pos_raw}), background {int((1 - y).sum())}")
    return X.astype(np.float32), y, g
