"""Part B: edits to real recordings. Does the classifier rely on the harmonic peaks, the smooth spectral shape, or
the timing? Each spectrum frame is split the way the detector splits it:

    P_db = shape + fine,   shape = droneears.detector.spectral_baseline(P_db)   (smooth, peak-ignoring),
                           fine  = P_db - shape                                   (peaks above it, valleys below)

then one part is edited and the edited power spectrum goes through the same log-mel -> window normalisation ->
int8 classifier path as the firmware. The rules are re-run on the same edited spectra for comparison.

Edits
  orig                 nothing (sanity: must match the unedited p)
  no_peaks             fine clipped to <= 0 everywhere: every peak flattened onto the shape
  no_peaks_<band>      peaks removed in one octave band only
  only_peaks_<band>    peaks removed everywhere except one octave band
  no_harm_peaks        peaks removed only within +-1 bin of k*f0 (the detector's f0 per frame)
  no_other_peaks       peaks removed everywhere except within +-1 bin of k*f0
  flat_shape           shape replaced by its mean level (fine structure kept)
  shape_of:<clip>      drone fine structure on a confuser's shape (and the reverse for the confuser clips)
  shuffle              frames shuffled in time within each classifier window
  time_avg             every frame of the window replaced by the window's mean spectrum

Usage: python -m interp.part_b [--confusers chainsaw,vacuum_cleaner,toilet_flush,engine]
"""
import argparse, json, os
import numpy as np
from interp import harness as H
from droneears import config as C
from droneears.detector import spectral_baseline, DroneDetector
from droneears.features import logmel, normalise_window

FREQ = np.arange(C.NBINS) * C.BIN_HZ
OCTAVES = [(100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 8000)]


def decompose(P):
    P_db = 10.0 * np.log10(P + 1e-12)
    shape = np.stack([spectral_baseline(f) for f in P_db]).astype(np.float32)
    return P_db, shape, (P_db - shape).astype(np.float32)


def to_power(db):
    return (10.0 ** (db / 10.0)).astype(np.float32)


def harm_near(f0):
    """(n, NBINS) bool: within +-1 bin of k*f0 for each frame's f0."""
    m = np.zeros((len(f0), C.NBINS), bool)
    for i, f in enumerate(f0):
        for k in range(1, int(C.FS / 2 / f) + 1):
            b = int(round(k * f / C.BIN_HZ))
            if b >= C.NBINS - 1:
                break
            m[i, max(0, b - 1): b + 2] = True
    return m


def window_ps(P, shuffle=False, time_avg=False, seed=0):
    """p(drone) of every classifier window (47 frames, step 23), exactly as the pipeline computes it."""
    clf = H.classifier()
    LM = logmel(P)
    rng = np.random.default_rng(seed)
    ps = []
    for s in range(0, len(LM) - C.CLS_WINDOW_FRAMES + 1, H.ML_EVERY):
        W = LM[s: s + C.CLS_WINDOW_FRAMES]
        if shuffle:
            W = W[rng.permutation(len(W))]
        if time_avg:
            W = np.repeat(10 * np.log10(np.mean(10 ** (W / 10), axis=0, keepdims=True)), len(W), axis=0)
        ps.append(clf(normalise_window(W)))
    return np.array(ps)


def rules_on(P):
    det = DroneDetector()
    ok, hits = [], []
    for f in P:
        d = det.feed(f)
        ok.append(d["frame_ok"]); hits.append(d["hits"])
    return dict(frame_ok_frac=float(np.mean(ok)), detected=bool(np.max(hits) >= C.PERSIST_ON), max_hits=int(np.max(hits)))


def edits_for(P, f0):
    P_db, shape, fine = decompose(P)
    peaks_gone = np.minimum(fine, 0.0)
    near = harm_near(f0)
    out = {"orig": P}
    out["no_peaks"] = to_power(shape + peaks_gone)
    for lo, hi in OCTAVES:
        band = (FREQ >= lo) & (FREQ < hi)
        out[f"no_peaks_{lo}-{hi}"] = to_power(shape + np.where(band, peaks_gone, fine))
        out[f"only_peaks_{lo}-{hi}"] = to_power(shape + np.where(band, fine, peaks_gone))
    out["no_harm_peaks"] = to_power(shape + np.where(near, peaks_gone, fine))
    out["no_other_peaks"] = to_power(shape + np.where(near, fine, peaks_gone))
    band = (FREQ >= 100) & (FREQ <= 8000)
    flat = shape[:, band].mean(axis=1, keepdims=True)
    out["flat_shape"] = to_power(np.where(band, flat, shape) + fine)
    return out, shape, fine


def tile_to(x, n):
    return np.concatenate([x] * int(np.ceil(n / len(x))))[:n]


def summarize(ps):
    return dict(median_p=float(np.median(ps)) if len(ps) else None, frac_gt_07=float(np.mean(ps > C.ML_THRESH)) if len(ps) else None,
                n=int(len(ps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusers", default="chainsaw,vacuum_cleaner,toilet_flush,engine")
    ap.add_argument("--per-confuser", type=int, default=3)
    ap.add_argument("--rules", choices=["shipped", "current"], default="shipped",
                    help="rules column: the shipped detector (global flatness, no DSP gate; what interp/README.md "
                         "reports) or droneears/config.py as it stands")
    args = ap.parse_args()
    if args.rules == "shipped":
        C.FLAT_MODE, C.GATE_ON = "global", False
    os.makedirs(H.OUT, exist_ok=True)
    rows = H.esc50_rows()
    # confuser clips: the ones the rules alone fire on (from Part C's records if present), else the first few
    conf_paths = []
    for cat in args.confusers.split(","):
        rs = [r for r in rows if r["category"] == cat]
        fired = []
        for r in rs:
            p = os.path.join(H.CACHE, "records", "esc50", r["filename"][:-4] + ".npz")
            if os.path.exists(p) and np.load(p)["hits"].max() >= C.PERSIST_ON:
                fired.append(r)
        pick = (fired or rs)[: args.per_confuser]
        conf_paths += [(cat, os.path.join(H.ESC50, "audio", r["filename"])) for r in pick]

    results = {}
    drone_parts = {}
    for path in H.PHONE_DRONE:
        name = os.path.splitext(os.path.basename(path))[0]
        y = H.load_cached(path)
        P = H.power_frames(y)
        f0 = H.record(y, clf=False, features=False)["f0"]
        eds, shape, fine = edits_for(P, f0)
        drone_parts[name] = (shape, fine)
        r = {}
        for k, Pe in eds.items():
            r[k] = dict(cls=summarize(window_ps(Pe)), rules=rules_on(Pe))
        r["shuffle"] = dict(cls=summarize(window_ps(P, shuffle=True)), rules=None)
        r["time_avg"] = dict(cls=summarize(window_ps(P, time_avg=True)), rules=None)
        results[f"drone:{name}"] = r
        print(f"{name:24s} orig p {r['orig']['cls']['median_p']:.2f}  no_peaks {r['no_peaks']['cls']['median_p']:.2f}  "
              f"no_harm {r['no_harm_peaks']['cls']['median_p']:.2f}  no_other {r['no_other_peaks']['cls']['median_p']:.2f}  "
              f"flat_shape {r['flat_shape']['cls']['median_p']:.2f}  shuffle {r['shuffle']['cls']['median_p']:.2f}  "
              f"time_avg {r['time_avg']['cls']['median_p']:.2f}", flush=True)

    # swaps: drone fine structure on confuser shape, confuser fine structure on drone shape
    for cat, path in conf_paths:
        cname = f"{cat}:{os.path.basename(path)[:-4]}"
        Pc = H.power_frames(H.load_cached(path))
        _, cshape, cfine = decompose(Pc)
        r = {"orig": dict(cls=summarize(window_ps(Pc)), rules=rules_on(Pc))}
        for name, (shape, fine) in drone_parts.items():
            n = len(cshape)
            Pd_on_c = to_power(cshape + tile_to(fine, n))
            Pc_on_d = to_power(tile_to(shape, n) + cfine)
            r[f"drone_fine_on_this_shape:{name}"] = dict(cls=summarize(window_ps(Pd_on_c)), rules=rules_on(Pd_on_c))
            r[f"this_fine_on_drone_shape:{name}"] = dict(cls=summarize(window_ps(Pc_on_d)), rules=rules_on(Pc_on_d))
        results[f"confuser:{cname}"] = r
        dfc = np.median([v["cls"]["median_p"] for k, v in r.items() if k.startswith("drone_fine")])
        cfd = np.median([v["cls"]["median_p"] for k, v in r.items() if k.startswith("this_fine")])
        print(f"{cname:34s} orig p {r['orig']['cls']['median_p']:.2f}  drone peaks on its shape {dfc:.2f}  "
              f"its peaks on drone shape {cfd:.2f}", flush=True)

    json.dump(results, open(os.path.join(H.OUT, "part_b.json"), "w"), indent=1)
    print("wrote", os.path.join(H.OUT, "part_b.json"))


if __name__ == "__main__":
    main()
