"""Part C: which cheap measurements separate real drones from the false alarms the rules raise (and the classifier
blocks)? Ranks every window feature from interp.features, picks gate thresholds on the dev split, and scores each
gate on the held-out groups by replaying the detection state machine (interp.harness.replay).

The gate works exactly like the classifier's veto today: the rules' persistence count must reach PERSIST_ON AND the
gate must be open at that frame. A clip therefore detects with gate "feature >= t" iff the feature's maximum over
its onset-candidate frames (hits >= PERSIST_ON) reaches t, which makes the clip-level threshold choice exact.

Splits
  dev     positives: phone recordings, phone mixes with dev backgrounds, DADS shards 5/12/20/28 (diverse drones)
          negatives: ESC-50 folds 1-4, false-alarm-prone ESC-50 categories mixed into real backgrounds (folds 1-4)
  test    positives: DADS shard 36 (never seen by the classifier or by gate selection), phone mixes with fold-5
          backgrounds; negatives: ESC-50 fold 5, its confusers in background noise, DADS no-drone, the phone ambient
  robust  mic-response tilts of phone / DADS drones and of rule-firing ESC-50 clips (reported only)

Usage: python -m interp.part_c [--flat-min 0.17] [--top 14]
"""
import argparse, collections, json, os
import numpy as np
from scipy.stats import rankdata
from interp import harness as H
from interp.collect import REC, load_record
from interp.features import window_features
from droneears import config as C

REPORT = [("phone:dev", "+", "phone drones"), ("phonemix:test", "+", "phone drones in noise (test bg)"),
          ("dads:test", "+", "DADS drones (held out)"), ("tilt+:robust", "+", "drones, mic tilt"),
          ("esc50:test", "-", "ESC-50 fold 5"), ("confmix:test", "-", "confusers in noise (fold 5)"),
          ("dadsbg:test", "-", "DADS no-drone"), ("ambient:test", "-", "phone ambient 237 s"),
          ("tilt-:robust", "-", "rule-firing ESC-50, mic tilt")]


# features that do not move with a smooth change of microphone response (prominence above the local shape, counts,
# f0 track, temporal statistics); the spectral-shape ones (slope, energy ratios, centroid, global flatness) do
TILT_INVARIANT = ["w_fund_half_db", "w_fund_f0_db", "w_fund_db", "w_low_comb_db", "w_harm_mean_db", "w_mid_mean_db",
                  "w_contrast_db", "w_decay_db", "w_harm_peaks", "w_inharm_peaks", "w_harm_frac_peaks", "w_extent_frac",
                  "w_top_harm_hz", "w_flux_db", "w_dlogf0", "w_level_std", "w_f0_jump_frac", "w_score", "w_n_harm",
                  "w_f0_steady", "w_f0", "w_flat_local", "w_flat_local_avg", "w_lf_peak_db"]


def auc(pos, neg):
    """P(pos > neg) with ties counted half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_all(groups=None):
    index = json.load(open(os.path.join(REC, "index.json")))
    clips = []
    for key, job in index.items():
        if groups and job["group"] not in groups:
            continue
        if not os.path.exists(os.path.join(REC, job["group"], job["id"] + ".npz")):
            continue
        rec = load_record(job)
        if "f:flat_local" not in rec:                  # stale record from before a feature was added
            continue
        rec.update(window_features(rec))
        clips.append(dict(job=job, rec=rec))
    return clips


def key_of(job):
    if job["group"] == "tilt":
        return f"tilt{'+' if job['label'] == 1 else '-'}:robust"
    return f"{job['group']}:{job['split']}"


def score_gate(clips, gate_fn):
    """Replay every clip with gate_fn(rec) -> (n,) bool (None = rules only). Returns per-group stats."""
    by = collections.defaultdict(lambda: dict(clips=0, any=0, frames=0, det_frames=0, first=[]))
    for c in clips:
        rec = c["rec"]
        det = H.replay(rec["frame_ok"], None if gate_fn is None else gate_fn(rec))
        b = by[key_of(c["job"])]
        b["clips"] += 1; b["any"] += int(det.any()); b["frames"] += len(det); b["det_frames"] += int(det.sum())
        if det.any():
            b["first"].append(int(np.argmax(det)) * C.HOP / C.FS)
    return {k: dict(clips=b["clips"], clips_detected=b["any"], clip_rate=b["any"] / max(1, b["clips"]),
                    frame_rate=b["det_frames"] / max(1, b["frames"]),
                    median_first_s=float(np.median(b["first"])) if b["first"] else None)
            for k, b in sorted(by.items())}


def onset_max(clips, fn):
    """Per clip: max of fn(rec) over frames where the rules would fire (hits >= PERSIST_ON), or None."""
    out = []
    for c in clips:
        m = c["rec"]["hits"] >= C.PERSIST_ON
        out.append(float(np.max(fn(c["rec"])[m])) if m.any() else None)
    return out


def dev_sets(clips):
    dev = [c for c in clips if c["job"]["split"] == "dev"]
    return dict(phone=[c for c in dev if c["job"]["group"] == "phone"],
                mix=[c for c in dev if c["job"]["group"] == "phonemix"],
                dads=[c for c in dev if c["job"]["group"] == "dadsdev"],
                neg=[c for c in dev if c["job"]["label"] == 0])


def choose_threshold(sets, fn, keep):
    """Largest share of dev negatives removed subject to keeping >= keep[name] of each dev positive set (among the
    clips the rules alone detect). Gate = fn(rec) >= t."""
    vals = {k: np.array([v for v in onset_max(cs, fn) if v is not None]) for k, cs in sets.items()}
    best = None
    for t in np.unique(np.concatenate(list(vals.values()))):
        if any(len(vals[k]) and np.mean(vals[k] >= t) < keep[k] for k in keep):
            continue
        removed = float(np.mean(vals["neg"] < t)) if len(vals["neg"]) else 0.0
        if best is None or removed > best[1]:
            best = (float(t), removed, {k: float(np.mean(vals[k] >= t)) if len(vals[k]) else None for k in keep})
    return best


def linear_gate(sets, feats, creg, keep):
    """L1 logistic regression on dev onset moments (frames with hits >= PERSIST_ON), each clip weighted 1 in total,
    negatives and positives balanced. Gate = sum_k a_k x_k >= b on raw rolling-mean features."""
    from sklearn.linear_model import LogisticRegression
    Xs, ys, ws = [], [], []
    for name, cs in sets.items():
        for c in cs:
            m = c["rec"]["hits"] >= C.PERSIST_ON
            if not m.any():
                continue
            X = np.stack([c["rec"][f][m] for f in feats], 1)
            Xs.append(X); ys.append(np.full(len(X), c["job"]["label"])); ws.append(np.full(len(X), 1.0 / len(X)))
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(penalty="l1", C=creg, solver="liblinear", class_weight="balanced", max_iter=5000)
    lr.fit((X - mu) / sd, y, sample_weight=w)
    coef = lr.coef_[0]
    keepf = np.flatnonzero(np.abs(coef) > 1e-6)
    if len(keepf) == 0:
        return None
    a = coef[keepf] / sd[keepf]
    names = [feats[k] for k in keepf]
    fn = lambda rec: np.stack([rec[f] for f in names], 1) @ a
    best = choose_threshold(sets, fn, keep)
    if best is None:
        return None
    return dict(kind="linear", C=creg, weights=dict(zip(names, map(float, a))), threshold=best[0],
                dev_neg_removed=best[1], dev_kept=best[2], fn=fn)


def fmt_row(name, res):
    cells = []
    for k, sign, _ in REPORT:
        v = res.get(k)
        cells.append(f"{v['clips_detected']}/{v['clips']}" if v else "-")
    return f"| {name} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat-min", type=float, default=C.FLAT_MIN, help="re-derive the rules with this tonality floor")
    ap.add_argument("--keep-phone", type=float, default=1.0)
    ap.add_argument("--keep-mix", type=float, default=0.95)
    ap.add_argument("--keep-dads", type=float, default=0.95)
    ap.add_argument("--flat-max", type=float, default=C.FLAT_THRESH)
    ap.add_argument("--flat-key", default="flat", help='"flat" (the detector) or "f:flat_local" (shape divided out)')
    ap.add_argument("--invariant", action="store_true", help="gate candidates: tilt-invariant features only")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(H.OUT, exist_ok=True)
    clips = load_all()
    for c in clips:                                  # rules re-derived offline (exact) with the requested floor
        c["rec"]["frame_ok"] = H.rules_frame_ok(c["rec"], flat_min=args.flat_min, flat_max=args.flat_max,
                                                flat_key=args.flat_key)
        c["rec"]["hits"] = H.hits_of(c["rec"]["frame_ok"])
    tag = "" if abs(args.flat_min - C.FLAT_MIN) < 1e-9 and args.flat_key == "flat" else \
        f"_{args.flat_key.replace('f:', '')}{args.flat_min:g}-{args.flat_max:g}"
    tag += "_inv" if args.invariant else ""
    print(f"tonality test {args.flat_key} in ({args.flat_min:g}, {args.flat_max:g}); {len(clips)} clips: " +
          ", ".join(f"{g} {n}" for g, n in sorted(collections.Counter(key_of(c['job']) for c in clips).items())))
    keep = dict(phone=args.keep_phone, mix=args.keep_mix, dads=args.keep_dads)
    sets = dev_sets(clips)

    rows = [("rules only", score_gate(clips, None)), ("rules + classifier (shipped fused)", score_gate(clips, lambda r: r["ml_ok"] > 0))]

    feats = sorted(k for k in clips[0]["rec"] if k.startswith("w_") and (not args.invariant or k in TILT_INVARIANT))
    pos = sets["phone"] + sets["mix"] + sets["dads"]
    ranking = []
    for f in feats:
        p_hi = [v for v in onset_max(pos, lambda r: r[f]) if v is not None]
        n_hi = [v for v in onset_max(sets["neg"], lambda r: r[f]) if v is not None]
        p_lo = [v for v in onset_max(pos, lambda r: -r[f]) if v is not None]
        n_lo = [v for v in onset_max(sets["neg"], lambda r: -r[f]) if v is not None]
        a_hi, a_lo = auc(p_hi, n_hi), auc(p_lo, n_lo)
        d, a = (">=", a_hi) if a_hi >= a_lo else ("<=", a_lo)
        ranking.append(dict(feature=f, direction=d, auc=a, n_pos=len(p_hi), n_neg=len(n_hi)))
    ranking.sort(key=lambda r: -r["auc"])
    print(f"\n== feature ranking on dev onset moments ({ranking[0]['n_pos']} drone clips, {ranking[0]['n_neg']} "
          f"false-alarm clips the rules fire on)")
    for r in ranking[: args.top]:
        print(f"  {r['feature']:22s} {r['direction']}  AUC {r['auc']:.3f}")

    gates = []
    for r in ranking[: args.top]:
        f, sgn = r["feature"], (1.0 if r["direction"] == ">=" else -1.0)
        fn = (lambda f=f, sgn=sgn: (lambda rec: sgn * rec[f]))()
        best = choose_threshold(sets, fn, keep)
        if best is None:
            continue
        g = dict(kind="single", feature=f, direction=r["direction"], threshold=best[0] * sgn,
                 dev_neg_removed=best[1], dev_kept=best[2], fn=fn, t=best[0])
        g["result"] = score_gate(clips, lambda rec, fn=fn, t=best[0]: fn(rec) >= t)
        gates.append(g)
        rows.append((f"{f[2:]} {r['direction']} {g['threshold']:.3g}", g["result"]))
    for creg in (0.01, 0.03, 0.1, 0.3):
        g = linear_gate(sets, feats, creg, keep)
        if g is None:
            continue
        g["result"] = score_gate(clips, lambda rec, fn=g["fn"], t=g["threshold"]: fn(rec) >= t)
        gates.append(g)
        rows.append((f"linear C={creg} ({len(g['weights'])} features)", g["result"]))

    head = "| gate | " + " | ".join(f"{'+' if s == '+' else '-'} {d}" for _, s, d in REPORT) + " |"
    print(f"\n== clips detected (+ rows: want high; - rows: false alarms, want low). Thresholds chosen on dev, keeping "
          f">= {keep['phone']:.0%} phone, {keep['mix']:.0%} dev mixes, {keep['dads']:.0%} DADS dev drones")
    print(head); print("|" + "---|" * (len(REPORT) + 1))
    for name, res in rows:
        print(fmt_row(name, res))
    for g in gates:
        if g["kind"] == "linear":
            print(f"\nlinear C={g['C']}: dev negatives removed {g['dev_neg_removed']:.0%}, kept {g['dev_kept']}; "
                  f"gate: " + " + ".join(f"{w:+.3g}*{f[2:]}" for f, w in sorted(g['weights'].items(), key=lambda kv: -abs(kv[1])))
                  + f" >= {g['threshold']:.3g}")
    for g in gates:
        g.pop("fn", None)
    json.dump(dict(flat_min=args.flat_min, flat_max=args.flat_max, flat_key=args.flat_key, invariant=args.invariant, keep=keep, rows=[dict(name=n, result=r) for n, r in rows],
                   ranking=ranking, gates=gates), open(os.path.join(H.OUT, f"part_c{tag}.json"), "w"), indent=1)
    print("wrote", os.path.join(H.OUT, f"part_c{tag}.json"))


if __name__ == "__main__":
    main()
