"""Retune the detector thresholds on held-out recordings (TRAINING_PLAN.md section 4, step 5).

Sweeps a grid of detector constants, runs the pipeline (fresh state per file, like scripts/run_file.py) on
held-out drone clips, held-out background clips and an ESC-50 sample, and recommends the config that detects
the most drone clips subject to: no background clip with any detection, and ESC-50 DETECTED frames <=
--max-esc-fa (default 0.5 % fused, 3 % rules-only). Ties: lower ESC false-alarm rate, then earlier median first
detection, then fewer changes from the current config.

Only constants that DroneDetector reads at call time (or in __init__, since every file gets a fresh Pipeline) can be
swept; they are set as droneears.config.<NAME> before each Pipeline is built. Constants consumed at import time
(the comb tables from F0_MIN/F0_MAX/F0_STEP/HARM_FMIN/N_HARMONICS, BASELINE_BINS and DET_BAND as default
arguments, the frame geometry) are refused.

Speed: the per-frame evidence (comb score, flatness, harmonic count, f0 steadiness, classifier outputs) does not
depend on the thresholds, and nothing upstream of the state machine reads the DETECTED state except the noise
floor, which only feeds 'strength'. So for threshold-only grids the pipeline runs once per file and the
frame_ok / persistence / release / ML-gate logic is replayed per config. The replay is checked against the real
pipeline's frame_ok, hits and detected on every file at the current config; any mismatch (or a grid over a
constant that changes the evidence itself, e.g. F0_MAX_JUMP) falls back to a full Pipeline rerun per config,
parallelised over configs with multiprocessing (fork).

Usage:
  python scripts/retune_holdout.py                                    # defaults below, classifier fused
  python scripts/retune_holdout.py --no-ml --grid HARM_THRESH_DB=5,6 --grid PERSIST_ON=60,70
  python scripts/retune_holdout.py --apply                            # write the recommendation into config.py
"""
import argparse, collections, difflib, glob, itertools, multiprocessing as mp, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from droneears import config as C
from droneears.audio import load
from droneears.pipeline import Pipeline

# read by DroneDetector.feed / set_ml at call time and only applied to per-frame evidence -> replayable from cache
REPLAYABLE = {"HARM_THRESH_DB", "FLAT_MIN", "FLAT_THRESH", "F0_STEADY_FRAC", "MIN_HARMONICS", "PERSIST_ON",
              "PERSIST_OFF", "RELEASE_SEC", "ML_THRESH", "ML_HOLD_FRAMES", "PERSIST_FRAMES", "FLAT_MODE",
              "FLAT_LOCAL_MIN", "FLAT_LOCAL_MAX", "GATE_ON", "GATE_THRESH"}
# read at call time / in __init__ but they change the evidence itself -> need a full rerun per config
RERUN = {"F0_MAX_JUMP", "F0_HYST_DB", "PROM_EMA", "F0_STEADY_MIN", "F0_HIST_FRAMES"}
# read at call time but cannot change DETECTED (noise floor feeds strength / snr only)
NO_EFFECT = {"FLOOR_SMOOTH", "FLOOR_BLOCK_FRAMES", "FLOOR_N_BLOCKS", "FLOOR_BIAS"}
# baked into module-level tables or default arguments when droneears.detector / features is imported
IMPORT_TIME = {"FS", "NFFT", "HOP", "NBINS", "BIN_HZ", "FRAMES_PER_SEC", "F0_MIN", "F0_MAX", "F0_STEP",
               "N_HARMONICS", "HARM_FMIN", "BASELINE_BINS", "DET_BAND", "NMEL", "MEL_FMIN", "MEL_FMAX",
               "CLS_WINDOW_FRAMES", "CLS_DB_FLOOR"}
DEFAULT_GRID = collections.OrderedDict([("HARM_THRESH_DB", [4.5, 5.5, 6.5, 7.5]),
                                        (("FLAT_LOCAL_MIN", [0.2, 0.25, 0.3, 0.35]) if C.FLAT_MODE == "local" else
                                         ("FLAT_MIN", [0.10, 0.14, 0.17, 0.20, 0.24])),
                                        ("PERSIST_ON", [60, 70, 80])])
DEFAULT_MODEL = os.path.join(ROOT, "artifacts", "drone_cls_int8.tflite")

AUDIO = []          # [(kind, name, y)] filled in the parent before forking workers (shared copy-on-write)
_CLF = None


# ---------------------------------------------------------------- inputs
def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p if os.path.isabs(p) else os.path.join(ROOT, p))) or sorted(glob.glob(p))
        if not hits and os.path.isfile(p):
            hits = [p]
        out += hits
    return sorted(set(os.path.abspath(h) for h in out))


def load_clip(path):
    """Preserve mono, legacy 3-channel and current hardware recordings; otherwise use channel 0."""
    y = load(path, mono=False)
    return y if y.shape[0] in (1, 3, C.N_MICS) else y[:1]


def esc_sample(per_class, seed):
    from droneears.dataset import esc50_clips
    clips = esc50_clips({1, 2, 3, 4, 5})
    by = collections.defaultdict(list)
    for path, cat in sorted(clips):
        by[cat].append(path)
    rng = np.random.default_rng(seed)
    return [p for cat in sorted(by) for p in rng.permutation(by[cat])[:per_class]]


def parse_grid(items):
    grid = collections.OrderedDict(DEFAULT_GRID)
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--grid wants NAME=v1,v2,...: {it!r}")
        name, vals = it.split("=", 1)
        name = name.strip().upper()
        if name in IMPORT_TIME:
            sys.exit(f"{name} is consumed when droneears.detector/features is imported (comb tables or default "
                     f"arguments); setting droneears.config.{name} at run time would have no effect. Not sweepable.")
        if name in NO_EFFECT:
            sys.exit(f"{name} only feeds the noise floor (strength / snr), never DETECTED; sweeping it is pointless.")
        if name not in REPLAYABLE | RERUN:
            sys.exit(f"{name}: not a known runtime detector constant (sweepable: "
                     f"{', '.join(sorted(REPLAYABLE | RERUN))})")
        cur = getattr(C, name)
        conv = int if isinstance(cur, (int, np.integer)) and not isinstance(cur, bool) else float
        grid[name] = [conv(v) for v in vals.split(",") if v.strip()]
    return grid


# ---------------------------------------------------------------- workers
def _init_worker(model, use_ml):
    global _CLF
    _CLF = None
    if use_ml:
        import warnings
        warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")
        from droneears.classifier import load_tflite_classifier
        _CLF = load_tflite_classifier(model)


def _run(y, cfg, record=False):
    """One fresh Pipeline over y with cfg applied. Returns per-frame arrays (record) or a compact summary."""
    old = {k: getattr(C, k) for k in cfg}
    try:
        for k, v in cfg.items():
            setattr(C, k, v)
        updates = []
        pipe = None
        clf = None
        if _CLF is not None:
            def clf(win):
                p = _CLF(win)
                updates.append((pipe.frame_i, p))
                return p
        pipe = Pipeline(classifier=clf)
        res = pipe.run(y)
    finally:
        for k, v in old.items():
            setattr(C, k, v)
    det = np.array([r["detected"] for r in res], bool)
    hits = np.array([r["hits"] for r in res], int)
    if not record:
        return dict(n=len(res), det=int(det.sum()), first=int(np.argmax(det)) if det.any() else None,
                    maxhits=int(hits.max()) if len(hits) else 0)
    g = lambda k, dt: np.array([r[k] for r in res], dt)
    return dict(score=g("score", np.float64), flat=g("flat", np.float64), flat_local=g("flat_local", np.float64),
                gate=g("gate", np.float64), n_harm=g("n_harm", int),
                steady=g("f0_steady", np.float64), frame_ok=g("frame_ok", bool), hits=hits, detected=det,
                updates=updates)


def _evidence_task(i):
    kind, name, y = AUDIO[i]
    return i, _run(y, {}, record=True)


def _config_task(args):
    ci, cfg = args
    return ci, [_run(y, cfg) for _, _, y in AUDIO]


# ---------------------------------------------------------------- replay
def replay(ev, cfg, use_ml):
    """Re-run the DroneDetector decision logic on cached evidence with the constants in cfg (others: current)."""
    g = lambda k: cfg.get(k, getattr(C, k))
    if g("FLAT_MODE") == "local":                 # tonality test on local flatness (config.py, interp/README.md)
        tonal = (g("FLAT_LOCAL_MIN") < ev["flat_local"]) & (ev["flat_local"] < g("FLAT_LOCAL_MAX"))
    else:
        tonal = (g("FLAT_MIN") < ev["flat"]) & (ev["flat"] < g("FLAT_THRESH"))
    ok = ((ev["score"] > g("HARM_THRESH_DB")) & tonal
          & (ev["steady"] >= g("F0_STEADY_FRAC")) & (ev["n_harm"] >= g("MIN_HARMONICS")))
    gate_ok = ev["gate"] >= g("GATE_THRESH") if g("GATE_ON") else np.ones(len(ok), bool)   # DSP gate at onset
    n = len(ok); pf = int(g("PERSIST_FRAMES"))
    cs = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(n)
    hits = cs[idx + 1] - cs[np.maximum(idx + 1 - pf, 0)]
    if use_ml:
        thr, hold = g("ML_THRESH"), int(g("ML_HOLD_FRAMES"))
        prob = np.zeros(n); last_q = np.full(n, -10 ** 9)
        for f, p in ev["updates"]:                 # updates are in frame order
            prob[f:] = p
            if p > thr:
                last_q[f:] = f
        ml_ok = (prob > thr) | (idx - last_q < hold)
    else:
        ml_ok = np.ones(n, bool)
    on, off, rel_n = int(g("PERSIST_ON")), int(g("PERSIST_OFF")), int(g("RELEASE_SEC") * C.FRAMES_PER_SEC)
    det = np.zeros(n, bool); d = False; rel = 0
    hl = hits.tolist(); ml = ml_ok.tolist(); gl = gate_ok.tolist()
    for i in range(n):
        h = hl[i]
        if not d:
            if h >= on and ml[i] and gl[i]:
                d = True; rel = 0
        elif h <= off:
            rel += 1
            if rel >= rel_n:
                d = False
        else:
            rel = 0
        det[i] = d
    return ok, hits, det


def summary_from_det(det, hits):
    return dict(n=len(det), det=int(det.sum()), first=int(np.argmax(det)) if det.any() else None,
                maxhits=int(hits.max()) if len(hits) else 0)


# ---------------------------------------------------------------- report
def score_config(per_file, kinds):
    out = {}
    D = [s for s, k in zip(per_file, kinds) if k == "drone"]
    B = [s for s, k in zip(per_file, kinds) if k == "bg"]
    E = [s for s, k in zip(per_file, kinds) if k == "esc"]
    firsts = [s["first"] * C.HOP / C.FS for s in D if s["first"] is not None]
    out["drone_n"] = sum(s["det"] > 0 for s in D); out["drone_N"] = len(D)
    out["first"] = float(np.median(firsts)) if firsts else None
    out["drone_minhits"] = min((s["maxhits"] for s in D), default=0)
    out["bg_fa"] = sum(s["det"] for s in B) / max(1, sum(s["n"] for s in B))
    out["bg_clips"] = sum(s["det"] > 0 for s in B); out["bg_N"] = len(B)
    out["bg_maxhits"] = max((s["maxhits"] for s in B), default=0)
    out["esc_fa"] = sum(s["det"] for s in E) / max(1, sum(s["n"] for s in E)) if E else None
    out["esc_clips"] = sum(s["det"] > 0 for s in E); out["esc_N"] = len(E)
    return out


def rewrite_config(path, values):
    src = open(path).read()
    new = src
    for name, v in values.items():
        pat = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]*?)(\s*)(#.*)?$", re.M)
        m = pat.search(new)
        if not m:
            print(f"  warning: no '{name} = ...' line in {path}; not changed"); continue
        old_tok = m.group(2)
        tok = repr(int(v)) if isinstance(v, (int, np.integer)) else repr(float(v))
        if old_tok.strip() == tok:
            continue
        gap = m.group(3)
        if m.group(4):                              # keep the comment column where possible
            width = len(old_tok) + len(gap)
            gap = " " * max(1, width - len(tok))
        new = new[: m.start()] + m.group(1) + tok + gap + (m.group(4) or "") + new[m.end():]
    return src, new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drone", nargs="+", default=["data/team/drone/*_holdout*"],
                    help="held-out drone clips (globs/paths; default data/team/drone/*_holdout*)")
    ap.add_argument("--bg", nargs="+", default=["data/team/background/*_holdout*", "data/non_drone/*.m4a"],
                    help="held-out background clips (default data/team/background/*_holdout* "
                         "data/non_drone/*.m4a)")
    ap.add_argument("--esc-per-class", type=int, default=8, help="ESC-50 clips per category, all folds (0: off)")
    ap.add_argument("--seed", type=int, default=0, help="ESC-50 sample seed")
    ap.add_argument("--no-ml", action="store_true", help="rules only (default: fused with the classifier)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="TFLite classifier (default artifacts/drone_cls_int8.tflite)")
    ap.add_argument("--grid", action="append", metavar="NAME=v1,v2,...",
                    help="override / add a grid axis (repeatable). Default: HARM_THRESH_DB=4.5,5.5,6.5,7.5 "
                         "FLAT_MIN=0.10,0.14,0.17,0.20,0.24 PERSIST_ON=60,70,80")
    ap.add_argument("--max-esc-fa", type=float, default=None,
                    help="max ESC-50 DETECTED frame fraction, percent (default 0.5 fused / 3 rules-only)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="worker processes")
    ap.add_argument("--full", action="store_true", help="skip the cache and rerun the Pipeline for every config")
    ap.add_argument("--top", type=int, default=0, help="print only the best N rows (default: all)")
    ap.add_argument("--apply", action="store_true", help="write the recommended values into --config and run "
                    "scripts/export_c_headers.py (export only when --config is the repo's config.py)")
    ap.add_argument("--config", default=os.path.join(ROOT, "droneears", "config.py"),
                    help="config file --apply rewrites (default droneears/config.py). The sweep itself always "
                         "uses the imported droneears.config as its base")
    args = ap.parse_args()

    use_ml = not args.no_ml
    if use_ml and not os.path.exists(args.model):
        sys.exit(f"classifier not found: {args.model} (pass --no-ml)")
    max_esc = (args.max_esc_fa if args.max_esc_fa is not None else (0.5 if use_ml else 3.0)) / 100.0
    grid = parse_grid(args.grid)
    names = list(grid)
    configs = [dict(zip(names, vals)) for vals in itertools.product(*grid.values())]
    current = {k: getattr(C, k) for k in names}

    drone = expand(args.drone); bg = expand(args.bg)
    esc = esc_sample(args.esc_per_class, args.seed) if args.esc_per_class > 0 else []
    if not drone:
        sys.exit(f"no drone clips match {args.drone}")
    files = [("drone", p) for p in drone] + [("bg", p) for p in bg] + [("esc", p) for p in esc]
    print(f"drone {len(drone)} clips, background {len(bg)}, ESC-50 {len(esc)} ({args.esc_per_class}/class, "
          f"seed {args.seed}); classifier {'fused: ' + os.path.relpath(args.model, ROOT) if use_ml else 'off'}; "
          f"{len(configs)} configs over {', '.join(f'{k}={grid[k]}' for k in names)}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:          # ffmpeg decodes are subprocesses: threads suffice
        ys = list(ex.map(load_clip, [p for _, p in files]))
    AUDIO[:] = [(k, os.path.basename(p), y) for (k, p), y in zip(files, ys)]
    kinds = [k for k, _, _ in AUDIO]
    secs = sum(y.shape[1] for _, _, y in AUDIO) / C.FS
    print(f"decoded {secs:.0f} s of audio in {time.time() - t0:.1f} s")

    need_full = args.full or any(k in RERUN for k in names)
    ctx = mp.get_context("fork")
    per_cfg = None
    if not need_full:
        t1 = time.time()
        print(f"pass 1: pipeline once per file at the current config ({args.jobs} workers) ...", flush=True)
        evs = [None] * len(AUDIO)
        with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(args.model, use_ml)) as pool:
            for k, (i, ev) in enumerate(pool.imap_unordered(_evidence_task, range(len(AUDIO))), 1):
                evs[i] = ev
                if k % 50 == 0 or k == len(AUDIO):
                    el = time.time() - t1
                    print(f"  {k}/{len(AUDIO)} files [{el:.0f} s, eta {el / k * (len(AUDIO) - k):.0f} s]", flush=True)
        bad = []
        for (kind, name, _), ev in zip(AUDIO, evs):
            ok, hits, det = replay(ev, {}, use_ml)
            if not (np.array_equal(ok, ev["frame_ok"]) and np.array_equal(hits, ev["hits"])
                    and np.array_equal(det, ev["detected"])):
                bad.append(name)
        if bad:
            print(f"  replay check FAILED on {len(bad)} file(s) ({', '.join(bad[:5])}...): the detector logic "
                  "differs from this script's replay; falling back to a full rerun per config")
            need_full = True
        else:
            print(f"  replay check: frame_ok, hits and detected identical to the pipeline on all {len(AUDIO)} files")
            per_cfg = []
            for cfg in configs:
                row = []
                for ev in evs:
                    _, hits, det = replay(ev, cfg, use_ml)
                    row.append(summary_from_det(det, hits))
                per_cfg.append(row)
            print(f"pass 2: replayed {len(configs)} configs in {time.time() - t1:.1f} s total")
    if need_full:
        t1 = time.time()
        est = secs / 100.0 * len(configs) / args.jobs
        print(f"full rerun: {len(configs)} configs x {secs:.0f} s audio, ~{est:.0f} s estimated on {args.jobs} "
              "workers ...", flush=True)
        per_cfg = [None] * len(configs)
        with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(args.model, use_ml)) as pool:
            for k, (ci, s) in enumerate(pool.imap_unordered(_config_task, list(enumerate(configs))), 1):
                per_cfg[ci] = s
                el = time.time() - t1
                print(f"  {k}/{len(configs)} configs [{el:.0f} s, eta {el / k * (len(configs) - k):.0f} s]", flush=True)

    rows = [(cfg, score_config(s, kinds)) for cfg, s in zip(configs, per_cfg)]
    def feasible(m):
        return m["bg_clips"] == 0 and (m["esc_fa"] is None or m["esc_fa"] <= max_esc)
    def key(r):
        cfg, m = r
        return (not feasible(m), -m["drone_n"], m["esc_fa"] or 0.0,
                m["first"] if m["first"] is not None else 1e9, sum(cfg[k] != current[k] for k in names))
    rows.sort(key=key)
    fmt = lambda v: f"{v:g}"
    pct = lambda v: "-" if v is None else f"{100 * v:.3f}%"
    sec1 = lambda v: "-" if v is None else f"{v:.1f}"
    hdr = (" | ".join(f"{k:>{max(len(k), 6)}s}" for k in names) +
           " | drone det | med 1st s | bg frames | bg clips | ESC frames | ESC clips | hits drone-min/bg-max")
    print("\n  " + hdr); print("  " + "-" * len(hdr))
    for cfg, m in rows[: args.top or None]:
        tag = ("*" if cfg == current else " ") + ("" if feasible(m) else "x")
        print(f"{tag:2s}" + " | ".join(f"{fmt(cfg[k]):>{max(len(k), 6)}s}" for k in names) +
              f" | {m['drone_n']:>4d}/{m['drone_N']:<4d} | {sec1(m['first']):>9s}"
              f" | {100 * m['bg_fa']:8.2f}% | {m['bg_clips']:>3d}/{m['bg_N']:<4d}"
              f" | {pct(m['esc_fa']):>10s}"
              f" | {m['esc_clips']:>3d}/{m['esc_N']:<5d} | {m['drone_minhits']:>3d}/{m['bg_maxhits']:<3d}")
    print("  (* = current config, x = infeasible: a background clip detects or ESC frames > "
          f"{100 * max_esc:g} %)")
    best_cfg, best = rows[0]
    if not feasible(best):
        print("\nno config satisfies bg clips == 0 and ESC frames <= "
              f"{100 * max_esc:g} %; nothing recommended (the first row is the least bad by drone count).")
        return
    cur_m = dict((tuple(sorted(c.items())), m) for c, m in rows).get(tuple(sorted(current.items())))
    print("\nrecommended: " + ", ".join(f"{k} = {fmt(best_cfg[k])}" for k in names) +
          f"  -> drone {best['drone_n']}/{best['drone_N']}, bg clips {best['bg_clips']}/{best['bg_N']}, "
          f"ESC frames {pct(best['esc_fa'])}"
          + ("  (same as current config)" if best_cfg == current else ""))
    if cur_m is not None and best_cfg != current:
        print("current:     " + ", ".join(f"{k} = {fmt(current[k])}" for k in names) +
              f"  -> drone {cur_m['drone_n']}/{cur_m['drone_N']}, bg clips {cur_m['bg_clips']}, ESC frames "
              f"{pct(cur_m['esc_fa'])}")
    if args.apply:
        path = os.path.abspath(args.config)
        old, new = rewrite_config(path, best_cfg)
        if old == new:
            print(f"--apply: {path} already has these values")
        else:
            open(path, "w").write(new)
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), path, path))
        if path == os.path.abspath(os.path.join(ROOT, "droneears", "config.py")):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_c_headers.py")], check=True)
        else:
            print("--apply: --config is not the repo's droneears/config.py; skipped scripts/export_c_headers.py")


if __name__ == "__main__":
    main()
