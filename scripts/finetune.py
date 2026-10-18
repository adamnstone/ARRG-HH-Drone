"""Fine-tune the shipped classifier on new board recordings (TRAINING_PLAN.md section 4, without a full retrain).

Starts from artifacts/drone_cls.keras and continues training at a low learning rate on
  * the new clips (default: data/team/{drone,background}/device_*.wav, the board's own 4-channel recordings),
    first 70 % of each clip in time; the last 30 % is held out (clips named *_holdout* are held out whole),
  * the original training set (droneears.dataset.build("train"), same defaults as scripts/train.py) as replay,
    so the model does not forget DADS / ESC-50 / the phone clips.
New-clip features match the firmware: the classifier sees the MEAN POWER of the mic channels
(droneears.pipeline.Pipeline.step), not a waveform downmix. Training adds each single channel and the new drone
audio mixed into the new ambient and into ESC-50 at -6..+18 dB, and oversamples the new windows.

Reports, for the shipped model and the fine-tuned one (float and int8): window-level p(drone) on the held-out part
of every new clip, the original test split (artifacts/features_test.npz), and the fused pipeline (rules AND
classifier) over each held-out segment. Writes artifacts/finetune/ (keras, int8 tflite, model_data.h, report);
the shipped model in artifacts/ is not touched. Copy by hand once you are happy with the numbers.

Usage:  python scripts/finetune.py [--epochs 10] [--lr 1e-4] [--oversample 8] [--clips 'data/team/*/device_*.wav']
        python scripts/finetune.py --eval-only          # baseline numbers for the shipped model only
"""
import argparse, glob, json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from droneears import config as C
from droneears.audio import load
from droneears.features import frames, power, classifier_windows
from droneears.pipeline import Pipeline, summarise

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=[os.path.join(ROOT, "data", "team", "drone", "device_*"),
                                               os.path.join(ROOT, "data", "team", "background", "device_*")])
ap.add_argument("--base", default=os.path.join(ROOT, "artifacts", "drone_cls.keras"))
ap.add_argument("--base-int8", default=os.path.join(ROOT, "artifacts", "drone_cls_int8.tflite"))
ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "finetune"))
ap.add_argument("--cache", default=os.path.join(ROOT, "data", "finetune_replay_cache.npz"),
                help="replay set cache (the original training windows); delete to rebuild")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--oversample", type=int, default=8, help="repeat the new-clip training windows this many times")
ap.add_argument("--train-frac", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval-only", action="store_true")
args = ap.parse_args()

import tensorflow as tf
import warnings; warnings.filterwarnings("ignore")
tf.random.set_seed(args.seed); np.random.seed(args.seed)
rng = np.random.default_rng(args.seed)


# ---------------------------------------------------------------- new clips
def label_of(path):
    parts = os.path.normpath(path).split(os.sep)
    if "drone" in parts: return 1
    if "background" in parts or "ambient" in parts: return 0
    sys.exit(f"{path}: put it under .../drone/ or .../background/ so the label is known")

def win_mean_power(y):
    """(ch, n) -> classifier windows from the mean of per-channel power, as the firmware / Pipeline computes it."""
    P = np.mean([power(np.fft.rfft(frames(ch), axis=1)) for ch in y], axis=0)
    return classifier_windows(P)

def load_ch(path):
    y = load(path, mono=False)
    return y if y.shape[0] in (1, 3, C.N_MICS) else y[:1]

clips = []
for pat in args.clips:
    for p in sorted(glob.glob(pat)):
        y = load_ch(p)
        lab = label_of(p)
        if "_holdout" in os.path.basename(p):
            tr, te = y[:, :0], y
        else:
            cut = int(args.train_frac * y.shape[1])
            tr, te = y[:, :cut], y[:, cut:]
        clips.append(dict(path=p, name=os.path.basename(p), label=lab, y=y, tr=tr, te=te))
if not clips:
    sys.exit("no clips matched " + " ".join(args.clips))
for c in clips:
    print(f"{c['name']}: {'drone' if c['label'] else 'background'}, {c['y'].shape[0]} ch, "
          f"{c['y'].shape[1] / C.FS:.1f} s (train {c['tr'].shape[1] / C.FS:.1f} s, held out {c['te'].shape[1] / C.FS:.1f} s)")


# ---------------------------------------------------------------- evaluation helpers
def int8_fn(tfl_bytes=None, path=None):
    it = tf.lite.Interpreter(model_content=tfl_bytes) if tfl_bytes is not None else tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = inp["quantization"]; s_out, z_out = outp["quantization"]
    def predict_batch(X):
        out = []
        for w in X:
            q = np.clip(np.round(w[None, ..., None] / s_in + z_in), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], q); it.invoke()
            out.append(float((it.get_tensor(outp["index"])[0, 1].astype(np.float32) - z_out) * s_out))
        return np.array(out)
    def predict_one(w):
        return float(predict_batch(np.asarray(w, np.float32)[None])[0])
    return predict_batch, predict_one

def metrics(p, y, thr):
    pred = p > thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(acc=(tp + tn) / max(1, len(y)), recall=tp / max(1, tp + fn) if (tp + fn) else None,
                fpr=fp / max(1, fp + tn) if (fp + tn) else None, tp=tp, fn=fn, fp=fp, tn=tn, n=int(len(y)))

def fmt(m):
    s = []
    if m["recall"] is not None: s.append(f"recall {m['recall']:.1%}")
    if m["fpr"] is not None: s.append(f"FPR {m['fpr']:.1%}")
    return f"acc {m['acc']:.1%}  " + "  ".join(s) + f"  (n {m['n']})"

def evaluate(tag, batch_fn, one_fn):
    """Window-level on new held-out parts + original test split; fused pipeline over each held-out segment."""
    rep = dict(clips={}, pipeline={})
    print(f"\n=== {tag}")
    for c in clips:
        W = win_mean_power(c["te"]); p = batch_fn(W); y = np.full(len(W), c["label"])
        r = {f"p>{t}": metrics(p, y, t) for t in (0.5, 0.7)}
        r["median_p"] = float(np.median(p)) if len(p) else None
        rep["clips"][c["name"]] = r
        print(f"  held-out windows {c['name']}: median p {r['median_p']:.2f}; p>0.5 {fmt(r['p>0.5'])}; p>0.7 {fmt(r['p>0.7'])}")
        res = Pipeline(classifier=one_fn).run(c["te"]); s = summarise(res)
        rep["pipeline"][c["name"]] = s
        print(f"  fused pipeline on held-out {c['name']}: DETECTED {s['detected_frac']:.1%} of frames, first at {s['first_detect_s']}")
    T = np.load(os.path.join(ROOT, "artifacts", "features_test.npz"))
    p = batch_fn(T["X"]); r = {f"p>{t}": metrics(p, T["y"], t) for t in (0.5, 0.7)}
    g = T["g"]
    for grp in ("team:drone", "team:background", "dads:drone", "dads:background", "synthetic"):
        m = g == grp
        if m.any(): r[grp] = metrics(p[m], T["y"][m], 0.5)
    esc = np.char.startswith(g.astype(str), "esc50:")
    r["esc50"] = metrics(p[esc], T["y"][esc], 0.5)
    rep["orig_test"] = r
    print(f"  original test split: p>0.5 {fmt(r['p>0.5'])}; p>0.7 {fmt(r['p>0.7'])}; ESC-50 FPR {r['esc50']['fpr']:.1%}")
    return rep

report = {}
b8, o8 = int8_fn(path=args.base_int8)
report["base_int8"] = evaluate("shipped model (int8)", b8, o8)
if args.eval_only:
    os.makedirs(args.out, exist_ok=True)
    json.dump(report, open(os.path.join(args.out, "baseline_report.json"), "w"), indent=1, default=float)
    sys.exit(0)


# ---------------------------------------------------------------- training set
def esc_bgs(n):
    from droneears.dataset import esc50_clips
    paths = esc50_clips([1, 2, 3, 4])
    return [load(paths[i][0]) for i in rng.choice(len(paths), n, replace=False)]

def mix3(sig, bg, snr_db):
    """sig (ch, n) drone; bg (ch, m) or (m,) background; per-channel RMS mix at snr_db."""
    bg = np.atleast_2d(bg)
    n = sig.shape[1]
    if bg.shape[1] < n: bg = np.tile(bg, (1, int(np.ceil(n / bg.shape[1]))))
    s0 = rng.integers(0, bg.shape[1] - n + 1); bg = bg[:, s0:s0 + n]
    if bg.shape[0] != sig.shape[0]: bg = np.repeat(bg[:1], sig.shape[0], axis=0)
    rs = np.sqrt(np.mean(sig ** 2)) + 1e-9; rb = np.sqrt(np.mean(bg ** 2)) + 1e-9
    return (sig / rs * 10 ** (snr_db / 20) + bg / rb).astype(np.float32)

Xn, yn = [], []
new_bg = [c["tr"] for c in clips if c["label"] == 0 and c["tr"].shape[1]]
escs = esc_bgs(40)
for c in clips:
    if not c["tr"].shape[1]: continue
    views = [c["tr"]] + ([c["tr"][i:i + 1] for i in range(c["tr"].shape[0])] if c["tr"].shape[0] > 1 else [])
    for v in views:
        W = win_mean_power(v); Xn.append(W); yn.append(np.full(len(W), c["label"]))
    if c["label"] == 1:
        for k in range(6):
            bg = new_bg[rng.integers(len(new_bg))] if (new_bg and k % 2 == 0) else escs[rng.integers(len(escs))]
            W = win_mean_power(mix3(c["tr"], bg, float(rng.uniform(-6, 18))))
            Xn.append(W); yn.append(np.ones(len(W), int))
Xn = np.concatenate(Xn); yn = np.concatenate(yn)
print(f"\nnew-clip training windows: {len(yn)} (drone {int(yn.sum())}, background {int((1 - yn).sum())})")

if os.path.exists(args.cache):
    z = np.load(args.cache); Xr, yr = z["X"], z["y"]
    print(f"replay set from cache {args.cache}: {len(yr)} windows")
else:
    from droneears.dataset import build
    t0 = time.time()
    Xr, yr, _ = build("train", n_synth=60, seed=0)   # same as `train.py --n-synth 60`; includes the new clips' 70 %
    np.savez(args.cache, X=Xr.astype(np.float16), y=yr)
    print(f"replay set built in {time.time() - t0:.0f}s: {len(yr)} windows (cached to {args.cache})")
Xr = Xr.astype(np.float32)

def augment(X):
    X = X.copy(); r = np.random.default_rng(1)
    for i in range(len(X)):
        if r.random() < 0.5:
            f0 = r.integers(0, C.NMEL - 6); X[i, :, f0:f0 + r.integers(1, 6)] = 0
        if r.random() < 0.5:
            t0_ = r.integers(0, C.CLS_WINDOW_FRAMES - 8); X[i, t0_:t0_ + r.integers(1, 8), :] = 0
    return X

Xtr = np.concatenate([Xr, augment(Xr)] + [Xn, augment(Xn)] * (args.oversample // 2 or 1))
ytr = np.concatenate([yr, yr] + [yn, yn] * (args.oversample // 2 or 1))
perm = rng.permutation(len(ytr)); Xtr, ytr = Xtr[perm], ytr[perm]
pos_w = (1 - ytr).sum() / max(1, ytr.sum())

# ---------------------------------------------------------------- fine-tune
model = tf.keras.models.load_model(args.base)
model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
hist = model.fit(Xtr[..., None], ytr, validation_split=0.1, epochs=args.epochs, batch_size=64,
                 class_weight={0: 1.0, 1: float(min(pos_w, 5.0))}, callbacks=cb, verbose=2)

os.makedirs(args.out, exist_ok=True)
model.save(os.path.join(args.out, "drone_cls.keras"))
def rep_ds():
    for i in np.random.default_rng(0).choice(len(Xr), 300, replace=False):
        yield [Xr[i][None, ..., None]]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]; conv.representative_dataset = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
tfl8 = conv.convert()
open(os.path.join(args.out, "drone_cls_int8.tflite"), "wb").write(tfl8)

f8, g8 = int8_fn(tfl_bytes=tfl8)
report["finetuned_int8"] = evaluate("fine-tuned model (int8)", f8, g8)

it = tf.lite.Interpreter(model_content=tfl8); it.allocate_tensors()
(s_in, z_in), (s_out, z_out) = it.get_input_details()[0]["quantization"], it.get_output_details()[0]["quantization"]
with open(os.path.join(args.out, "model_data.h"), "w") as f:
    f.write("// Auto-generated by scripts/finetune.py. int8 TFLite model: input (1,47,40,1) int8, output (1,2) int8.\n")
    f.write(f"// input quant: scale {s_in:.8g} zero {z_in};  output quant: scale {s_out:.8g} zero {z_out}\n")
    f.write("#pragma once\n#include <stdint.h>\n")
    f.write(f"#define MODEL_IN_SCALE {s_in:.8g}f\n#define MODEL_IN_ZERO {z_in}\n#define MODEL_OUT_SCALE {s_out:.8g}f\n#define MODEL_OUT_ZERO {z_out}\n")
    f.write(f"const unsigned int drone_cls_int8_tflite_len = {len(tfl8)};\n")
    f.write("alignas(16) const unsigned char drone_cls_int8_tflite[] = {\n")
    for i in range(0, len(tfl8), 16):
        f.write("  " + ", ".join(f"0x{b:02x}" for b in tfl8[i:i + 16]) + ",\n")
    f.write("};\n")
report.update(epochs_run=len(hist.history["loss"]), lr=args.lr, oversample=args.oversample,
              new_train_windows=int(len(yn)), replay_windows=int(len(yr)), tflite_int8_bytes=len(tfl8),
              clips=[c["name"] for c in clips])
json.dump(report, open(os.path.join(args.out, "finetune_report.json"), "w"), indent=1, default=float)
print("\nwrote", args.out)
