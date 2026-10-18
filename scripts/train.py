"""Train the drone/background classifier, evaluate, export float + int8 TFLite and a C array for the ESP32-S3.

Usage:  python scripts/train.py [--epochs 25] [--n-synth 120] [--n-dads-pos 200] [--n-dads-neg 100] [--esc-per-fold 200] [--out artifacts]
"""
import argparse, json, os, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from droneears import config as C
from droneears.dataset import build

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--n-synth", type=int, default=120)
ap.add_argument("--esc-per-fold", type=int, default=None)
ap.add_argument("--n-dads-pos", type=int, default=200, help="DADS 2.5 s drone segments for training (test uses 1/4); 0 disables")
ap.add_argument("--n-dads-neg", type=int, default=100, help="DADS no-drone clips for training (test uses 1/4); 0 disables")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "artifacts"))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import tensorflow as tf
tf.random.set_seed(args.seed); np.random.seed(args.seed)

t0 = time.time()
Xtr, ytr, gtr = build("train", n_synth=args.n_synth, esc_per_fold=args.esc_per_fold, seed=args.seed,
                      n_dads_pos=args.n_dads_pos, n_dads_neg=args.n_dads_neg)
Xte, yte, gte = build("test", n_synth=max(20, args.n_synth // 5), esc_per_fold=args.esc_per_fold, seed=args.seed,
                      n_dads_pos=args.n_dads_pos // 4, n_dads_neg=args.n_dads_neg // 4)
print(f"data built in {time.time() - t0:.0f}s")
np.savez_compressed(os.path.join(args.out, "features_test.npz"), X=Xte, y=yte, g=gte)


def make_model():
    """~12k parameters. Input (47 frames, 40 mel, 1). Designed for TFLite Micro: conv2d / maxpool / dense only."""
    inp = tf.keras.Input(shape=(C.CLS_WINDOW_FRAMES, C.NMEL, 1))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="drone_cls")


model = make_model()
model.summary()
pos_w = (1 - ytr).sum() / max(1, ytr.sum())
class_weight = {0: 1.0, 1: float(min(pos_w, 5.0))}
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# simple SpecAugment-style masking as an augmentation
def augment(X):
    X = X.copy()
    rng = np.random.default_rng(1)
    for i in range(len(X)):
        if rng.random() < 0.5:
            f0 = rng.integers(0, C.NMEL - 6); X[i, :, f0:f0 + rng.integers(1, 6)] = 0
        if rng.random() < 0.5:
            t0_ = rng.integers(0, C.CLS_WINDOW_FRAMES - 8); X[i, t0_:t0_ + rng.integers(1, 8), :] = 0
    return X
Xtr_aug = np.concatenate([Xtr, augment(Xtr)]); ytr_aug = np.concatenate([ytr, ytr])
# validation_split takes the *tail* of the array. The pool is ordered backgrounds-then-positives, so with the
# full ESC-50 the tail would be all masked positives (and training would never see one): shuffle first.
perm = np.random.default_rng(args.seed).permutation(len(Xtr_aug))
Xtr_aug, ytr_aug = Xtr_aug[perm], ytr_aug[perm]
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
hist = model.fit(Xtr_aug[..., None], ytr_aug, validation_split=0.1, epochs=args.epochs, batch_size=64,
                 class_weight=class_weight, callbacks=cb, verbose=2)

# ---------------------------------------------------------------- evaluation
def report(name, probs, y, groups):
    pred = (probs > 0.5).astype(int)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    print(f"\n{name}: acc {acc:.3f}  recall {tp / max(1, tp + fn):.3f}  false-positive rate {fp / max(1, fp + tn):.3f}  (TP {tp} FN {fn} FP {fp} TN {tn})")
    per = collections.defaultdict(lambda: [0, 0])
    for p_, y_, g_ in zip(pred, y, groups):
        per[g_][0] += int(p_ != y_); per[g_][1] += 1
    worst = sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:8]
    print("  worst groups (error rate, n):", ", ".join(f"{g} {e / n:.2f} ({n})" for g, (e, n) in worst if e))
    return dict(acc=acc, recall=tp / max(1, tp + fn), fpr=fp / max(1, fp + tn), tp=tp, fn=fn, fp=fp, tn=tn)

probs = model.predict(Xte[..., None], verbose=0)[:, 1]
res_float = report("float model, test split", probs, yte, gte)

# ---------------------------------------------------------------- export
model.save(os.path.join(args.out, "drone_cls.keras"))
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl_float = conv.convert()
open(os.path.join(args.out, "drone_cls_float.tflite"), "wb").write(tfl_float)

def rep():
    idx = np.random.default_rng(0).choice(len(Xtr), min(300, len(Xtr)), replace=False)
    for i in idx:
        yield [Xtr[i][None, ..., None].astype(np.float32)]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl_int8 = conv.convert()
p8 = os.path.join(args.out, "drone_cls_int8.tflite")
open(p8, "wb").write(tfl_int8)
print(f"\nTFLite float {len(tfl_float) / 1024:.1f} KB, int8 {len(tfl_int8) / 1024:.1f} KB")

# int8 accuracy via the TFLite interpreter
interp = tf.lite.Interpreter(model_content=tfl_int8)
interp.allocate_tensors()
inp = interp.get_input_details()[0]; outp = interp.get_output_details()[0]
s_in, z_in = inp["quantization"]; s_out, z_out = outp["quantization"]
probs8 = []
t = time.time()
for i in range(len(Xte)):
    q = np.clip(np.round(Xte[i][None, ..., None] / s_in + z_in), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q); interp.invoke()
    o = interp.get_tensor(outp["index"]).astype(np.float32)
    probs8.append(float((o[0, 1] - z_out) * s_out))
probs8 = np.array(probs8)
res_int8 = report("int8 model, test split", probs8, yte, gte)
print(f"int8 interpreter: {(time.time() - t) / len(Xte) * 1000:.2f} ms/window on this CPU")

# C array for the firmware
with open(os.path.join(args.out, "model_data.h"), "w") as f:
    f.write("// Auto-generated by scripts/train.py. int8 TFLite model: input (1,47,40,1) int8, output (1,2) int8.\n")
    f.write(f"// input quant: scale {s_in:.8g} zero {z_in};  output quant: scale {s_out:.8g} zero {z_out}\n")
    f.write("#pragma once\n#include <stdint.h>\n")
    f.write(f"#define MODEL_IN_SCALE {s_in:.8g}f\n#define MODEL_IN_ZERO {z_in}\n#define MODEL_OUT_SCALE {s_out:.8g}f\n#define MODEL_OUT_ZERO {z_out}\n")
    f.write(f"const unsigned int drone_cls_int8_tflite_len = {len(tfl_int8)};\n")
    f.write("alignas(16) const unsigned char drone_cls_int8_tflite[] = {\n")
    for i in range(0, len(tfl_int8), 16):
        f.write("  " + ", ".join(f"0x{b:02x}" for b in tfl_int8[i:i + 16]) + ",\n")
    f.write("};\n")

json.dump(dict(train_windows=int(len(ytr)), test_windows=int(len(yte)), float=res_float, int8=res_int8,
               tflite_float_bytes=len(tfl_float), tflite_int8_bytes=len(tfl_int8), params=model.count_params(),
               epochs_run=len(hist.history["loss"]), in_quant=[float(s_in), int(z_in)], out_quant=[float(s_out), int(z_out)]),
          open(os.path.join(args.out, "train_report.json"), "w"), indent=1)
print("\nwrote", args.out)
