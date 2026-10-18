# Training plan: drone / background classifier

This is the plan for the learned half of the detector. The physics half (harmonic comb search, bearing) needs no
training and is covered in `README.md`.

## 1. What the model does and does not do

The classifier answers one question twice a second: *does this one-second window contain a multirotor?* It outputs
`p(drone)`. It does **not** find direction (that is pure signal processing on the three mic channels) and it does not
decide alone: the firmware declares a detection only when the harmonic-comb detector and the classifier agree
(`fusion = harmonic persistence AND p(drone) > 0.7`). The split is deliberate: the rules give explainability and
zero-data operation, the model gives recall on textures the rules miss and rejects the tonal confusers (sirens,
bells, animals, chainsaws) that fool a comb search.

**On the board the classifier is off (`USE_TFLM 0`), so since `richard-interp` a DSP gate does its veto there**
(`GATE_*` in `droneears/config.py`): six rolling measurements chosen by probing what the classifier listens to and
what separates drones from the rules' false alarms. The probes also showed that the model relies on the drone's
fundamental in 200–400 Hz and only accepts f0 of 150–350 Hz (the f0 range of the synthetic training drones, below:
`uniform(150, 360)`), so a retrain should widen the synthetic f0 range. Details: `interp/README.md`.

Input: a 47 x 40 log-mel image (1.0 s, 23 ms hops, 40 mel bands 100-8000 Hz) computed from the same 2048-point FFT
the rest of the pipeline uses. Normalised to the window's loudest bin and clipped to -80 dB, so absolute gain
(phone mic vs the MP34DT01-M PDM mics, distance) does not matter. Model: 4 conv layers + global pool + dense, 15 698 parameters,
int8 TFLite, 24 KB flash, 0.06 ms/window on a laptop CPU and an estimated 60-150 ms on the ESP32-S3.

## 2. Data

| Source | Role | Amount now | Licence |
|---|---|---|---|
| Team recordings (`data/drone`, `data/non_drone`, plus `data/team/*`) | **real** positives / negatives, same environment as the demo | 96 s drone (9 clips) + 237 s ambient (phone), committed | ours |
| ESC-50 (Piczak) | backgrounds incl. the hard confusers: chainsaw, airplane, engine, siren, church bells, insects, crickets, wind, rain, crowd | 2000 x 5 s clips, 50 classes | CC BY-NC 3.0 |
| DADS (`data/hf-drone/`, 6 of 39 parquet shards) | **real** positives from six drone recording campaigns + urban/scene negatives | 200 x 2.46 s drone segments + 100 no-drone clips train (50 + 25 test, shard 36 / last 20 % of shard 0) | MIT (check per-source licences) |
| Synthetic quadcopters (`droneears.simulate.synth_drone`) | positives for diversity of f0, harmonic decay, motor count, RPM wander | generated on demand | ours |
| Field session (with the real mics on the headband) | real positives at 10/25/50/100 m, behind building, transits; real negatives incl. our own motors | target 15-20 min drone, 40 min background | ours |

Public drone data: DADS (*drone-audio-detection-samples*, Hugging Face, `python scripts/get_dads.py`) is used. The
dataset is MIT-licensed on Hugging Face but aggregates six drone sources and four background sources with their own
licences; its card lists Al-Emadi et al.'s *DroneAudioDataset*, whose licence excludes military/defence/security use.
This is a hackathon project, not a defence product, so that is fine; re-check the per-source licences before any
commercial use. Its drone rows are 0.5 s chunks; five consecutive chunk ids (adjacent slices of one recording) are joined with a 10 ms crossfade
into one 2.46 s segment. No-drone clips of 4.9-5.1 s are skipped because DADS's negatives include ESC-50, whose
fold 5 is our test set. A random half of all non-DADS clips is band-limited to 8 kHz like DADS audio, so
bandwidth cannot act as a label shortcut. ESC-50 is CC BY-NC, fine for a non-commercial hackathon; if the project
ever goes commercial, replace the negatives with your own recordings (you will have them by then).

Every positive clip is also **mixed** into random backgrounds at SNRs from -6 to +18 dB and sometimes low-passed
(2.5/4/6 kHz) to imitate distance, so the model sees drones *under* ambient rather than clean. SpecAugment-style
time and frequency masking is applied during training.

Splits are by source, never by window: ESC-50 folds 1-4 train / fold 5 test; team clips split 70/30 **in time**
(first 70 % train, last 30 % test); synthetic clips by seed. Windows from one recording never appear on both sides.

## 3. Procedure, now (laptop)

```
pip install -r requirements.txt
# put your recordings in place (any format ffmpeg reads):
#   data/team/drone/<name>.m4a         data/team/background/<name>.m4a
python scripts/get_dads.py                                   # once: public DADS drone audio (~1.7 GB) -> data/hf-drone/
python scripts/train.py --epochs 30 --n-synth 60             # ~4 min on a laptop CPU
python scripts/run_file.py data/team/drone/<name>.m4a         # watch the fused detector on your own clip
python scripts/demo_direction.py --drone data/team/drone/<name>.m4a --bg data/team/background/<name>.m4a
```

`train.py` prints accuracy, recall and false-positive rate on the held-out split for the float and the int8
model, the worst-performing background categories, and writes `artifacts/drone_cls_int8.tflite` plus
`artifacts/model_data.h` (the C array the firmware includes) and `artifacts/train_report.json`.

Current result (`--epochs 30 --n-synth 60`, defaults `--n-dads-pos 200 --n-dads-neg 100`, 2026-10-08). Positives:
the 9 committed phone recordings in `data/drone/` (~96 s: two fly-bys, hovering at 5 m and 10 m, two
fly-bys at 50 m height, descending, assorted motion, immediate vicinity), each split 70/30 in time; 200 DADS drone
segments; 60 synthetic clips. Training windows: 124 raw + 372 mixed `team:drone`, 612 + 1 836 `dads:drone`,
300 + 900 synthetic (4 144 positive); negatives 14 400 ESC-50 (folds 1-4), 336 `team:background` (first 70 % of
`ambient_adam.m4a`, 237 s), 1 737 `dads:background`. Test split (5 363 windows): ESC-50 fold 5, the last 30 % of
every team clip, DADS shard 36 + last 20 % of shard 0, 20 synthetic clips. int8 model, 18 epochs (early stop,
best val_loss 0.085), 24 320 bytes, 15 698 parameters, 0.06 ms/window on a laptop CPU:

| int8, test split | p>0.5 | p>0.7 | n |
|---|---|---|---|
| all: acc / recall / FPR | 94.9 % / 88.2 % / 3.2 % (TP 1058 FN 142 FP 132 TN 4031) | 95.0 % / 86.1 % / 2.5 % | 5 363 |
| `team:drone` recall | 84 % | 82 % | 44 |
| `team:drone:mix` recall | 64 % | 55 % | 132 |
| `team:background` FPR | 0 % | 0 % | 143 |
| `dads:drone` / `dads:drone:mix` recall | 92 % / 87 % | 90 % / 86 % | 156 / 468 |
| `dads:background` FPR | 13.6 % | 12.4 % | 420 |
| synthetic / synthetic:mix recall | 100 % / 95 % | 100 % / 95 % | 100 / 300 |
| ESC-50 fold 5 FPR | 2.1 % | 1.4 % (airplane, chainsaw, insects, door creaks 11-13 %) | 3 600 |

The classifier (`ml` column of `scripts/run_file.py`, rows from t = 1.5 s) on the whole real recordings (so 70 %
of each drone file was in training):

| file | dur s | median p(drone) | p10-p90 | frac p>0.7 | previous run median |
|---|---|---|---|---|---|
| Immediate vicinity | 5.3 | 0.60 | 0.11-0.99 | 0.50 | 0.02 |
| Hovering 5m ahead | 7.9 | 1.00 | 0.94-1.00 | 0.92 | 0.24 |
| Hovering 10m away | 7.1 | 0.98 | 0.91-0.99 | 1.00 | 0.99 |
| Fly by 1 | 6.5 | 0.98 | 0.54-0.99 | 0.82 | 0.87 |
| Fly by 2 | 8.4 | 0.67 | 0.12-0.96 | 0.43 | 0.85 |
| Assorted motion | 17.7 | 0.95 | 0.84-0.99 | 0.91 | 0.19 |
| Descending | 11.5 | 0.98 | 0.88-0.99 | 1.00 | 0.11 |
| Fly by 50m height 1 | 16.3 | 0.78 | 0.20-1.00 | 0.55 | 0.96 |
| Fly by 50m height 2 | 15.3 | 1.00 | 0.99-1.00 | 1.00 | 1.00 |
| ambient_adam (no drone) | 236.6 | 0.00 | 0.00-0.01 | 0.00 | 0.00 |

DADS fixed the close-range clips (immediate vicinity, hovering at 5 m, descending, assorted motion) at some cost on
Fly by 2 and the first 50 m fly-by, and on `dads:background` (13.6 % FPR on the 25-clip test sample; a fresh
60-clip sample gives 0.7 %, so this is dominated by a few engine-like clips). Caveat: the team test windows share
recordings with training (different time segments of the same files), so a whole-session holdout (`*_holdout.*`,
step 2 below) is still the real test.

The fused detector (harmonic rules AND classifier, `scripts/run_file.py` default) on the whole real recordings,
channel 0, after the rules-path retune described in `README.md` step 2:

| file | dur s | fused DETECTED % | first detection s | rules-only DETECTED % | max hits (need 70) |
|---|---|---|---|---|---|
| Immediate vicinity | 5.3 | 25.5 | 3.9 | 27.1 | 89 |
| Hovering 5m ahead | 7.9 | 61.6 | 3.0 | 61.6 | 77 |
| Hovering 10m away | 7.1 | 32.0 | 4.8 | 32.0 | 74 |
| Fly by 1 | 6.5 | 59.7 | 2.6 | 59.7 | 91 |
| Fly by 2 | 8.4 | 49.1 | 3.9 | 49.1 | 80 |
| Assorted motion | 17.7 | 49.8 | 8.9 | 49.8 | 86 |
| Descending | 11.5 | 18.7 | 9.3 | 18.7 | 89 |
| Fly by 50m height 1 | 16.3 | 81.8 | 3.0 | 81.8 | 81 |
| Fly by 50m height 2 | 15.3 | 70.8 | 4.5 | 70.8 | 89 |
| ambient_adam (no drone) | 236.6 | 0.0 | - | 0.0 | 63 |

False positives after the harmonic gate: ESC-50, all 2000 clips, 0.19 % of frames and 11 clips (chainsaw 4,
toilet flush 4, washing machine 1, door creaks 1, insects 1); rules-only 2.2 % of frames / 5.6 % of clips.
60 held-out DADS no-drone clips: fused 0 clips / 0.00 % of frames (rules-only 1 clip, 0.31 %). The detector
thresholds were tuned on these same 9 clips and the margins over the 70-hit persistence are thin (Hovering 10 m
peaks at 74, ambient at 63), so treat 9/9 as optimistic until a held-out session exists.

Note on the simulator: after this model was trained, `droneears.simulate.synth_drone` was made more realistic
(pink airflow noise, blade-rate amplitude modulation, slower harmonic decay) so that the detector's flatness floor
could be raised. A retrain with the new synthetic defaults (same command, 2026-10-08) scored worse on the held-out
split (int8 accuracy 92.9 %, recall 84.8 %, FPR 4.7 %, early-stopped at epoch 12), so the shipped model is the one
trained on the old synthetic positives. The next retrain should revisit `--n-synth` (try 0-30) or the new
`simulate.py` defaults; the fused tests in `tests/` will tell you whether the 9 clips still detect.

## 4. Procedure, after the field session (fine-tune)

1. Copy the recordings into `data/team/drone/` and `data/team/background/` (keep distance/condition/bearing in the
   file names; trim the drone files to flying segments only; leave passing cars *in* the drone files).
2. Keep one full session's files as a held-out test by naming them `*_holdout.*` (the loader puts those entirely
   in the test split).
3. Retrain from scratch: `python scripts/train.py --epochs 30 --n-synth 60`. With real positives available,
   synthetic ones are reduced to a minority so they cannot dominate.
4. Read `train_report.json`: recall on `team:drone` windows is the headline; false positives on
   `team:background` (which now include your own motors running) is the number that decides the demo.
5. Tune the two thresholds on the held-out session, not on training data: `HARM_THRESH_DB` and `ML_THRESH` in
   `droneears/config.py`. Look at the score distributions `scripts/run_file.py` prints for a drone file and a
   background file and put each threshold in the gap.
6. `python scripts/export_c_headers.py` and copy `artifacts/model_data.h` to `firmware/esp32s3/`. Rebuild.

Budget: 20 minutes of compute, an hour including the labelling.

**Shipped model since 2026-10-15: fine-tuned, not retrained.** `python scripts/finetune.py` starts from
`artifacts/drone_cls.keras` and trains on, at learning rate 1e-4, the first 70 % of each board recording in
`data/team/*/device_*` plus the original training set as replay. Inputs are built from the mean power of the
three mic channels, as the firmware computes them. Output goes to `artifacts/finetune/` and is copied over the shipped
files by hand; `--eval-only` reports the current model on new clips. It was run on the first board session (indoors,
60 s drone `device_20261015_033001_drone_indoors.wav`, 30 s ambient `device_20261015_032227_ambient_indoors1.wav`),
evaluated on the last 30 % of each clip:

| int8 | before | after |
|---|---|---|
| board drone windows, recall at p>0.7 (n 35) | 46 % (median p 0.67) | 100 % (median p 1.00) |
| board ambient windows, FPR (n 17) | 0 % | 0 % |
| original test split, acc / recall / FPR at p>0.5 | 94.9 % / 88.2 % / 3.2 % | 93.3 % / 88.7 % / 5.4 % |
| fused, all 2000 ESC-50 clips | 0.19 % of frames, 11 clips | 0.20 % of frames, 11 clips |
| fused, held-out board drone / ambient | 83 % / 0 % of frames DETECTED | 83 % / 0 % |

One room, one session, and the held-out windows come from the same recordings as the training windows, so the
100 % is optimistic. An outdoor session with `_holdout` files is the next real test. `artifacts/train_report.json`
still describes the from-scratch model the fine-tune started from; `artifacts/finetune_report.json` is this run.

## 5. What to watch for

* **Domain shift** is the failure mode. Phone recordings and MP34DT01-M (PDM mic) recordings differ in frequency response and
  noise; the normalisation removes gain but not shape. The field session through the real mics fixes this; until
  then expect the model to be optimistic.
* **Far drones** are faint and low-passed by air. The `*_100m*` recordings are the ones to check recall on.
* **Chainsaws, mowers, generators** share the comb structure. Record a lawnmower if one can be found; it is worth
  more than another hour of drone audio.
* **Your own motors** vibrate the headband. Record negatives with the motors cycling. If they trigger detections,
  the fix is mechanical (foam under the mic pods) before it is algorithmic.
* **Over-trusting accuracy numbers.** A random window split would report 99 % and mean nothing. Keep the splits by
  source.

## 6. Stretch (only after the above)

* Add an "approach/recede" output from f0 drift (Doppler) — no model needed.
* Multiclass (quad vs fixed-wing vs none) if you record a fixed-wing.
* Knowledge distillation from a larger laptop-side model into the 16 k-parameter one if recall at 100 m is the
  bottleneck.
