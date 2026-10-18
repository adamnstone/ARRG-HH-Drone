# Drone detection + direction estimation with Hugging Face audio

Run from your project's directory; both scripts read JSON config files **next to the scripts**.

## 1. Setup

1. Keep your existing `data/drone` and `data/non_drone` folders (M4A/WAV/FLAC/MP3).
2. Install Python requirements: `python -m pip install -r requirements.txt`.
3. Install FFmpeg and ensure `ffmpeg -version` works.
4. Edit `config.json` and `direction_config.json` for your own paths.
5. Keep `huggingface.enabled` true. First run downloads about **244 MB** of public audio (requires internet); subsequent runs reuse local files.

Hugging Face source: https://huggingface.co/datasets/imm61/itu-aris-lab-acoustic-drone-dataset
Published label mapping: `audio/drone` = drone; `audio/background` = non-drone.
License: CC BY 4.0. Attribute the source when presenting your results.

The public dataset contains 5,491 one-second clips recorded in one extended session and its clips overlap by 50%. The pipeline uses the publisher's `splits/split_interval_disjoint.json` mapping to prevent adjacent overlapping clips from being divided across training/validation/test. The resulting public test scores are still *not* independent microphone sessions. We do not mix train/test audio across splits.

## 2. Detection

```
python train_compare.py
```

The script performs TWO sets of model runs, three architectures each (six trainings):

- `results/`: **original own-data-only** experiment (preserves all prior outputs and charts)
- `results/combined_with_huggingface/`: **combined own + Hugging Face** experiment

The combined experiment evaluates **window-level** predictions separately on:

- `own_test`: only your original held-out audio windows
- `huggingface_test`: only official held-out public audio windows
- `combined_test`: both together

Outputs include per-source confusion matrices, per-window CSVs, `comparison_by_source.csv`, `comparison_by_source.json`, checkpoints, loss charts and ROC curves. Test thresholds are selected only on validation windows.

To use all public windows set:

```
"max_files_per_class_per_split": null
```

For a smaller CPU-speed smoke run, set e.g. `50`. Cap sampling is deterministic within each original split and does not mix split membership. This is only for quick debugging; train on all for final plots.

## 3. Triangulation simulation

```
python triangulation.py
```

Two simulations are run, without training:

- `results_direction/`: original local-audio simulation; honors existing `max_source_windows`
- `results_direction/huggingface/`: independent simulation using **all** Hugging Face drone clips, **one synthetic direction per clip** by default; each direction is tested at every configured SNR

Both save CSV/JSON, error-vs-SNR plots and two animated GIFs. Public recordings are monophonic, so **three synchronized virtual microphone channels are simulated** by shifting copies and adding independent Gaussian noise. These are *not* measured physical-array directions. The GIF's radius is illustrative; the algorithm does not estimate distance. The moving-path GIF uses a fixed drone audio clip as the source while changing its synthetic direction.

The public triangulation simulation uses every drone-labeled file in the publisher's official split mapping regardless of train/validation/test affiliation; this is safe because triangulation has **no learned model**.

When your three synchronized INMP441 channels are available, a separate real-array input path and calibration will be required for measured DOA evaluation.

## 4. Important cautions

- Your original single long non-drone recording triggers temporal-block splitting. Do not report its held-out one-second windows as independent recording sessions.
- The public clips overlap by 50% and come from one recording session; published interval-disjoint partitions reduce overlap leakage, but do not establish external-environment generalization.
- For credible final evaluation collect independent microphone sessions from both classes, preferably on the INMP441 setup.
- Full public dataset training and simulation can be slow on CPU. Disable GIF rendering during large batch tests with `"make_gifs": false` in `direction_config.json`.
