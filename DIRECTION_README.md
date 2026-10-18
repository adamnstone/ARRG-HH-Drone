# Four-microphone direction estimation

`sig_proc_tri.py` adapts the all-pairs GCC-PHAT algorithm from branch `rachel`
to main's noncoplanar B/L/R/T headset. Geometry is shared with the running
pipeline through `droneears/config.py`: B=(0,-0.10,0), L=(-0.09,0,0),
R=(0.09,0,0), T=(0,0,0.09), in metres. These remain placeholder dimensions;
update MIC_POS with measured hardware coordinates before evaluating recordings.

Input is synchronized audio shaped `(4, N)`, ordered B, L, R, T,
with one dominant drone source per window. Public APIs default to the
hardware rate of 48 kHz; pass `fs` for recordings at other rates. The optional
dataset CLI resamples to `SIM_FS=16000` and passes that rate explicitly. Coordinates
are x right, y forward, z up. Azimuth is 0 ahead, +90 right, ±180 behind.
The solver band-limits and PHAT-normalizes all six cross-spectra, searches
physically bounded delays with interpolation, then solves the six-by-three
least-squares system. The crown microphone measures signed elevation directly.

```python
from sig_proc_tri import estimate_direction, azimuth_from_direction, elevation_from_direction

u, confidence, grid, fit = estimate_direction(frames, fs=48000)
if u is not None:
    right, forward, up = u  # three components, with unit length
    azimuth = azimuth_from_direction(u)
    elevation = elevation_from_direction(u)
```

`estimate_direction` now returns a **3D unit direction vector**, replacing its
previous scalar azimuth return. The components are dimensionless and relative
to the headset; this far-field model does not determine drone distance or a
position in metres. A position requires a separate distance estimate plus an
origin. `estimate_direction_3d` remains an equivalent compatibility name.
Use `estimate_azimuth` for the previous horizontal-only return format.

Silence returns None and zero confidence. Directly overhead and underneath
sources retain valid 3D vectors near `[0, 0, 1]` and `[0, 0, -1]`;
`azimuth_from_direction` and `estimate_azimuth` return None near those directions.
The CLI prints the 3D vector and both angles, scores **3D angular error**, and
plots true/estimated direction arrows in 3D. The returned grid/fit are optional
horizontal projection diagnostics; the solver itself fits all three axes.

Confidence is a coherence/geometry heuristic, not a detection probability.
It is limited by the weakest usable microphone pair, with a penalty when
pairs are excluded, so clean pairs cannot conceal a low-coherence baseline.
It is not a microphone-health guarantee: unrelated noise can occasionally
produce coherent-looking peaks, particularly in short windows.
The estimator can return a low-confidence direction for noise; callers must
apply a calibrated confidence threshold. The CLI suppresses turn instructions
below `--min-confidence` (default 0.2 is a demo setting).
`estimate_direction` uses `min_magnitude` to reject a near-zero fitted vector;
`estimate_azimuth` separately uses `min_horizontal` for overhead azimuth.
`estimate_direction_details` exposes the actual fitted pair delays and peaks.
A missing/silent mic leaves insufficient 3D geometry and returns None.
Pairs with a second distinct in-bound GCC peak at least 90% as high as the
best peak are marked ambiguous and excluded from the fit. Short windows also
get an ambiguity check using only spectrally supported bins; this prevents
PHAT-amplified window leakage from hiding a competing period alias. The
original full-band curve remains the source of fitted delays. If the remaining baselines do
not span 3D, the vector is None and confidence is zero. This catches periodic
aliases that can otherwise produce a confidently reversed direction; it does
not eliminate all errors from tones, noise or reflections.
`estimate_direction_details(...).ambiguous_pairs` identifies excluded pairs.
The standalone CLI retains Rachel's optional dataset experiment:
`python3 sig_proc_tri.py --clips 20 --elevation 40` (requires datasets,
soundfile and scipy; downloads the dataset). Core estimation needs only NumPy.

`python3 direction_triangulation.py` runs the existing mono-source simulation,
now with the same four-mic geometry and solver. Configure audio paths, noise,
and trials in `direction_config.json`. It writes CSV/JSON, plots and optional
GIFs to output_dir. CSV delays use channel indices in B/L/R/T order and the
convention arrival_i minus arrival_j; these are the exact delays used by the
solver. Undefined trials have blank estimates and direction_valid=False.
Metrics count undefined trials separately and include them as failures in the
fraction within ten degrees; angular-error statistics use defined estimates. Azimuth now follows the hardware
convention above. The former inferred-positive-elevation and horizon-projection
outputs have been removed. Plotting/decoding needs matplotlib and FFmpeg; device/file tools also need FFprobe.

Run deterministic checks without downloads:
`python3 -m unittest discover -s tests -p test_triangulation.py`.

The live detector continues to use `droneears/bearing.py` and its harmonic mask;
this standalone experiment does not replace that firmware-matched path.
Both estimate far-field direction, not source position or distance. Simulated
accuracy does not establish hardware accuracy: synchronization, channel order,
geometry, echoes, wind and tonal ambiguity must be checked on actual recordings.

The latest post-review 100-clip benchmark outputs are in
`results_hf_four_mic_review/` (13,200 scenarios at 16 kHz) and
`results_hf_four_mic_48k_review/` (2,000 supplemental scenarios). Reproduce with
`python3 scripts/benchmark_four_mic.py --out results_hf_four_mic_review`.
The input cache `benchmark_results/clips.npz` is hash-verified against Rachel's
original Hugging Face manifest. The 48 kHz check resamples the same 16 kHz
source clips and tests sample-rate compatibility rather than native hardware
recordings. Its white-noise bandwidth also differs from the 16 kHz run.

The missing-microphone benchmark is `scripts/benchmark_missing_mic.py`.
For each of the same 100 hash-verified clips it removes the source from each
of B/L/R/T in turn, across five elevations and four nominal baseline SNRs.
Use `--mode silent` to set the failed channel to zero, or `--mode noise_only`
to retain only its original independent added noise. The other three channels
stay identical to the paired intact input. For the 2,048-sample hardware-sized
window, run:

```sh
python3 scripts/benchmark_missing_mic.py --mode silent --samples 2048 --out results_hf_missing_mic_silent_2048
python3 scripts/benchmark_missing_mic.py --mode noise_only --samples 2048 --out results_hf_missing_mic_noise_2048
```

Each mode tests 8,000 damaged arrays at 48 kHz. All silent-channel cases return
None with confidence zero. In the noise-only cases, 20/8,000 clear the unchanged
0.2 demo gate; ten have 3D errors above 15 degrees, including a near reversal of
an overhead direction. The gate withholds 99.75% of these faults, but does not
guarantee microphone-health detection. Three source-bearing microphones supply
only two independent delay constraints, so the general 3D direction has a mirror
ambiguity across their plane. There is no automatic three-mic fallback in this
solver. Results, trial-level diagnostics, source provenance and independent
audits are in the two output directories above. These are simulated total-source
loss tests, rather than measured head shadow or partially attenuated hardware
signals, and apply to `sig_proc_tri.py` rather than the live bearing estimator.

The separate experimental copy is **`sig_proc_tri_robust.py`**. The original
`sig_proc_tri.py` is preserved. The copy retains the original simulation, data
loader, bounded GCC-PHAT delay calculation and periodic-alias rejection, and
adds multi-segment spectral coherence and robust weighted delay fitting.
At a 2,048-sample window it uses seven overlapping 512-sample Hann segments;
coherence is calculated after averaging cross/auto spectra across segments.
Its finite-average noise bias is corrected before forming pair weights.
Each pair's weight also reflects its GCC peak and its Huber fit residual.
Period-alias checks use both PHAT and energy-weighted correlation curves;
the latter prevents weak spectral leakage from hiding competing tone peaks
in short windows. Accepted lag estimates still come from the PHAT curve.
Windows shorter than 10 ms deliberately return `unreliable`: very short Hann
windows can still hide aliases and previously produced accepted wrong vectors.
The duration floor is an engineering restriction, not proof that every longer
tonal signal is unambiguous. The 2,048-sample hardware window is about 42.7 ms.
The weighted SVD checks effective rank and conditioning after downweighting.
Output confidence also includes the weakest retained GCC peak: uniformly poor
correlation peaks must reduce confidence even when their relative weights agree.

```python
from sig_proc_tri_robust import estimate_direction_details

result = estimate_direction_details(frames, fs=48000)
if result.vector is not None and result.confidence >= 0.2:
    direction = result.vector  # [right, forward, up], unit length
elif result.status == "ambiguous":
    candidates = result.candidates  # two hypotheses; do not issue a unique bearing
```

`result.status` is `full_3d`, `constrained_3d`, `ambiguous`, or `unreliable`.
Ambiguous/unreliable results have `vector=None` and output confidence zero.
Rank-2 evidence yields the two unit directions mirrored across the surviving
microphone plane when the measured delays admit such candidates. The solver
rejects impossible projections instead of normalizing them to force a result.
`support_confidence` describes retained evidence even when direction is ambiguous;
it does not make a candidate a unique localization. `pair_weights`,
`pair_coherence`, `mic_reliability`, `reliable_pairs`, `geometry_rank`,
`geometry_condition`, and `reason` expose the quality decisions.

`assume_source_facing=True` is an opt-in (default `False`; CLI
`--source-facing-prior`) that assumes three surviving microphones are on the
source-facing side of the array. For exactly three survivors whose excluded mic is
incoherent with all the others, the plane normal points away from the excluded mic,
and a separated branch on that side can return `constrained_3d`. With the configured
geometry, missing T selects the **-z** side of the B/L/R plane; missing B selects **+y**.
This is relative to the surviving plane, not automatically above the helmet. It is off
by default because it gives confident wrong answers for dead, disconnected or noisy
mics (a dead T always picks the below-plane mirror); the committed default had been
switched on after the verification run, which is why 14 robust tests failed. Low
channel quality or reflections do not prove the source is on the opposite side; the
inference assumes the three working mics are closest.
Near-plane uncertainty still leaves the direction ambiguous.
`constraint_normal`, `excluded_mic`, and `constraint_source` identify the prior.

The optional `hemisphere_normal=[0, 0, 1]` explicitly assumes the source is
above the helmet's x/y plane. When used alone, it selects a rank-2 branch only
when the hemisphere separates both branches throughout a conservative sensitivity
interval based on the existing three-tolerance delay bound. A noisy candidate
just below the boundary cannot by itself rule out that branch. The result is marked
`constrained_3d` with `constraint_applied=True`. With `assume_source_facing=False`,
an absent or ambiguous hemisphere prior leaves both branches unresolved. When
both priors are enabled, they must support the same branch or it is withheld.
The normal must be expressed in the same
coordinate frame as `mics`; gravity/world-up needs transforming with the actual
helmet orientation. An above-only prior cannot generally resolve front/back
ambiguity when the back mic fails. The sensitivity interval is an engineering
guard, not a calibrated probability or a guarantee that real delay errors stay
within that bound. This guard deliberately withholds more near-boundary cases.

The robust controls are keyword arguments to the detail and tuple APIs:
`min_pair_coherence=0.35` (bias-corrected coherence), `max_condition=20.0`,
`delay_tolerance_samples=0.5` (Huber scale, not FFT interpolation resolution),
`assume_source_facing=False`, and `hemisphere_normal=None`. Residuals above three Huber scales are excluded;
a unit-vector fit must also satisfy the retained-delay residual check. These
are documented starting settings, not fitted helmet thresholds. The output
confidence and the 0.2 example gate remain uncalibrated heuristics.
Finite nonzero hemisphere normals may be supplied at any representable scale;
their magnitude does not change the prior. Invalid turn thresholds or deadbands
raise `ValueError` rather than bypassing the turn-confidence gate.

The copy still assumes free-field arrival delays. It cannot identify a coherent
reflection as a reflection, or recover missing information from a dead mic.
Before deployment, measure the mounted coordinates and record known directions
with the actual helmet, including head tilt, wind and distant sources.
Systematic direction-dependent helmet scattering requires a measured or
helmet-aware steering model; quality weighting alone does not correct it.
The separate copy does not change the existing live/firmware bearing path.

Run `python3 scripts/benchmark_robust_mics.py --out results_hf_robust_mics_2048`
for the paired 100-clip regression. The saved results below predate the latest
code-review fixes; consult their recorded source hashes and the
[saved review verification](results_robust_code_review/VERIFICATION.txt)
before comparing them with a new run. Its 18,000 cases include 2,000 intact arrays
and 16,000 silent/noise-only microphone faults. The robust copy withholds a
unique vector in all 16,000 fault cases, exposing mirror candidates when the
remaining evidence supports them. At the unchanged 0.2 gate, intact-array
acceptance is 1,261/2,000 (63.05%), versus the original's 1,461/2,000 (73.05%);
all accepted vectors in both versions are within 15 degrees. This is improved
fault withholding with a coverage cost, not proof of higher raw accuracy.
The additional partial-attenuation check and its reproduction script are in
`results_hf_robust_attenuation_2048/`. These recordings were used during
development to uncover a confidence-quality omission, so the final comparisons
are regression checks, not held-out performance estimates. See each report's
scope notes before interpreting these as physical helmet performance.
