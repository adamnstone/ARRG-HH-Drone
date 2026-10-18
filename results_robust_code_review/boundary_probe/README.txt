EXPLORATORY HEMISPHERE-BOUNDARY GUARD CHECK

This folder preserves the exact numerical protocol and recorded output from the
420-case synthetic boundary probe. Preservation did not rerun the estimator.

Protocol: one fixed seed144 Gaussian source, 2048 samples at48 kHz, 10 dB added
white-noise SNR; one muted L/R/T microphone; seven azimuths; true elevations
0/0.1/1/3 degrees; five noise seeds. The explicit hemisphere is z>=0 and the
unchanged demo confidence gate is0.2. There are no Hugging Face recordings here.

historical_result.json is the recorded run, using solver SHA256:
b2aacc21f763a3003e4d6139c608f29167c90afc40ae604b750ba75c8f8733e5

The previous point-hemisphere rule was reconstructed on the same current
unconstrained candidate fits and support scores. It was not a separate replay
of a historical solver file. Its48 bad acceptances match the earlier direct
old-code probe. This reconstruction accepted158 directions within15 degrees,
accepted48 over15 degrees, and withheld214. The new interval guard withheld
all420 (328 ambiguous,92 unreliable), accepting0 correct and0 incorrect.

This is an explicit coverage cost: the guard also withholds the158 previously
accepted accurate near-boundary estimates. Existing separate far-from-boundary
regressions still resolve supported branches. These selected synthetic cases do
not estimate a calibrated physical-headset failure rate or establish a general
absence of bad acceptances.

The source changed after this recorded run to add a10 ms minimum analysis
window. This folder's historical output is not relabeled as a run of that newer
source. The2048-sample windows exceed that floor, but no rerun was performed as
part of preservation. probe.py reports the actual source hash on future runs.

Reproduce against the current repository code, writing a separate result:

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/doa-benchmark-venv/bin/python \
  results_robust_code_review/boundary_probe/probe.py \
  --out results_robust_code_review/boundary_probe/rerun_result.json
