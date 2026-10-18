"""Frozen supplemental partial-attenuation check; no solver parameters tuned.

Run from any directory with NumPy/SciPy installed. Set BLAS thread variables to
one in the shell, as recorded in provenance.json. Outputs remain beside script.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.signal import resample_poly

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
import sig_proc_tri as original
import sig_proc_tri_robust as robust


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    output = dict(trials=len(rows), robust_status_counts=dict(Counter(r['robust_status'] for r in rows)))
    for prefix in ['original_', 'robust_']:
        defined = [r for r in rows if r[prefix + 'vector_defined']]
        accepted = [r for r in rows if r[prefix + 'accepted']]
        errors = [r[prefix + 'error_deg'] for r in accepted]
        output.update({prefix + 'defined': len(defined),
                       prefix + 'accepted': len(accepted),
                       prefix + 'accepted_over15': sum(e > 15 for e in errors),
                       prefix + 'accepted_over30': sum(e > 30 for e in errors),
                       prefix + 'accepted_max_error_deg': max(errors) if errors else None,
                       prefix + 'accepted_median_error_deg': float(np.median(errors)) if errors else None})
    for name in ['weak_pair_coherence', 'healthy_pair_coherence',
                 'weak_pair_weight', 'healthy_pair_weight',
                 'weak_mic_reliability', 'weak_pairs_reliable', 'healthy_pairs_reliable']:
        values = [r[name] for r in rows if r[name] is not None]
        output[name + '_median'] = float(np.median(values)) if values else None
    return output


def evaluate(frames, truth, info):
    old = original.estimate_direction_details(frames, fs=48000)
    new = robust.estimate_direction_details(frames, fs=48000)
    row = dict(info)
    for label, result in [('original', old), ('robust', new)]:
        vector = result.vector
        if vector is not None:
            if not np.all(np.isfinite(vector)) or not np.isclose(np.linalg.norm(vector), 1):
                raise AssertionError('Expected finite unit vector')
        row.update({label + '_vector_defined': vector is not None,
                    label + '_confidence': float(result.confidence),
                    label + '_accepted': bool(vector is not None and result.confidence >= 0.2),
                    label + '_error_deg': original.direction_error_deg(vector, truth),
                    label + '_vector_json': json.dumps(vector.tolist()) if vector is not None else None})
    row.update(robust_status=new.status, robust_reason=new.reason,
               robust_geometry_rank=int(new.geometry_rank),
               robust_geometry_condition=float(new.geometry_condition) if np.isfinite(new.geometry_condition) else None,
               robust_support_confidence=float(new.support_confidence),
               robust_constraint_applied=bool(new.constraint_applied),
               robust_pair_coherence_json=json.dumps(new.pair_coherence.tolist()),
               robust_pair_weights_json=json.dumps(new.pair_weights.tolist()),
               robust_reliable_pairs_json=json.dumps(new.reliable_pairs.tolist()),
               robust_mic_reliability_json=json.dumps(new.mic_reliability.tolist()),
               original_pair_peaks_json=json.dumps(old.peaks.tolist()))
    mic = info['attenuated_channel']
    if mic is None:
        row.update({name: None for name in ['weak_pair_coherence', 'healthy_pair_coherence',
                                           'weak_pair_weight', 'healthy_pair_weight',
                                           'weak_mic_reliability', 'weak_pairs_reliable',
                                           'healthy_pairs_reliable']})
    else:
        incident = np.array([mic in pair for pair in original.PAIRS])
        row.update(weak_pair_coherence=float(np.median(new.pair_coherence[incident])),
                   healthy_pair_coherence=float(np.median(new.pair_coherence[~incident])),
                   weak_pair_weight=float(np.median(new.pair_weights[incident])),
                   healthy_pair_weight=float(np.median(new.pair_weights[~incident])),
                   weak_mic_reliability=float(new.mic_reliability[mic]),
                   weak_pairs_reliable=int(np.count_nonzero(new.reliable_pairs[incident])),
                   healthy_pairs_reliable=int(np.count_nonzero(new.reliable_pairs[~incident])))
    if new.vector is None and new.confidence != 0:
        raise AssertionError('Undefined vector must have zero confidence')
    if new.constraint_applied:
        raise AssertionError('No hemisphere assumption was supplied')
    return row


def main():
    started = time.perf_counter()
    reference_dir = ROOT / 'results_hf_robust_mics_2048'
    reference = json.loads((reference_dir / 'provenance.json').read_text())
    manifest = json.loads((reference_dir / 'clip_manifest.json').read_text())
    cache_path = ROOT / 'benchmark_results' / 'clips.npz'
    hashes_before = {name: fingerprint(ROOT / name) for name in ['sig_proc_tri.py', 'sig_proc_tri_robust.py']}
    if hashes_before['sig_proc_tri.py'] != reference['original_estimator_sha256']:
        raise ValueError('Original solver differs from matching reference protocol')
    if hashes_before['sig_proc_tri_robust.py'] != reference['robust_estimator_sha256']:
        raise ValueError('Robust solver differs from frozen reference protocol')
    with np.load(cache_path, allow_pickle=False) as cache:
        source_clips = cache['clips']
        if source_clips.shape != (100, 8000) or int(cache['seed']) != reference['seed'] or float(cache['win']) != 0.5:
            raise ValueError('Wrong cached clip shape or protocol')
        if json.loads(str(cache['manifest'])) != manifest:
            raise ValueError('Cached manifest differs from verified protocol')
    hashes = [hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest() for x in source_clips]
    if hashes != [item['normalized_waveform_sha256'] for item in manifest['clips']] or len(set(hashes)) != 100:
        raise ValueError('All 100 original clip hashes must match')
    clips = resample_poly(source_clips, 3, 1, axis=1)
    crop_start = (clips.shape[1] - 2048) // 2
    clips = clips[:, crop_start:crop_start + 2048]
    seed = reference['seed']
    angles = original.wrap_deg(-180 + np.random.default_rng(
        np.random.SeedSequence([seed, 0])).uniform(0, 360, (100, 1)))[:, 0]
    provenance = dict(dataset=reference['dataset'], source_commit=reference['source_commit'],
                      source_clips=100, all_source_hashes_verified=True,
                      clip_manifest_sha256=fingerprint(reference_dir / 'clip_manifest.json'),
                      source_cache_sha256=fingerprint(cache_path),
                      reference_provenance_sha256=fingerprint(reference_dir / 'provenance.json'),
                      estimator_sha256=hashes_before, script_sha256=fingerprint(Path(__file__)),
                      fs=48000, samples=2048, window_seconds=2048 / 48000,
                      source_fs=16000, source_samples=8000,
                      resampling='scipy.signal.resample_poly 3:1; then central2048 crop before propagation',
                      crop_start_resampled_samples=crop_start,
                      seed=seed, azimuth_protocol='wrap_deg(-180+default_rng(SeedSequence([seed,0])).uniform(0,360,(100,1)))[:,0]',
                      elevation_protocol='[0,30,60,80,90][clip_index % 5]',
                      baseline_snr_db=20,
                      baseline_noise_seed_protocol='SeedSequence([seed,1,clip_index,0,elevation,20])',
                      attenuation_signal_gains_db=[-6, -12, -20, -30],
                      damaged_channel='gain*clean[mic]+(baseline[mic]-clean[mic]); other3 unchanged',
                      damaged_nominal_snr_db=[14, 8, 0, -10],
                      microphone_positions=original.MICS.tolist(), microphone_names=original.hardware.MIC_NAMES,
                      acceptance='defined unit vector and unchanged confidence >=0.2',
                      hemisphere_prior=False, parameter_tuning=False,
                      limitations=['Uniform source-only amplitude attenuation; not a helmet/head acoustic model',
                                   'Real mono clips with simulated free-field far-field propagation and independent white noise',
                                   'Baseline SNR is relative to whole recording, including existing background',
                                   'Small selected supplement: one direction/elevation/noise draw per clip'],
                      numpy_version=np.__version__, started_utc=datetime.now(timezone.utc).isoformat(),
                      thread_environment={key: os.environ.get(key) for key in
                                          ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                           'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']},
                      command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 '
                              '/private/tmp/doa-benchmark-venv/bin/python results_hf_robust_attenuation_2048/benchmark.py')
    write_json('provenance.json', provenance)
    write_json('clip_manifest.json', manifest)
    rows = []
    for clip_index, source in enumerate(clips):
        elevation = [0, 30, 60, 80, 90][clip_index % 5]
        azimuth = float(angles[clip_index]) if elevation != 90 else 0.0
        truth = original.unit_vector(azimuth, elevation)
        noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 1, clip_index, 0, elevation, 20]))
        baseline = original.simulate_array(source, azimuth, 20, noise_rng, fs=48000, elevation_deg=elevation)
        clean = original.simulate_array(source, azimuth, np.inf, np.random.default_rng(0),
                                        fs=48000, elevation_deg=elevation)
        info = dict(clip_index=clip_index, true_azimuth_deg=azimuth if elevation != 90 else None,
                    true_elevation_deg=elevation, baseline_snr_db=20,
                    attenuated_mic=None, attenuated_channel=None, attenuation_db=0,
                    damaged_nominal_snr_db=None)
        rows.append(evaluate(baseline, truth, info))
        for attenuation in [-6, -12, -20, -30]:
            gain = 10 ** (attenuation / 20)
            for mic, name in enumerate(original.hardware.MIC_NAMES):
                damaged = baseline.copy()
                damaged[mic] = gain * clean[mic] + (baseline[mic] - clean[mic])
                healthy = [other for other in range(4) if other != mic]
                if not np.array_equal(damaged[healthy], baseline[healthy]):
                    raise AssertionError('Healthy channels changed')
                rows.append(evaluate(damaged, truth, dict(info, attenuation_db=attenuation,
                                                         attenuated_mic=name, attenuated_channel=mic,
                                                         damaged_nominal_snr_db=20 + attenuation)))
        if (clip_index + 1) % 20 == 0:
            print(f'{clip_index + 1}/100 clips; {len(rows)} paired cases', flush=True)
    if len(rows) != 1700:
        raise AssertionError('Expected 100 controls plus1600 attenuated cases')
    if {name: fingerprint(ROOT / name) for name in hashes_before} != hashes_before:
        raise AssertionError('Solver changed during the run')
    if fingerprint(Path(__file__)) != provenance['script_sha256']:
        raise AssertionError('Benchmark script changed during the run')
    by_attenuation = [dict(attenuation_db=db, **metrics([r for r in rows if r['attenuation_db'] == db]))
                      for db in [0, -6, -12, -20, -30]]
    by_mic = [dict(attenuation_db=db, attenuated_mic=name,
                   **metrics([r for r in rows if r['attenuation_db'] == db and r['attenuated_mic'] == name]))
              for db in [-6, -12, -20, -30] for name in original.hardware.MIC_NAMES]
    write_csv('trials.csv', rows)
    write_csv('by_attenuation.csv', by_attenuation)
    write_csv('by_mic.csv', by_mic)
    write_json('summary.json', dict(by_attenuation=by_attenuation, by_mic=by_mic))
    provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), paired_cases=len(rows),
                      intact_controls=100, attenuated_cases=1600,
                      estimates_per_solver=1700, unchanged_solver_hashes=True,
                      elapsed_seconds=time.perf_counter() - started)
    write_json('provenance.json', provenance)
    lines = ['SUPPLEMENTAL PARTIAL-ATTENUATION CHECK: 100 HASH-VERIFIED HF CLIPS',
             '', '48 kHz, central2048 samples (42.67 ms), nominal intact20 dB SNR.',
             'One full-circle azimuth per clip, elevations cycling0/30/60/80/90 degrees.',
             'Attenuate source component of one channel; preserve its original noise and other3 channels.',
             'Both current frozen solvers use unchanged defaults and confidence gate0.2; no hemisphere prior.',
             '', 'atten_dB cases original_accepted original_accepted_over15 robust_accepted robust_accepted_over15 statuses']
    for item in by_attenuation:
        lines.append(f"{item['attenuation_db']:8d} {item['trials']:5d} {item['original_accepted']:17d} "
                     f"{item['original_accepted_over15']:24d} {item['robust_accepted']:15d} "
                     f"{item['robust_accepted_over15']:22d} {item['robust_status_counts']}")
    lines.extend(['', 'Weak-pair diagnostics (medians across cases; each case takes median of3 incident pairs):'])
    for item in by_attenuation[1:]:
        lines.append(f"{item['attenuation_db']:3d} dB: coherence weak/healthy "
                     f"{item['weak_pair_coherence_median']:.4f}/{item['healthy_pair_coherence_median']:.4f}; "
                     f"weights {item['weak_pair_weight_median']:.4f}/{item['healthy_pair_weight_median']:.4f}; "
                     f"reliable weak/healthy {item['weak_pairs_reliable_median']:.0f}/{item['healthy_pairs_reliable_median']:.0f}")
    lines.extend(['', 'LIMITS: Flat signal attenuation is not a measured helmet/head transfer function.',
                  'This does not test real hardware, wind, synchronization error, reflections or phase distortion.',
                  'Withheld mirror candidates are not credited as unique output accuracy.',
                  'No thresholds or solver code changed. Detailed per-case results and hashes accompany this report.'])
    (OUT / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
