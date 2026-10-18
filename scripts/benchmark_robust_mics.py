"""Compare the separate robust solver with saved original 2048-sample results.

The default protocol uses all 100 hash-verified clips and the exact central
source crop, angles, noise draws, and failed-channel construction of
benchmark_missing_mic.py: 2,000 intact and 16,000 damaged-array estimates.
--clips N selects the first N clips for development; it is not the full run.
Neither solver receives the known direction or a hemisphere constraint.
The robust solver's source-facing assumption is explicitly disabled to preserve
this historical comparison without priors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sig_proc_tri as original


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def case_key(row):
    return tuple(int(row[name]) for name in
                 ('clip_index', 'azimuth_index', 'true_elevation_deg', 'snr_db'))


def saved_result(row, prefix=''):
    return dict(vector_defined=row[prefix+'vector_defined'] == 'True',
                confidence=float(row[prefix+'confidence']),
                error_3d_deg=float(row[prefix+'error_3d_deg']) if row[prefix+'error_3d_deg'] else None,
                accepted=row[prefix+'accepted'] == 'True')


def load_protocol(args):
    """Verify historical waveforms and both completed original fault sweeps."""
    references, provenance, files = {}, {}, {}
    baseline = {}
    for mode, folder in [('silent', args.silent_dir), ('noise_only', args.noise_dir)]:
        meta = json.loads((folder/'provenance.json').read_text())
        manifest = json.loads((folder/'clip_manifest.json').read_text())
        if (meta['mode'] != mode or meta['samples'] != 2048 or meta['fs'] != 48000
                or meta['clips'] != 100 or not meta.get('finished_utc')):
            raise ValueError('Completed 100-clip, 2048-sample, 48 kHz references required')
        if fingerprint(ROOT/'sig_proc_tri.py') != meta['estimator_sha256']:
            raise ValueError('Original solver changed since the reference runs')
        historical = json.loads(subprocess.check_output([
            'git', 'show', meta['source_commit']+':benchmark_results/clip_manifest.json'], cwd=ROOT))
        if historical != manifest:
            raise ValueError('Reference source manifest differs from historical branch')
        with (folder/'trials.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        keyed = {(case_key(row), row['missing_mic']): row for row in rows}
        expected_keys = {(i, 0, el, snr) for i in range(100)
                         for el in [0, 30, 60, 80, 90] for snr in [30, 20, 10, 0]}
        if (len(rows) != 8000 or len(keyed) != 8000 or
                set(keyed) != {(k, name) for k in expected_keys for name in ['B', 'L', 'R', 'T']}):
            raise ValueError('Incomplete or duplicated original fault coverage')
        if (meta['mic_names'] != original.hardware.MIC_NAMES or
                meta['mics'] != original.MICS.tolist() or meta['azimuths_per_clip'] != 1 or
                meta['elevations'] != [0, 30, 60, 80, 90] or
                meta['nominal_baseline_snrs'] != [30, 20, 10, 0]):
            raise ValueError('Reference array or conditions differ from protocol')
        for row in rows:
            if int(row['fs']) != 48000 or int(row['samples']) != 2048 or row['mode'] != mode:
                raise ValueError('Reference row does not match protocol')
            key = case_key(row)
            base = saved_result(row, 'baseline_')
            if key in baseline and baseline[key] != base:
                raise ValueError('Original paired intact results disagree')
            baseline[key] = base
            for result in (base, saved_result(row)):
                if result['accepted'] != (result['vector_defined'] and
                                           result['confidence'] >= meta['min_confidence']):
                    raise ValueError('Original reference acceptance is inconsistent')
        references[mode] = keyed
        provenance[mode] = meta
        files[mode] = dict(directory=str(folder),
                           provenance_sha256=fingerprint(folder/'provenance.json'),
                           trials_sha256=fingerprint(folder/'trials.csv'),
                           manifest_sha256=fingerprint(folder/'clip_manifest.json'))
    for field in ('source_commit', 'source_fs', 'seed', 'min_confidence', 'estimator_sha256'):
        if provenance['silent'][field] != provenance['noise_only'][field]:
            raise ValueError('Reference protocols disagree: '+field)
    meta = provenance['silent']
    with np.load(args.clip_cache, allow_pickle=False) as cache:
        clips = cache['clips']
        if (clips.shape != (100, 8000) or int(cache['seed']) != meta['seed'] or
                float(cache['win']) != 0.5 or json.loads(str(cache['manifest'])) != manifest):
            raise ValueError('Cached source metadata differs from protocol')
    hashes = [hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest() for x in clips]
    if hashes != [r['normalized_waveform_sha256'] for r in manifest['clips']] or len(set(hashes)) != 100:
        raise ValueError('Cached original waveform hashes do not match')
    if meta['source_fs'] != 16000:
        raise ValueError('Expected 16 kHz historical cached source recordings')
    from scipy.signal import resample_poly
    clips = resample_poly(clips, 3, 1, axis=1)
    start = (clips.shape[1]-2048)//2
    clips = clips[:args.clips, start:start+2048]
    angles = original.wrap_deg(-180 + np.random.default_rng(
        np.random.SeedSequence([meta['seed'], 0])).uniform(0, 360, (100, 1)))[:, 0]
    return clips, angles, meta, manifest, baseline, references, files


def summary(rows):
    out = dict(trials=len(rows), clips=len({r['clip_index'] for r in rows}),
               status_counts=dict(Counter(r['status'] for r in rows)))
    for prefix in ('original_', 'robust_'):
        defined = [r for r in rows if r[prefix+'vector_defined']]
        errors = [r[prefix+'error_3d_deg'] for r in defined]
        accepted = [r for r in rows if r[prefix+'accepted']]
        accepted_errors = [r[prefix+'error_3d_deg'] for r in accepted]
        values = dict(defined_count=len(defined), undefined_count=len(rows)-len(defined),
                      raw_within15_count=sum(e <= 15 for e in errors),
                      raw_within15_all_pct=100*sum(e <= 15 for e in errors)/len(rows),
                      median_3d_error_deg=float(np.median(errors)) if errors else None,
                      p90_3d_error_deg=float(np.percentile(errors, 90)) if errors else None,
                      accepted_count=len(accepted), accepted_pct=100*len(accepted)/len(rows),
                      accepted_over15_count=sum(e > 15 for e in accepted_errors),
                      accepted_over30_count=sum(e > 30 for e in accepted_errors),
                      accepted_within15_pct=100*sum(e <= 15 for e in accepted_errors)/len(accepted) if accepted else None,
                      accepted_max_3d_error_deg=max(accepted_errors) if accepted else None)
        out.update({prefix+k: v for k, v in values.items()})
    out.update(candidate_set_count=sum(r['candidate_count'] > 0 for r in rows),
               candidate_set_within15_count=sum(r['candidate_best_error_3d_deg'] is not None and
                                               r['candidate_best_error_3d_deg'] <= 15 for r in rows),
               mean_estimator_ms=float(np.mean([r['estimator_ms'] for r in rows])))
    return out


def estimate_row(robust, frames, old, truth, info, threshold):
    started = time.perf_counter_ns()
    result = robust.estimate_direction_details(frames, fs=48000,
                                               assume_source_facing=False)
    elapsed = (time.perf_counter_ns()-started)/1e6
    vector = result.vector
    candidates = np.asarray(result.candidates, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or not np.all(np.isfinite(candidates)):
        raise ValueError('Robust candidates must have finite shape (K, 3)')
    candidate_errors = [original.direction_error_deg(u, truth) for u in candidates]
    if any(e is None for e in candidate_errors):
        raise ValueError('Robust solver returned a zero candidate')
    confidence = float(result.confidence)
    status = str(result.status)
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Robust confidence must be in [0, 1]')
    if vector is not None:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.isclose(np.linalg.norm(vector), 1):
            raise ValueError('Robust output vector must be a finite 3D unit vector')
    if status in ('ambiguous', 'unreliable') and vector is not None:
        raise ValueError('Nonunique/unreliable status must withhold the output vector')
    accepted = vector is not None and confidence >= threshold
    row = dict(info)
    row.update({('original_'+k): v for k, v in old.items()})
    row.update(status=status, reason=str(result.reason), robust_vector_defined=vector is not None,
               robust_ux=float(vector[0]) if vector is not None else None,
               robust_uy=float(vector[1]) if vector is not None else None,
               robust_uz=float(vector[2]) if vector is not None else None,
               robust_error_3d_deg=original.direction_error_deg(vector, truth),
               robust_azimuth_deg=original.azimuth_from_direction(vector),
               robust_elevation_deg=original.elevation_from_direction(vector),
               robust_confidence=confidence, robust_accepted=bool(accepted),
               support_confidence=float(result.support_confidence),
               constraint_applied=bool(result.constraint_applied),
               geometry_rank=int(result.geometry_rank),
               geometry_condition=float(result.geometry_condition) if np.isfinite(result.geometry_condition) else None,
               candidate_count=len(candidates), candidate_vectors_json=json.dumps(candidates.tolist()),
               candidate_errors_3d_deg_json=json.dumps(candidate_errors),
               candidate_best_error_3d_deg=min(candidate_errors) if candidate_errors else None,
               candidate_worst_error_3d_deg=max(candidate_errors) if candidate_errors else None,
               pair_weights_json=json.dumps(np.asarray(result.pair_weights).tolist()),
               pair_coherence_json=json.dumps(np.asarray(result.pair_coherence).tolist()),
               reliable_pairs_json=json.dumps(np.asarray(result.reliable_pairs, dtype=bool).tolist()),
               mic_reliability_json=json.dumps(np.asarray(result.mic_reliability).tolist()),
               estimator_ms=elapsed)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', type=int, default=100, help='First N clips; values below 100 are development subsets')
    parser.add_argument('--clip-cache', type=Path, default=ROOT/'benchmark_results'/'clips.npz')
    parser.add_argument('--silent-dir', type=Path, default=ROOT/'results_hf_missing_mic_silent_2048')
    parser.add_argument('--noise-dir', type=Path, default=ROOT/'results_hf_missing_mic_noise_2048')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.clips <= 100:
        parser.error('--clips must be from 1 to 100')
    clips, angles, reference, manifest, baseline, faults, reference_files = load_protocol(args)
    robust = importlib.import_module('sig_proc_tri_robust')
    original_hash = fingerprint(ROOT/'sig_proc_tri.py')
    robust_hash = fingerprint(ROOT/'sig_proc_tri_robust.py')
    script_hash = fingerprint(Path(__file__))
    meta = dict(dataset=reference['dataset'], source_commit=reference['source_commit'],
                verified_source_clips=100, evaluated_source_clips=args.clips,
                development_subset=args.clips != 100, all_source_hashes_verified=True,
                source_fs=16000, fs=48000, samples=2048, window_seconds=2048/48000,
                resampling='scipy.signal.resample_poly, 3:1; central source crop before propagation',
                seed=reference['seed'], azimuths_per_clip=1, elevations=reference['elevations'],
                nominal_baseline_snrs=reference['nominal_baseline_snrs'],
                mics=original.MICS.tolist(), mic_names=original.hardware.MIC_NAMES,
                min_confidence=reference['min_confidence'], confidence_threshold_calibrated=False,
                original_estimator_sha256=original_hash, robust_estimator_sha256=robust_hash,
                script_sha256=script_hash, reference_files=reference_files,
                original_results='Saved matching original fault rows and their fresh 2048-sample intact baselines',
                original_estimator_unchanged=True, parameter_tuning=False,
                hemisphere_prior_applied=False, assume_source_facing=False,
                truth_used_for_estimation=False,
                candidate_error_role='Diagnostic set coverage only; never credited as unique output accuracy',
                numpy_version=np.__version__, started_utc=datetime.now(timezone.utc).isoformat(),
                command='python3 scripts/benchmark_robust_mics.py '+' '.join(sys.argv[1:]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'provenance.json', meta)
    write_json(args.out/'clip_manifest.json', manifest)
    rows, conditions = [], []
    print(f'Verified 100 original clips; evaluating {args.clips}, 2048 samples at 48000 Hz.', flush=True)
    for elevation in reference['elevations']:
        for snr in reference['nominal_baseline_snrs']:
            group = []
            for clip_index, source in enumerate(clips):
                az = float(angles[clip_index]) if elevation != 90 else 0.0
                key = (clip_index, 0, elevation, snr)
                if elevation != 90 and abs(float(faults['silent'][key, 'B']['true_azimuth_deg'])-az) > 1e-9:
                    raise ValueError('Reconstructed direction differs from reference')
                truth = original.unit_vector(az, elevation)
                rng = np.random.default_rng(np.random.SeedSequence([
                    reference['seed'], 1, clip_index, 0, elevation, snr]))
                frames = original.simulate_array(source, az, snr, rng, fs=48000, elevation_deg=elevation)
                clean = original.simulate_array(source, az, np.inf, np.random.default_rng(0),
                                                 fs=48000, elevation_deg=elevation)
                info = dict(mode='intact', missing_mic=None, missing_channel=None,
                            clip_index=clip_index, azimuth_index=0, fs=48000, samples=2048,
                            true_azimuth_deg=az if elevation != 90 else None,
                            true_elevation_deg=elevation, snr_db=snr)
                group.append(estimate_row(robust, frames, baseline[key], truth, info, reference['min_confidence']))
                for mode in ('silent', 'noise_only'):
                    for mic, name in enumerate(original.hardware.MIC_NAMES):
                        damaged = frames.copy()
                        damaged[mic] = 0.0 if mode == 'silent' else frames[mic]-clean[mic]
                        healthy = [i for i in range(4) if i != mic]
                        if not np.array_equal(damaged[healthy], frames[healthy]):
                            raise AssertionError('Healthy channel changed')
                        fault_info = dict(info, mode=mode, missing_mic=name, missing_channel=mic)
                        old = saved_result(faults[mode][key, name])
                        group.append(estimate_row(robust, damaged, old, truth, fault_info, reference['min_confidence']))
            rows.extend(group)
            for mode in ('intact', 'silent', 'noise_only'):
                selected = [r for r in group if r['mode'] == mode]
                metrics = dict(mode=mode, elevation_deg=elevation, snr_db=snr, **summary(selected))
                conditions.append(metrics)
                print(f"elev {elevation:2d}, SNR {snr:2d}, {mode}: {len(selected)} trials; "
                      f"accepted {metrics['robust_accepted_count']}; accepted >15deg "
                      f"{metrics['robust_accepted_over15_count']}; statuses {metrics['status_counts']}", flush=True)
            write_csv(args.out/'by_condition.csv', conditions)
    expected = args.clips*20*9
    if len(rows) != expected or len({(r['mode'], r['missing_mic'], r['clip_index'],
                                   r['true_elevation_deg'], r['snr_db']) for r in rows}) != expected:
        raise AssertionError('Incomplete or duplicated comparison coverage')
    if (fingerprint(ROOT/'sig_proc_tri.py') != original_hash or
            fingerprint(ROOT/'sig_proc_tri_robust.py') != robust_hash or
            fingerprint(Path(__file__)) != script_hash):
        raise AssertionError('A solver or benchmark changed during the run')
    by_mode = [dict(mode=mode, **summary([r for r in rows if r['mode'] == mode]))
               for mode in ('intact', 'silent', 'noise_only')]
    by_mic = [dict(mode=mode, missing_mic=name, **summary([
        r for r in rows if r['mode'] == mode and r['missing_mic'] == name]))
        for mode in ('silent', 'noise_only') for name in original.hardware.MIC_NAMES]
    by_snr = [dict(mode=mode, snr_db=snr, **summary([
        r for r in rows if r['mode'] == mode and r['snr_db'] == snr]))
        for mode in ('intact', 'silent', 'noise_only') for snr in reference['nominal_baseline_snrs']]
    write_csv(args.out/'trials.csv', rows)
    write_csv(args.out/'by_mode.csv', by_mode)
    write_csv(args.out/'by_mic.csv', by_mic)
    write_csv(args.out/'by_snr.csv', by_snr)
    write_json(args.out/'summary.json', dict(by_mode=by_mode, by_mic=by_mic, by_snr=by_snr, by_condition=conditions))
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), trials=len(rows),
                intact_trials=args.clips*20, fault_trials=args.clips*160)
    write_json(args.out/'provenance.json', meta)
    lines = ['ROBUST VS ORIGINAL: PAIRED 2048-SAMPLE FOUR-MIC COMPARISON', '',
             f"Evaluated {args.clips}/100 hash-verified source clips; {len(rows):,} trials.",
             'Development subset: '+str(args.clips != 100),
             '48 kHz, central 2048-sample source crop; no hemisphere prior.',
             f"Fixed confidence gate: {reference['min_confidence']} (uncalibrated).", '',
             'Mode         Cases  Original accepted/bad>15  Robust accepted/bad>15  Robust status counts']
    for r in by_mode:
        lines.append(f"{r['mode']:11} {r['trials']:6d}  {r['original_accepted_count']:7d}/{r['original_accepted_over15_count']:<6d} "
                     f"{r['robust_accepted_count']:14d}/{r['robust_accepted_over15_count']:<6d} {r['status_counts']}")
    lines += ['', 'Interpretation:',
              'Candidate sets describe ambiguity. Their truth-nearness is diagnostic only and is',
              'never counted as a uniquely accepted direction. Withheld results are not successes.',
              'Three source-bearing microphones generally leave mirror ambiguity across their plane.',
              'Silent faults set one channel to exactly zero. Noise-only faults subtract its identical',
              'clean source from the noisy baseline, retaining the original independent added noise.',
              'Healthy channels, source crops, angles, noise seeds and the 0.2 gate match the saved',
              'original comparison; original numeric results are reused without rerunning or changing it.',
              'SNR labels describe the intact baseline; the failed noise-only channel has no drone.',
              'Real mono source recordings were upsampled from 16 to 48 kHz; no native source',
              'bandwidth is added. Whole-band white-noise SNR does not equal in-band SNR and',
              'must not be used to infer a sample-rate advantage over the 16 kHz sweep.',
              'These are simulated synchronized delays and independent white noise. Original',
              'recording noise propagates coherently and is removed with the failed-channel source.',
              'No helmet, hardware, diffuse-noise, reverberation or live/firmware validation is claimed.',
              'Repeated scenarios reuse source recordings. No benchmark parameter tuning occurred.',
              '', 'Reproduce: '+meta['command']]
    (args.out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(by_mode, indent=2), flush=True)


if __name__ == '__main__':
    main()
