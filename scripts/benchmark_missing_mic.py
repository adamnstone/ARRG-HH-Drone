"""Replay the 100 verified HF clips with one silent or noise-only microphone.

Uses the paired 48 kHz review benchmark as its intact reference. By default,
each mode tests 2,000 source/direction/SNR conditions with each of B/L/R/T
missing: 8,000 damaged-array estimates. --samples 2048 instead tests a central
hardware-sized source window and computes fresh paired intact estimates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sig_proc_tri as doa
from benchmark_four_mic import summarize, write_csv, write_json


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nullable_float(value):
    return float(value) if value != '' else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['silent', 'noise_only'], required=True)
    parser.add_argument('--baseline-dir', type=Path, default=ROOT/'results_hf_four_mic_48k_review')
    parser.add_argument('--clip-cache', type=Path, default=ROOT/'benchmark_results'/'clips.npz')
    parser.add_argument('--samples', type=int, help='Central source window length at the analysis rate')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.samples is not None and args.samples < 4:
        parser.error('--samples must be at least four')
    reference = json.loads((args.baseline_dir/'provenance.json').read_text())
    manifest = json.loads((args.baseline_dir/'clip_manifest.json').read_text())
    historical_manifest = json.loads(subprocess.check_output(
        ['git', 'show', reference['source_commit']+':benchmark_results/clip_manifest.json'], cwd=ROOT))
    if manifest != historical_manifest or not reference.get('finished_utc'):
        raise ValueError('Completed reference and historical source manifest required')
    solver_hash = fingerprint(ROOT/'sig_proc_tri.py')
    if solver_hash != reference['estimator_sha256']:
        raise ValueError('Solver differs from the intact reference benchmark')
    if reference['mics'] != doa.MICS.tolist() or reference['sound_speed'] != doa.C:
        raise ValueError('Reference geometry differs from current solver')
    original = reference['source_metadata']
    with np.load(args.clip_cache, allow_pickle=False) as cache:
        clips = cache['clips']
        if (int(cache['seed']) != reference['seed'] or
                float(cache['win']) != original['window_seconds'] or
                json.loads(str(cache['manifest'])) != manifest):
            raise ValueError('Cache metadata differs from verified reference')
    hashes = [hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest() for x in clips]
    if (clips.shape != (100, round(original['fs']*original['window_seconds'])) or
            hashes != [r['normalized_waveform_sha256'] for r in manifest['clips']] or
            len(set(hashes)) != 100):
        raise ValueError('Expected all 100 distinct original waveform hashes')
    fs, seed = reference['fs'], reference['seed']
    if fs != original['fs']:
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(fs, original['fs'])
        clips = resample_poly(clips, fs//divisor, original['fs']//divisor, axis=1)
    if args.samples is not None:
        if args.samples > clips.shape[1]:
            raise ValueError('Requested source window exceeds cached clip duration')
        start = (clips.shape[1]-args.samples)//2
        clips = clips[:, start:start+args.samples]
    with (args.baseline_dir/'trials.csv').open(newline='') as stream:
        baseline_rows = list(csv.DictReader(stream))
    baseline = {(int(r['clip_index']), int(r['azimuth_index']),
                 int(r['true_elevation_deg']), int(r['snr_db'])): r for r in baseline_rows}
    count_az = reference['azimuths_per_clip']
    elevations, snrs = reference['elevations'], reference['snrs']
    expected_base = 100*((len(elevations)-1)*count_az+1)*len(snrs)
    if len(baseline) != expected_base or len(baseline_rows) != expected_base:
        raise ValueError('Incomplete intact reference coverage')
    spacing = 360/count_az
    angles = doa.wrap_deg(-180 + np.arange(count_az)[None, :]*spacing +
                          np.random.default_rng(np.random.SeedSequence([seed, 0])).uniform(0, spacing, (100, 1)))
    threshold = reference['min_confidence']
    names = doa.hardware.MIC_NAMES
    meta = dict(dataset=reference['dataset'], mode=args.mode, clips=100,
                all_source_hashes_verified=True, source_commit=reference['source_commit'],
                source_fs=original['fs'], fs=fs, samples=clips.shape[1],
                window_seconds=clips.shape[1]/fs, source_window='central crop' if args.samples else 'full cached clip',
                azimuths_per_clip=count_az, elevations=elevations, nominal_baseline_snrs=snrs,
                mics=doa.MICS.tolist(), mic_names=names, min_confidence=threshold,
                confidence_threshold_calibrated=False, seed=seed, parameter_tuning=False,
                estimator_sha256=solver_hash, script_sha256=fingerprint(Path(__file__)),
                helper_sha256=fingerprint(ROOT/'scripts/benchmark_four_mic.py'),
                baseline_provenance=str(args.baseline_dir/'provenance.json'),
                baseline_provenance_sha256=fingerprint(args.baseline_dir/'provenance.json'),
                baseline_trials_sha256=fingerprint(args.baseline_dir/'trials.csv'),
                baseline_method='recomputed for cropped source' if args.samples else 'paired saved review output',
                failure_model=('Selected channel set to exactly zero' if args.mode == 'silent' else
                               'Selected channel = baseline noisy channel minus identical clean source; added noise retained'),
                numpy_version=np.__version__, started_utc=datetime.now(timezone.utc).isoformat(),
                command='python3 scripts/benchmark_missing_mic.py '+' '.join(sys.argv[1:]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'provenance.json', meta)
    write_json(args.out/'clip_manifest.json', manifest)
    rows, conditions = [], []
    print(f'Verified 100 clips; {args.mode}, {fs} Hz, {clips.shape[1]} samples.', flush=True)
    for elevation in elevations:
        for snr in snrs:
            group = []
            for clip_index, signal in enumerate(clips):
                for ai in range(1 if elevation == 90 else count_az):
                    az = float(angles[clip_index, ai]) if elevation != 90 else 0.0
                    old = baseline[clip_index, ai, elevation, snr]
                    if (int(old['fs']) != fs or
                            (elevation != 90 and abs(float(old['true_azimuth_deg'])-az) > 1e-9)):
                        raise ValueError('Reference trial does not match paired input')
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, clip_index, ai, elevation, snr]))
                    frames = doa.simulate_array(signal, az, snr, rng, fs=fs, elevation_deg=elevation)
                    clean = doa.simulate_array(signal, az, np.inf, np.random.default_rng(0),
                                               fs=fs, elevation_deg=elevation)
                    if args.samples:
                        intact = doa.estimate_direction_details(frames, fs=fs)
                        base_defined = intact.vector is not None
                        base_confidence = intact.confidence
                        base_error = doa.direction_error_deg(intact.vector, doa.unit_vector(az, elevation))
                        base_accepted = base_defined and base_confidence >= threshold
                    else:
                        base_defined = old['vector_defined'] == 'True'
                        base_confidence = float(old['confidence'])
                        base_error = nullable_float(old['error_3d_deg'])
                        base_accepted = old['accepted'] == 'True'
                    for mic, name in enumerate(names):
                        damaged = frames.copy()
                        damaged[mic] = 0.0 if args.mode == 'silent' else frames[mic]-clean[mic]
                        retained = [i for i in range(4) if i != mic]
                        if not np.array_equal(damaged[retained], frames[retained]):
                            raise AssertionError('A healthy channel was altered')
                        started = time.perf_counter_ns()
                        result = doa.estimate_direction_details(damaged, fs=fs)
                        elapsed = (time.perf_counter_ns()-started)/1e6
                        vector = result.vector
                        est_az = doa.azimuth_from_direction(vector)
                        est_el = doa.elevation_from_direction(vector)
                        valid = np.isfinite(result.delays)
                        geometry = np.array([doa.MICS[j]-doa.MICS[i] for i, j in doa.PAIRS])
                        rank = int(np.linalg.matrix_rank(geometry[valid])) if np.any(valid) else 0
                        row = dict(mode=args.mode, missing_mic=name, missing_channel=mic,
                                   clip_index=clip_index, azimuth_index=ai, fs=fs, samples=clips.shape[1],
                                   true_azimuth_deg=az if elevation != 90 else None,
                                   true_elevation_deg=elevation, snr_db=snr, vector_defined=vector is not None,
                                   estimated_ux=float(vector[0]) if vector is not None else None,
                                   estimated_uy=float(vector[1]) if vector is not None else None,
                                   estimated_uz=float(vector[2]) if vector is not None else None,
                                   estimated_azimuth_deg=est_az, estimated_elevation_deg=est_el,
                                   azimuth_error_deg=doa.ang_err(est_az, az) if elevation != 90 and est_az is not None else None,
                                   elevation_error_deg=abs(est_el-elevation) if est_el is not None else None,
                                   error_3d_deg=doa.direction_error_deg(vector, doa.unit_vector(az, elevation)),
                                   confidence=result.confidence,
                                   accepted=bool(vector is not None and result.confidence >= threshold),
                                   usable_pair_count=int(np.count_nonzero(valid)), usable_geometry_rank=rank,
                                   ambiguous_pair_count=int(np.count_nonzero(result.ambiguous_pairs)),
                                   baseline_vector_defined=base_defined, baseline_confidence=base_confidence,
                                   baseline_error_3d_deg=base_error, baseline_accepted=base_accepted,
                                   estimator_ms=elapsed)
                        group.append(row)
            rows.extend(group)
            condition = dict(elevation_deg=elevation, snr_db=snr, **summarize(group))
            conditions.append(condition)
            write_csv(args.out/'by_condition.csv', conditions)
            print(f"elev {elevation:2d}, baseline SNR {snr:2d}: {len(group)} faults; "
                  f"undefined {condition['undefined_count']}; accepted {condition['accepted_count']}; "
                  f"accepted >15deg {condition['accepted_over15_count']}", flush=True)
    if len(rows) != 4*expected_base or {r['clip_index'] for r in rows} != set(range(100)):
        raise AssertionError('Incomplete fault coverage')
    if fingerprint(ROOT/'sig_proc_tri.py') != solver_hash:
        raise AssertionError('Solver changed during benchmark')
    by_mic = [dict(missing_mic=name, **summarize([r for r in rows if r['missing_mic'] == name])) for name in names]
    by_snr = [dict(snr_db=snr, **summarize([r for r in rows if r['snr_db'] == snr])) for snr in snrs]
    by_mic_snr = [dict(missing_mic=name, snr_db=snr, **summarize([
        r for r in rows if r['missing_mic'] == name and r['snr_db'] == snr])) for name in names for snr in snrs]
    overall = summarize(rows)
    write_csv(args.out/'trials.csv', rows)
    write_csv(args.out/'by_mic.csv', by_mic)
    write_csv(args.out/'by_snr.csv', by_snr)
    write_csv(args.out/'by_mic_snr.csv', by_mic_snr)
    write_json(args.out/'summary.json', dict(overall=overall, by_mic=by_mic, by_snr=by_snr,
                                            by_mic_snr=by_mic_snr, by_condition=conditions))
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), trials=len(rows))
    write_json(args.out/'provenance.json', meta)
    lines = ['100 HF CLIPS: ONE MICROPHONE WITHOUT THE DRONE', '',
             f"Mode: {args.mode}; {fs} Hz; {clips.shape[1]} samples ({clips.shape[1]/fs:g} seconds).",
             f"Trials: {len(rows):,}; each of B/L/R/T absent in every paired source/direction/SNR condition.",
             f"Defined: {overall['defined_count']}; undefined: {overall['undefined_count']}.",
             f"Accepted at confidence >= {threshold}: {overall['accepted_count']} ({overall['accepted_pct']:.3f}%).",
             f"Accepted with 3D error >15 degrees: {overall['accepted_over15_count']}.",
             '', 'Missing    Trials  Undefined  Accepted  Accepted>15deg']
    for r in by_mic:
        lines.append(f"{r['missing_mic']:>7} {r['trials']:9d} {r['undefined_count']:10d} "
                     f"{r['accepted_count']:9d} {r['accepted_over15_count']:15d}")
    lines += ['', 'Interpretation:',
              'Three source-bearing microphones provide only two independent delay constraints.',
              'An unrestricted 3D direction is generally ambiguous across their microphone plane.',
              'Undefined is safe withholding, not successful localization. Even an accepted estimate',
              'within 15 degrees does not establish unique 3D observability from three microphones.',
              'The noise-only failed channel has no drone signal; reported SNR is the intact-array',
              'baseline SNR. Its independent added noise is retained, not replaced by a new draw.',
              'Recorded mono background noise is propagated with the source and removed from the',
              'failed channel with that source. This does not model partial acoustic attenuation,',
              'real head shadow, reverberation or diffuse noise. No hardware recordings are tested.',
              'The 48 kHz sources are upsampled from 16 kHz and add no native source information.',
              'Confidence 0.2 is an unchanged, uncalibrated demo threshold. No parameter tuning.',
              'These results concern sig_proc_tri.py, not the separate live/firmware bearing path.',
              'Trials reuse 100 source recordings; repeated conditions are not independent recordings.',
              '', 'Reproduce: '+meta['command']]
    (args.out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == '__main__':
    main()
