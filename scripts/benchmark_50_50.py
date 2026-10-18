"""Seeded, disjoint 50-intact/50-degraded source comparison of both solvers.

The frozen primary comparison has 100 cases at 20 dB nominal baseline SNR.
The complete sweep contains 1,200 paired inputs and 2,400 fresh solver calls.
Sources are real mono recordings with simulated synchronized propagation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sig_proc_tri as original
import sig_proc_tri_robust as robust
from benchmark_robust_mics import estimate_row, fingerprint, summary, write_csv, write_json

SOURCE_COMMIT = '312bcbee5fc84ced6652aac4df36ec7ce2c6eb0d'
ORIGINAL_SHA256 = '0e79e8661976912d04b20807f73985a4b58f2da4aefbb206f7f2d48d44e2a8ac'
ELEVATIONS = [0, 30, 60, 80, 90]
SNRS = [30, 20, 10, 0]
GAINS_DB = [-10, -20, -30]
FS = 48000
SAMPLES = 2048
CONFIDENCE_THRESHOLD = 0.2


def waveform_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()


def read_sources(cache_path):
    def historical(path):
        return json.loads(subprocess.check_output(['git', 'show', SOURCE_COMMIT+':'+path], cwd=ROOT))
    manifest = historical('benchmark_results/clip_manifest.json')
    metadata = historical('benchmark_results/metadata.json')
    with np.load(cache_path, allow_pickle=False) as cache:
        sources = cache['clips']
        if (sources.shape != (100, 8000) or int(cache['seed']) != metadata['seed'] or
                float(cache['win']) != metadata['window_seconds'] or
                json.loads(str(cache['manifest'])) != manifest):
            raise ValueError('Cache does not match the pinned original benchmark metadata')
    hashes = [waveform_hash(source) for source in sources]
    if hashes != [entry['normalized_waveform_sha256'] for entry in manifest['clips']] or len(set(hashes)) != 100:
        raise ValueError('Expected all 100 distinct original source waveform hashes')
    if metadata['fs'] != 16000 or metadata['window_seconds'] != 0.5:
        raise ValueError('Expected original 0.5-second sources at 16 kHz')
    from scipy.signal import resample_poly
    sources = resample_poly(sources, 3, 1, axis=1)
    start = (sources.shape[1]-SAMPLES)//2
    return sources[:, start:start+SAMPLES], manifest, metadata, start


def make_assignments(seed, sources, manifest):
    order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(100)
    angles = np.random.default_rng(np.random.SeedSequence([seed, 1])).uniform(-180, 180, 100)
    names = original.hardware.MIC_NAMES
    assignments = []
    for position, clip_index in enumerate(order):
        group = 'intact' if position < 50 else 'degraded'
        j = position % 50
        elevation = ELEVATIONS[j % 5]
        selected_channel = None
        primary_mode = 'intact'
        if group == 'degraded':
            if j < 40:
                selected_channel = (j//5) % 4
                primary_mode = 'noise_only' if j < 20 else 'weak_-30'
            elif j < 45:
                selected_channel = [0, 1, 2, 3, 0][j-40]
                primary_mode = 'noise_only'
            else:
                selected_channel = [1, 2, 3, 0, 1][j-45]
                primary_mode = 'weak_-30'
        clip_index = int(clip_index)
        entry = manifest['clips'][clip_index]
        assignments.append(dict(
            clip_index=clip_index, dataset_row=entry['dataset_row'], group=group,
            group_index=j, shuffled_position=position, true_elevation_deg=elevation,
            true_azimuth_deg=float(angles[clip_index]) if elevation != 90 else None,
            generated_azimuth_deg=float(angles[clip_index]),
            selected_channel=selected_channel,
            selected_mic=names[selected_channel] if selected_channel is not None else None,
            primary_mode=primary_mode, primary_nominal_snr_db=20,
            original_waveform_sha256=entry['normalized_waveform_sha256'],
            source_crop_sha256=waveform_hash(sources[clip_index])))
    first = {r['clip_index'] for r in assignments if r['group'] == 'intact'}
    second = {r['clip_index'] for r in assignments if r['group'] == 'degraded'}
    if len(first) != 50 or len(second) != 50 or first & second or first | second != set(range(100)):
        raise AssertionError('Source groups must be disjoint and cover all 100 original clips')
    for group in ['intact', 'degraded']:
        members = [r for r in assignments if r['group'] == group]
        if Counter(r['true_elevation_deg'] for r in members) != {el: 10 for el in ELEVATIONS}:
            raise AssertionError('Elevations must be balanced within each source group')
    degraded = [r for r in assignments if r['group'] == 'degraded']
    if Counter(r['selected_mic'] for r in degraded) != {'B': 13, 'L': 13, 'R': 12, 'T': 12}:
        raise AssertionError('Primary damaged microphones do not match the frozen assignment')
    if Counter(r['primary_mode'] for r in degraded) != {'noise_only': 25, 'weak_-30': 25}:
        raise AssertionError('Primary degradation models must each have 25 clips')
    if Counter((r['primary_mode'], r['true_elevation_deg']) for r in degraded) != {
            (mode, el): 5 for mode in ['noise_only', 'weak_-30'] for el in ELEVATIONS}:
        raise AssertionError('Primary degradation/elevation cells must each contain five clips')
    return assignments


def source_noise_metrics(clean, noise, gains, selected_channel):
    source_power = np.mean((clean*gains[:, None])**2, axis=1)
    noise_power = np.mean(noise**2, axis=1)
    snrs = [float(10*np.log10(s/n)) if s > 0 and n > 0 else None
            for s, n in zip(source_power, noise_power)]
    gain_db = [float(20*np.log10(gain)) if gain > 0 else None for gain in gains]
    mic = selected_channel
    return dict(
        channel_source_gains_linear_json=json.dumps(gains.tolist()),
        channel_source_gains_db_json=json.dumps(gain_db),
        channel_source_power_json=json.dumps(source_power.tolist()),
        channel_noise_power_json=json.dumps(noise_power.tolist()),
        measured_channel_snrs_db_json=json.dumps(snrs),
        selected_source_gain_linear=float(gains[mic]) if mic is not None else None,
        selected_source_gain_db=gain_db[mic] if mic is not None else None,
        selected_source_power=float(source_power[mic]) if mic is not None else None,
        selected_noise_power=float(noise_power[mic]) if mic is not None else None,
        measured_selected_channel_snr_db=snrs[mic] if mic is not None else None,
        selected_channel_has_no_source=bool(mic is not None and gains[mic] == 0))


def run_both(frames, truth, info):
    """Run both solvers fresh against the same read-only input; verify immutability."""
    frames = np.asarray(frames, dtype=float)
    frames.setflags(write=False)
    before = waveform_hash(frames)
    started = time.perf_counter_ns()
    old = original.estimate_direction_details(frames, fs=FS)
    elapsed = (time.perf_counter_ns()-started)/1e6
    old_values = dict(
        vector_defined=old.vector is not None, confidence=old.confidence,
        error_3d_deg=original.direction_error_deg(old.vector, truth),
        accepted=bool(old.vector is not None and old.confidence >= CONFIDENCE_THRESHOLD),
        ux=float(old.vector[0]) if old.vector is not None else None,
        uy=float(old.vector[1]) if old.vector is not None else None,
        uz=float(old.vector[2]) if old.vector is not None else None,
        azimuth_deg=original.azimuth_from_direction(old.vector),
        elevation_deg=original.elevation_from_direction(old.vector), estimator_ms=elapsed)
    if waveform_hash(frames) != before:
        raise AssertionError('Original solver changed the shared input')
    row = estimate_row(robust, frames, old_values, truth, info, CONFIDENCE_THRESHOLD)
    if waveform_hash(frames) != before:
        raise AssertionError('Robust solver changed the shared input')
    row.update(input_sha256=before, inputs_unchanged_after_both_solvers=True,
               robust_estimator_ms=row['estimator_ms'])
    return row


def aggregates(rows, field, values):
    return [dict(**{field: value}, **summary([r for r in rows if r[field] == value]))
            for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--clip-cache', type=Path, default=ROOT/'benchmark_results'/'clips.npz')
    parser.add_argument('--out', type=Path, default=ROOT/'results_hf_50_50')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    hashes = dict(original_estimator=fingerprint(ROOT/'sig_proc_tri.py'),
                  robust_estimator=fingerprint(ROOT/'sig_proc_tri_robust.py'),
                  benchmark_script=fingerprint(Path(__file__)),
                  benchmark_helper=fingerprint(ROOT/'scripts/benchmark_robust_mics.py'))
    if hashes['original_estimator'] != ORIGINAL_SHA256:
        raise ValueError('Original solver differs from the preserved implementation')
    sources, manifest, original_metadata, crop_start = read_sources(args.clip_cache)
    assignments = make_assignments(args.seed, sources, manifest)
    meta = dict(
        dataset=original_metadata['dataset'], source_commit=SOURCE_COMMIT,
        source_cache=str(args.clip_cache), source_cache_sha256=fingerprint(args.clip_cache),
        original_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        original_source_clips=100, all_source_hashes_verified=True, disjoint_groups_verified=True,
        source_group_sizes={'intact': 50, 'degraded': 50}, source_fs=16000, fs=FS,
        samples=SAMPLES, source_crop_start_resampled_samples=crop_start,
        source_crop_seconds=SAMPLES/FS, seed=args.seed,
        rng_streams=dict(membership='SeedSequence([seed,0]) permutation(100)',
                         angles='SeedSequence([seed,1]) uniform(-180,180,100), indexed by original clip_index',
                         added_noise='SeedSequence([seed,2,clip_index,nominal_snr_db])'),
        config=dict(elevations=ELEVATIONS, nominal_baseline_snrs_db=SNRS,
                    weak_source_gains_db=GAINS_DB, primary_nominal_snr_db=20,
                    primary_case_count=100, confidence_threshold=CONFIDENCE_THRESHOLD,
                    primary_degraded_model_counts={'noise_only': 25, 'weak_-30': 25},
                    primary_degraded_mic_counts={'B': 13, 'L': 13, 'R': 12, 'T': 12},
                    primary_degraded_assignment=(
                        'j<40: el=ELEVATIONS[j%5], mic=(j//5)%4, noise_only for j<20 else weak_-30; '
                        'j40..44: noise_only, mic=[0,1,2,3,0]; '
                        'j45..49: weak_-30, mic=[1,2,3,0,1]; el=ELEVATIONS[j%5]'),
                    channel_order=original.hardware.MIC_NAMES, mics_xyz_m=original.MICS.tolist(),
                    sound_speed_m_s=original.C, hemisphere_prior=None,
                    assume_source_facing=False,
                    solver_options=('Original defaults with fs=48000; robust defaults with '
                                    'fs=48000 and assume_source_facing=False for this historical no-prior protocol')),
        expected_scenarios=1200, expected_solver_calls=2400,
        confidence_threshold_calibrated=False, parameter_tuning=False,
        truth_used_for_estimation=False, input_immutability_method='Read-only arrays plus before/after SHA256 for both estimators',
        noise_only_snr_representation='Null SNR/gain dB means zero source power (-infinity); linear gain and powers remain explicit',
        comparison_scope='New disjoint within-run source groups; these recordings were used in earlier algorithm development',
        hashes_sha256=hashes, numpy_version=np.__version__,
        started_utc=datetime.now(timezone.utc).isoformat(),
        command='python3 scripts/benchmark_50_50.py '+' '.join(sys.argv[1:]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'provenance.json', meta)
    write_json(args.out/'clip_manifest.json', manifest)
    write_csv(args.out/'clip_assignments.csv', assignments)
    rows = []
    print('Verified 100 distinct sources; disjoint groups and primary model/mic/elevation balance passed.', flush=True)
    for assignment in assignments:
        clip_index = assignment['clip_index']
        source = sources[clip_index]
        elevation = assignment['true_elevation_deg']
        azimuth = float(assignment['true_azimuth_deg'] or 0)
        truth = original.unit_vector(azimuth, elevation)
        clean = original.simulate_array(source, azimuth, np.inf, np.random.default_rng(0),
                                        fs=FS, elevation_deg=elevation)
        for snr in SNRS:
            stream = [args.seed, 2, clip_index, snr]
            baseline = original.simulate_array(source, azimuth, snr, np.random.default_rng(
                np.random.SeedSequence(stream)), fs=FS, elevation_deg=elevation)
            noise = baseline-clean
            selected = assignment['selected_channel']
            modes = [('intact', 1.0)]
            if assignment['group'] == 'degraded':
                modes += [('noise_only', 0.0)] + [(f'weak_{gain}', 10**(gain/20)) for gain in GAINS_DB]
            control_id = f'clip_{clip_index:03d}_snr_{snr}_intact'
            healthy = [i for i in range(4) if i != selected]
            control_healthy_hash = waveform_hash(baseline[healthy])
            for mode, gain in modes:
                frames = baseline.copy()
                gains = np.ones(4)
                if selected is not None and mode != 'intact':
                    frames[selected] = gain*clean[selected]+noise[selected]
                    gains[selected] = gain
                if not np.array_equal(frames[healthy], baseline[healthy]):
                    raise AssertionError('A healthy channel differs from the matched intact control')
                case_id = f'clip_{clip_index:03d}_snr_{snr}_{mode}'
                primary = snr == 20 and mode == assignment['primary_mode']
                info = dict(
                    case_id=case_id, group=assignment['group'], group_index=assignment['group_index'],
                    mode=mode, is_primary=primary, is_matched_control=assignment['group'] == 'degraded' and mode == 'intact',
                    matched_control_id=control_id, clip_index=clip_index,
                    dataset_row=assignment['dataset_row'], fs=FS, samples=SAMPLES,
                    true_azimuth_deg=assignment['true_azimuth_deg'], true_elevation_deg=elevation,
                    snr_db=snr, selected_channel=selected, selected_mic=assignment['selected_mic'],
                    damaged_channel=selected if mode != 'intact' else None,
                    damaged_mic=assignment['selected_mic'] if mode != 'intact' else None,
                    original_waveform_sha256=assignment['original_waveform_sha256'],
                    source_crop_sha256=assignment['source_crop_sha256'],
                    added_noise_stream_json=json.dumps(stream), added_noise_sha256=waveform_hash(noise),
                    healthy_channels_sha256=waveform_hash(frames[healthy]),
                    control_healthy_channels_sha256=control_healthy_hash,
                    healthy_channels_identical_to_control=True,
                    **source_noise_metrics(clean, noise, gains, selected))
                rows.append(run_both(frames, truth, info))
        if (assignment['shuffled_position']+1) % 10 == 0:
            print(f"Completed {assignment['shuffled_position']+1}/100 source clips; {len(rows)} scenarios.", flush=True)
    primary = [r for r in rows if r['is_primary']]
    if (len(rows) != 1200 or len(primary) != 100 or len({r['case_id'] for r in rows}) != 1200
            or Counter(r['group'] for r in primary) != {'intact': 50, 'degraded': 50}):
        raise AssertionError('Complete sweep/primary coverage mismatch')
    if any(fingerprint(path) != hashes[key] for key, path in [
            ('original_estimator', ROOT/'sig_proc_tri.py'), ('robust_estimator', ROOT/'sig_proc_tri_robust.py'),
            ('benchmark_script', Path(__file__)), ('benchmark_helper', ROOT/'scripts/benchmark_robust_mics.py')]):
        raise AssertionError('A solver, benchmark, or helper changed during the run')
    primary_summary = [dict(selection='all_100', **summary(primary))]
    primary_summary += [dict(selection=group, **summary([r for r in primary if r['group'] == group]))
                        for group in ['intact', 'degraded']]
    fault_rows = [r for r in rows if r['mode'] != 'intact']
    result = dict(
        overall=summary(rows), by_primary=primary_summary,
        by_group=aggregates(rows, 'group', ['intact', 'degraded']),
        by_mode=aggregates(rows, 'mode', ['intact', 'noise_only', 'weak_-10', 'weak_-20', 'weak_-30']),
        by_snr=aggregates(rows, 'snr_db', SNRS),
        by_mic=aggregates(fault_rows, 'damaged_mic', original.hardware.MIC_NAMES),
        by_elevation=aggregates(rows, 'true_elevation_deg', ELEVATIONS),
        by_primary_mode=aggregates(primary, 'mode', ['intact', 'noise_only', 'weak_-30']),
        by_primary_elevation=aggregates(primary, 'true_elevation_deg', ELEVATIONS),
        by_group_snr=[dict(group=group, snr_db=snr, **summary([
            r for r in rows if r['group'] == group and r['snr_db'] == snr]))
            for group in ['intact', 'degraded'] for snr in SNRS],
        by_mode_snr=[dict(mode=mode, snr_db=snr, **summary([
            r for r in rows if r['mode'] == mode and r['snr_db'] == snr]))
            for mode in ['intact', 'noise_only', 'weak_-10', 'weak_-20', 'weak_-30'] for snr in SNRS])
    write_csv(args.out/'trial_level.csv', rows)
    write_csv(args.out/'primary_100.csv', primary)
    write_json(args.out/'summary.json', result)
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), scenarios=len(rows), solver_calls=2*len(rows),
                primary_cases=len(primary), source_groups_disjoint=True, all_inputs_unchanged=True,
                source_group_membership={group: [r['clip_index'] for r in assignments if r['group'] == group]
                                         for group in ['intact', 'degraded']})
    write_json(args.out/'provenance.json', meta)
    lines = ['SEEDED 50/50 FOUR-MIC COMPARISON', '',
             '100 distinct source clips, 1,200 paired scenarios, 2,400 fresh estimator calls.',
             'Primary: 20 dB baseline SNR; 50 intact and 50 degraded inputs from disjoint source groups.',
             'Selection      Cases  Original accepted/bad>15  Robust accepted/bad>15  Robust status counts']
    for metrics in primary_summary:
        lines.append(f"{metrics['selection']:12} {metrics['trials']:5d} "
                     f"{metrics['original_accepted_count']:10d}/{metrics['original_accepted_over15_count']:<6d} "
                     f"{metrics['robust_accepted_count']:13d}/{metrics['robust_accepted_over15_count']:<6d} "
                     f"{metrics['status_counts']}")
    lines += ['', 'No hemisphere prior or known direction supplied to either solver. Confidence 0.2 is uncalibrated.',
              'Candidate sets are ambiguity diagnostics, not uniquely accepted localization successes.',
              'These previously used mono sources have simulated delays/noise; no helmet or hardware validation.',
              'The disjoint within-run groups are not recordings held out from algorithm development.',
              '', 'Reproduce: '+meta['command']]
    (args.out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(primary_summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
