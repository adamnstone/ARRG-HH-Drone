"""Replay Rachel's saved three-mic failures against the current four-mic solver.

Uses the exact cached real mono waveforms; no download or model is needed.
Outputs are matched simulated scenarios, not recordings from a physical array.
Run: python3 scripts/replay_rachel_failures.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sig_proc_tri as current


def branch_bytes(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def waveform_hash(signal):
    return hashlib.sha256(np.asarray(signal, dtype='<f8').tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def noise_rng(meta, row):
    return np.random.default_rng(np.random.SeedSequence([
        meta['seed'], 1, row['clip_index'], row['azimuth_index'],
        row['elevation_deg'], row['snr_db']]))


def normalize_trial(row):
    return dict(clip_index=int(row['clip_index']), azimuth_index=int(row['azimuth_index']),
                azimuth_deg=float(row.get('azimuth_deg', row.get('azimuth_true_deg'))),
                elevation_deg=int(row['elevation_deg']), snr_db=int(row['snr_db']))


def trial_key(row):
    return tuple(row[key] for key in ('clip_index', 'azimuth_index', 'elevation_deg', 'snr_db'))


def valid_angle(angle):
    return angle is not None and np.isfinite(angle)


def score_trial(row, clips, meta, old, threshold):
    signal = clips[row['clip_index']]
    az, el = row['azimuth_deg'], row['elevation_deg']
    old_frames = old.simulate_array(signal, az, row['snr_db'], noise_rng(meta, row),
                                    fs=meta['fs'], elevation_deg=el)
    old_az, old_confidence, _, _ = old.estimate_direction(old_frames, fs=meta['fs'])
    old_error = old.ang_err(old_az, az) if valid_angle(old_az) else None
    new_frames = current.simulate_array(signal, az, row['snr_db'], noise_rng(meta, row),
                                       fs=meta['fs'], elevation_deg=el)
    result = current.estimate_direction_details(new_frames, fs=meta['fs'])
    vector = result.vector
    new_az = current.azimuth_from_direction(vector)
    new_el = current.elevation_from_direction(vector)
    error = current.direction_error_deg(vector, current.unit_vector(az, el))
    accepted = vector is not None and result.confidence >= threshold
    status = 'undefined' if vector is None else ('accepted' if accepted else 'low_confidence')
    return dict(**row,
                old_azimuth_deg=float(old_az) if valid_angle(old_az) else None,
                old_azimuth_error_deg=old_error, old_confidence=float(old_confidence),
                new_azimuth_deg=new_az, new_elevation_deg=new_el,
                new_azimuth_error_deg=current.ang_err(new_az, az) if new_az is not None else None,
                new_3d_error_deg=error, new_confidence=result.confidence,
                new_ux=float(vector[0]) if vector is not None else None,
                new_uy=float(vector[1]) if vector is not None else None,
                new_uz=float(vector[2]) if vector is not None else None,
                ambiguous_pair_count=int(np.count_nonzero(result.ambiguous_pairs)),
                old_accepted_at_demo_threshold=bool(valid_angle(old_az) and old_confidence >= threshold),
                new_status=status, new_accepted=accepted,
                new_azimuth_within_15_deg=bool(new_az is not None and current.ang_err(new_az, az) <= 15),
                new_3d_within_15_deg=bool(error is not None and error <= 15),
                new_accepted_3d_within_15_deg=bool(accepted and error <= 15))


def summarize(rows):
    errors = [r['new_3d_error_deg'] for r in rows if r['new_3d_error_deg'] is not None]
    az_errors = [r['new_azimuth_error_deg'] for r in rows if r['new_azimuth_error_deg'] is not None]
    accepted = [r for r in rows if r['new_accepted']]
    return dict(trials=len(rows), new_undefined=sum(r['new_status'] == 'undefined' for r in rows),
                new_low_confidence=sum(r['new_status'] == 'low_confidence' for r in rows),
                new_accepted=len(accepted),
                old_accepted_at_demo_threshold=sum(r['old_accepted_at_demo_threshold'] for r in rows),
                new_raw_azimuth_within15_count=sum(r['new_azimuth_within_15_deg'] for r in rows),
                new_raw_3d_within15_count=sum(r['new_3d_within_15_deg'] for r in rows),
                new_accepted_3d_within15_count=sum(r['new_accepted_3d_within_15_deg'] for r in rows),
                new_accepted_3d_over15_count=sum(r['new_3d_error_deg'] > 15 for r in accepted),
                new_median_3d_error_deg=float(np.median(errors)) if errors else None,
                new_p90_3d_error_deg=float(np.percentile(errors, 90)) if errors else None,
                new_median_azimuth_error_deg=float(np.median(az_errors)) if az_errors else None)


def periodic_stress(old, threshold):
    """Rachel's periodic characterization, preserving its sequential RNG stream."""
    rng = np.random.default_rng(20261016)
    times = np.arange(4096) / old.FS
    signal = sum(np.cos(2 * np.pi * frequency * times + phase) for frequency, phase in
                 [(2200, 0.2), (4400, 1.1), (6600, 0.7)])
    signal /= np.sqrt(np.mean(signal ** 2))
    rows = []
    for snr in [30, 0]:
        for elevation in [0, 60]:
            for azimuth in [-170, -135, -100, -90, -55, -15, 0, 25, 60, 90, 120, 155]:
                four_rng = np.random.default_rng()
                four_rng.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
                old_frames = old.simulate_array(signal, azimuth, snr, rng, elevation_deg=elevation)
                old_az, old_conf, _, _ = old.estimate_direction(old_frames)
                frames = current.simulate_array(signal, azimuth, snr, four_rng, fs=old.FS,
                                               elevation_deg=elevation)
                vector, conf, _, _ = current.estimate_direction(frames, fs=old.FS)
                az = current.azimuth_from_direction(vector)
                error = current.direction_error_deg(vector, current.unit_vector(azimuth, elevation))
                accepted = vector is not None and conf >= threshold
                rows.append(dict(azimuth_deg=azimuth, elevation_deg=elevation, snr_db=snr,
                                 old_azimuth_error_deg=old.ang_err(old_az, azimuth) if valid_angle(old_az) else None,
                                 old_confidence=old_conf,
                                 new_azimuth_error_deg=current.ang_err(az, azimuth) if az is not None else None,
                                 new_3d_error_deg=error, new_confidence=conf,
                                 new_status='undefined' if vector is None else ('accepted' if accepted else 'low_confidence'),
                                 new_accepted=accepted, new_3d_within_15_deg=bool(error is not None and error <= 15)))
    return rows


def write_report(out, saved, rows, threshold, periodic=None):
    summary = summarize(rows)
    lines = [
        "RACHEL THREE-MIC FAILURES: CURRENT FOUR-MIC REPLAY",
        "",
        "Exact old results reproduced using all 100 hash-verified cached mono waveforms.",
        "Selection: old GCC non-overhead azimuth error >15 degrees or undefined.",
        "Evaluation: same clips, directions, 16 kHz rate, SNR and noise seed recipe;",
        "new B/L/R/T geometry and channel count require newly simulated arrays.",
        "",
        "SIX SAVED EXAMPLES (all 0 dB added noise)",
        "Case    Old az error    New az error    New 3D error    Confidence    Decision",
    ]
    def fmt(value):
        return 'undefined' if value is None else f'{value:.2f}'
    for row in saved:
        lines.append(f"{row['case']:4d} {fmt(row['old_azimuth_error_deg']):>15} "
                     f"{fmt(row['new_azimuth_error_deg']):>15} {fmt(row['new_3d_error_deg']):>15} "
                     f"{row['new_confidence']:13.6f}    {row['new_status']}")
    lines += [
        "",
        f"Saved cases withheld at demo threshold {threshold}: "
        f"{len(saved) - sum(r['new_accepted'] for r in saved)}/{len(saved)}.",
        f"Saved raw 3D estimates within 15 degrees: "
        f"{sum(r['new_3d_within_15_deg'] for r in saved)}/{len(saved)}.",
        f"The clean-control maximum 3D error is {max(r['new_clean_3d_error_deg'] for r in saved):.8f} degrees.",
        "",
        f"ALL {len(rows):,} PREVIOUSLY FAILED TRIALS",
        f"Raw azimuth within 15 degrees: {summary['new_raw_azimuth_within15_count']}/{len(rows)}",
        f"Raw 3D direction within 15 degrees: {summary['new_raw_3d_within15_count']}/{len(rows)}",
        f"Accepted at demo threshold: {summary['new_accepted']}/{len(rows)}",
        f"Accepted with 3D error >15 degrees: {summary['new_accepted_3d_over15_count']}",
        f"Low confidence: {summary['new_low_confidence']}; undefined 3D: {summary['new_undefined']}",
        f"Raw 3D error median: {summary['new_median_3d_error_deg']:.3f} degrees; "
        f"p90: {summary['new_p90_3d_error_deg']:.3f} degrees (defined vectors only)",
        "",
        "INTERPRETATION",
        "This is conditional replay of earlier failures, not overall accuracy.",
        f"The old estimator would withhold "
        f"{len(rows) - summary['old_accepted_at_demo_threshold']}/{len(rows)} "
        f"at confidence {threshold}.",
        "Withholding a result is not a correct direction; the demo threshold is uncalibrated.",
        "Historical azimuth error and new 3D error measure different things; both new",
        "azimuth and 3D errors are included. Near-overhead azimuth is ill-conditioned.",
        "The arrays have different geometry, channel order/noise assignment and a slight",
        "common-delay alignment change; this does not isolate the effect of mic count.",
        "These are real mono recordings with simulated propagation, not hardware array recordings.",
        "",
        "Reproduce: python3 scripts/replay_rachel_failures.py",
        "Requires branch rachel and benchmark_results/clips.npz; core execution needs only NumPy.",
        "See provenance.json, saved_failure_comparison.csv, all_failed_trials.csv and summary.json.",
    ]
    if periodic is not None:
        lines += ["", "HISTORICAL PERIODIC STRESS CHARACTERIZATION",
                  "2200/4400/6600 Hz source, N=4096, fs=16000, seed=20261016;",
                  "12 azimuths x 2 elevations (0/60) x 2 SNRs (30/0).",
                  "Old sequential noise stream preserved; each state copied for new four-channel audio."]
        for snr in [30, 0]:
            group = [r for r in periodic if r['snr_db'] == snr]
            errs = [r['new_3d_error_deg'] for r in group if r['new_3d_error_deg'] is not None]
            lines.append(f"SNR {snr}: raw 3D within 15 degrees "
                         f"{sum(r['new_3d_within_15_deg'] for r in group)}/{len(group)}; "
                         f"accepted {sum(r['new_accepted'] for r in group)}; "
                         f"median 3D error {float(np.median(errs)):.2f} degrees.")
        lines += ["This historical section had no accuracy assertion; it characterizes a known limitation.",
                  "Four mics still do not resolve this strongly periodic source; apply confidence gating.",
                  "See periodic_stress.csv for all raw results."]
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref', default='rachel')
    parser.add_argument('--clip-cache', type=Path, default=ROOT / 'benchmark_results' / 'clips.npz')
    parser.add_argument('--out', type=Path, default=ROOT / 'results_rachel_replay')
    parser.add_argument('--min-confidence', type=float, default=0.2,
                        help='uncalibrated demo threshold, reported separately from raw accuracy')
    args = parser.parse_args()
    if not np.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        parser.error('--min-confidence must be in [0,1]')
    # Pin the branch once so concurrent commits cannot mix historical artifacts.
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.source_ref + '^{commit}'],
                                     cwd=ROOT, text=True).strip()
    meta = json.loads(branch_bytes(commit, 'benchmark_results/metadata.json'))
    manifest = json.loads(branch_bytes(commit, 'benchmark_results/clip_manifest.json'))
    cases = json.loads(branch_bytes(commit, 'benchmark_results/failure_cases.json'))
    source = branch_bytes(commit, 'sig_proc_tri.py')
    if hashlib.sha256(source).hexdigest() != meta['estimator_sha256']:
        raise ValueError('Historical estimator source no longer matches the recorded benchmark')
    with np.load(args.clip_cache, allow_pickle=False) as cache:
        clips = cache['clips']
        if int(cache['seed']) != meta['seed'] or float(cache['win']) != meta['window_seconds']:
            raise ValueError('Clip cache seed/window does not match historical metadata')
        if json.loads(str(cache['manifest'])) != manifest:
            raise ValueError('Clip-cache manifest differs from historical manifest')
    if clips.shape != (meta['clips'], round(meta['window_seconds'] * meta['fs'])):
        raise ValueError('Clip cache dimensions differ from the historical run')
    for signal, record in zip(clips, manifest['clips']):
        if waveform_hash(signal) != record['normalized_waveform_sha256']:
            raise ValueError(f"Clip hash mismatch at {record['clip_index']}")
    old = types.ModuleType('rachel_three_mic_baseline')
    exec(compile(source, f'{commit}:sig_proc_tri.py', 'exec'), old.__dict__)
    trials = list(csv.DictReader(io.StringIO(branch_bytes(commit, 'benchmark_results/trials.csv').decode())))
    failures = [r for r in trials if r['method'] == 'gcc_ls' and int(r['elevation_deg']) != 90
                and (r['valid'] != 'True' or float(r['error_deg']) > 15)]
    print(f'Verified all {len(clips)} cached waveform hashes. Replaying {len(failures)} historical failures.', flush=True)
    scored, saved = [], []
    by_key = {}
    for index, recorded in enumerate(failures):
        row = score_trial(normalize_trial(recorded), clips, meta, old, args.min_confidence)
        np.testing.assert_allclose(row['old_confidence'], float(recorded['confidence']), atol=1e-12, rtol=0)
        if recorded['valid'] == 'True':
            np.testing.assert_allclose(row['old_azimuth_deg'], float(recorded['azimuth_est_deg']), atol=1e-9, rtol=0)
        elif row['old_azimuth_deg'] is not None:
            raise AssertionError('Historical undefined result did not reproduce')
        scored.append(row)
        by_key[trial_key(row)] = row
        if (index + 1) % 200 == 0:
            print(f'Replayed {index + 1}/{len(failures)}', flush=True)
    for case in cases:
        row = dict(by_key[trial_key(normalize_trial(case))])
        signal = clips[row['clip_index']]
        clean_frames = current.simulate_array(signal, row['azimuth_deg'], np.inf, np.random.default_rng(0),
                                             fs=meta['fs'], elevation_deg=row['elevation_deg'])
        clean = current.estimate_direction_details(clean_frames, fs=meta['fs'])
        saved.append(dict(case=case['case'], **row,
                          new_clean_3d_error_deg=current.direction_error_deg(
                              clean.vector, current.unit_vector(row['azimuth_deg'], row['elevation_deg'])),
                          new_clean_confidence=clean.confidence))
    grouped = []
    for elevation, snr in sorted({(r['elevation_deg'], r['snr_db']) for r in scored}):
        group = [r for r in scored if r['elevation_deg'] == elevation and r['snr_db'] == snr]
        grouped.append(dict(elevation_deg=elevation, snr_db=snr, **summarize(group)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'saved_failure_comparison.csv', saved)
    write_json(args.out / 'saved_failure_comparison.json', saved)
    write_csv(args.out / 'all_failed_trials.csv', scored)
    write_json(args.out / 'summary.json', dict(saved_cases=summarize(saved),
                                             all_historical_failures=summarize(scored), by_condition=grouped))
    write_json(args.out / 'provenance.json', dict(
        source_ref=args.source_ref, source_commit=commit, source_metadata=meta,
        cache_path=str(args.clip_cache), all_waveform_hashes_verified=True,
        original_failed_trials_exactly_reproduced=len(scored),
        current_estimator_sha256=hashlib.sha256((ROOT / 'sig_proc_tri.py').read_bytes()).hexdigest(),
        replay_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        four_mic_positions=current.MICS.tolist(), fs=meta['fs'], current_sound_speed=current.C,
        current_numpy_version=np.__version__, current_python_version=sys.version,
        confidence_threshold=args.min_confidence, confidence_threshold_calibrated=False,
        failure_selection='Historical gcc_ls, non-overhead, azimuth error >15 deg OR undefined',
        comparison_scope='Conditional replay of previously failed scenarios, not overall accuracy',
        simulation='Same exact real mono clips, truth, SNR and SeedSequence; new four-mic geometry/channel order',
        caveats=['New arrays cannot receive identical three-channel input; per-channel noise mapping changes',
                 'Current simulator also removes common delay, slightly changing source alignment/cropping',
                 'Historical errors are azimuth only; new azimuth and full 3D errors are reported separately',
                 'Rejected and low-confidence vectors are withheld, not counted as correct directions',
                 'This is simulated propagation, not real synchronized microphone-array audio']))
    periodic = periodic_stress(old, args.min_confidence)
    write_csv(args.out / 'periodic_stress.csv', periodic)
    write_json(args.out / 'periodic_stress.json', periodic)
    write_report(args.out, saved, scored, args.min_confidence, periodic)
    print('Saved six-case results:')
    for row in saved:
        def fmt(value): return 'undefined' if value is None else f'{value:.2f}'
        print(f"case {row['case']}: old az error {fmt(row['old_azimuth_error_deg'])}; "
              f"new az {fmt(row['new_azimuth_error_deg'])}, 3D {fmt(row['new_3d_error_deg'])}; "
              f"confidence {row['new_confidence']:.4f}; {row['new_status']}")
    print(json.dumps(summarize(scored), indent=2))
    print(f'Results: {args.out}')


if __name__ == '__main__':
    main()
