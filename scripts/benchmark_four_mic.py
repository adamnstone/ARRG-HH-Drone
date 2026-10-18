"""Evaluate all 100 cached Hugging Face drone clips with four-mic propagation.

Default protocol matches Rachel's original 13,200 scenarios, including zenith.
Cached source waveforms are verified against the historical branch manifest.
No fitting or parameter tuning occurs. Requires only NumPy at 16 kHz; other
rates also require scipy for resampling. Run from the repo:
    python3 scripts/benchmark_four_mic.py
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

ELEVATIONS = [0, 30, 60, 80, 90]
SNRS = [30, 20, 10, 0]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    errors = np.array([r['error_3d_deg'] for r in rows if r['vector_defined']])
    elevation_errors = [r['elevation_error_deg'] for r in rows if r['vector_defined']]
    az_errors = [r['azimuth_error_deg'] for r in rows if r['azimuth_error_deg'] is not None]
    accepted = [r for r in rows if r['accepted']]
    accepted_errors = [r['error_3d_deg'] for r in accepted]
    within15 = sum(r['vector_defined'] and r['error_3d_deg'] <= 15 for r in rows)
    accepted_good = sum(r['error_3d_deg'] <= 15 for r in accepted)
    high = [r for r in rows if r['true_elevation_deg'] >= 80]
    low = [r for r in rows if r['true_elevation_deg'] <= 30]
    return dict(trials=len(rows), clips=len({r['clip_index'] for r in rows}),
                defined_count=len(errors), undefined_count=len(rows)-len(errors),
                median_3d_error_deg=float(np.median(errors)) if len(errors) else None,
                p90_3d_error_deg=float(np.percentile(errors, 90)) if len(errors) else None,
                max_3d_error_deg=float(np.max(errors)) if len(errors) else None,
                raw_within15_count=within15, raw_within15_all_pct=100*within15/len(rows),
                raw_over30_count=int(np.count_nonzero(errors > 30)),
                raw_over90_count=int(np.count_nonzero(errors > 90)),
                median_azimuth_error_deg=float(np.median(az_errors)) if az_errors else None,
                median_elevation_error_deg=float(np.median(elevation_errors)) if elevation_errors else None,
                accepted_count=len(accepted), accepted_pct=100*len(accepted)/len(rows),
                accepted_within15_count=accepted_good,
                accepted_within15_pct=100*accepted_good/len(accepted) if accepted else None,
                accepted_within15_all_pct=100*accepted_good/len(rows),
                accepted_over15_count=len(accepted)-accepted_good,
                accepted_over30_count=sum(e > 30 for e in accepted_errors),
                accepted_max_3d_error_deg=max(accepted_errors) if accepted_errors else None,
                high_elevation_trials=len(high),
                high_elevation_pointed_above60_count=sum(r['vector_defined'] and r['estimated_elevation_deg'] > 60 for r in high),
                high_elevation_accepted_above60_count=sum(r['accepted'] and r['estimated_elevation_deg'] > 60 for r in high),
                low_elevation_trials=len(low),
                low_elevation_pointed_above60_count=sum(r['vector_defined'] and r['estimated_elevation_deg'] > 60 for r in low),
                low_elevation_accepted_above60_count=sum(r['accepted'] and r['estimated_elevation_deg'] > 60 for r in low),
                mean_estimator_ms=float(np.mean([r['estimator_ms'] for r in rows])))


def report(out, meta, by_snr, overall):
    lines = ["100 HUGGING FACE CLIPS: FOUR-MIC DIRECTION BENCHMARK", "",
             f"Source: {meta['dataset']}; 100 distinct hash-verified original waveforms.",
             f"Window: {meta['window_seconds']} s; analysis rate: {meta['fs']} Hz.",
             f"Geometry: B/L/R/T {meta['mics']} metres, x right / y forward / z up.",
             f"Azimuths per clip per non-overhead elevation: {meta['azimuths_per_clip']}.",
             "Elevations: 0, 30, 60, 80, 90 degrees; SNR: 30, 20, 10, 0 dB.",
             f"Total estimator calls: {overall['trials']:,}; confidence threshold: {meta['min_confidence']} (uncalibrated).",
             "", "ALL ELEVATIONS COMBINED (3D angular error in degrees)",
             "SNR   Trials   Median     P90   Raw<=15%  Accepted%  Accepted<=15%  AcceptedBad>15"]
    for row in by_snr:
        quality = f"{row['accepted_within15_pct']:.2f}" if row['accepted_within15_pct'] is not None else 'undefined'
        lines.append(f"{row['snr_db']:3d} {row['trials']:8d} {row['median_3d_error_deg']:8.3f} "
                     f"{row['p90_3d_error_deg']:7.3f} {row['raw_within15_all_pct']:10.2f} "
                     f"{row['accepted_pct']:10.2f} {quality:>14} {row['accepted_over15_count']:14d}")
    lines += ["", "Scope and interpretation:",
              "Real MONO drone recordings, with simulated synchronized four-mic arrival delays",
              "and independent added white noise. Existing recording noise propagates coherently",
              "with the drone source. This is not hardware or diffuse-noise validation.",
              "All input clips are used; no historical-failure selection or parameter tuning.",
              "Raw<=15 uses ALL trials: an undefined vector is a failure. Median/P90 use defined vectors.",
              "Accepted means defined vector and confidence >= threshold; withheld results are not successes.",
              "Accepted<=15 uses accepted trials only; consult accepted% to see coverage.",
              "Azimuth is not scored at zenith; the 3D direction and elevation are scored there.",
              "Above60 counters are instantaneous elevation checks, not the live stateful overhead decision.",
              "Per-clip repetitions share source audio and are not independent recordings.",
              "", "Detailed results: trials.csv, by_condition.csv, by_snr.csv, by_clip.csv, summary.json, provenance.json.",
              "Reproduce: " + meta['command']]
    (out / 'REPORT.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-cache', type=Path, default=ROOT/'benchmark_results'/'clips.npz')
    parser.add_argument('--source-ref', default='rachel')
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--azimuths', type=int, default=8)
    parser.add_argument('--min-confidence', type=float, default=0.2)
    parser.add_argument('--out', type=Path, default=ROOT/'results_hf_four_mic')
    args = parser.parse_args()
    if args.fs < 8000 or args.azimuths < 1:
        parser.error('sample rate >=8000 and at least one azimuth required')
    if not np.isfinite(args.min_confidence) or not 0 <= args.min_confidence <= 1:
        parser.error('confidence threshold must be in [0,1]')
    commit = subprocess.check_output(['git','rev-parse','--verify',args.source_ref+'^{commit}'],cwd=ROOT,text=True).strip()
    def historical(path):
        return json.loads(subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=ROOT))
    original = historical('benchmark_results/metadata.json')
    manifest = historical('benchmark_results/clip_manifest.json')
    with np.load(args.clip_cache, allow_pickle=False) as cache:
        clips = cache['clips']
        if int(cache['seed']) != original['seed'] or float(cache['win']) != original['window_seconds']:
            raise ValueError('Cache seed/window differs from original dataset benchmark')
        if json.loads(str(cache['manifest'])) != manifest:
            raise ValueError('Cached and original manifests differ')
    if len(clips) != 100 or clips.shape != (100, round(original['fs']*original['window_seconds'])):
        raise ValueError('Expected all 100 original 0.5-second clips')
    hashes = [hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest() for x in clips]
    if hashes != [r['normalized_waveform_sha256'] for r in manifest['clips']] or len(set(hashes)) != 100:
        raise ValueError('Source waveform hashes must match and be distinct')
    source_fs = original['fs']
    if args.fs != source_fs:
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(args.fs, source_fs)
        clips = resample_poly(clips, args.fs//divisor, source_fs//divisor, axis=1)
    seed = original['seed']
    spacing = 360 / args.azimuths
    angles = doa.wrap_deg(-180 + np.arange(args.azimuths)[None,:]*spacing +
                          np.random.default_rng(np.random.SeedSequence([seed,0])).uniform(0,spacing,(100,1)))
    meta = dict(dataset=original['dataset'], source_ref=args.source_ref, source_commit=commit,
                source_metadata=original, clip_cache=str(args.clip_cache), all_source_hashes_verified=True,
                source_fs=source_fs, fs=args.fs, resampled=args.fs != source_fs,
                resampling='scipy.signal.resample_poly' if args.fs != source_fs else None,
                clips=100, window_seconds=original['window_seconds'], seed=seed,
                azimuths_per_clip=args.azimuths, elevations=ELEVATIONS, snrs=SNRS,
                mics=doa.MICS.tolist(), sound_speed=doa.C, min_confidence=args.min_confidence,
                confidence_threshold_calibrated=False, parameter_tuning=False,
                angle_design='Even full-circle spacing with seeded independent per-clip offset; zenith azimuth0',
                noise_design='SeedSequence([seed,1,clip_index,azimuth_index,elevation,snr]); independent white noise',
                estimator_sha256=hashlib.sha256((ROOT/'sig_proc_tri.py').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=np.__version__,
                started_utc=datetime.now(timezone.utc).isoformat(),
                command='python3 scripts/benchmark_four_mic.py '+ ' '.join(sys.argv[1:]))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'clip_manifest.json',manifest)
    write_json(args.out/'provenance.json',meta)
    rows, conditions = [], []
    print(f'Verified 100 source clips. Running at {args.fs} Hz.',flush=True)
    for elevation in ELEVATIONS:
        for snr in SNRS:
            group = []
            for clip_index, signal in enumerate(clips):
                for ai in range(1 if elevation == 90 else args.azimuths):
                    az = float(angles[clip_index,ai]) if elevation != 90 else 0.0
                    rng = np.random.default_rng(np.random.SeedSequence([seed,1,clip_index,ai,elevation,snr]))
                    frames = doa.simulate_array(signal,az,snr,rng,fs=args.fs,elevation_deg=elevation)
                    start = time.perf_counter_ns()
                    result = doa.estimate_direction_details(frames,fs=args.fs)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    vector = result.vector
                    estimated_az = doa.azimuth_from_direction(vector)
                    estimated_el = doa.elevation_from_direction(vector)
                    error = doa.direction_error_deg(vector,doa.unit_vector(az,elevation))
                    row = dict(clip_index=clip_index, azimuth_index=ai, fs=args.fs,
                               true_azimuth_deg=az if elevation != 90 else None, true_elevation_deg=elevation,
                               snr_db=snr, vector_defined=vector is not None,
                               estimated_ux=float(vector[0]) if vector is not None else None,
                               estimated_uy=float(vector[1]) if vector is not None else None,
                               estimated_uz=float(vector[2]) if vector is not None else None,
                               estimated_azimuth_deg=estimated_az,estimated_elevation_deg=estimated_el,
                               azimuth_error_deg=doa.ang_err(estimated_az,az) if elevation != 90 and estimated_az is not None else None,
                               elevation_error_deg=abs(estimated_el-elevation) if estimated_el is not None else None,
                               error_3d_deg=error,confidence=result.confidence,
                               accepted=bool(vector is not None and result.confidence >= args.min_confidence),
                               ambiguous_pair_count=int(np.count_nonzero(result.ambiguous_pairs)),estimator_ms=elapsed)
                    group.append(row)
            rows.extend(group)
            condition=dict(elevation_deg=elevation,snr_db=snr,**summarize(group))
            conditions.append(condition)
            write_csv(args.out/'by_condition.csv',conditions)
            print(f"elev {elevation:2d}, SNR {snr:2d}: {len(group)} trials; "
                  f"3D<=15 {condition['raw_within15_all_pct']:.1f}%; accepted {condition['accepted_pct']:.1f}%",flush=True)
    expected = 100 * (4*args.azimuths+1)*len(SNRS)
    if len(rows) != expected or {r['clip_index'] for r in rows} != set(range(100)):
        raise AssertionError('Incomplete clip/trial coverage')
    by_snr = [dict(snr_db=snr,**summarize([r for r in rows if r['snr_db']==snr])) for snr in SNRS]
    by_clip = [dict(clip_index=i,**summarize([r for r in rows if r['clip_index']==i])) for i in range(100)]
    overall = summarize(rows)
    write_csv(args.out/'trials.csv',rows)
    write_csv(args.out/'by_snr.csv',by_snr)
    write_csv(args.out/'by_clip.csv',by_clip)
    write_json(args.out/'summary.json',dict(overall=overall,by_snr=by_snr,by_condition=conditions))
    meta.update(finished_utc=datetime.now(timezone.utc).isoformat(), trials=len(rows))
    write_json(args.out/'provenance.json',meta)
    report(args.out,meta,by_snr,overall)
    print(json.dumps(by_snr,indent=2))
    print(f'Results: {args.out}',flush=True)


if __name__ == '__main__':
    main()
