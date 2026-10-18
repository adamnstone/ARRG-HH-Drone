"""Simulated 4-microphone drone direction estimation (no ML).

Run:
    python triangulation.py

Config:
    direction_config.json in the same folder.

This script:
- reads one-second drone audio windows from your existing data folder
- simulates four virtual microphones in the hardware B/L/R/T array
- applies direction-dependent arrival-time offsets and Gaussian noise
- estimates direction with GCC-PHAT + least-squares TDOA geometry
- saves summary CSV/JSON results
- generates plots and animated GIFs

Important: this estimates direction-of-arrival, not source distance.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from hf_public import resolve_public_dataset, public_splits

FORMATS = {'.m4a', '.wav', '.flac', '.mp3'}
ROOT = Path(__file__).resolve().parent


def load_config():
    cfg_path = ROOT / 'direction_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing {cfg_path}. Use the included example.')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    common_path = ROOT / 'config.json'
    if common_path.exists():
        common = json.loads(common_path.read_text(encoding='utf-8'))
        if not cfg.get('drone_dir'):
            cfg['drone_dir'] = common.get('drone_dir')
    for key in ['drone_dir', 'output_dir']:
        raw = Path(cfg[key]).expanduser()
        cfg[key] = (raw if raw.is_absolute() else ROOT / raw).resolve()

    cfg.setdefault('simulations_per_window', 3)
    cfg.setdefault('max_source_windows', 100)
    cfg.setdefault('noise_snr_db', [30, 20, 10, 0])
    cfg.setdefault('azimuth_range_degrees', [-180, 180])
    cfg.setdefault('elevation_range_degrees', [5, 85])
    cfg.setdefault('gcc_analysis_seconds', 0.5)
    cfg.setdefault('gcc_interp', 8)
    cfg.setdefault('random_seed', 42)
    cfg.setdefault('make_gifs', True)
    cfg.setdefault('gif_example_frames', 30)
    cfg.setdefault('moving_demo_frames', 72)
    cfg.setdefault('moving_demo_snr_db', 10)
    cfg.setdefault('plot_sample_snr_db', 10)
    cfg.setdefault('visual_demo_radius_m', 10.0)

    if cfg['speed_of_sound_m_s'] <= 0 or cfg['gcc_interp'] < 2:
        raise ValueError('Positive sound speed and gcc_interp >= 2 required.')
    return cfg


def decode_audio(path, sample_rate):
    proc = subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1',
        '-ar', str(sample_rate), '-'
    ], capture_output=True, check=True)
    data = np.frombuffer(proc.stdout, dtype='<f4').copy()
    if data.size == 0:
        raise ValueError(f'Empty audio: {path}')
    return data


def one_second_windows(audio, samples):
    for pos in range(0, len(audio) - samples + 1, samples):
        part = audio[pos:pos + samples].astype(np.float64)
        part -= part.mean()
        if np.sqrt(np.mean(part ** 2)) > 1e-6:
            yield pos, part


def microphone_positions(d=None):
    """Hardware B/L/R/T geometry; d is retained for old callers only."""
    from droneears.config import MIC_POS
    return MIC_POS.copy()


def direction_from_angles(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el),
    ])


def angles_from_direction(u):
    from sig_proc_tri import azimuth_from_direction, elevation_from_direction
    return azimuth_from_direction(u), elevation_from_direction(u)


def simulate_channels(mono, positions, direction, fs, c, snr_db, rng):
    delays = -(positions @ direction) / c
    delays -= np.min(delays)
    n = len(mono)
    t = np.arange(n, dtype=float)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    noise_sd = rms * 10 ** (-float(snr_db) / 20.0)
    outputs = []
    for delay in delays:
        shifted = np.interp(t - delay * fs, t, mono, left=0.0, right=0.0)
        shifted += rng.normal(0, noise_sd, size=n)
        outputs.append(shifted)
    return np.stack(outputs), delays


def estimate_direction(channels, positions, fs, c, analysis_seconds, interp):
    """Return (direction or None, fitted pair delays, invalid flag)."""
    from sig_proc_tri import estimate_direction_details, C as solver_c
    channels = np.asarray(channels, dtype=float)
    if channels.ndim != 2 or channels.shape[0] != 4:
        raise ValueError('channels must have shape (4, N) in B/L/R/T order')
    if not np.isfinite(c) or c <= 0 or not np.isfinite(analysis_seconds) or analysis_seconds <= 0:
        raise ValueError('positive finite sound speed and analysis_seconds required')
    n = min(channels.shape[1], int(fs * analysis_seconds))
    start = (channels.shape[1] - n) // 2
    segment = channels[:, start:start + n]
    # Scaling the geometry preserves configurable sound speed in this legacy API.
    result = estimate_direction_details(
        segment, fs=fs, mics=np.asarray(positions) * solver_c / c,
        interp=interp, fmax=min(3000.0, fs / 2))
    return result.vector, result.delays, result.vector is None


def angular_error_degrees(a, b):
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def summarize(rows, snrs):
    result = {}
    for snr in snrs:
        trials = [r for r in rows if r['snr_db'] == float(snr)]
        errors = np.array([r['angular_error_deg'] for r in trials
                           if r['angular_error_deg'] is not None], dtype=float)
        result[str(snr)] = {
            'n': len(trials),
            'valid_estimates': int(errors.size),
            'undefined_estimates': len(trials) - int(errors.size),
            'mean_error_deg': float(np.mean(errors)) if len(errors) else None,
            'median_error_deg': float(np.median(errors)) if len(errors) else None,
            'p90_error_deg': float(np.percentile(errors, 90)) if len(errors) else None,
            'fraction_within_10_deg': float(np.count_nonzero(errors <= 10) / len(trials)) if trials else None,
        }
    return result


def plot_metrics(summary, out):
    snrs = [float(k) for k in summary.keys()]
    med = [summary[str(k)]['median_error_deg'] for k in snrs]
    p90 = [summary[str(k)]['p90_error_deg'] for k in snrs]
    within10 = [summary[str(k)]['fraction_within_10_deg'] for k in snrs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snrs, med, marker='o', label='Median error')
    ax.plot(snrs, p90, marker='s', label='90th percentile error')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Angular error (degrees)')
    ax.set_title('Direction-estimation error vs. noise')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'error_vs_snr.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snrs, np.array(within10) * 100.0, marker='o')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Predictions within 10° (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Fraction of estimates within 10°')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'within_10deg_vs_snr.png', dpi=180)
    plt.close(fig)


def plot_scatter(rows, chosen_snr, out):
    subset = [r for r in rows if r['snr_db'] == float(chosen_snr)
              and r['angular_error_deg'] is not None and r['estimated_azimuth_deg'] is not None]
    if not subset:
        return
    true_az = np.array([r['true_azimuth_deg'] for r in subset])
    est_az = np.array([r['estimated_azimuth_deg'] for r in subset])
    err = np.array([r['angular_error_deg'] for r in subset])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(true_az, est_az, c=err, s=16, cmap='viridis')
    ax.plot([-180, 180], [-180, 180], linestyle='--', linewidth=1)
    ax.set_xlabel('True azimuth (deg)')
    ax.set_ylabel('Estimated azimuth (deg)')
    ax.set_title(f'Azimuth estimates at {chosen_snr} dB SNR')
    fig.colorbar(sc, ax=ax, label='Angular error (deg)')
    fig.tight_layout()
    fig.savefig(out / f'azimuth_scatter_{int(chosen_snr)}dB.png', dpi=180)
    plt.close(fig)


def _setup_3d_axes(ax, radius):
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 0.7))


def make_single_example_gif(example_rows, positions, out_path, radius):
    if not example_rows:
        return
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        _setup_3d_axes(ax, radius)
        row = example_rows[frame]
        true_vec = np.array([row['true_ux'], row['true_uy'], row['true_uz']])
        est_vec = np.array([row['estimated_ux'], row['estimated_uy'], row['estimated_uz']])
        true_pt = true_vec * radius
        est_pt = est_vec * radius
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=40, marker='o', label='Mics')
        ax.quiver(0, 0, 0, true_pt[0], true_pt[1], true_pt[2], length=1.0, normalize=False, label='True')
        ax.quiver(0, 0, 0, est_pt[0], est_pt[1], est_pt[2], length=1.0, normalize=False, label='Estimated')
        ax.scatter([true_pt[0]], [true_pt[1]], [true_pt[2]], s=50)
        ax.set_title(f"Single examples | SNR {row['snr_db']:.0f} dB | error {row['angular_error_deg']:.2f}°")
        ax.legend(loc='upper left')
        ax.view_init(elev=20, azim=35)

    ani = FuncAnimation(fig, update, frames=len(example_rows), interval=600)
    ani.save(out_path, writer=PillowWriter(fps=1.5))
    plt.close(fig)


def make_moving_demo_gif(mono, positions, cfg, rng, out_path):
    fs = int(cfg['sample_rate'])
    c = float(cfg['speed_of_sound_m_s'])
    snr = float(cfg['moving_demo_snr_db'])
    n_frames = int(cfg['moving_demo_frames'])
    radius = float(cfg['visual_demo_radius_m'])
    azs = np.linspace(-120, 120, n_frames)
    els = 35 + 20 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
    true_points = []
    est_points = []
    errors = []
    for az, el in zip(azs, els):
        truth = direction_from_angles(az, el)
        channels, _ = simulate_channels(mono, positions, truth, fs, c, snr, rng)
        estimated, _, _ = estimate_direction(channels, positions, fs, c, float(cfg['gcc_analysis_seconds']), int(cfg['gcc_interp']))
        true_points.append(truth * radius)
        est_points.append(estimated * radius if estimated is not None else np.full(3, np.nan))
        errors.append(angular_error_degrees(truth, estimated) if estimated is not None else float('nan'))
    true_points = np.asarray(true_points)
    est_points = np.asarray(est_points)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        _setup_3d_axes(ax, radius)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=40, marker='o', label='Mics')
        ax.plot(true_points[:frame + 1, 0], true_points[:frame + 1, 1], true_points[:frame + 1, 2], linewidth=1.5, label='True path')
        ax.plot(est_points[:frame + 1, 0], est_points[:frame + 1, 1], est_points[:frame + 1, 2], linewidth=1.5, linestyle='--', label='Estimated path')
        tp = true_points[frame]
        ep = est_points[frame]
        ax.quiver(0, 0, 0, tp[0], tp[1], tp[2], length=1.0, normalize=False, label='True now')
        ax.quiver(0, 0, 0, ep[0], ep[1], ep[2], length=1.0, normalize=False, label='Estimated now')
        ax.scatter([tp[0]], [tp[1]], [tp[2]], s=45)
        ax.scatter([ep[0]], [ep[1]], [ep[2]], s=45)
        ax.set_title(f"Moving drone demo | frame {frame+1}/{n_frames} | error {errors[frame]:.2f}°")
        ax.legend(loc='upper left', fontsize=8)
        ax.view_init(elev=22, azim=35)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=120)
    ani.save(out_path, writer=PillowWriter(fps=8))
    plt.close(fig)


def run(folder_override=None, out_override=None, file_list=None, limit_override=None, trials_override=None):
    cfg = load_config()
    folder, out = folder_override or cfg['drone_dir'], out_override or cfg['output_dir']
    if not folder.is_dir():
        raise FileNotFoundError(f'Drone folder not found: {folder}')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('FFmpeg is not installed or missing from PATH')
    files = sorted(Path(p) for p in file_list) if file_list is not None else sorted(p for p in folder.rglob('*') if p.suffix.lower() in FORMATS)
    if not files:
        raise ValueError(f'No supported audio recordings found under {folder}')
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.get('random_seed', 42)))
    fs = int(cfg['sample_rate'])
    samples = round(fs * float(cfg['window_seconds']))
    positions = microphone_positions()
    c = float(cfg['speed_of_sound_m_s'])
    azrange = cfg['azimuth_range_degrees']
    elrange = cfg['elevation_range_degrees']
    snrs = [float(v) for v in cfg['noise_snr_db']]
    limit = (len(files) * 10**7 if limit_override is None and file_list is not None else int(limit_override if limit_override is not None else cfg['max_source_windows']))

    trials_per_window = int(trials_override if trials_override is not None else cfg['simulations_per_window'])
    rows = []
    used = 0
    demo_mono = None
    for path in files:
        if used >= limit:
            break
        audio = decode_audio(path, fs)
        for start, mono in one_second_windows(audio, samples):
            if used >= limit:
                break
            used += 1
            if demo_mono is None:
                demo_mono = mono.copy()
            for trial in range(trials_per_window):
                az = float(rng.uniform(*azrange))
                el = float(rng.uniform(*elrange))
                truth = direction_from_angles(az, el)
                for snr in snrs:
                    channels, _ = simulate_channels(mono, positions, truth, fs, c, snr, rng)
                    estimated, dt, invalid = estimate_direction(channels, positions, fs, c, float(cfg['gcc_analysis_seconds']), int(cfg['gcc_interp']))
                    az_pred, el_pred = angles_from_direction(estimated) if estimated is not None else (None, None)
                    est_xyz = estimated if estimated is not None else (None, None, None)
                    rows.append({
                        'source_file': str(path),
                        'start_seconds': round(start / fs, 6),
                        'trial': trial,
                        'snr_db': snr,
                        'true_azimuth_deg': az,
                        'true_elevation_deg': el,
                        'estimated_azimuth_deg': az_pred,
                        'estimated_elevation_deg': el_pred,
                        'true_ux': truth[0], 'true_uy': truth[1], 'true_uz': truth[2],
                        'estimated_ux': est_xyz[0], 'estimated_uy': est_xyz[1], 'estimated_uz': est_xyz[2],
                        'direction_valid': not invalid,
                        'angular_error_deg': angular_error_degrees(truth, estimated) if estimated is not None else None,
                        **{f'delay_{i}_vs_{j}_us': dt[k] * 1e6
                           for k, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))},
                    })
    if not rows:
        raise ValueError('No usable non-silent 1-second source windows.')

    with (out / 'direction_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, snrs)
    metrics_obj = {
        'count_source_windows': used,
        'microphones_xyz_m': positions.tolist(),
        'assumptions': ['far_field_plane_wave', 'synchronized_sample_clocks'],
        'metrics_by_snr': summary,
    }
    (out / 'direction_metrics.json').write_text(json.dumps(metrics_obj, indent=2), encoding='utf-8')

    plot_metrics(summary, out)
    plot_scatter(rows, cfg['plot_sample_snr_db'], out)

    if cfg.get('make_gifs', True) and demo_mono is not None:
        chosen_snr = float(cfg['plot_sample_snr_db'])
        subset = [r for r in rows if r['snr_db'] == chosen_snr and r['direction_valid']][:int(cfg['gif_example_frames'])]
        if subset:
            make_single_example_gif(subset, positions, out / 'single_examples_direction.gif', float(cfg['visual_demo_radius_m']))
        make_moving_demo_gif(demo_mono, positions, cfg, rng, out / 'moving_drone_demo.gif')

    for snr, metrics in summary.items():
        median = metrics['median_error_deg']
        median_text = f'{median:.2f}' if median is not None else 'undefined'
        print(f'SNR {snr} dB | n={metrics["n"]} | undefined={metrics["undefined_estimates"]} | median error={median_text} deg | within 10 deg={metrics["fraction_within_10_deg"]:.1%}')
    print(f'Wrote {out / "direction_predictions.csv"} and direction_metrics.json')


def main():
    """Run local and enabled public-data experiments from either entrypoint."""
    run()  # Keep the original local-data simulation and outputs intact.
    cfg = load_config()
    hf_cfg = cfg.get('huggingface', {})
    if hf_cfg.get('enabled', False):
        hf_root = resolve_public_dataset(hf_cfg, ROOT)
        # For triangulation, process ALL unique positive clips, including official
        # train/val/test members. No ML fitting happens in this script.
        hf_partitions = public_splits(hf_root, None)
        hf_drone_files = sorted({r['path'] for rs in hf_partitions.values() for r in rs if r['label'] == 1})
        print(f'Running independent simulation on {len(hf_drone_files)} Hugging Face drone clips...')
        run(folder_override=hf_root / 'audio' / 'drone',
            out_override=cfg['output_dir'] / 'huggingface',
            file_list=hf_drone_files, trials_override=int(hf_cfg.get('simulations_per_window', 1)))


if __name__ == '__main__':
    main()

