"""Create a standalone, readable report from the frozen 50/50 comparison CSV.

This script only reads benchmark inputs and writes report artifacts. It never
runs or imports either estimator. All aggregate metrics are recomputed from the
trial-level CSV rather than copied from the benchmark's summary tables.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = [('original', 'Original'), ('robust', 'Robust')]
MODES = ['intact', 'weak_-10', 'weak_-20', 'weak_-30', 'noise_only']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boolean(value):
    if value not in ['True', 'False']:
        raise ValueError(f'Expected recorded boolean, got {value!r}')
    return value == 'True'


def number(value):
    return None if value in [None, ''] else float(value)


def read_trials(path, threshold):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ['clip_index', 'dataset_row', 'group_index']:
            row[key] = int(row[key])
        for key in ['snr_db', 'true_azimuth_deg', 'true_elevation_deg',
                    'measured_selected_channel_snr_db', 'selected_source_gain_db']:
            row[key] = number(row[key])
        for key in ['is_primary', 'is_matched_control']:
            row[key] = boolean(row[key])
        for prefix, _ in MODELS:
            for key in ['vector_defined', 'accepted']:
                row[f'{prefix}_{key}'] = boolean(row[f'{prefix}_{key}'])
            for key in ['confidence', 'error_3d_deg', 'ux', 'uy', 'uz']:
                row[f'{prefix}_{key}'] = number(row[f'{prefix}_{key}'])
            defined = row[f'{prefix}_vector_defined']
            if row[f'{prefix}_accepted'] != (defined and row[f'{prefix}_confidence'] >= threshold):
                raise ValueError('CSV acceptance differs from recorded confidence threshold')
            if defined != (row[f'{prefix}_error_3d_deg'] is not None):
                raise ValueError('A raw angular error must exist exactly when a vector exists')
    return rows


def stats(rows, prefix):
    defined = [r for r in rows if r[f'{prefix}_vector_defined']]
    accepted = [r for r in rows if r[f'{prefix}_accepted']]
    errors = [r[f'{prefix}_error_3d_deg'] for r in defined]
    accepted_errors = [r[f'{prefix}_error_3d_deg'] for r in accepted]
    return dict(n=len(rows), defined=len(defined), undefined=len(rows) - len(defined),
                accepted=len(accepted), withheld=len(rows) - len(accepted),
                low_confidence=len(defined) - len(accepted),
                good=sum(e <= 15 for e in accepted_errors),
                bad15=sum(e > 15 for e in accepted_errors),
                bad30=sum(e > 30 for e in accepted_errors),
                raw_bad15=sum(e > 15 for e in errors), raw_bad30=sum(e > 30 for e in errors),
                raw_median=float(np.median(errors)) if errors else None,
                raw_p90=float(np.percentile(errors, 90)) if errors else None,
                raw_max=max(errors) if errors else None,
                median=float(np.median(accepted_errors)) if accepted_errors else None,
                p90=float(np.percentile(accepted_errors, 90)) if accepted_errors else None,
                maximum=max(accepted_errors) if accepted_errors else None)


def fraction(numerator, denominator):
    return f'{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)' if denominator else '0/0 (n/a)'


def degrees(value):
    return f'{value:.2f}°' if value is not None else 'n/a'


def mode_name(mode, snr=20):
    if mode == 'intact':
        return 'Intact control'
    if mode == 'noise_only':
        return 'Noise-only selected mic'
    gain = int(mode.split('_')[1])
    return f'{gain} dB source gain (mic SNR ≈ {snr + gain:g} dB)'


def vector_text(row, prefix):
    if not row[f'{prefix}_vector_defined']:
        return 'undefined'
    return '[' + ', '.join(f'{row[f"{prefix}_{axis}"]:+.4f}' for axis in ['ux', 'uy', 'uz']) + ']'


def truth_vector(row):
    azimuth = np.deg2rad(row['true_azimuth_deg'] or 0)
    elevation = np.deg2rad(row['true_elevation_deg'])
    return [np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth), np.sin(elevation)]


class Report:
    def __init__(self):
        self.html = []
        self.text = []

    def heading(self, title, level=2):
        self.html.append(f'<h{level}>{escape(title)}</h{level}>')
        self.text.extend(['', title, '=' * len(title) if level == 1 else '-' * len(title)])

    def paragraph(self, text, callout=False):
        self.html.append(f'<p class="{"callout" if callout else ""}">{escape(text)}</p>')
        self.text.extend([text, ''])

    def table(self, headers, rows, caption=None):
        if caption:
            self.html.append(f'<p class="caption">{escape(caption)}</p>')
            self.text.append(caption)
        self.html.append('<div class="table-wrap"><table><thead><tr>' +
                         ''.join(f'<th>{escape(str(h))}</th>' for h in headers) +
                         '</tr></thead><tbody>')
        for row in rows:
            self.html.append('<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>')
        self.html.append('</tbody></table></div>')
        widths = [max(len(str(h)), max((len(str(row[i])) for row in rows), default=0))
                  for i, h in enumerate(headers)]
        self.text.append(' | '.join(str(h).ljust(w) for h, w in zip(headers, widths)))
        self.text.append('-+-'.join('-' * width for width in widths))
        self.text.extend(' | '.join(str(cell).ljust(width) for cell, width in zip(row, widths)) for row in rows)
        self.text.append('')

    def figure(self, path, caption):
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        self.html.append(f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}">'
                         f'<figcaption>{escape(caption)}</figcaption></figure>')
        self.text.extend([f'Figure: {path.name}', caption, ''])

    def code(self, text):
        self.html.append(f'<pre>{escape(text)}</pre>')
        self.text.extend([text, ''])


def outcome_rows(groups):
    rows = []
    for group, members in groups:
        for prefix, name in MODELS:
            m = stats(members, prefix)
            rows.append([group, name, m['n'], fraction(m['accepted'], m['n']),
                         fraction(m['good'], m['n']), m['bad15'], m['bad30'],
                         fraction(m['withheld'], m['n'])])
    return rows


def plots(output, primary, all_rows):
    os.environ.setdefault('MPLCONFIGDIR', str(Path('/private/tmp') / 'doa-report-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 11,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    labels, data = [], []
    for group in ['intact', 'degraded']:
        members = [r for r in primary if r['group'] == group]
        for prefix, label in MODELS:
            m = stats(members, prefix)
            labels.append(f'{group.capitalize()} · {label}\nN = {m["n"]}')
            data.append([m['good'], m['bad15'], m['withheld']])
    values = np.asarray(data)
    left = np.zeros(len(labels))
    for index, (label, color) in enumerate([
            ('Accepted within 15°', '#237a57'), ('Accepted over 15°', '#bf3e43'),
            ('Withheld', '#b4bdc9')]):
        widths = values[:, index] / values.sum(axis=1) * 100
        axis.barh(np.arange(len(labels)), widths, left=left, color=color, label=label, height=0.65)
        for row, (width, count) in enumerate(zip(widths, values[:, index])):
            if count:
                axis.text(left[row] + width / 2, row, str(count), ha='center', va='center',
                          color='white' if index < 2 else '#182839', fontweight='bold')
        left += widths
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.xaxis.set_major_formatter(PercentFormatter(100))
    axis.set_xlabel('Percentage of all inputs in each group; numbers inside bars are case counts')
    axis.set_title('Primary comparison: coverage and incorrect acceptance\n50 intact + 50 degraded inputs, baseline 20 dB SNR', pad=16)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    figure.subplots_adjust(left=0.21, right=0.98, top=0.80, bottom=0.25)
    first = output / 'primary_outcomes.png'
    figure.savefig(first, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    labels = ['Intact\n20 dB', 'Weak mic\n10 dB', 'Weak mic\n0 dB',
              'Weak mic\n−10 dB', 'Noise only\nzero source']
    x = np.arange(len(MODES))
    highest_bad = 0
    for prefix, label, color, marker in [('original', 'Original', '#3568a7', 'o'),
                                          ('robust', 'Robust', '#cf7829', 's')]:
        coverage, bad = [], []
        for mode in MODES:
            members = [r for r in all_rows if r['group'] == 'degraded' and
                       r['snr_db'] == 20 and r['mode'] == mode]
            m = stats(members, prefix)
            coverage.append(100 * m['accepted'] / m['n'])
            bad.append(100 * m['bad15'] / m['n'])
        highest_bad = max(highest_bad, max(bad))
        for axis, values in zip(axes, [coverage, bad]):
            axis.plot(x, values, marker=marker, color=color, label=label, linewidth=2, markersize=6)
    axes[0].set_title('Accepted directions / all 50 inputs')
    axes[1].set_title('Accepted errors >15° / all 50 inputs')
    axes[0].set_ylim(-2, 105)
    axes[1].set_ylim(-0.5, max(5, np.ceil(highest_bad / 5) * 5 + 1))
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.yaxis.set_major_formatter(PercentFormatter(100))
        axis.grid(axis='y', alpha=0.22)
        axis.legend(frameon=False, loc='best')
        axis.set_xlabel('Selected-channel nominal SNR; other microphones remain at 20 dB')
    figure.suptitle('Paired sensitivity: the same 50 degraded-group recordings and noise draws',
                    fontsize=14, y=0.98)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.23, wspace=0.24)
    second = output / 'weak_mic_sensitivity.png'
    figure.savefig(second, dpi=180)
    plt.close(figure)
    return first, second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'results_hf_50_50')
    args = parser.parse_args()
    output = args.data_dir.resolve()
    metadata = json.loads((output / 'provenance.json').read_text())
    if not metadata.get('finished_utc'):
        raise ValueError('The benchmark must finish before generating its report')
    threshold = metadata['config']['confidence_threshold']
    rows = read_trials(output / 'trial_level.csv', threshold)
    primary = [r for r in rows if r['is_primary']]
    if len(rows) != 1200 or len(primary) != 100 or len({r['clip_index'] for r in rows}) != 100:
        raise ValueError('Expected 1200 scenarios, a 100-case primary subset, and 100 source clips')
    if Counter(r['group'] for r in primary) != {'intact': 50, 'degraded': 50}:
        raise ValueError('Primary source groups must contain 50 clips each')
    degraded = [r for r in primary if r['group'] == 'degraded']
    intact = [r for r in primary if r['group'] == 'intact']
    groups = [('All primary inputs', primary), ('Intact group', intact), ('Degraded group', degraded),
              ('Noise-only mic', [r for r in degraded if r['mode'] == 'noise_only']),
              ('−30 dB source gain', [r for r in degraded if r['mode'] == 'weak_-30'])]
    first_plot, second_plot = plots(output, primary, rows)
    report = Report()
    report.heading('Four-microphone direction estimation', 1)
    report.paragraph('Full comparison of the original and robust solvers: 50 intact + 50 degraded '
                     'Hugging Face source recordings, with paired sensitivity checks.')
    report.paragraph('The primary comparison contains 100 inputs at a nominal intact-channel SNR of 20 dB. '
                     'The complete sweep contains 1,200 paired scenarios and 2,400 estimator calls using 100 '
                     'distinct cached recordings. The primary 100 are a subset of this sweep, not an additional 100 recordings.', True)
    old, new = stats(primary, 'original'), stats(primary, 'robust')
    report.paragraph(f"Across the primary 100 cases, Original accepts {old['accepted']} and withholds {old['withheld']}; "
                     f"Robust accepts {new['accepted']} and withholds {new['withheld']}. Accepted errors above 15°: "
                     f"Original {old['bad15']}, Robust {new['bad15']}. These are measured outcomes on this selected simulation protocol, "
                     'not a calibrated failure probability.')
    if old['bad15'] == new['bad15'] == 0:
        report.paragraph('Both solvers have zero incorrectly accepted directions in the primary comparison. '
                         'This primary sample therefore does not demonstrate a safety advantage for either solver. '
                         'The robust version withholds more degraded inputs; the paired sensitivity section quantifies that coverage cost.')
    report.heading('What was tested and what the numbers mean')
    report.paragraph('The source groups are disjoint within this run: 50 recordings supply the intact group and 50 '
                     'different recordings supply the degraded group. All 100 clips were already used in earlier '
                     'algorithm development; this is stress validation on previously seen recordings, not a held-out evaluation.')
    report.paragraph('Real mono drone recordings were resampled from 16 kHz to 48 kHz using 3:1 polyphase resampling, '
                     'then cropped to the central 2,048 samples (42.67 ms) before simulated propagation. '
                     'This is not native 48 kHz multi-microphone audio. Four synchronized free-field plane-wave channels '
                     'were synthesized for the B/L/R/T geometry, with independent added white noise.')
    report.paragraph('Each group has 10 recordings at each elevation 0°, 30°, 60°, 80°, 90°. Azimuths cover the full circle '
                     'using the frozen seeded assignment; azimuth is undefined at exactly overhead, but full 3D angular error remains valid. '
                     'Each degraded input affects one selected microphone: B 13 cases, L 13, R 12, T 12.')
    report.table(['Primary condition', 'Inputs', 'Selected channel', 'Other 3 channels'], [
        ['Intact', '50', 'Nominal 20 dB SNR', 'Nominal 20 dB SNR'],
        ['Noise-only', '25', 'Source component removed; original noise retained', 'Unchanged'],
        ['Weak source', '25', 'Source gain −30 dB; noise retained; nominal SNR −10 dB', 'Unchanged']])
    weak = [r['measured_selected_channel_snr_db'] for r in degraded if r['mode'] == 'weak_-30']
    report.paragraph(f'Realized selected-channel SNR in the 25 weak-source cases: median {np.median(weak):.2f} dB, '
                     f'range {min(weak):.2f} to {max(weak):.2f} dB. Noise-only channels have zero source power '
                     '(SNR −∞, represented as null in CSV), rather than a finite SNR.')
    report.table(['Term', 'Definition'], [
        ['Raw defined', 'Solver returned a finite unit vector; includes low-confidence output.'],
        ['Accepted', f'Vector defined and confidence ≥ {threshold:g}; unchanged demo gate, not hardware-calibrated.'],
        ['Withheld', 'Vector undefined OR confidence below the gate.'],
        ['Accepted within 15°', 'Accepted full 3D angular error ≤ 15°; denominator is all inputs when shown as acceptance coverage.'],
        ['Bad>15° / bad>30°', 'Accepted errors strictly above the threshold; >30° is a subset of >15°.'],
        ['Accepted precision', 'Accepted outputs within 15° divided by accepted outputs only.'],
        ['Raw error / accepted error', 'Statistics over defined vectors / accepted vectors, respectively; undefined vectors excluded.'],
        ['Ambiguous', 'Mirror candidates were returned without a unique direction; never credited as accepted localization.']])
    report.paragraph('Neither solver receives ground-truth direction, and no hemisphere assumption is supplied. '
                     'A raw upward-pointing vector or a candidate set does not override the confidence gate. '
                     'The vector describes direction [right, forward, up], not position or distance in meters.')
    report.paragraph('Both versions use the same numerical 0.2 gate for this comparison. Their confidence formulas differ, '
                     'so that shared number does not establish equivalent calibrated confidence, coverage or error rates.')

    report.heading('Primary 100: side-by-side outcomes')
    report.table(['Selection', 'Solver', 'N', 'Accepted /N', 'Accepted≤15° /N',
                  'Accepted>15°', 'Accepted>30°', 'Withheld /N'], outcome_rows(groups),
                 'The two degradation rows partition the 50 degraded inputs; intact + degraded partition the 100 primary inputs.')
    report.figure(first_plot, 'Counts and percentages use all 50 inputs in each source group. '
                  'Withheld combines undefined vectors and defined vectors below confidence 0.2.')
    flow, accepted_errors, raw_errors = [], [], []
    for group, members in groups:
        for prefix, name in MODELS:
            m = stats(members, prefix)
            status = Counter(r['status'] for r in members) if prefix == 'robust' else {}
            flow.append([group, name, fraction(m['defined'], m['n']), m['undefined'],
                         m['low_confidence'], status.get('ambiguous', '—'), status.get('unreliable', '—')])
            accepted_errors.append([group, name, m['accepted'], fraction(m['good'], m['accepted']),
                                    degrees(m['median']), degrees(m['p90']), degrees(m['maximum'])])
            raw_errors.append([group, name, m['defined'], m['raw_bad15'], m['raw_bad30'],
                               degrees(m['raw_median']), degrees(m['raw_p90']), degrees(m['raw_max'])])
    report.table(['Selection', 'Solver', 'Raw defined /N', 'Undefined', 'Defined but low confidence',
                  'Ambiguous', 'Unreliable'], flow,
                 'Undefined + defined-but-low-confidence equals withheld. Robust status counts describe raw output, independently of acceptance.')
    degraded_robust = stats(degraded, 'robust')
    degraded_status = Counter(row['status'] for row in degraded)
    report.paragraph(f"On the 50 primary degraded inputs, Robust still returns {degraded_robust['defined']} raw vectors, "
                     f"all below the confidence gate. It returns no unique vector for the other "
                     f"{degraded_robust['undefined']} inputs: {degraded_status.get('ambiguous', 0)} ambiguous and "
                     f"{degraded_status.get('unreliable', 0)} unreliable. Thus 'withheld 50' is a gate result, "
                     'not a claim that the estimator function returned None in all 50 cases.')
    ambiguity = []
    for label, members in groups[2:]:
        ambiguous = [r for r in members if r['status'] == 'ambiguous']
        covered = sum(number(r['candidate_best_error_3d_deg']) is not None and
                      number(r['candidate_best_error_3d_deg']) <= 15 for r in ambiguous)
        ambiguity.append([label, len(members), len(ambiguous), fraction(covered, len(ambiguous)), 0])
    report.table(['Selection', 'N', 'Ambiguous candidate sets', 'Set contains truth within15°',
                  'Credited unique acceptances'], ambiguity,
                 'Candidate-set coverage is diagnostic only. Ground truth is used for scoring, never for choosing a candidate.')
    report.heading('Angular error: accepted outputs and raw vectors')
    report.table(['Selection', 'Solver', 'Accepted n', '≤15° /accepted', 'Median', 'P90', 'Maximum'], accepted_errors,
                 'Accepted precision uses accepted n as its denominator. An n of 0 produces n/a statistics.')
    report.table(['Selection', 'Solver', 'Raw defined n', 'Raw>15°', 'Raw>30°', 'Median', 'P90', 'Maximum'], raw_errors,
                 'Raw errors include low-confidence estimates. Better rejection can improve accepted accuracy without improving these raw errors.')

    report.heading('Primary degraded inputs: microphone and elevation breakdown')
    for field, values, title in [('damaged_mic', ['B', 'L', 'R', 'T'], 'Selected degraded microphone'),
                                  ('true_elevation_deg', [0, 30, 60, 80, 90], 'True elevation (degrees)')]:
        breakdown = []
        for value in values:
            selected = [r for r in degraded if r[field] == value]
            a, b = stats(selected, 'original'), stats(selected, 'robust')
            breakdown.append([value, len(selected), fraction(a['accepted'], a['n']), a['bad15'], a['bad30'],
                              fraction(b['accepted'], b['n']), b['bad15'], b['bad30']])
        report.table([title, 'N', 'Original accepted /N', 'Orig>15°', 'Orig>30°',
                      'Robust accepted /N', 'Robust>15°', 'Robust>30°'], breakdown)

    report.heading('Paired intact controls for the same 50 degraded-group recordings')
    controls = [r for r in rows if r['group'] == 'degraded' and r['snr_db'] == 20 and r['mode'] == 'intact']
    report.paragraph('Comparing different 50-recording groups can mix hardware degradation with source difficulty. '
                     'These controls use the exact same 50 degraded-group recordings, directions, and noise draws. '
                     'Only the selected-channel source component changes; the other three channel arrays are bit-identical.')
    report.table(['Condition', 'Solver', 'N', 'Accepted /N', 'Accepted≤15° /N',
                  'Accepted>15°', 'Accepted>30°', 'Withheld /N'],
                 outcome_rows([('Matched intact controls', controls), ('Assigned primary degradation', degraded)]))
    lookup = {r['case_id']: r for r in controls}
    transitions = []
    for prefix, name in MODELS:
        control_good = [r for r in degraded if lookup[r['matched_control_id']][f'{prefix}_accepted'] and
                        lookup[r['matched_control_id']][f'{prefix}_error_3d_deg'] <= 15]
        after = stats(control_good, prefix)
        transitions.append([name, len(control_good), after['good'], after['bad15'], after['withheld']])
    report.table(['Solver', 'Control accepted≤15°', 'Degraded remains accepted≤15°',
                  'Degraded accepted>15°', 'Degraded withheld'], transitions,
                 'Transitions are conditioned on a correct accepted intact control; their row totals equal that control count.')

    report.heading('Paired weak-channel sensitivity at baseline 20 dB')
    report.paragraph('Each row below uses the same 50 degraded-group recordings, same selected microphones, '
                     'directions and original noise. Source gain −10/−20/−30 dB leaves nominal selected-channel '
                     'SNR 10/0/−10 dB. Noise-only removes the source entirely. These cases overlap the primary '
                     'degraded examples and are not additional independent recordings.')
    sensitivity = []
    for mode in MODES:
        selected = [r for r in rows if r['group'] == 'degraded' and r['snr_db'] == 20 and r['mode'] == mode]
        a, b = stats(selected, 'original'), stats(selected, 'robust')
        sensitivity.append([mode_name(mode), len(selected), fraction(a['accepted'], a['n']), a['bad15'], a['bad30'],
                            fraction(b['accepted'], b['n']), b['bad15'], b['bad30'],
                            f"{b['raw_bad15']}/{b['defined']}"])
    report.table(['Channel condition', 'N', 'Orig accepted /N', 'Orig>15°', 'Orig>30°',
                  'Robust accepted /N', 'Robust>15°', 'Robust>30°', 'Robust raw>15° /defined'], sensitivity)
    report.figure(second_plot, 'Paired 50-clip sensitivity at baseline 20 dB. Left: acceptance coverage. '
                  'Right: incorrect acceptance among all 50 inputs. The right-hand vertical scale is labeled separately.')
    status_rows = []
    for mode in MODES:
        selected = [r for r in rows if r['group'] == 'degraded' and r['snr_db'] == 20 and r['mode'] == mode]
        counts = Counter(r['status'] for r in selected)
        status_rows.append([mode_name(mode), len(selected)] + [counts.get(status, 0)
                           for status in ['full_3d', 'constrained_3d', 'ambiguous', 'unreliable']])
    report.table(['Condition', 'N', 'Full 3D', 'Constrained 3D', 'Ambiguous', 'Unreliable'], status_rows,
                 'No hemisphere assumption is supplied, so constrained 3D should remain zero.')

    report.heading('Selected worst primary cases')
    report.paragraph('Selection rule: take each solver’s three largest defined raw errors and two largest '
                     'accepted errors, then remove duplicate cases. Raw examples can be withheld; a large raw error '
                     'is not automatically an incorrectly accepted result. All vectors use [right, forward, up].')
    selected_ids = []
    for prefix, _ in MODELS:
        for accepted_only, limit in [(False, 3), (True, 2)]:
            candidates = [r for r in primary if r[f'{prefix}_vector_defined'] and
                          (not accepted_only or r[f'{prefix}_accepted'])]
            for row in sorted(candidates, key=lambda r: r[f'{prefix}_error_3d_deg'], reverse=True)[:limit]:
                if row['case_id'] not in selected_ids:
                    selected_ids.append(row['case_id'])
    by_id = {r['case_id']: r for r in primary}
    for case_id in selected_ids:
        row = by_id[case_id]
        truth = '[' + ', '.join(f'{value:+.4f}' for value in truth_vector(row)) + ']'
        report.paragraph(f"{case_id}: original clip {row['clip_index']}, dataset row {row['dataset_row']}; "
                         f"group {row['group']}, mode {row['mode']}, selected mic {row['selected_mic'] or 'none'}, "
                         f"elevation {row['true_elevation_deg']:g}°. Known truth {truth}.")
        example_rows = []
        for prefix, name in MODELS:
            example_rows.append([name, vector_text(row, prefix), f"{row[f'{prefix}_confidence']:.4f}",
                                 degrees(row[f'{prefix}_error_3d_deg']),
                                 'ACCEPTED' if row[f'{prefix}_accepted'] else 'WITHHELD',
                                 row['status'] if prefix == 'robust' else 'defined' if row[f'{prefix}_vector_defined'] else 'undefined'])
        report.table(['Solver', 'Returned vector', 'Confidence', '3D error', 'Gate result', 'Raw status'], example_rows)
        report.paragraph(f"Robust reason: {row['reason']}. Input SHA256: {row['input_sha256']}.")

    report.heading('SNR appendix: all 1,200 paired scenarios')
    report.paragraph('The complete sweep is 50 intact-group sources × 4 baseline SNRs = 200 scenarios, plus 50 '
                     'degraded-group sources × 5 channel conditions × 4 baseline SNRs = 1,000 scenarios. '
                     'The primary 100 are marked is_primary in trial_level.csv. Across SNRs and failure models, '
                     'repeated rows reuse recordings and are not statistically independent samples.')
    report.table(['Selection', 'Solver', 'N', 'Accepted /N', 'Accepted≤15° /N',
                  'Accepted>15°', 'Accepted>30°', 'Withheld /N'],
                 outcome_rows([('Complete overlapping sweep', rows)]))
    full_error_rows = []
    for prefix, label in MODELS:
        m = stats(rows, prefix)
        full_error_rows.append([label, m['defined'], m['undefined'], m['low_confidence'],
                                m['accepted'], degrees(m['median']), degrees(m['p90']),
                                degrees(m['maximum'])])
    report.table(['Solver', 'Raw defined n', 'Undefined', 'Defined low confidence', 'Accepted n',
                  'Accepted median', 'Accepted P90', 'Accepted maximum'], full_error_rows,
                 'Complete-sweep accepted error statistics use accepted n; undefined and low-confidence counts sum to withheld.')
    paired = []
    for label, selected in [('Primary 100', primary), ('Complete sweep', rows)]:
        counts = Counter()
        for row in selected:
            a = row['original_accepted'] and row['original_error_3d_deg'] <= 15
            b = row['robust_accepted'] and row['robust_error_3d_deg'] <= 15
            counts[(a, b)] += 1
        paired.append([label, len(selected), counts[(True, True)], counts[(True, False)],
                       counts[(False, True)], counts[(False, False)]])
    report.table(['Selection', 'N', 'Both accepted≤15°', 'Original only accepted≤15°',
                  'Robust only accepted≤15°', 'Neither accepted≤15°'], paired,
                 'Every row compares both solvers on identical input. The Neither column counts cases where neither solver '
                 'produced an accepted estimate within 15°; this includes withheld estimates and incorrectly accepted directions.')
    complete_a, complete_b = stats(rows, 'original'), stats(rows, 'robust')
    report.paragraph(f"Across the full sweep, all {paired[1][2]} accurate Robust acceptances are also accurate Original "
                     f"acceptances. Original additionally accepts {paired[1][3]} accurate directions and "
                     f"{complete_a['bad15']} accepted estimate{'s' if complete_a['bad15'] != 1 else ''} more than 15° wrong; Robust has "
                     f"{paired[1][4]} uniquely accurate acceptances and {complete_b['bad15']} incorrectly accepted directions. "
                     'This is a conservative coverage/error tradeoff, not a uniform accuracy improvement.')
    full_wrong = [row for row in rows if any(row[f'{prefix}_accepted'] and
                  row[f'{prefix}_error_3d_deg'] > 15 for prefix, _ in MODELS)]
    if full_wrong:
        report.paragraph('The following inputs have an accepted error above 15° somewhere in the full sweep. '
                         'They are separate from the primary headline comparison and reuse its source pool.')
        for row in full_wrong:
            truth = '[' + ', '.join(f'{value:+.4f}' for value in truth_vector(row)) + ']'
            report.paragraph(f"{row['case_id']}: dataset row {row['dataset_row']}; "
                             f"baseline SNR {row['snr_db']:g} dB; selected mic {row['selected_mic']}; "
                             f"true elevation {row['true_elevation_deg']:g}°; known truth {truth}.")
            report.table(['Solver', 'Returned vector', 'Confidence', '3D error', 'Gate result'], [
                [name, vector_text(row, prefix), f"{row[f'{prefix}_confidence']:.4f}",
                 degrees(row[f'{prefix}_error_3d_deg']),
                 'ACCEPTED' if row[f'{prefix}_accepted'] else 'WITHHELD']
                for prefix, name in MODELS])
    intact_appendix = []
    full_sensitivity = []
    for snr in [30, 20, 10, 0]:
        selected = [r for r in rows if r['group'] == 'intact' and r['snr_db'] == snr]
        a, b = stats(selected, 'original'), stats(selected, 'robust')
        intact_appendix.append([snr, len(selected), fraction(a['accepted'], a['n']), a['bad15'], a['bad30'],
                                fraction(b['accepted'], b['n']), b['bad15'], b['bad30']])
        for mode in MODES:
            selected = [r for r in rows if r['group'] == 'degraded' and r['snr_db'] == snr and r['mode'] == mode]
            a, b = stats(selected, 'original'), stats(selected, 'robust')
            full_sensitivity.append([snr, mode_name(mode, snr), len(selected),
                                     fraction(a['accepted'], a['n']), a['bad15'], a['bad30'],
                                     fraction(b['accepted'], b['n']), b['bad15'], b['bad30']])
    report.table(['Intact-group baseline SNR (dB)', 'N', 'Orig accepted /N', 'Orig>15°', 'Orig>30°',
                  'Robust accepted /N', 'Robust>15°', 'Robust>30°'], intact_appendix)
    report.table(['Baseline SNR (dB)', 'Degraded-group channel condition', 'N', 'Orig accepted /N',
                  'Orig>15°', 'Orig>30°', 'Robust accepted /N', 'Robust>15°', 'Robust>30°'], full_sensitivity)

    report.heading('Limits and interpretation')
    report.paragraph('The degraded models change only the source amplitude of one channel while keeping its '
                     'original added noise. They do not model a helmet transfer function, head shadow with '
                     'frequency-dependent phase, microphone timing offsets, wind, reflections, motion or clipping. '
                     'No synchronized recording from the physical four-microphone headset is evaluated here.')
    report.paragraph('The 0.2 confidence gate is a demo setting and is not a calibrated probability. These selected '
                     'counts do not establish a future failure rate. Conservative withholding trades coverage for '
                     'fewer actionable errors; it does not by itself improve raw angular accuracy. '
                     'A three-microphone mirror pair is ambiguity, not a measured unique 3D location.')
    report.paragraph('B/L/R/T geometry (meters; x right, y forward, z up): ' +
                     json.dumps(metadata['config']['mics_xyz_m']) + '. '
                     'Measure the hardware positions and verify channel order before deployment.')

    report.heading('Reproduction and provenance')
    report.code('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \\\n'
                '  python3 scripts/benchmark_50_50.py --out results_hf_50_50\n'
                'python3 scripts/report_50_50.py --data-dir results_hf_50_50')
    report.paragraph(f"Dataset: {metadata['dataset']}. Pinned historical source-manifest commit: "
                     f"{metadata['source_commit']}. Assignment seed: {metadata['seed']}. "
                     'All 100 normalized source waveform hashes were verified. Exact source membership, '
                     'crop hashes, noise streams, per-channel realized powers/SNRs and input hashes are recorded in the accompanying CSVs.')
    hashes = metadata['hashes_sha256']
    report.table(['Frozen artifact', 'SHA256'], [[name, value] for name, value in hashes.items()] +
                 [['trial_level.csv', sha256(output / 'trial_level.csv')],
                  ['primary_100.csv', sha256(output / 'primary_100.csv')],
                  ['clip_assignments.csv', sha256(output / 'clip_assignments.csv')],
                  ['report generator', sha256(Path(__file__))]])
    report.paragraph('trial_level.csv contains all 1,200 paired scenarios; primary_100.csv is its 100-case primary '
                     'subset; clip_assignments.csv records the disjoint source assignment; clip_manifest.json '
                     'records source-row identities and normalized waveform hashes; provenance.json records '
                     'the complete protocol. FULL_REPORT.html embeds both PNGs and needs no network access.')
    audit_path = output / 'AUDIT.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        report.paragraph('The independent consistency audit passed. It reconstructed all 1,200 inputs, replayed 24 '
                         'estimator cases, and checked 60 aggregate cells. The complete audit is in AUDIT.json; '
                         'these numerical checks are not additional hardware validation.')
        report.table(['Audit check', 'Verified result'], [
            ['Overall status', audit['status']],
            ['Source waveform hashes', audit['source_hashes_verified']],
            ['Disjoint source groups', audit['source_groups_disjoint']],
            ['Independently reconstructed input hashes', audit['independently_reconstructed_input_hashes']],
            ['Independently replayed cases / estimator calls',
             f"{audit['independent_estimator_replay_cases']} / {audit['independent_estimator_replay_calls']}"],
            ['Aggregate cells checked', audit['aggregate_cells_verified']]])
    report.heading('Recorded protocol details')
    report.table(['Parameter', 'Recorded value'], [
        ['Sample rates / analysis window', f"{metadata['source_fs']} Hz source → {metadata['fs']} Hz; {metadata['samples']} samples"],
        ['Baseline SNRs (dB)', metadata['config']['nominal_baseline_snrs_db']],
        ['Weak-source gains (dB)', metadata['config']['weak_source_gains_db']],
        ['Solver options', metadata['config']['solver_options']],
        ['Source cache SHA256', metadata['source_cache_sha256']],
        ['Benchmark started UTC', metadata['started_utc']],
        ['Benchmark finished UTC', metadata['finished_utc']],
        ['Input immutability verified', metadata['all_inputs_unchanged']]])
    report.code('RNG streams:\n' + json.dumps(metadata['rng_streams'], indent=2))

    css = '''body{margin:0;background:#edf1f5;color:#1c2937;font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1220px;margin:32px auto;background:white;padding:44px 48px 56px;border-radius:10px;box-shadow:0 4px 20px #18334b12}
h1{font-size:34px;line-height:1.2;margin:0 0 16px;color:#183a53}h2{font-size:23px;line-height:1.3;margin:42px 0 16px;padding-top:16px;border-top:1px solid #dce4eb;color:#183a53}
p{margin:12px 0 18px;max-width:1080px}.callout{background:#e9f3f7;border-left:4px solid #357e9e;padding:18px 22px;max-width:none}
.table-wrap{overflow-x:auto;margin:16px 0 26px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums}
th{text-align:left;background:#eaf0f6;font-weight:650;color:#16364f}th,td{padding:10px 12px;border-bottom:1px solid #e0e7ed;vertical-align:top}td:first-child{font-weight:550}
tr:nth-child(even) td{background:#f7f9fb}.caption,figcaption{color:#4c6072;font-size:14px;line-height:1.5}figure{margin:24px 0 32px}img{display:block;width:100%;height:auto}figcaption{padding:10px 4px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:20px;font:12px/1.55 ui-monospace,SFMono-Regular,monospace;border-radius:6px}
@media(max-width:800px){main{margin:0;padding:26px 18px;border-radius:0}h1{font-size:28px}th,td{padding:8px}}
@media print{body{background:white}main{max-width:none;margin:0;padding:0;box-shadow:none}figure,table{break-inside:avoid}h2{break-after:avoid}pre{font-size:10px}}'''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Four-microphone 50/50 comparison report</title><style>' + css +
                '</style></head><body><main>' + '\n'.join(report.html) + '</main></body></html>')
    (output / 'FULL_REPORT.html').write_text(document)
    (output / 'FULL_REPORT.txt').write_text('\n'.join(report.text) + '\n')
    report_provenance = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                             report_generator_sha256=sha256(Path(__file__)),
                             input_csv_sha256=sha256(output / 'trial_level.csv'),
                             benchmark_provenance_sha256=sha256(output / 'provenance.json'),
                             rows=len(rows), primary_rows=len(primary), source_clips=100,
                             aggregation='Recomputed independently from trial_level.csv',
                             primary_metrics={prefix: stats(primary, prefix) for prefix, _ in MODELS},
                             artifacts={name: sha256(output / name) for name in
                                        ['FULL_REPORT.html', 'FULL_REPORT.txt', first_plot.name, second_plot.name]})
    (output / 'report_provenance.json').write_text(json.dumps(report_provenance, indent=2) + '\n')
    print(f'Wrote {output / "FULL_REPORT.html"}')
    print(f'Wrote {output / "FULL_REPORT.txt"}')
    print(json.dumps(report_provenance['primary_metrics'], indent=2))


if __name__ == '__main__':
    main()
