"""Exploratory hemisphere-boundary coverage probe, saved from the executed code.

The historical result used solver SHA256 b2aacc21f763a3003e4d6139c608f29167c90afc40ae604b750ba75c8f8733e5.
The previous rule is reconstructed from this run's unconstrained candidates and
support scores. It is not a separate replay of an old estimator file. The
initial direct old-code probe independently found the same48 bad acceptances.
This is a selected synthetic stress probe, not a calibrated failure-rate study.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import sig_proc_tri_robust as robust


def run():
    source = np.random.default_rng(144).normal(size=2048)
    counts = {'previous_point_rule': Counter(), 'current_interval_guard': Counter()}
    per_mic = {i: {'previous_point_rule': Counter(), 'current_interval_guard': Counter()}
               for i in [1, 2, 3]}
    statuses = Counter()
    scenarios = 0
    for missing in [1, 2, 3]:
        for azimuth in [-170, -120, -70, -20, 30, 80, 130]:
            for elevation in [0, 0.1, 1, 3]:
                for seed in range(5):
                    truth = robust.unit_vector(azimuth, elevation)
                    frames = robust.simulate_array(source, azimuth, 10,
                                                   np.random.default_rng(seed),
                                                   elevation_deg=elevation)
                    frames[missing] = 0
                    base = robust.estimate_direction_details(frames)
                    current = robust.estimate_direction_details(
                        frames, hemisphere_normal=[0, 0, 1])
                    statuses[current.status] += 1
                    previous_vector = None
                    previous_confidence = 0.
                    if base.geometry_rank == 2 and len(base.candidates) == 2:
                        allowed = np.flatnonzero(base.candidates[:, 2] >= -1e-9)
                        if len(allowed) == 1:
                            previous_vector = base.candidates[allowed[0]]
                            previous_confidence = base.support_confidence
                    for label, vector, confidence in [
                            ('previous_point_rule', previous_vector, previous_confidence),
                            ('current_interval_guard', current.vector, current.confidence)]:
                        if vector is None or confidence < 0.2:
                            outcome = 'withheld'
                        elif robust.direction_error_deg(vector, truth) <= 15:
                            outcome = 'accepted_within15'
                        else:
                            outcome = 'accepted_over15'
                        counts[label][outcome] += 1
                        per_mic[missing][label][outcome] += 1
                    scenarios += 1
    return dict(
        scenarios=scenarios, source_seed=144, source_samples=2048, fs=48000, snr_db=10,
        missing_channels=[1, 2, 3], azimuths=[-170, -120, -70, -20, 30, 80, 130],
        elevations=[0, 0.1, 1, 3], noise_seeds=list(range(5)), hemisphere=[0, 0, 1], gate=0.2,
        comparison_note=('Previous point-hemisphere selection reconstructed from current unconstrained '
                         'candidates/support; underlying measurement pipeline is current code. Earlier '
                         'direct old-code run also found48 accepted_over15.'),
        totals={key: dict(value) for key, value in counts.items()},
        per_mic={key: {name: dict(value) for name, value in item.items()}
                 for key, item in per_mic.items()},
        current_statuses=dict(statuses),
        solver_sha256=hashlib.sha256((ROOT / 'sig_proc_tri_robust.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Optional new JSON output; historical_result.json is archival')
    args = parser.parse_args()
    output = json.dumps(run(), indent=2) + '\n'
    if args.out:
        args.out.write_text(output)
    print(output, end='')
