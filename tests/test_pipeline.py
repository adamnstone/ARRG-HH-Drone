"""pytest: the physics half of the pipeline. Run with `pytest -q`."""
import os, sys, subprocess, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from droneears import config as C
from droneears.simulate import synth_drone, arrivals, add_noise, tilt
from droneears.pipeline import Pipeline, summarise
from droneears.bearing import haptic_weights, solve_direction, A
from droneears.features import mel_filterbank


def _run(theta, snr=10, seed=0, elev=0.0):
    x = synth_drone(4.0, f0=230.0, seed=seed)
    y = add_noise(arrivals(x, theta, elev), snr, rng=np.random.default_rng(seed + 11))
    res = Pipeline().run(y)
    th = [r["theta"] for r in res if r["detected"] and r["theta"] is not None]
    return res, (np.median(th[-30:]) if th else None)


def test_bearing_all_quadrants():
    for theta in [0, 60, 120, 180, -120, -60]:
        res, est = _run(theta)
        assert est is not None, f"no detection at {theta}"
        err = abs(((est - theta + 180) % 360) - 180)
        assert err < 4.0, f"bearing error {err:.1f} deg at {theta}"
        assert not any(r["overhead"] for r in res), f"overhead reported for a source at {theta} deg in the plane"
        det = [r for r in res if r["detected"]]
        # conf = the robust solver's confidence (bearing.robust_direction; ~0.4-0.8 for this clean synthetic case)
        assert np.median([r["conf"] for r in det]) > 0.3
        assert np.mean([r["status"] == "full_3d" and r["mused"] == 15 for r in det]) > 0.9


@pytest.mark.parametrize("elev", [0, 30, 60, 85])
def test_elevation_recovered(elev):
    res, est = _run(60, elev=elev)
    el = [r["elev"] for r in res if r["detected"] and r["elev"] is not None]
    assert el, f"no detection at elev {elev}"
    err = abs(np.median(el) - elev)
    assert err < 5.0, f"elevation error {err:.1f} deg at elev {elev}"
    if elev < 80:                                    # azimuth is ill-conditioned straight up; fine elsewhere
        assert abs(((est - 60 + 180) % 360) - 180) < 4.0


def test_overhead_drone_buzzes_all_motors():
    # nearly straight overhead: the top mic hears it first, elevation ~85 deg
    res, _ = _run(30, elev=85)
    det = [i for i, r in enumerate(res) if r["detected"]]
    assert det, "overhead drone not detected"
    ov = [i for i, r in enumerate(res) if r["overhead"]]
    assert ov and ov[0] - det[0] <= int(C.FRAMES_PER_SEC), f"overhead not declared within 1 s of detection: {ov[:1]}"
    assert all(r["detected"] for r in res[ov[0]:]) and all(r["overhead"] for r in res[ov[0]:])
    assert all(r["haptics"] == {k: 1.0 for k in C.MOTOR_ANGLES} for r in res[ov[0]:])
    assert not any(r["overhead"] for r in res if not r["detected"])


def test_elevated_but_not_overhead():
    res, _ = _run(-45, elev=30)
    assert any(r["detected"] for r in res)
    assert not any(r["overhead"] for r in res)


def test_detects_within_two_seconds():
    res, _ = _run(30)
    s = summarise(res)
    assert s["first_detect_s"] is not None and s["first_detect_s"] < 2.5


def test_white_noise_does_not_detect():
    y = np.random.default_rng(0).standard_normal((C.N_MICS, C.FS * 4)).astype(np.float32)
    res = Pipeline().run(y)
    assert not any(r["detected"] for r in res)


def test_least_squares_recovers_unit_vector():
    from droneears.simulate import unit_vector
    from droneears.bearing import elevation_deg, bearing_deg
    assert A.shape == (C.N_PAIRS, 3)
    for el in [-30, 0, 20, 45, 70, 89]:
        for th in np.linspace(-180, 180, 13):
            u_true = unit_vector(th, el)
            u, conf = solve_direction(A @ u_true)
            assert abs(conf - 1) < 1e-6 and np.allclose(u, u_true, atol=1e-6)
            assert abs(elevation_deg(u) - el) < 1e-4
            if el < 89:
                assert abs(((bearing_deg(u) - th + 180) % 360) - 180) < 1e-4


def test_haptic_weights_sum_to_one_and_pan():
    names = list(C.MOTOR_ANGLES)
    assert names == ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
    for th in np.linspace(-180, 180, 73):
        w = haptic_weights(th)
        assert list(w) == names
        assert abs(sum(w.values()) - 1) < 1e-9
        assert sum(v > 0 for v in w.values()) <= 2
    for th, m in [(0, "F"), (45, "FR"), (90, "R"), (135, "BR"), (180, "B"), (-180, "B"), (-135, "BL"), (-90, "L"), (-45, "FL")]:
        w = haptic_weights(th)
        assert w[m] == 1.0 and sum(w.values()) == 1.0, (th, w)


def test_haptic_weights_eight_motors():
    w = haptic_weights(70)                           # between FR (45) and R (90): linear pan
    assert abs(w["R"] - 25 / 45) < 1e-9 and abs(w["FR"] - 20 / 45) < 1e-9
    assert all(v == 0 for k, v in w.items() if k not in ("FR", "R"))
    w = haptic_weights(-160)                         # between B (180) and BL (-135)
    assert abs(w["B"] - 25 / 45) < 1e-9 and abs(w["BL"] - 20 / 45) < 1e-9
    for th in [-170, -90, 0, 60, 180]:
        assert haptic_weights(th, overhead=True) == {k: 1.0 for k in C.MOTOR_ANGLES}


def test_mel_filterbank_rows_normalised():
    fb = mel_filterbank()
    assert fb.shape == (C.NMEL, C.NBINS)
    assert np.allclose(fb.sum(axis=1), 1.0, atol=1e-5)


def test_cpp_core_matches_python():
    root = os.path.join(os.path.dirname(__file__), "..")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "compare_core.py")], capture_output=True, text=True)
    assert "CORE MATCHES PYTHON" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


# ---- real phone recordings (data/drone, data/non_drone): channel 0, as scripts/run_file.py (--no-ml for rules-only)
_RICH = os.path.join(os.path.dirname(__file__), "..", "data")
_REAL_DETECTED = ["Immediate vicinity", "Hovering 5m ahead", "Hovering 10m away", "Fly by 1", "Fly by 2", "Descending",
                  "Assorted motion", "Fly by 50m height 1", "Fly by 50m height 2"]
_MODEL = os.path.join(os.path.dirname(__file__), "..", "artifacts", "drone_cls_int8.tflite")


def _load_ch0(path):
    if not os.path.exists(path):
        pytest.skip(f"missing {path}")
    from droneears.audio import load
    return load(path, mono=False)[:1]


def _rules_only(path):
    return Pipeline().run(_load_ch0(path))


def _fused(path):
    y = _load_ch0(path)
    if not os.path.exists(_MODEL):
        pytest.skip(f"missing {_MODEL}")
    try:
        from droneears.classifier import load_tflite_classifier
        clf = load_tflite_classifier(_MODEL)
    except Exception as e:                                   # no TFLite runtime installed
        pytest.skip(f"classifier unavailable: {e}")
    return Pipeline(classifier=clf).run(y)


@pytest.mark.parametrize("name", _REAL_DETECTED)
def test_real_drone_recording_detected(name):
    res = _rules_only(os.path.join(_RICH, "drone", name + ".m4a"))
    assert any(r["detected"] for r in res), f"{name}: max hits {max(r['hits'] for r in res)}"


@pytest.mark.parametrize("name", _REAL_DETECTED)
def test_real_drone_recording_detected_fused(name):
    res = _fused(os.path.join(_RICH, "drone", name + ".m4a"))
    assert any(r["detected"] for r in res), f"{name}: max hits {max(r['hits'] for r in res)}, max p {max(r['ml'] for r in res):.2f}"


def test_real_ambient_recording_not_detected():
    res = _rules_only(os.path.join(_RICH, "non_drone", "ambient_adam.m4a"))
    assert not any(r["detected"] for r in res)


def test_real_ambient_recording_not_detected_fused():
    res = _fused(os.path.join(_RICH, "non_drone", "ambient_adam.m4a"))
    assert not any(r["detected"] for r in res)


# ---- interp findings (interp/README.md): local-flatness tonality test and the DSP gate
@pytest.mark.parametrize("db_per_oct", [-6, -3, 3, 6])
def test_detection_survives_mic_tilt(db_per_oct):
    # a different microphone / mounting response must not stop detection (the global-flatness test lost every phone
    # drone recording at -6 dB/octave)
    x = synth_drone(5.0, f0=230.0, seed=1)
    y = add_noise(arrivals(tilt(x, db_per_oct), 45.0), 10, rng=np.random.default_rng(3))
    res = Pipeline().run(y)
    assert any(r["detected"] for r in res), f"tilt {db_per_oct:+d} dB/oct: max hits {max(r['hits'] for r in res)}"


def test_dsp_gate_blocks_the_rules_false_alarm_on_ambient(monkeypatch):
    # with local flatness the rules alone reach DETECTED on the phone ambient recording; the DSP gate must veto it
    y = _load_ch0(os.path.join(_RICH, "non_drone", "ambient_adam.m4a"))
    monkeypatch.setattr(C, "GATE_ON", False)
    assert any(r["detected"] for r in Pipeline().run(y)), "rules alone no longer fire on ambient: test is moot"
    monkeypatch.setattr(C, "GATE_ON", True)
    assert not any(r["detected"] for r in Pipeline().run(y))
