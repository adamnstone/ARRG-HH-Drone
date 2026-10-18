"""Dataset-free regression checks for Rachel's four-microphone solver."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest import mock

import numpy as np
import sig_proc_tri
from sig_proc_tri import (C, MICS, FS, PAIRS, simulate_array, estimate_direction_3d,
                          estimate_direction_details, estimate_direction,
                          estimate_azimuth, azimuth_from_direction,
                          elevation_from_direction, turn_instruction, ang_err)


class FourMicTests(unittest.TestCase):
    def test_signed_elevation_and_all_quadrants(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(size=4096)
        for elevation in [-60, 0, 30, 80, 90]:
            for azimuth in [-170, -90, 0, 45, 90, 170]:
                frames = simulate_array(signal, azimuth, 30, rng, elevation_deg=elevation)
                u, confidence, _, _ = estimate_direction_3d(frames)
                self.assertIsNotNone(u)
                measured_el = np.degrees(np.arctan2(u[2], np.linalg.norm(u[:2])))
                self.assertLess(abs(measured_el - elevation), 1)
                self.assertGreater(confidence, 0.5)
                az = estimate_azimuth(frames)[0]
                if elevation == 90:
                    self.assertIsNone(az)
                else:
                    self.assertLess(ang_err(az, azimuth), 2)

    def test_primary_api_returns_normalized_signed_3d_direction(self):
        rng = np.random.default_rng(12)
        signal = rng.normal(size=4096)
        for fs in [16000, 48000]:
            for azimuth, elevation in [(-90, -60), (40, 35), (170, 0),
                                       (0, 90), (0, -90)]:
                with self.subTest(fs=fs, azimuth=azimuth, elevation=elevation):
                    frames = simulate_array(signal, azimuth, 30, rng,
                                            fs=fs, elevation_deg=elevation)
                    u, confidence, grid, fit = estimate_direction(frames, fs=fs)
                    self.assertIsNotNone(u)
                    self.assertEqual(u.shape, (3,))
                    self.assertAlmostEqual(np.linalg.norm(u), 1.0, places=12)
                    az, el = np.deg2rad([azimuth, elevation])
                    truth = np.array([np.cos(el) * np.sin(az),
                                      np.cos(el) * np.cos(az), np.sin(el)])
                    error = np.degrees(np.arccos(np.clip(truth @ u, -1, 1)))
                    self.assertLess(error, 1.0)
                    self.assertGreater(confidence, 0.5)
                    legacy = estimate_direction_3d(frames, fs=fs)
                    for actual, expected in zip((u, confidence, grid, fit), legacy):
                        np.testing.assert_array_equal(actual, expected)

    def test_hardware_rate_defaults_match_explicit_48khz(self):
        from droneears import config
        self.assertEqual(config.FS, 48000)
        signal = np.random.default_rng(13).normal(size=8192)
        frames = simulate_array(signal, -63, 30, np.random.default_rng(14),
                                fs=48000, elevation_deg=-35)
        default_simulation = simulate_array(signal, -63, 30, np.random.default_rng(14),
                                            elevation_deg=-35)
        np.testing.assert_array_equal(default_simulation, frames)
        default = estimate_direction(frames)
        explicit = estimate_direction(frames, fs=48000)
        self.assertIsNotNone(default[0])
        for actual, expected in zip(default, explicit):
            np.testing.assert_array_equal(actual, expected)
        self.assertLess(ang_err(azimuth_from_direction(default[0]), -63), 1)
        self.assertLess(abs(elevation_from_direction(default[0]) + 35), 1)

    def test_ambiguous_overhead_tone_cannot_report_reversed_direction(self):
        # Construct continuous waveforms directly, independently of the FFT
        # simulator. The top mic hears this overhead tone earlier, but a 2 kHz
        # period admits multiple physically possible inter-mic delays.
        times = np.arange(8192) / 48000
        truth = np.array([0., 0., 1.])
        frames = np.sin(2 * np.pi * 2000 *
                        (times[None, :] + (MICS @ truth / C)[:, None]))
        result = estimate_direction_details(frames, fs=48000)
        self.assertIsNone(result.vector)
        self.assertEqual(result.confidence, 0)
        self.assertTrue(np.any(result.ambiguous_pairs))
        self.assertTrue(np.all(np.isnan(result.delays[result.ambiguous_pairs])))
        self.assertIsNone(estimate_direction(frames, fs=48000)[0])

    def test_short_window_periodic_aliases_cannot_report_reversed_direction(self):
        # A short Hann window broadens a pure tone's FFT peak. Leakage must
        # not make the multiple physically possible 2 kHz delays look unique.
        for length in [1024, 2048]:
            times = np.arange(length) / 48000
            for truth in [np.array([0., -1., 0.]), np.array([0., 0., 1.])]:
                with self.subTest(length=length, truth=truth):
                    frames = np.sin(2 * np.pi * 2000 *
                                    (times[None, :] + (MICS @ truth / C)[:, None]))
                    result = estimate_direction_details(frames, fs=48000)
                    self.assertIsNone(result.vector)
                    self.assertEqual(result.confidence, 0)

    def test_multiharmonic_sources_retain_signed_3d_direction(self):
        times = np.arange(8192) / 48000
        for fundamental in [130., 220., 350., 500.]:
            for azimuth, elevation in [(45, 30), (-110, -40), (0, 90)]:
                with self.subTest(fundamental=fundamental, azimuth=azimuth,
                                  elevation=elevation):
                    az, el = np.deg2rad([azimuth, elevation])
                    truth = np.array([np.cos(el) * np.sin(az),
                                      np.cos(el) * np.cos(az), np.sin(el)])
                    wave_times = times[None, :] + (MICS @ truth / C)[:, None]
                    frames = sum(np.sin(2 * np.pi * fundamental * k * wave_times + 0.37 * k)
                                 / k ** 0.8 for k in range(1, int(2800 // fundamental) + 1))
                    result = estimate_direction_details(frames, fs=48000)
                    self.assertIsNotNone(result.vector)
                    error = np.degrees(np.arccos(np.clip(truth @ result.vector, -1, 1)))
                    self.assertLess(error, 0.1)
                    self.assertGreater(result.confidence, 0.9)

    def test_direction_angle_helpers(self):
        self.assertIsNone(azimuth_from_direction(None))
        self.assertIsNone(elevation_from_direction(None))
        for sign in [-1, 1]:
            vertical = np.array([0.0, 0.0, float(sign)])
            self.assertIsNone(azimuth_from_direction(vertical))
            self.assertAlmostEqual(elevation_from_direction(vertical), sign * 90)
        for azimuth, elevation in [(-135, -40), (-90, 0), (0, 30), (90, 45),
                                   (179, -80)]:
            az, el = np.deg2rad([azimuth, elevation])
            u = np.array([np.cos(el) * np.sin(az),
                          np.cos(el) * np.cos(az), np.sin(el)])
            self.assertLess(ang_err(azimuth_from_direction(u), azimuth), 1e-10)
            self.assertAlmostEqual(elevation_from_direction(u), elevation)

    def test_silence_and_geometry(self):
        silent = np.zeros((4, 4096))
        self.assertIsNone(estimate_direction(silent)[0])
        self.assertIsNone(estimate_direction_3d(silent)[0])
        self.assertIsNone(estimate_azimuth(silent)[0])
        with self.assertRaises(ValueError):
            estimate_direction(np.ones((3, 4096)))
        with self.assertRaises(ValueError):
            estimate_direction(np.ones((4, 4096)), mics=np.zeros((4, 3)))

    def test_custom_geometry(self):
        rng = np.random.default_rng(3)
        mics = MICS * 1.3 + [1, -2, 0.5]
        frames = simulate_array(rng.normal(size=4096), -72, 40, rng, mics=mics, elevation_deg=-30)
        self.assertLess(ang_err(estimate_azimuth(frames, mics=mics)[0], -72), 1)

    @staticmethod
    def _independent_placement_capture(mics, truth, fs):
        # Evaluate a continuous source at each microphone's arrival time.
        # This deliberately bypasses mic_delays/simulate_array so a shared
        # coordinate or sign error cannot make both the input and fit agree.
        times = np.arange(8192) / fs
        wave_times = times[None, :] + (mics @ truth / C)[:, None]
        rng = np.random.default_rng(712)
        frequencies = rng.uniform(250, 2800, 48)
        phases = rng.uniform(-np.pi, np.pi, len(frequencies))
        return sum(np.sin(2 * np.pi * frequency * wave_times + phase)
                   for frequency, phase in zip(frequencies, phases))

    @staticmethod
    def _rotated_hardware_placement():
        # Unequal measured-style placements, then a tilt and a translation.
        # None of the baseline directions is tied to a coordinate axis.
        mics = np.array([[-0.012, -0.112, 0.015],
                         [-0.086, 0.018, -0.008],
                         [0.103, 0.004, 0.010],
                         [0.008, -0.003, 0.112]])
        yaw, tilt = np.deg2rad([23, -17])
        rotate_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotate_x = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                             [0, np.sin(tilt), np.cos(tilt)]])
        return mics @ (rotate_z @ rotate_x).T + [1.4, -0.7, 0.2]

    def test_independent_rotated_placement_retains_full_3d_direction(self):
        mics = self._rotated_hardware_placement()
        for fs in [16000, 48000]:
            for azimuth, elevation in [(-117, -43), (63, 37), (0, 90), (0, -90)]:
                with self.subTest(fs=fs, azimuth=azimuth, elevation=elevation):
                    az, el = np.deg2rad([azimuth, elevation])
                    truth = np.array([np.cos(el) * np.sin(az),
                                      np.cos(el) * np.cos(az), np.sin(el)])
                    frames = self._independent_placement_capture(mics, truth, fs)
                    result = estimate_direction_details(frames, fs=fs, mics=mics)
                    self.assertIsNotNone(result.vector)
                    self.assertAlmostEqual(np.linalg.norm(result.vector), 1.0, places=12)
                    error = np.degrees(np.arccos(np.clip(truth @ result.vector, -1, 1)))
                    self.assertLess(error, 0.1)
                    self.assertGreater(result.confidence, 0.9)
                    expected_delays = np.array([(mics[j] - mics[i]) @ truth / C
                                                for i, j in PAIRS])
                    # Finite Hann windows slightly bias the fitted delays;
                    # require sub-sample agreement with the analytic arrivals.
                    np.testing.assert_allclose(result.delays, expected_delays, atol=0.1 / fs)

    def test_consistent_channel_and_coordinate_permutations_preserve_direction(self):
        mics = self._rotated_hardware_placement()
        truth = np.array([0.36, -0.48, 0.8])
        for fs in [16000, 48000]:
            frames = self._independent_placement_capture(mics, truth, fs)
            reference = estimate_direction_details(frames, fs=fs, mics=mics)
            self.assertIsNotNone(reference.vector)
            for order in [[2, 0, 3, 1], [3, 2, 1, 0]]:
                with self.subTest(fs=fs, order=order):
                    result = estimate_direction_details(frames[order], fs=fs,
                                                        mics=mics[order])
                    self.assertIsNotNone(result.vector)
                    np.testing.assert_allclose(result.vector, reference.vector, atol=1e-12)
                    self.assertAlmostEqual(result.confidence, reference.confidence, places=12)
                    error = np.degrees(np.arccos(np.clip(truth @ result.vector, -1, 1)))
                    self.assertLess(error, 0.1)
                    expected_delays = np.array([(mics[order[j]] - mics[order[i]]) @ truth / C
                                                for i, j in PAIRS])
                    np.testing.assert_allclose(result.delays, expected_delays, atol=0.1 / fs)

    def test_coplanar_blrf_placement_rejects_ambiguous_3d_geometry(self):
        # Four valid, distinct microphones still cannot determine the sign of
        # elevation when the crown microphone is moved to the front plane.
        mics = np.array([[0, -0.10, 0], [-0.09, 0, 0],
                         [0.09, 0, 0], [0, 0.10, 0]])
        truth = np.array([0.36, -0.48, 0.8])
        frames = self._independent_placement_capture(mics, truth, 48000)
        with self.assertRaisesRegex(ValueError, 'span three dimensions'):
            estimate_direction_details(frames, fs=48000, mics=mics)

    def test_independent_integer_arrivals_and_delay_signs(self):
        # Coordinates in units of one sample's propagation distance. Arrival
        # offsets below are constructed directly, without using the simulator.
        mics = np.array([[0, 0, 0], [-4, 0, 0], [2, 3, 0], [1, 0, 3]]) * C / FS
        rng = np.random.default_rng(19)
        source = rng.normal(size=8192)
        for axis in np.eye(3):
            for sign in [-1, 1]:
                truth = sign * axis
                offsets = np.rint(-mics @ truth * FS / C).astype(int)
                offsets -= offsets.min()
                frames = np.stack([source[20-d:20-d+4096] for d in offsets])
                result = estimate_direction_details(frames, mics=mics)
                self.assertIsNotNone(result.vector)
                np.testing.assert_allclose(result.vector, truth, atol=0.002)
                expected = np.array([(offsets[i] - offsets[j]) / FS for i, j in PAIRS])
                np.testing.assert_allclose(result.delays, expected, atol=0.01 / FS)
                self.assertGreater(result.confidence, 0.98)

    def test_each_silent_mic_rejects_3d_solution(self):
        rng = np.random.default_rng(24)
        frames = simulate_array(rng.normal(size=4096), 45, 40, rng, elevation_deg=30)
        for mic in range(4):
            broken = frames.copy()
            broken[mic] = 0
            result = estimate_direction_details(broken)
            self.assertIsNone(result.vector)
            self.assertIsNone(estimate_direction(broken)[0])
            self.assertEqual(result.confidence, 0)
            for k, pair in enumerate(PAIRS):
                self.assertEqual(np.isnan(result.delays[k]), mic in pair)

    def test_each_noise_only_mic_prevents_confident_3d_solution(self):
        # Three clean channels give three coherent pairs, but they span only
        # a plane. Their strong scores must not hide an uncorrelated fourth
        # channel that supplies the missing dimension of the 3D solution.
        rng = np.random.default_rng(933)
        frames = simulate_array(rng.normal(size=2048), 35, np.inf, rng,
                                fs=48000, elevation_deg=50)
        for mic in range(4):
            with self.subTest(mic=mic):
                broken = frames.copy()
                broken[mic] = np.random.default_rng(1000 + mic).normal(size=2048)
                result = estimate_direction_details(broken, fs=48000)
                self.assertLess(result.confidence, 0.2)
                azimuth = azimuth_from_direction(result.vector)
                self.assertIn('no turn', turn_instruction(azimuth,
                                                          confidence=result.confidence))

    def test_hardware_sample_rate_and_endfire(self):
        rng = np.random.default_rng(31)
        signal = rng.normal(size=4096)
        for fs in [16000, 48000]:
            for i, j in PAIRS:
                truth = MICS[j] - MICS[i]
                truth /= np.linalg.norm(truth)
                az = np.degrees(np.arctan2(truth[0], truth[1]))
                el = np.degrees(np.arctan2(truth[2], np.linalg.norm(truth[:2])))
                frames = simulate_array(signal, az, np.inf, rng, fs=fs, elevation_deg=el)
                result = estimate_direction_details(frames, fs=fs)
                error = np.degrees(np.arccos(np.clip(truth @ result.vector, -1, 1)))
                self.assertLess(error, 0.1)
                self.assertGreater(result.confidence, 0.99)
                bounds = np.array([np.linalg.norm(MICS[b]-MICS[a]) / C for a, b in PAIRS])
                self.assertTrue(np.all(np.abs(result.delays) <= bounds))

    def test_simulation_snr_scales_with_signal_and_ignores_origin(self):
        signal = np.random.default_rng(10).normal(size=4096)
        base = simulate_array(signal, 45, 10, np.random.default_rng(4), elevation_deg=30)
        louder = simulate_array(10 * signal, 45, 10, np.random.default_rng(4), elevation_deg=30)
        translated = simulate_array(signal, 45, 10, np.random.default_rng(4),
                                    mics=MICS + [1000, -1000, 500], elevation_deg=30)
        np.testing.assert_allclose(louder, base * 10, atol=1e-12)
        np.testing.assert_allclose(translated, base, atol=1e-8)

    def test_rejects_invalid_parameters(self):
        frames = np.ones((4, 4096))
        for kwargs in [{'fs': np.nan}, {'fs': np.inf}, {'step_deg': np.nan},
                       {'min_magnitude': np.nan}, {'interp': 2.5}]:
            with self.assertRaises(ValueError):
                estimate_direction_3d(frames, **kwargs)
        with self.assertRaises(ValueError):
            estimate_azimuth(frames, min_horizontal=np.nan)
        with self.assertRaises(ValueError):
            azimuth_from_direction([0, 1, 0], min_horizontal=np.nan)
        with self.assertRaises(ValueError):
            estimate_direction_3d(np.ones((4, 4)), fs=48000)

    def test_low_confidence_cannot_issue_demo_turn(self):
        frames = np.random.default_rng(11).normal(size=(4, 4096))
        az, confidence, _, _ = estimate_azimuth(frames)
        self.assertLess(confidence, 0.2)
        self.assertIn('no turn', turn_instruction(az, confidence=confidence))
        self.assertIn('no turn', turn_instruction(None))
        self.assertEqual(turn_instruction(45, confidence=0.9), 'Turn right 45 deg')

    def test_cli_reports_3d_direction_and_handles_overhead(self):
        clips = [np.random.default_rng(55).normal(size=8192)]
        for elevation in [35, 90]:
            with self.subTest(elevation=elevation):
                output = io.StringIO()
                argv = ['sig_proc_tri.py', '--clips', '1', '--snr', '30',
                        '--elevation', str(elevation), '--seed', '0']
                with mock.patch.object(sig_proc_tri, 'load_drone_clips', return_value=clips), \
                     mock.patch.object(sig_proc_tri, 'make_plots') as plots, \
                     mock.patch('sys.argv', argv), contextlib.redirect_stdout(output):
                    sig_proc_tri.main()
                text = output.getvalue()
                self.assertIn('direction [right, forward, up]', text)
                self.assertIn('3d angular error', text.lower())
                plots.assert_called_once()
                medians, p90s, demo = plots.call_args.args[1:]
                # A zenith direction is a valid 3D estimate even though its
                # horizontal bearing is undefined; it must count in the sweep.
                self.assertTrue(np.all(np.isfinite(medians)))
                self.assertTrue(np.all(np.isfinite(p90s)))
                self.assertLess(max(medians), 1.0)
                if elevation == 90:
                    self.assertIn('no turn instruction', text)
                    self.assertIsNotNone(demo[1])
                    self.assertGreater(demo[1][2], 0.99)

    def test_file_entrypoint_preserves_current_hardware_channels(self):
        from droneears import config
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_file.py'
        rng = np.random.default_rng(26)
        for channels in [1, 3, 4, 5]:
            with self.subTest(channels=channels):
                audio = rng.normal(size=(channels, config.NFFT + config.HOP)).astype(np.float32)
                output = io.StringIO()
                with mock.patch('droneears.audio.load', return_value=audio), \
                     mock.patch('droneears.classifier.load_tflite_classifier') as classifier, \
                     mock.patch.object(sys, 'argv', [str(script), 'capture.wav', '--no-ml']), \
                     mock.patch.object(sys, 'path', list(sys.path)), \
                     contextlib.redirect_stdout(output):
                    namespace = runpy.run_path(str(script), run_name='__main__')
                classifier.assert_not_called()
                expected = audio if channels in (1, 4) else audio[:1]
                np.testing.assert_array_equal(namespace['y'], expected)
                self.assertEqual(namespace['multi'], channels == 4)
                self.assertEqual('theta' in namespace['res'][0], channels == 4)
                self.assertEqual('elev' in namespace['res'][0], channels == 4)
                self.assertIn(f'{expected.shape[0]} ch, classifier off', output.getvalue())

    def test_live_entrypoint_opens_all_four_synchronous_hardware_channels(self):
        from droneears import config
        script = Path(__file__).resolve().parents[1] / 'scripts' / 'live_mac.py'

        class StopBeforeCapture(Exception):
            pass

        for available_channels in [1, 3, 4, 8]:
            with self.subTest(available_channels=available_channels):
                audio_device = types.ModuleType('sounddevice')
                audio_device.query_devices = mock.Mock(return_value={
                    'name': 'Test interface', 'max_input_channels': available_channels})
                audio_device.InputStream = mock.Mock(side_effect=StopBeforeCapture)
                with mock.patch.dict(sys.modules, {'sounddevice': audio_device}), \
                     mock.patch('droneears.classifier.load_tflite_classifier') as classifier, \
                     mock.patch.object(sys, 'argv', [str(script), '--device', '2', '--no-ml']), \
                     mock.patch.object(sys, 'path', list(sys.path)), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaises(StopBeforeCapture):
                    runpy.run_path(str(script), run_name='__main__')
                classifier.assert_not_called()
                audio_device.query_devices.assert_called_once_with(2, 'input')
                audio_device.InputStream.assert_called_once()
                stream = audio_device.InputStream.call_args.kwargs
                self.assertEqual(stream['channels'], 4 if available_channels >= 4 else 1)
                self.assertEqual(stream['samplerate'], config.FS)
                self.assertEqual(stream['blocksize'], config.HOP)

    def test_legacy_uses_actual_fitted_delays_and_handles_undefined(self):
        from triangulation import estimate_direction as legacy_estimate, summarize
        rng = np.random.default_rng(6)
        frames = simulate_array(rng.normal(size=4096), 50, 10, rng, elevation_deg=-40)
        # Include content outside the solver band so an unmasked, separate GCC
        # calculation cannot masquerade as the delays used in the actual fit.
        frames[0] += np.sin(2 * np.pi * 5000 * np.arange(4096) / FS)
        result = estimate_direction_details(frames, interp=8)
        u, delays, invalid = legacy_estimate(frames, MICS, FS, C, 1.0, 8)
        self.assertFalse(invalid)
        np.testing.assert_array_equal(delays, result.delays)
        np.testing.assert_array_equal(u, result.vector)
        u, _, invalid = legacy_estimate(np.zeros_like(frames), MICS, FS, C, 1.0, 8)
        self.assertTrue(invalid)
        self.assertIsNone(u)
        metrics = summarize([{'snr_db': 10., 'angular_error_deg': None},
                             {'snr_db': 10., 'angular_error_deg': 2.}], [10])['10']
        self.assertEqual(metrics['undefined_estimates'], 1)
        self.assertEqual(metrics['fraction_within_10_deg'], 0.5)


if __name__ == '__main__':
    unittest.main()
