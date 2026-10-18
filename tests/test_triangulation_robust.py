"""Independent geometry and rejection checks for the optional robust solver."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np

import sig_proc_tri_robust as robust


class RobustFourMicTests(unittest.TestCase):
    @staticmethod
    def _capture(mics, truth, fs=48000, length=8192):
        """Continuous broadband arrivals, independent of the solver simulator."""
        times = np.arange(length) / fs
        wave_times = times[None, :] + (mics @ truth / robust.C)[:, None]
        rng = np.random.default_rng(712)
        frequencies = rng.uniform(250, 2800, 48)
        phases = rng.uniform(-np.pi, np.pi, len(frequencies))
        return sum(np.sin(2 * np.pi * frequency * wave_times + phase)
                   for frequency, phase in zip(frequencies, phases))

    @staticmethod
    def _rotated_mics():
        positions = np.array([[-0.012, -0.112, 0.015],
                              [-0.086, 0.018, -0.008],
                              [0.103, 0.004, 0.010],
                              [0.008, -0.003, 0.112]])
        yaw, tilt = np.deg2rad([23, -17])
        rotate_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotate_x = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                             [0, np.sin(tilt), np.cos(tilt)]])
        return positions @ (rotate_z @ rotate_x).T + [1.4, -0.7, 0.2]

    @staticmethod
    def _angular_error(actual, expected):
        return float(np.degrees(np.arccos(np.clip(actual @ expected, -1, 1))))

    def _assert_metadata(self, result):
        self.assertIn(result.status, {'full_3d', 'constrained_3d',
                                      'ambiguous', 'unreliable'})
        self.assertEqual(result.pair_weights.shape, (6,))
        self.assertEqual(result.pair_coherence.shape, (6,))
        self.assertEqual(result.reliable_pairs.shape, (6,))
        self.assertEqual(result.reliable_pairs.dtype, np.dtype(bool))
        self.assertEqual(result.mic_reliability.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.pair_weights)))
        self.assertTrue(np.all(result.pair_weights >= 0))
        self.assertTrue(np.all(np.isfinite(result.pair_coherence)))
        self.assertTrue(np.all(np.isfinite(result.mic_reliability)))
        self.assertEqual(result.candidates.ndim, 2)
        self.assertEqual(result.candidates.shape[1], 3)
        self.assertTrue(np.all(np.isfinite(result.candidates)))
        if len(result.candidates):
            np.testing.assert_allclose(np.linalg.norm(result.candidates, axis=1), 1,
                                       atol=1e-10)
        self.assertIn(result.geometry_rank, [0, 1, 2, 3])
        self.assertFalse(np.isnan(result.geometry_condition))
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)
        self.assertGreaterEqual(result.support_confidence, 0)
        self.assertLessEqual(result.support_confidence, 1)
        self.assertIsInstance(result.reason, str)
        self.assertTrue(result.reason)
        if result.vector is None:
            self.assertEqual(result.confidence, 0)
            self.assertFalse(result.constraint_applied)
        else:
            self.assertEqual(result.vector.shape, (3,))
            self.assertAlmostEqual(np.linalg.norm(result.vector), 1, places=12)
            self.assertEqual(result.constraint_applied, result.status == 'constrained_3d')

    def _assert_candidates_include(self, result, expected, max_error=1.0):
        # Sub-sample finite-window GCC bias affects the inferred null-space
        # component of a rank-2 solution more than normalized full-3D vectors.
        # The analytic delay checks below bound that error independently.
        self.assertGreater(len(result.candidates), 0)
        self.assertLess(min(self._angular_error(candidate, expected)
                            for candidate in result.candidates), max_error)

    def _assert_rank_two_candidate_delays(self, result, mics, truth, fs=48000):
        self.assertEqual(result.geometry_rank, 2)
        geometry = np.array([mics[j] - mics[i] for i, j in robust.PAIRS])
        reliable = result.reliable_pairs
        self.assertTrue(np.any(reliable))
        expected = geometry @ truth / robust.C
        np.testing.assert_allclose(result.delays[reliable], expected[reliable],
                                   atol=0.1 / fs)
        # Both branches must reproduce the same measured delays; the sign of
        # the missing normal component is not contained in those measurements.
        for candidate in result.candidates:
            predicted = geometry[reliable] @ candidate / robust.C
            np.testing.assert_allclose(predicted, result.delays[reliable],
                                       atol=0.1 / fs)

    def test_clean_independent_arrivals_return_signed_full_3d_at_both_rates(self):
        for fs in [16000, 48000]:
            for truth in [np.array([0.36, 0.48, 0.8]),
                          np.array([-0.36, 0.48, -0.8]),
                          np.array([0., 0., 1.]), np.array([0., 0., -1.])]:
                with self.subTest(fs=fs, truth=truth):
                    frames = self._capture(robust.MICS, truth, fs)
                    result = robust.estimate_direction_details(frames, fs=fs)
                    self._assert_metadata(result)
                    self.assertEqual(result.status, 'full_3d')
                    self.assertEqual(result.geometry_rank, 3)
                    self.assertIsNotNone(result.vector)
                    self.assertLess(self._angular_error(result.vector, truth), 0.2)
                    self.assertTrue(np.all(result.reliable_pairs))
                    self.assertTrue(np.all(result.pair_weights > 0))
                    self._assert_candidates_include(result, truth)

    def test_rotated_translated_geometry_and_channel_permutations(self):
        mics = self._rotated_mics()
        truth = np.array([0.36, -0.48, 0.8])
        for fs in [16000, 48000]:
            frames = self._capture(mics, truth, fs)
            reference = robust.estimate_direction_details(frames, fs=fs, mics=mics)
            self._assert_metadata(reference)
            self.assertEqual(reference.status, 'full_3d')
            self.assertLess(self._angular_error(reference.vector, truth), 0.2)
            for order in [[2, 0, 3, 1], [3, 2, 1, 0]]:
                with self.subTest(fs=fs, order=order):
                    result = robust.estimate_direction_details(frames[order], fs=fs,
                                                               mics=mics[order])
                    self._assert_metadata(result)
                    self.assertEqual(result.status, 'full_3d')
                    np.testing.assert_allclose(result.vector, reference.vector, atol=1e-10)
                    self.assertAlmostEqual(result.confidence, reference.confidence, places=10)

    def test_each_dead_mic_exposes_two_mirror_candidates_without_assumptions(self):
        # Stay away from every survivor plane so both mirror branches are
        # distinct; tangent directions are a separate numerical boundary.
        truth = np.array([0.48, 0.64, 0.6])
        frames = self._capture(robust.MICS, truth)
        for missing in range(4):
            with self.subTest(missing=missing):
                broken = frames.copy()
                broken[missing] = 0
                result = robust.estimate_direction_details(broken, fs=48000)
                self._assert_metadata(result)
                self.assertEqual(result.status, 'ambiguous')
                self.assertEqual(result.geometry_rank, 2)
                self.assertIsNone(result.vector)
                self.assertEqual(result.candidates.shape, (2, 3))
                survivors = np.delete(robust.MICS, missing, axis=0)
                normal = np.cross(survivors[1] - survivors[0],
                                  survivors[2] - survivors[0])
                normal /= np.linalg.norm(normal)
                mirror = truth - 2 * (truth @ normal) * normal
                self._assert_candidates_include(result, truth)
                self._assert_candidates_include(result, mirror)
                self._assert_rank_two_candidate_delays(result, robust.MICS, truth)
                incident = np.array([missing in pair for pair in robust.PAIRS])
                self.assertFalse(np.any(result.reliable_pairs[incident]))

    def test_explicit_upper_hemisphere_resolves_top_dropout_and_records_assumption(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        frames[3] = 0
        result = robust.estimate_direction_details(frames, fs=48000,
                                                   hemisphere_normal=[0, 0, 1])
        self._assert_metadata(result)
        self.assertEqual(result.status, 'constrained_3d')
        self.assertTrue(result.constraint_applied)
        self.assertEqual(result.geometry_rank, 2)
        self.assertIsNotNone(result.vector)
        self.assertLess(self._angular_error(result.vector, truth), 1.0)
        self.assertGreater(result.vector[2], 0)
        self._assert_rank_two_candidate_delays(result, robust.MICS, truth)

    def test_opposite_explicit_hemisphere_selects_opposite_mirror_branch(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        frames[3] = 0
        result = robust.estimate_direction_details(frames, fs=48000,
                                                   hemisphere_normal=[0, 0, -1])
        self._assert_metadata(result)
        self.assertEqual(result.status, 'constrained_3d')
        self.assertTrue(result.constraint_applied)
        self.assertLess(self._angular_error(result.vector, truth * [1, 1, -1]), 1.0)
        self._assert_rank_two_candidate_delays(result, robust.MICS, truth)

    def test_rotated_survivor_plane_uses_hemisphere_in_coordinate_frame(self):
        mics = self._rotated_mics()
        normal = np.cross(mics[1] - mics[0], mics[2] - mics[0])
        normal /= np.linalg.norm(normal)
        tangent = mics[1] - mics[0]
        tangent /= np.linalg.norm(tangent)
        truth = 0.6 * normal + 0.8 * tangent
        frames = self._capture(mics, truth)
        frames[3] = 0
        result = robust.estimate_direction_details(frames, fs=48000, mics=mics,
                                                   hemisphere_normal=normal)
        self._assert_metadata(result)
        self.assertEqual(result.status, 'constrained_3d')
        self.assertEqual(result.geometry_rank, 2)
        self.assertLess(self._angular_error(result.vector, truth), 1.0)
        self.assertGreater(result.vector @ normal, 0)
        self._assert_rank_two_candidate_delays(result, mics, truth)

    def test_above_assumption_cannot_resolve_front_back_mirror_after_back_dropout(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        frames[0] = 0
        result = robust.estimate_direction_details(frames, fs=48000,
                                                   hemisphere_normal=[0, 0, 1])
        self._assert_metadata(result)
        self.assertEqual(result.status, 'ambiguous')
        self.assertIsNone(result.vector)
        self.assertEqual(result.candidates.shape, (2, 3))
        self.assertTrue(np.all(result.candidates[:, 2] > 0))
        self._assert_candidates_include(result, truth)
        self._assert_candidates_include(result, truth * [1, -1, 1])
        self._assert_rank_two_candidate_delays(result, robust.MICS, truth)

    def test_noisy_hemisphere_boundary_cannot_confidently_select_wrong_mirror(self):
        # The source really is above the helmet. A small delay error moves
        # its nearby candidate just below z=0, while the far mirror remains
        # above. The hemisphere must not be treated as excluding that nearby
        # branch when measurement uncertainty still permits it on the boundary.
        source = np.random.default_rng(144).normal(size=2048)
        frames = robust.simulate_array(source, -70, 10, np.random.default_rng(1),
                                       fs=48000, elevation_deg=1)
        frames[1] = 0
        result = robust.estimate_direction_details(frames, fs=48000,
                                                   hemisphere_normal=[0, 0, 1])
        self._assert_metadata(result)
        self.assertEqual(result.status, 'ambiguous')
        self.assertEqual(result.geometry_rank, 2)
        self.assertIsNone(result.vector)
        self.assertFalse(result.constraint_applied)
        self.assertEqual(result.candidates.shape, (2, 3))
        self._assert_candidates_include(result, robust.unit_vector(-70, 1), max_error=15)
        self.assertIn('no turn', robust.turn_instruction(
            robust.azimuth_from_direction(result.vector), confidence=result.confidence))

    def test_explicit_upper_hemisphere_rejects_contradicting_full_rank_direction(self):
        truth = np.array([0.36, 0.48, -0.8])
        frames = self._capture(robust.MICS, truth)
        unconstrained = robust.estimate_direction_details(frames, fs=48000)
        self._assert_metadata(unconstrained)
        self.assertEqual(unconstrained.status, 'full_3d')
        self.assertFalse(unconstrained.constraint_applied)
        self.assertLess(self._angular_error(unconstrained.vector, truth), 0.2)
        rejected = robust.estimate_direction_details(frames, fs=48000,
                                                     hemisphere_normal=[0, 0, 1])
        self._assert_metadata(rejected)
        self.assertEqual(rejected.status, 'unreliable')
        self.assertEqual(rejected.geometry_rank, 3)
        self.assertIsNone(rejected.vector)
        self.assertEqual(rejected.confidence, 0)
        # A compatible explicit prior does not turn a measured full-rank
        # solution into a branch selected by an assumption.
        compatible = robust.estimate_direction_details(frames, fs=48000,
                                                       hemisphere_normal=[0, 0, -1])
        self._assert_metadata(compatible)
        self.assertEqual(compatible.status, 'full_3d')
        self.assertFalse(compatible.constraint_applied)
        np.testing.assert_array_equal(compatible.vector, unconstrained.vector)

    def test_zero_projection_rank_two_direction_still_has_two_unit_candidates(self):
        for missing, truth, constrained_status in [
                (3, np.array([0., 0., 1.]), 'constrained_3d'),
                (0, np.array([0., 1., 0.]), 'ambiguous')]:
            with self.subTest(missing=missing):
                frames = self._capture(robust.MICS, truth)
                frames[missing] = 0
                result = robust.estimate_direction_details(frames, fs=48000)
                self._assert_metadata(result)
                self.assertEqual(result.status, 'ambiguous')
                self.assertIsNone(result.vector)
                self.assertEqual(result.candidates.shape, (2, 3))
                self._assert_candidates_include(result, truth)
                self._assert_candidates_include(result, -truth)
                self._assert_rank_two_candidate_delays(result, robust.MICS, truth)
                constrained = robust.estimate_direction_details(
                    frames, fs=48000, hemisphere_normal=[0, 0, 1])
                self._assert_metadata(constrained)
                self.assertEqual(constrained.status, constrained_status)
                if constrained_status == 'constrained_3d':
                    self.assertLess(self._angular_error(constrained.vector, truth), 1.0)
                else:
                    self.assertIsNone(constrained.vector)

    def test_coplanar_blrf_is_explicitly_ambiguous_until_hemisphere_supplied(self):
        mics = np.array([[0, -0.10, 0], [-0.09, 0, 0],
                         [0.09, 0, 0], [0, 0.10, 0]])
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(mics, truth)
        result = robust.estimate_direction_details(frames, fs=48000, mics=mics)
        self._assert_metadata(result)
        self.assertEqual(result.status, 'ambiguous')
        self.assertEqual(result.geometry_rank, 2)
        self.assertIsNone(result.vector)
        self._assert_candidates_include(result, truth)
        self._assert_candidates_include(result, truth * [1, 1, -1])
        self._assert_rank_two_candidate_delays(result, mics, truth)
        constrained = robust.estimate_direction_details(frames, fs=48000, mics=mics,
                                                        hemisphere_normal=[0, 0, 1])
        self._assert_metadata(constrained)
        self.assertEqual(constrained.status, 'constrained_3d')
        self.assertLess(self._angular_error(constrained.vector, truth), 1.0)
        self._assert_rank_two_candidate_delays(constrained, mics, truth)

    def test_nearly_coplanar_geometry_does_not_claim_unconstrained_full_3d(self):
        mics = robust.MICS.copy()
        mics[3, 2] = 0.00001
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(mics, truth)
        result = robust.estimate_direction_details(frames, fs=48000, mics=mics)
        self._assert_metadata(result)
        self.assertNotEqual(result.status, 'full_3d')
        self.assertIsNone(result.vector)
        self.assertFalse(result.constraint_applied)

    def test_silence_and_only_two_working_microphones_are_unreliable(self):
        truth = np.array([0.36, 0.48, 0.8])
        two_working = self._capture(robust.MICS, truth)
        two_working[2:] = 0
        for frames in [np.zeros((4, 8192)), two_working]:
            with self.subTest(working=np.count_nonzero(np.any(frames, axis=1))):
                result = robust.estimate_direction_details(frames, fs=48000)
                self._assert_metadata(result)
                self.assertEqual(result.status, 'unreliable')
                self.assertIsNone(result.vector)
                self.assertLessEqual(result.geometry_rank, 1)
                self.assertEqual(result.candidates.shape, (0, 3))

    def test_four_independent_noise_channels_do_not_produce_a_direction(self):
        frames = np.random.default_rng(981).normal(size=(4, 8192))
        result = robust.estimate_direction_details(frames, fs=48000)
        self._assert_metadata(result)
        self.assertEqual(result.status, 'unreliable')
        self.assertIsNone(result.vector)

    def test_noise_only_fourth_channel_does_not_hide_behind_three_clean_pairs(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        for mic in range(4):
            with self.subTest(mic=mic):
                broken = frames.copy()
                broken[mic] = np.random.default_rng(1000 + mic).normal(size=frames.shape[1])
                result = robust.estimate_direction_details(broken, fs=48000)
                self._assert_metadata(result)
                self.assertIn(result.status, {'ambiguous', 'unreliable'})
                self.assertIsNone(result.vector)
                self.assertLessEqual(result.geometry_rank, 2)
                incident = np.array([mic in pair for pair in robust.PAIRS])
                self.assertLess(np.median(result.pair_weights[incident]),
                                np.median(result.pair_weights[~incident]))

    def test_channel_gain_and_dc_offsets_preserve_clean_spatial_information(self):
        truth = np.array([0.36, -0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        reference = robust.estimate_direction_details(frames, fs=48000)
        gains = np.array([0.3, 2., 1.2, 0.7])[:, None]
        offsets = np.array([10., -3., 4., 2.])[:, None]
        result = robust.estimate_direction_details(frames * gains + offsets, fs=48000)
        self._assert_metadata(result)
        self.assertEqual(result.status, 'full_3d')
        self.assertLess(self._angular_error(result.vector, truth), 0.2)
        np.testing.assert_allclose(result.vector, reference.vector, atol=1e-10)

    def test_noise_added_to_one_real_channel_reduces_its_pair_quality(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        rng = np.random.default_rng(548)
        frames[3] += rng.normal(scale=0.5 * np.sqrt(np.mean(frames[3] ** 2)),
                                size=frames.shape[1])
        result = robust.estimate_direction_details(frames, fs=48000)
        self._assert_metadata(result)
        incident = np.array([3 in pair for pair in robust.PAIRS])
        self.assertLess(np.median(result.pair_weights[incident]),
                        np.median(result.pair_weights[~incident]))
        # Either a supported accurate solution or explicit withholding is
        # valid; quality weighting must not manufacture a confident wrong one.
        if result.vector is not None:
            self.assertLess(self._angular_error(result.vector, truth), 2)

    def test_absolute_pair_quality_survives_weighted_fit_scale_invariance(self):
        truth = np.array([0.36, -0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        reference = robust.estimate_direction_details(frames, fs=48000)
        observation = robust._pair_delay_estimate(frames, fs=48000)
        # A weighted fit is unchanged when every weight has the same scale.
        # Confidence must still retain evidence that all six pair peaks are
        # weak. Preserve the actual delays and actual Welch-coherence input;
        # vary only the absolute observed correlation quality.
        weaker_observation = replace(observation, peaks=observation.peaks * 0.05)
        with mock.patch.object(robust, '_pair_delay_estimate', return_value=weaker_observation):
            weaker = robust.estimate_direction_details(frames, fs=48000)
        self._assert_metadata(reference)
        self._assert_metadata(weaker)
        self.assertEqual(reference.status, 'full_3d')
        self.assertEqual(weaker.status, 'full_3d')
        np.testing.assert_allclose(weaker.vector, reference.vector, atol=1e-10)
        np.testing.assert_array_equal(weaker.pair_coherence, reference.pair_coherence)
        np.testing.assert_allclose(weaker.pair_weights / weaker.pair_weights.sum(),
                                   reference.pair_weights / reference.pair_weights.sum(),
                                   atol=1e-10)
        self.assertGreater(weaker.confidence, 0)
        self.assertLess(weaker.confidence, reference.confidence * 0.1)
        self.assertLess(weaker.support_confidence, reference.support_confidence * 0.1)

    def test_noisy_tonal_source_cannot_issue_a_confident_wrong_direction(self):
        # Stable narrow-band content can have high Welch coherence while the
        # broad-band GCC delay peaks remain weak. Such coherence alone must
        # not make an inaccurate fitted vector actionable.
        times = np.arange(2048) / 48000
        signal = (np.sin(2 * np.pi * 220 * times + 0.37) +
                  0.1 * np.random.default_rng(144).normal(size=2048))
        frames = robust.simulate_array(signal, 35, 0, np.random.default_rng(20),
                                       fs=48000, elevation_deg=50)
        result = robust.estimate_direction_details(frames, fs=48000)
        self._assert_metadata(result)
        if result.vector is not None and result.confidence >= 0.2:
            self.assertLessEqual(self._angular_error(result.vector,
                                                     robust.unit_vector(35, 50)), 15)
        else:
            azimuth = robust.azimuth_from_direction(result.vector)
            self.assertIn('no turn', robust.turn_instruction(azimuth,
                                                            confidence=result.confidence))

    def test_short_window_periodic_aliases_do_not_become_accepted_3d_vectors(self):
        for length in [1024, 2048]:
            times = np.arange(length) / 48000
            for truth in [np.array([0., -1., 0.]), np.array([0., 0., 1.])]:
                with self.subTest(length=length, truth=truth):
                    frames = np.sin(2 * np.pi * 2000 *
                                    (times[None, :] + (robust.MICS @ truth / robust.C)[:, None]))
                    result = robust.estimate_direction_details(frames, fs=48000)
                    self._assert_metadata(result)
                    self.assertIn(result.status, {'ambiguous', 'unreliable'})
                    self.assertIsNone(result.vector)
                    self.assertTrue(np.any(result.ambiguous_pairs))

    def test_short_noninteger_periodic_tones_do_not_return_wrong_3d_vectors(self):
        cases = [(512, 1750., -120., 0., 0.),
                 (1024, 1788.376989596092, -94.89043578987696,
                  -28.04894205123877, 1.9947965927886955)]
        for length, frequency, azimuth, elevation, phase in cases:
            with self.subTest(length=length, frequency=frequency):
                truth = robust.unit_vector(azimuth, elevation)
                times = np.arange(length) / 48000
                frames = np.sin(2 * np.pi * frequency *
                                (times[None, :] + (robust.MICS @ truth / robust.C)[:, None]) + phase)
                result = robust.estimate_direction_details(frames, fs=48000)
                self._assert_metadata(result)
                self.assertIsNone(result.vector)
                self.assertEqual(result.confidence, 0)

    def test_underresolved_windows_withhold_known_wrong_tonal_estimates(self):
        # A 1 ms tone was previously accepted more than 110 degrees away from
        # truth. Verify the duration floor at both rates, including through
        # the tuple API; zero padding must not be used to invent observation time.
        truth = robust.unit_vector(-120, 0)
        for fs, cases in [(48000, [(32, 1500), (48, 1500), (64, 1500), (128, 1750), (256, 1750)]),
                          (16000, [(16, 1750), (32, 1750), (64, 1750)])]:
            for length, frequency in cases:
                with self.subTest(fs=fs, length=length):
                    times = np.arange(length)/fs
                    frames = np.sin(2*np.pi*frequency*(
                        times[None, :] + (robust.MICS @ truth/robust.C)[:, None]))
                    result = robust.estimate_direction_details(frames, fs=fs)
                    self._assert_metadata(result)
                    self.assertEqual(result.status, 'unreliable')
                    self.assertIsNone(result.vector)
                    self.assertEqual(result.confidence, 0)
                    self.assertIn('10 ms', result.reason)
                    vector, confidence, _, _ = robust.estimate_direction(frames, fs=fs)
                    self.assertIsNone(vector)
                    self.assertEqual(confidence, 0)

    def test_tuple_wrapper_preserves_vector_confidence_and_ambiguous_rejection(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        for missing in [None, 3]:
            capture = frames.copy()
            if missing is not None:
                capture[missing] = 0
            with self.subTest(missing=missing):
                details = robust.estimate_direction_details(capture, fs=48000)
                vector, confidence, grid, fit = robust.estimate_direction(capture, fs=48000)
                if details.vector is None:
                    self.assertIsNone(vector)
                else:
                    np.testing.assert_array_equal(vector, details.vector)
                self.assertEqual(confidence, details.confidence)
                np.testing.assert_array_equal(grid, details.grid)
                np.testing.assert_array_equal(fit, details.fit)

    def test_invalid_hemisphere_normals_are_rejected(self):
        frames = self._capture(robust.MICS, np.array([0.36, 0.48, 0.8]))
        for normal in [[0, 0, 0], [0, 1], [np.nan, 0, 1], [0, np.inf, 1]]:
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                robust.estimate_direction_details(frames, fs=48000,
                                                   hemisphere_normal=normal)

    def test_extreme_finite_hemisphere_scaling_preserves_prior_in_both_ranks(self):
        above = np.array([0.36, 0.48, 0.8])
        below = above * [1, 1, -1]
        full_rank = self._capture(robust.MICS, below)
        rank_two = self._capture(robust.MICS, above)
        rank_two[3] = 0
        reference = robust.estimate_direction_details(rank_two, fs=48000,
                                                      hemisphere_normal=[0, 0, 1])
        for scale in [1e-200, 1e-160, 1e200]:
            with self.subTest(scale=scale):
                normal = np.array([0., 0., scale])
                rejected = robust.estimate_direction_details(full_rank, fs=48000,
                                                             hemisphere_normal=normal)
                self._assert_metadata(rejected)
                self.assertEqual(rejected.status, 'unreliable')
                self.assertIsNone(rejected.vector)
                selected = robust.estimate_direction_details(rank_two, fs=48000,
                                                             hemisphere_normal=normal)
                self._assert_metadata(selected)
                self.assertEqual(selected.status, 'constrained_3d')
                np.testing.assert_array_equal(selected.vector, reference.vector)
                self.assertEqual(selected.confidence, reference.confidence)

    def test_public_angle_helpers_normalize_extreme_finite_vectors(self):
        first = np.array([-0.36, 0.48, 0.8])
        second = np.array([0.48, 0.36, -0.8])
        azimuth = robust.azimuth_from_direction(first)
        elevation = robust.elevation_from_direction(first)
        error = robust.direction_error_deg(first, second)
        for scale in [1e-200, 1e-160, 1e200]:
            with self.subTest(scale=scale):
                self.assertAlmostEqual(robust.azimuth_from_direction(first * scale), azimuth)
                self.assertAlmostEqual(robust.elevation_from_direction(first * scale), elevation)
                actual_error = robust.direction_error_deg(first * scale, second / scale)
                self.assertIsNotNone(actual_error)
                self.assertAlmostEqual(actual_error, error)
                self_error = robust.direction_error_deg(first * scale, first)
                self.assertIsNotNone(self_error)
                self.assertLess(self_error, 1e-5)

    def test_invalid_turn_gate_and_deadband_parameters_cannot_issue_instructions(self):
        parameters = [{'min_confidence': value}
                      for value in [np.nan, np.inf, -np.inf, -0.01, 1.01]]
        parameters += [{'deadband': value}
                       for value in [np.nan, np.inf, -np.inf, -0.01, 180.01]]
        for kwargs in parameters:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                robust.turn_instruction(90, confidence=0.001, **kwargs)

    def test_vector_wrappers_forward_explicit_hemisphere_option(self):
        truth = np.array([0.36, 0.48, 0.8])
        frames = self._capture(robust.MICS, truth)
        frames[3] = 0
        details = robust.estimate_direction_details(frames, fs=48000,
                                                    hemisphere_normal=[0, 0, 1])
        for estimator in [robust.estimate_direction, robust.estimate_direction_3d]:
            with self.subTest(estimator=estimator.__name__):
                vector, confidence, grid, fit = estimator(frames, fs=48000,
                                                          hemisphere_normal=[0, 0, 1])
                np.testing.assert_array_equal(vector, details.vector)
                self.assertEqual(confidence, details.confidence)
                np.testing.assert_array_equal(grid, details.grid)
                np.testing.assert_array_equal(fit, details.fit)


if __name__ == '__main__':
    unittest.main()
