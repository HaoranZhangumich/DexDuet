import hashlib
from pathlib import Path
import tempfile
import unittest

from dexmachina.eval.protocol import (
    build_evaluation_metadata, evaluation_folder_name,
    policy_controlled_env_indices, prepare_evaluation_directory, validate_evaluation_options,
    verify_common_reset_state,
)


class EvalProtocolTests(unittest.TestCase):
    def test_common_reset_is_frozen_and_changed_arrays_are_rejected(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "common_reset.json"
            states = {"solver0.qpos": np.arange(12, dtype=np.float32).reshape(3, 4),
                      "environment.episode_start_buf": np.zeros(3, dtype=np.int32)}
            first = verify_common_reset_state(path, states, seed=1042, num_envs=3)
            second = verify_common_reset_state(path, states, seed=1042, num_envs=3)
            self.assertEqual(first, second)
            self.assertEqual(first["initial_state_distribution"], "fixed_reference_start_repeated_across_environments")
            states["solver0.qpos"][0, 0] = 0.01
            with self.assertRaisesRegex(RuntimeError, "reset differs"):
                verify_common_reset_state(path, states, seed=1042, num_envs=3)

    def test_common_reset_rejects_seed_changes_and_file_tampering(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "common_reset.json"
            states = {"qpos": np.zeros((2, 3), dtype=np.float32)}
            metadata = verify_common_reset_state(path, states, seed=1042, num_envs=2)
            with self.assertRaisesRegex(RuntimeError, "reset differs"):
                verify_common_reset_state(path, states, seed=1043, num_envs=2)
            Path(metadata["state_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                verify_common_reset_state(path, states, seed=1042, num_envs=2)

    def test_training_and_single_environment_defaults_are_unchanged(self):
        self.assertEqual(policy_controlled_env_indices(100, is_eval=False), list(range(100)))
        self.assertEqual(policy_controlled_env_indices(1, is_eval=True), [0])

    def test_legacy_eval_preserves_last_unused_environment(self):
        self.assertEqual(policy_controlled_env_indices(100, is_eval=True), list(range(99)))
        self.assertEqual(evaluation_folder_name('inspire_hand.pth'), 'inspire_hand_eval')

    def test_opt_in_controls_all_and_preserves_legacy_artifact_path(self):
        self.assertEqual(policy_controlled_env_indices(100, is_eval=True, eval_reference_env=False),
                         list(range(100)))
        self.assertEqual(evaluation_folder_name('inspire_hand.pth', all_policy_envs=True),
                         'inspire_hand_eval_all_policy')

    def test_rejects_reference_visualization_in_all_policy_protocol(self):
        with self.assertRaisesRegex(ValueError, 'show_reference'):
            validate_evaluation_options(100, 1, all_policy_envs=True, show_reference=True)
        validate_evaluation_options(100, 1, all_policy_envs=False, show_reference=True)
        for environments, episodes in ((0, 1), (1, 0), (-1, 1), (1, -1)):
            with self.assertRaises(ValueError):
                validate_evaluation_options(environments, episodes, all_policy_envs=True, show_reference=False)

    def test_metadata_counts_trials_and_binds_exact_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='eval-protocol-test-') as directory:
            checkpoint = Path(directory) / 'test.pth'
            checkpoint.write_bytes(b'trusted test checkpoint bytes')
            legacy = build_evaluation_metadata(checkpoint, num_envs=100,
                                               eval_episodes=2, all_policy_envs=False)
            current = build_evaluation_metadata(checkpoint, num_envs=100,
                                                eval_episodes=2, all_policy_envs=True)
            self.assertEqual(legacy['policy_controlled_trials'], 198)
            self.assertEqual(legacy['uncontrolled_env_ids'], [99])
            self.assertEqual(current['policy_controlled_trials'], 200)
            self.assertEqual(current['num_trials'], 200)
            self.assertEqual(current['uncontrolled_env_ids'], [])
            self.assertEqual(current['checkpoint_sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(current['protocol'], 'all_policy_controlled_v1')
            self.assertNotEqual(current['protocol'], legacy['protocol'])

    def test_opt_in_refuses_to_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory(prefix='eval-protocol-test-') as directory:
            path = Path(directory) / 'checkpoint_eval_all_policy'
            prepare_evaluation_directory(path, all_policy_envs=True)
            sentinel = path / 'eval_ep0.npy'
            sentinel.write_bytes(b'previous result')
            with self.assertRaises(FileExistsError):
                prepare_evaluation_directory(path, all_policy_envs=True)
            self.assertEqual(sentinel.read_bytes(), b'previous result')
            # Default retains the previous evaluator's directory semantics.
            prepare_evaluation_directory(path, all_policy_envs=False)


if __name__ == '__main__':
    unittest.main()
