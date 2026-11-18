"""Evaluation publishing contracts, with no network or W&B uploads."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from dexmachina.eval import publish_final_eval
from dexmachina.eval.protocol import build_evaluation_metadata


class EvaluationPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='publish-final-eval-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / 'run'
        (self.run_dir / 'params').mkdir(parents=True)
        (self.run_dir / 'params/env.yaml').write_text('wandb:\n  run_id: offline-test\n')
        self.checkpoint = self.run_dir / 'model.pth'
        self.checkpoint.write_bytes(b'not-a-real-checkpoint-test-hash-only')
        self.eval_dir = self.run_dir / 'model_eval_all_policy'
        self.eval_dir.mkdir()
        self.stats = self.eval_dir / 'add_stats.json'
        self.stats.write_text(json.dumps({'auc': {'overall': .8}, 'mean_add': {'overall': .02}}))
        self.protocol = build_evaluation_metadata(
            self.checkpoint, num_envs=100, eval_episodes=1, all_policy_envs=True,
        )
        self.protocol.update(status='complete', completed_episodes=1)
        self.save_protocol()
        self.run = SimpleNamespace(log=mock.Mock(), summary={}, finish=mock.Mock())
        self.wandb = SimpleNamespace(init=mock.Mock(return_value=self.run))

    def save_protocol(self):
        (self.eval_dir / 'eval_protocol.json').write_text(json.dumps(self.protocol))

    def publish(self, *extra):
        argv = ['publish', '--run-dir', str(self.run_dir), '--num-trials', '100',
                '--stats-path', str(self.stats), '--checkpoint', str(self.checkpoint), *extra]
        with mock.patch.object(sys, 'argv', argv), mock.patch.dict(sys.modules, {'wandb': self.wandb}):
            publish_final_eval.main()

    def test_records_control_count_hash_and_protocol(self):
        self.publish()
        self.assertEqual(self.run.summary['final_eval/policy_controlled_trials'], 100)
        self.assertEqual(self.run.summary['final_eval/protocol'], 'all_policy_controlled_v1')
        self.assertEqual(self.run.summary['final_eval/checkpoint_sha256'], self.protocol['checkpoint_sha256'])
        self.run.finish.assert_called_once()

    def test_separate_prefix_preserves_historical_summary(self):
        self.run.summary['final_eval/add_auc'] = .6696
        self.publish('--metric-prefix', 'all_policy_eval')
        self.assertEqual(self.run.summary['final_eval/add_auc'], .6696)
        self.assertEqual(self.run.summary['all_policy_eval/add_auc'], .8)

    def test_incomplete_or_wrong_count_rejected_before_network(self):
        for overrides in ({'status': 'running'}, {'completed_episodes': 0},
                          {'num_trials': 99}, {'policy_controlled_trials': 99}):
            with self.subTest(overrides=overrides):
                old = dict(self.protocol)
                self.protocol.update(overrides)
                self.save_protocol()
                with self.assertRaises(ValueError):
                    self.publish()
                self.wandb.init.assert_not_called()
                self.protocol = old

    def test_changed_checkpoint_rejected_before_network(self):
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.publish()
        self.wandb.init.assert_not_called()

    def test_nonfinite_statistics_rejected_before_network(self):
        self.stats.write_text(json.dumps({'auc': {'overall': float('nan')}, 'mean_add': {'overall': .02}}))
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.publish()
        self.wandb.init.assert_not_called()

    def test_external_checkpoint_rejected_before_network(self):
        outside = self.root / 'external.pth'
        outside.write_bytes(b'outside')
        with self.assertRaisesRegex(ValueError, 'belong'):
            self.publish('--checkpoint', str(outside))
        self.wandb.init.assert_not_called()


if __name__ == '__main__':
    unittest.main()
