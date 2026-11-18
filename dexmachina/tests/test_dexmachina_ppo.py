"""Parity with the actual pinned learner, and restored module controls."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.torch_runner import Runner

from dexmachina.rl.ablation_ppo import (
    ALGORITHM_NAME, DEXMACHINA_ALGORITHM_NAME, configure_ablation_ppo,
    register_ablation_ppo,
)
from dexmachina.rl.per_hand_ppo import per_hand_neglogp
from dexmachina.tests.test_ablation_ppo import make_config as make_v2_config


def make_config(directory, *, ratio_mode="joint", value_mode="scalar", **overrides):
    config = make_v2_config(directory, ratio_mode=ratio_mode, value_mode=value_mode)
    options = config["params"]["config"]
    options.update(ablation_protocol="dexmachina", learning_rate=3e-4,
                   lr_schedule="adaptive", schedule_type="standard", kl_threshold=0.008,
                   max_epochs=10, score_to_win=100000, save_best_after=10,
                   horizon_length=8, minibatch_size=8, mini_epochs=14, print_stats=False)
    options.update(overrides)
    uenv = options["vec_env"].env.unwrapped
    uenv.curriculum.num_epoch_since_zero = 0
    uenv.start_recording = lambda: None
    uenv.get_recorded_frames = lambda **kwargs: None
    uenv.dt = 1 / 30
    configure_ablation_ppo(config, ratio_mode=ratio_mode, value_mode=value_mode)
    return config


class SchedulerSpy:
    def __init__(self, scheduler):
        self.scheduler, self.calls = scheduler, []

    def update(self, current_lr, entropy, epoch, frames, kl):
        output = self.scheduler.update(current_lr, entropy, epoch, frames, kl)
        self.calls.append((current_lr, entropy, epoch, frames, kl, *output))
        return output


class DexMachinaPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="native-ppo-parity-")
        self.agents = []

    def tearDown(self):
        for agent in self.agents:
            agent.writer.close()
        self.directory.cleanup()

    def agent(self, *, baseline=False, config=None, **options):
        cfg = config or make_config(self.directory.name, **options)
        cfg = deepcopy(cfg)
        if baseline:
            cfg["params"]["algo"]["name"] = "a2c_continuous"
        runner = Runner()
        original = runner.algo_factory._builders["a2c_continuous"]
        register_ablation_ppo(runner)
        self.assertIs(runner.algo_factory._builders["a2c_continuous"], original)
        runner.load(cfg)
        agent = runner.algo_factory.create(cfg["params"]["algo"]["name"],
                                            base_name="test", params=runner.params)
        agent.algo_observer.wandb_after_print_stats = lambda *args: {}
        self.agents.append(agent)
        return agent

    @staticmethod
    def rollout(agent):
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.set_eval()
        with torch.no_grad():
            return agent.play_steps()

    def test_joint_scalar_matches_pinned_agent_through_all_56_updates(self):
        config = make_config(self.directory.name)
        baseline = self.agent(baseline=True, config=config)
        restored = self.agent(config=config)
        for key, expected in baseline.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], expected, rtol=0, atol=0)
        batch = self.rollout(baseline)
        self.rollout(restored)
        outputs, scheduler_calls, parameters_after_steps = [], [], []
        for agent in (baseline, restored):
            agent.play_steps = lambda: deepcopy(batch)
            agent.scheduler = SchedulerSpy(agent.scheduler)
            snapshots = []
            optimizer_step = agent.optimizer.step

            def record_step(*args, _step=optimizer_step, _agent=agent, **kwargs):
                result = _step(*args, **kwargs)
                snapshots.append({name: value.detach().clone()
                                  for name, value in _agent.model.named_parameters()})
                return result

            with patch.object(agent.optimizer, "step", side_effect=record_step):
                outputs.append(agent.train_epoch())
            scheduler_calls.append(agent.scheduler.calls)
            parameters_after_steps.append(snapshots)
        self.assertEqual(len(parameters_after_steps[0]), 56)
        self.assertEqual(len(parameters_after_steps[1]), 56)
        self.assertEqual(len(scheduler_calls[0]), 14)
        self.assertEqual(scheduler_calls[0], scheduler_calls[1])
        # Establish that this exercised adaptation rather than fixed LR.
        self.assertGreater(len({call[0] for call in scheduler_calls[0]}), 1)
        for old_step, new_step in zip(*parameters_after_steps):
            for key in old_step:
                torch.testing.assert_close(new_step[key], old_step[key], rtol=0, atol=0)
        # Timing fields differ; losses, KL history and LR must be exact.
        for index in (4, 5, 6, 7, 8):
            for expected, actual in zip(outputs[0][index], outputs[1][index]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(outputs[0][9:], outputs[1][9:])
        self.assertEqual(baseline.optimizer.state_dict()["param_groups"],
                         restored.optimizer.state_dict()["param_groups"])
        for state_id, expected in baseline.optimizer.state_dict()["state"].items():
            for key, value in expected.items():
                torch.testing.assert_close(restored.optimizer.state_dict()["state"][state_id][key],
                                           value, rtol=0, atol=0)

    def test_ratio_control_duplicates_exact_baseline_advantage(self):
        joint = self.agent()
        per_hand = self.agent(ratio_mode="per_hand")
        batch = self.rollout(joint)
        for agent in (joint, per_hand):
            agent.set_train()
            agent.prepare_dataset(deepcopy(batch))
        expected = joint.dataset.values_dict["advantages"]
        actual = per_hand.dataset.values_dict["advantages"]
        self.assertTrue(torch.equal(actual[:, 0], expected))
        self.assertTrue(torch.equal(actual[:, 1], expected))
        for key in ("old_values", "returns"):
            torch.testing.assert_close(per_hand.dataset.values_dict[key],
                                       joint.dataset.values_dict[key], rtol=0, atol=0)

    def test_scheduler_uses_mutable_reference_but_ratio_denominator_stays_frozen(self):
        agent = self.agent(ratio_mode="per_hand")
        batch = self.rollout(agent)
        agent.set_train()
        agent.prepare_dataset(batch)
        data = agent.dataset.values_dict
        fixed_nll = data["old_logp_actions"].clone()
        fixed_mu = data["ablation_old_mu"].clone()
        minibatch = agent.dataset[0]
        # Mimic a previous visit's policy; make the scale small enough that the
        # pinned epsilon formula cannot be confused with exact Gaussian KL.
        reference_mu = minibatch["mu"].clone() + 0.01
        reference_sigma = torch.full_like(minibatch["sigma"], 0.02)
        agent.dataset.update_mu_sigma(reference_mu, reference_sigma)
        minibatch = agent.dataset[0]
        agent.calc_gradients(minibatch)
        mu, sigma = agent.train_result[6:8]
        expected = torch_ext.policy_kl(mu, sigma, reference_mu, reference_sigma)
        torch.testing.assert_close(agent.train_result[3], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(fixed_nll, data["old_logp_actions"]))
        self.assertTrue(torch.equal(fixed_mu, data["ablation_old_mu"]))
        # Actual current parameters equal rollout parameters before this first
        # optimizer step; the frozen-forward diagnostic is therefore zero.
        torch.testing.assert_close(agent.aux_loss_dict["ablation_ppo/frozen_rollout_forward_kl"][0],
                                   torch.tensor(0.0), atol=1e-6, rtol=0)
        self.assertGreater(expected.item(), 1)

    def test_per_hand_first_order_actor_gradient_matches_baseline(self):
        baseline, hand = self.agent(baseline=True), self.agent(ratio_mode="per_hand")
        batch = self.rollout(baseline)
        grads = []
        for agent in (baseline, hand):
            agent.prepare_dataset(deepcopy(batch))
            minibatch = agent.dataset[0]
            result = agent.model({"is_train": True, "prev_actions": minibatch["actions"],
                                   "obs": minibatch["obs"]})
            if agent is baseline:
                loss = common_losses.actor_loss(minibatch["old_logp_actions"],
                                                result["prev_neglogp"], minibatch["advantages"],
                                                True, agent.e_clip).mean()
            else:
                loss = common_losses.actor_loss(
                    minibatch["old_logp_actions"],
                    per_hand_neglogp(minibatch["actions"], result["mus"], result["sigmas"]),
                    minibatch["advantages"], True, agent.e_clip,
                ).sum(-1).mean()
            loss.backward()
            grads.append({name: value.grad.clone() for name, value in agent.model.named_parameters()
                          if value.grad is not None})
        self.assertEqual(set(grads[0]), set(grads[1]))
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name], grads[1][name], rtol=2e-5, atol=1e-7)

    def test_per_hand_value_mode_keeps_two_value_streams(self):
        agent = self.agent(ratio_mode="per_hand", value_mode="per_hand")
        action = torch.tensor([[0.5, 0.5, 0., 0.]]).expand(4, -1)
        agent.init_tensors()
        agent.env_reset()
        rewards = agent.env_step(action)[1]
        # Both value streams learn from the same scalar team reward.
        self.assertEqual(rewards.shape, (4, 2))
        self.assertTrue(torch.equal(rewards[:, 0], rewards[:, 1]))
        agent.obs = agent.env_reset()
        result = agent.train_epoch()
        self.assertEqual(len(result[4]), 56)
        self.assertEqual(agent.dataset.values_dict["returns"].shape, (32, 2))

    def test_native_train_retains_upcoming_epoch_and_early_stop_decisions(self):
        baseline, restored = self.agent(baseline=True), self.agent()
        for agent in (baseline, restored):
            agent.vec_env.env.unwrapped.curriculum.num_epoch_since_zero = 1501
        with patch("rl_games.common.a2c_common.wandb.log"):
            original_result = baseline.train()
            restored_result = restored.train()
        self.assertEqual(original_result[1], 1)
        self.assertEqual(restored_result[1], original_result[1])
        self.assertEqual(restored.vec_env.curriculum_calls, [1])
        self.assertEqual(restored.training_stop_reason, "native_zero_gain_patience")
        self.assertEqual(restored.get_full_state_weights()["training_termination"]["stop_reason"],
                         "native_zero_gain_patience")

    def test_protocol_contract_rejects_fixed_lr_and_incompatible_resume(self):
        config = make_config(self.directory.name)
        self.assertEqual(config["params"]["algo"]["name"], DEXMACHINA_ALGORITHM_NAME)
        option = dict(ratio_mode="joint", value_mode="scalar")
        for key, value in (("lr_schedule", "identity"), ("schedule_type", "legacy"),
                            ("kl_threshold", 0.004), ("learning_rate", 1e-4)):
            changed = deepcopy(config)
            changed["params"]["config"][key] = value
            with self.assertRaisesRegex(ValueError, key):
                configure_ablation_ppo(changed, **option)
        native = self.agent(config=config)
        self.rollout(native)
        state = native.get_full_state_weights()
        v2 = self.agent(config=make_v2_config(self.directory.name, **option))
        with self.assertRaisesRegex(ValueError, "resume metadata"):
            v2.set_full_state_weights(state)
        registered = Runner()
        register_ablation_ppo(registered)
        self.assertIn(ALGORITHM_NAME, registered.algo_factory._builders)
        self.assertIn(DEXMACHINA_ALGORITHM_NAME, registered.algo_factory._builders)


if __name__ == "__main__":
    unittest.main()
