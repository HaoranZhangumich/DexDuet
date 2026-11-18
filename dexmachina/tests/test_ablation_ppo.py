"""CPU contracts for the per-hand learner's ratio and value options and the exact budget."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import gym
import numpy as np
import torch
import yaml
from rl_games.common import common_losses
from rl_games.torch_runner import Runner

from dexmachina.rl.ablation_ppo import (
    ALGORITHM_NAME, _options, actor_advantages, clipped_actor_loss, configure_ablation_ppo,
    expand_shared_reward, register_ablation_ppo,
)
from dexmachina.rl.per_hand_ppo import per_hand_entropy, per_hand_kl, per_hand_neglogp
from dexmachina.tests.test_coordinate_network_builder import make_obs, make_params


class FakeVecEnv:
    """Deterministic bimanual dynamics, avoiding Genesis or CUDA in tests."""
    def __init__(self, batch=4):
        self.batch = batch
        self.t = 0
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(per_hand_ppo=True))

    def reset(self):
        self.t = 0
        return make_obs(self.batch)

    def step(self, actions):
        self.t += 1
        obs = make_obs(self.batch)
        obs[:, :4] += actions * 0.05
        reward = 1.0 - actions.square().mean(-1)
        return obs, reward, torch.full((self.batch,), self.t % 3 == 0), {
            "episode": {"task_rew": reward},
        }

    def set_train_info(self, *args):
        pass

    def get_env_state(self):
        return {"t": self.t}

    def set_env_state(self, state):
        self.t = state["t"]


class FakeAblationVecEnv(FakeVecEnv):
    def __init__(self):
        super().__init__()
        self.curriculum_calls = []
        self.env.unwrapped.set_curriculum = self.set_curriculum

    def set_curriculum(self, completed_updates):
        self.curriculum_calls.append(completed_updates)
        return False


def make_config(directory, *, ratio_mode="per_hand", value_mode="per_hand", max_epochs=3):
    cfg_path = Path(__file__).parents[1] / "rl/configs/rl_games_ppo_cfg.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["params"]["network"] = make_params()
    cfg["params"]["network"]["name"] = "coordinate_actor_critic"
    env = FakeAblationVecEnv()
    # These trigger both hardcoded legacy early stops; v2 must ignore them.
    env.env.unwrapped.use_curriculum = True
    env.env.unwrapped.curriculum = SimpleNamespace(num_epoch_since_zero=99999)
    cfg["params"]["config"].update({
        "device": "cpu", "device_name": "cpu", "num_actors": env.batch,
        "horizon_length": 4, "minibatch_size": 8, "mini_epochs": 2,
        "seq_length": 1, "train_dir": directory, "full_experiment_name": "test",
        "vec_env": env, "lr_schedule": "identity", "learning_rate": 3e-4,
        "max_epochs": max_epochs, "max_frames": -1, "score_to_win": -999999,
        "save_best_after": 0, "env_info": {
            "value_size": 2 if value_mode == "per_hand" else 1, "agents": 1,
            "observation_space": gym.spaces.Box(-np.inf, np.inf, (19,)),
            "action_space": gym.spaces.Box(-1.0, 1.0, (4,)),
        },
    })
    configure_ablation_ppo(cfg, ratio_mode=ratio_mode, value_mode=value_mode)
    return cfg


class AblationPPOMathTests(unittest.TestCase):
    def test_ratio_only_control_has_identical_unclipped_first_order_gradient(self):
        torch.manual_seed(19)
        old_mu, actions = torch.randn(7, 4), torch.randn(7, 4)
        old_logstd = torch.randn(7, 4) * 0.2
        old_nll = per_hand_neglogp(actions, old_mu, old_logstd.exp())
        raw = torch.arange(7, dtype=torch.float32)[:, None]
        gradients = []
        for mode in ("joint", "per_hand"):
            mu = old_mu.clone().requires_grad_()
            logstd = old_logstd.clone().requires_grad_()
            loss, _, ratios = clipped_actor_loss(
                old_nll, per_hand_neglogp(actions, mu, logstd.exp()),
                actor_advantages(raw, mode), mode, 0.2,
            )
            torch.testing.assert_close(ratios, torch.ones_like(ratios))
            gradients.append(torch.autograd.grad(loss, (mu, logstd)))
        for joint, per_hand in zip(*gradients):
            torch.testing.assert_close(joint, per_hand, rtol=1e-6, atol=1e-7)

    def test_ratio_only_advantage_exactly_duplicates_joint_scalar_advantage(self):
        raw = torch.tensor([[1.0], [2.0], [4.0]])
        joint = actor_advantages(raw, "joint")
        per_hand = actor_advantages(raw, "per_hand")
        self.assertTrue(torch.equal(per_hand[:, :1], joint))
        self.assertTrue(torch.equal(per_hand[:, 1:], joint))
        # Two learned critics are intentionally a distinct, later module.
        dual = actor_advantages(torch.cat((raw, -raw), -1), "per_hand")
        self.assertFalse(torch.equal(dual[:, 0], dual[:, 1]))

    def test_shared_reward_is_detached_and_repeated_per_value_column(self):
        shared = torch.tensor([1.0, 2.0], requires_grad=True)
        per_hand = expand_shared_reward(shared, "per_hand")
        self.assertTrue(torch.equal(per_hand, torch.tensor([[1.0, 1.0], [2.0, 2.0]])))
        self.assertFalse(per_hand.requires_grad)
        scalar = expand_shared_reward(shared[:, None], "scalar")
        self.assertTrue(torch.equal(scalar, torch.tensor([[1.0], [2.0]])))
        # A copy, not a view of the shared reward.
        per_hand[0, 0] = 5.0
        self.assertEqual(shared[0].item(), 1.0)
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            expand_shared_reward(torch.tensor([1.0, float("nan")]), "per_hand")
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            expand_shared_reward(torch.ones(2, 2), "per_hand")
        with self.assertRaisesRegex(ValueError, "value_mode"):
            expand_shared_reward(shared, "per_agent")

    def test_full_action_kl_entropy_and_mean_critic_reductions(self):
        torch.manual_seed(22)
        old_mu, mu = torch.randn(9, 4), torch.randn(9, 4)
        old_sigma, sigma = torch.rand(9, 4) + 0.3, torch.rand(9, 4) + 0.3
        old, new = torch.distributions.Normal(old_mu, old_sigma), torch.distributions.Normal(mu, sigma)
        torch.testing.assert_close(per_hand_kl(old_mu, old_sigma, mu, sigma).sum(-1).mean(),
                                   torch.distributions.kl_divergence(old, new).sum(-1).mean())
        torch.testing.assert_close(per_hand_kl(old_mu, old_sigma, old_mu, old_sigma),
                                   torch.zeros(9, 2), atol=1e-7, rtol=0)
        torch.testing.assert_close(per_hand_entropy(sigma).sum(-1).mean(), new.entropy().sum(-1).mean())
        values, returns = torch.randn(9, 1), torch.randn(9, 1)
        single = common_losses.default_critic_loss(values, values, 0.2, returns, True).mean()
        dual = common_losses.default_critic_loss(values.expand(-1, 2), values.expand(-1, 2),
                                                 0.2, returns.expand(-1, 2), True).mean()
        torch.testing.assert_close(single, dual)

    def test_hand_clipping_does_not_couple_the_other_hand(self):
        mu = torch.tensor([[0.3, 0.0, 0.2, 0.0]], requires_grad=True)
        actions, sigma = torch.ones_like(mu), torch.ones_like(mu)
        loss, _, _ = clipped_actor_loss(per_hand_neglogp(actions, torch.zeros_like(mu), sigma),
                                        per_hand_neglogp(actions, mu, sigma),
                                        torch.tensor([[1.0, -1.0]]), "per_hand", 0.2)
        loss.backward()
        torch.testing.assert_close(mu.grad[:, :2], torch.zeros(1, 2))
        self.assertGreater(mu.grad[:, 2:].abs().sum().item(), 0)

    def test_config_rejects_confounded_or_incompatible_controls(self):
        option = dict(ratio_mode="joint", value_mode="scalar")
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory, **option)
            for key, value, message in (("lr_schedule", "adaptive", "fixed learning"),
                                         ("max_epochs", -1, "positive exact"),
                                         ("max_frames", 20, "max_epochs only"),
                                         ("normalize_input", True, "normalize_input")):
                changed = deepcopy(config)
                changed["params"]["config"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    configure_ablation_ppo(changed, **option)
        # Saved configs and checkpoints carry credit_mode and credit_coefficient as fixed schema fields.
        self.assertEqual(_options("per_hand", "per_hand", "none", 0.0),
                         dict(ratio_mode="per_hand", value_mode="per_hand", credit_mode="none",
                              credit_coefficient=0.0))
        for credit_mode, coefficient in (("individual", 0.0), ("shared", 0.0), ("none", 0.3),
                                         ("individual", 1.0), ("none", float("nan"))):
            with self.assertRaisesRegex(ValueError, "credit is not supported"):
                _options("per_hand", "per_hand", credit_mode, coefficient)
        with self.assertRaisesRegex(ValueError, "ratio_mode"):
            _options("per_agent", "per_hand")


class AblationPPOIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="ablation-ppo-test-")
        self.agents = []

    def tearDown(self):
        for agent in self.agents:
            agent.writer.close()
        self.directory.cleanup()

    def agent(self, **options):
        cfg = make_config(self.directory.name, **options)
        runner = Runner()
        original = runner.algo_factory._builders["a2c_continuous"]
        register_ablation_ppo(runner)
        self.assertIs(runner.algo_factory._builders["a2c_continuous"], original)
        runner.load(cfg)
        agent = runner.algo_factory.create(ALGORITHM_NAME, base_name="test", params=runner.params)
        agent.algo_observer.wandb_after_print_stats = lambda *args: {}
        agent._test_runner = runner
        self.agents.append(agent)
        return agent

    def test_all_ratio_value_combinations_rollout_update_and_freeze_old_policy(self):
        for ratio_mode in ("joint", "per_hand"):
            for value_mode in ("scalar", "per_hand"):
                with self.subTest(ratio_mode=ratio_mode, value_mode=value_mode):
                    agent = self.agent(ratio_mode=ratio_mode, value_mode=value_mode)
                    agent.init_tensors()
                    agent.obs = agent.env_reset()
                    result = agent.train_epoch()
                    self.assertEqual(len(result[4]), 4)
                    self.assertEqual(agent.curr_frames, 16)
                    data = agent.dataset.values_dict
                    self.assertEqual(data["old_values"].shape, (16, agent.value_size))
                    self.assertEqual(data["advantages"].shape, (16, 1 if ratio_mode == "joint" else 2))
                    frozen = data["ablation_old_mu"].clone()
                    agent.dataset[0]
                    agent.dataset.update_mu_sigma(torch.full((8, 4), 100.0), torch.ones(8, 4))
                    self.assertTrue(torch.equal(frozen, data["ablation_old_mu"]))
                    self.assertTrue(all(torch.isfinite(value) for value in result[4]))

    def test_resume_preserves_contract_and_rejects_different_value_mode(self):
        agent = self.agent()
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.train_epoch()
        checkpoint = deepcopy(agent.get_full_state_weights())
        self.assertEqual(checkpoint["ablation_ppo"]["credit_mode"], "none")
        self.assertEqual(checkpoint["ablation_ppo"]["credit_coefficient"], 0.0)
        restored = self.agent()
        restored.set_full_state_weights(checkpoint)
        for name, value in agent.model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[name])
        with self.assertRaisesRegex(ValueError, "resume metadata"):
            self.agent(value_mode="scalar").set_full_state_weights(checkpoint)

    def test_saved_config_with_reward_credit_is_rejected(self):
        for credit_mode, coefficient in (("individual", 1.0), ("shared", 0.0), ("none", 0.5)):
            cfg = make_config(self.directory.name)
            cfg["params"]["config"]["ablation_ppo"].update(credit_mode=credit_mode,
                                                           credit_coefficient=coefficient)
            runner = Runner()
            register_ablation_ppo(runner)
            runner.load(cfg)
            with self.assertRaisesRegex(ValueError, "credit is not supported"):
                runner.algo_factory.create(ALGORITHM_NAME, base_name="test", params=runner.params)

    def test_exact_budget_counts_all_updates_and_ignores_legacy_early_stops(self):
        agent = self.agent(max_epochs=3)
        result = agent.train()
        self.assertEqual(result[1], 3)
        self.assertEqual(agent.frame, 48)
        self.assertEqual(agent.vec_env.curriculum_calls, [0, 1, 2])
        self.assertEqual(agent.last_lr, 3e-4)
        checkpoint = torch.load(Path(agent.nn_dir) / "ablation_final.pth", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["frame"], 48)

    def test_evaluation_player_accepts_both_value_shapes(self):
        for value_mode in ("scalar", "per_hand"):
            agent = self.agent(value_mode=value_mode)
            player = agent._test_runner.create_player()
            player.model.load_state_dict(agent.get_full_state_weights()["model"])
            player.has_batch_dimension = True
            actual = player.get_action(make_obs(4), is_deterministic=True)
            agent.set_eval()
            with torch.no_grad():
                expected = agent.model({"is_train": False, "obs": make_obs(4)})["mus"].clamp(-1, 1)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
