"""CPU tests of the per-hand PPO pieces: hand-wise likelihood, entropy and KL, per-hand ratio clipping and advantages,
and the two value heads."""

from copy import deepcopy
import unittest

import torch
from rl_games.common import common_losses
from rl_games.torch_runner import Runner

from dexmachina.rl.ablation_ppo import actor_advantages, register_ablation_ppo
from dexmachina.rl.coordinate_network_builder import CoordinateActorCriticBuilder
from dexmachina.rl.per_hand_ppo import NUM_HANDS, per_hand_entropy, per_hand_kl, per_hand_neglogp
from dexmachina.tests.test_coordinate_network_builder import make_obs, make_params


class PerHandPPOMathTests(unittest.TestCase):
    def test_hand_likelihoods_entropy_kl_sum_to_joint(self):
        torch.manual_seed(12)
        actions, mu = torch.randn(7, 24), torch.randn(7, 24)
        sigma = torch.rand(7, 24) + 0.2
        old_mu, old_sigma = torch.randn(7, 24), torch.rand(7, 24) + 0.2
        normal = torch.distributions.Normal(mu, sigma)
        old = torch.distributions.Normal(old_mu, old_sigma)
        self.assertEqual(NUM_HANDS, 2)
        self.assertEqual(per_hand_neglogp(actions, mu, sigma).shape, (7, 2))
        torch.testing.assert_close(per_hand_neglogp(actions, mu, sigma).sum(-1),
                                   -normal.log_prob(actions).sum(-1))
        torch.testing.assert_close(per_hand_entropy(sigma).sum(-1), normal.entropy().sum(-1))
        torch.testing.assert_close(per_hand_kl(old_mu, old_sigma, mu, sigma).sum(-1),
                                   torch.distributions.kl_divergence(old, normal).sum(-1))
        with self.assertRaisesRegex(ValueError, "equal hand blocks"):
            per_hand_neglogp(actions[:, :23], mu[:, :23], sigma[:, :23])

    def test_ratio_clipping_and_gradient_are_per_hand(self):
        mu = torch.tensor([[0.3, 0.0, 0.2, 0.0]], requires_grad=True)
        action, sigma = torch.ones_like(mu), torch.ones_like(mu)
        old_nll = per_hand_neglogp(action, torch.zeros_like(mu), sigma)
        nll = per_hand_neglogp(action, mu, sigma)
        advantage = torch.tensor([[1.0, -1.0]])
        ratio = (old_nll - nll).exp()
        self.assertGreater(ratio[0, 0].item(), 1.2)
        loss = common_losses.actor_loss(old_nll, nll, advantage, True, 0.2)
        loss.sum().backward()
        # Positive left advantage is clipped; negative right advantage is not.
        torch.testing.assert_close(mu.grad[:, :2], torch.zeros(1, 2))
        self.assertGreater(mu.grad[:, 2:].abs().sum().item(), 0)
        # A shared scalar actor loss would incorrectly couple these two ratios.
        self.assertEqual(tuple(loss.shape), (1, 2))

    def test_advantages_are_normalized_independently(self):
        a = torch.tensor([[1.0, 100.0], [2.0, 300.0], [4.0, -100.0]])
        result = actor_advantages(a, "per_hand")
        torch.testing.assert_close(result.mean(0), torch.zeros(2), atol=1e-7, rtol=0)
        torch.testing.assert_close(result.std(0), torch.ones(2))
        changed = a.clone()
        changed[:, 1] *= -100
        torch.testing.assert_close(actor_advantages(changed, "per_hand")[:, 0], result[:, 0])

    def test_default_network_and_algorithm_are_unchanged(self):
        params = make_params()
        outputs = []
        for explicit_default in (False, True):
            trial = deepcopy(params)
            if explicit_default:
                trial["coordinate"]["per_hand_value"] = False
            torch.manual_seed(7)
            builder = CoordinateActorCriticBuilder()
            builder.load(trial)
            network = builder.build("coordinate", actions_num=4, input_shape=(19,), value_size=1)
            outputs.append(network({"obs": make_obs()}))
        for old, new in zip(outputs[0][:3], outputs[1][:3]):
            self.assertTrue(torch.equal(old, new))
        runner = Runner()
        original = runner.algo_factory._builders["a2c_continuous"]
        register_ablation_ppo(runner)
        self.assertIs(runner.algo_factory._builders["a2c_continuous"], original)

    def test_two_values_require_explicit_env_contract(self):
        params = make_params()
        params["coordinate"]["per_hand_value"] = True
        builder = CoordinateActorCriticBuilder()
        builder.load(params)
        with self.assertRaisesRegex(ValueError, "value_size"):
            builder.build("coordinate", actions_num=4, input_shape=(19,), value_size=1)
        network = builder.build("coordinate", actions_num=4, input_shape=(19,), value_size=2)
        self.assertEqual(network({"obs": make_obs()})[2].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
