import unittest

import torch

from dexmachina.rl.coordinate_network_builder import (
    CoordinateActorCriticBuilder,
)


def make_params(message_enabled=True):
    return {
        "separate": True,
        "space": {
            "continuous": {
                "mu_activation": "None",
                "sigma_activation": "None",
                "mu_init": {"name": "default"},
                "sigma_init": {"name": "const_initializer", "val": 0},
                "fixed_sigma": True,
            }
        },
        "mlp": {
            "units": [32, 16],
            "activation": "elu",
            "initializer": {"name": "default"},
            "d2rl": False,
        },
        "coordinate": {
            "hand_obs_slices": [[0, 4], [4, 8]],
            "hand_feature_indices": [[1, 2, 3], [5, 6, 7]],
            "shared_feature_indices": [8, 9],
            "previous_action_slices": [[10, 12], [12, 14]],
            "active_mask_slice": [14, 16],
            "presence_mask_slice": [16, 18],
            "message_gate_slice": [18, 19],
            "message_enabled": message_enabled,
            "encoder_dim": 16,
            "message_hidden_dim": 8,
            "message_dim": 8,
            "actor_units": [16, 8],
            "critic_units": [32, 16],
            "activation": "elu",
        },
    }


def make_network(message_enabled=True):
    torch.manual_seed(7)
    builder = CoordinateActorCriticBuilder()
    builder.load(make_params(message_enabled))
    return builder.build(
        "coordinate", actions_num=4, input_shape=(19,), value_size=1
    )


def make_obs(batch=3, message_gate=1.0):
    torch.manual_seed(11)
    obs = torch.randn(batch, 19)
    obs[:, 14:16] = 1.0
    obs[:, 16:18] = 1.0
    obs[:, 18:19] = message_gate
    return obs


def swap_hands(obs):
    swapped = obs.clone()
    swapped[:, 0:4] = obs[:, 4:8]
    swapped[:, 4:8] = obs[:, 0:4]
    swapped[:, 10:12] = obs[:, 12:14]
    swapped[:, 12:14] = obs[:, 10:12]
    swapped[:, 14:15] = obs[:, 15:16]
    swapped[:, 15:16] = obs[:, 14:15]
    swapped[:, 16:17] = obs[:, 17:18]
    swapped[:, 17:18] = obs[:, 16:17]
    return swapped


class CoordinateNetworkBuilderTests(unittest.TestCase):
    def test_output_shapes_and_fixed_logstd(self):
        network = make_network()
        mu, logstd, value, states = network({"obs": make_obs()})
        self.assertEqual(mu.shape, (3, 4))
        self.assertEqual(logstd.shape, (3, 4))
        self.assertEqual(value.shape, (3, 1))
        self.assertIsNone(states)
        torch.testing.assert_close(logstd, torch.zeros_like(logstd))

    def test_shared_actor_is_permutation_equivariant(self):
        network = make_network()
        obs = make_obs()
        mu, _, _, _ = network({"obs": obs})
        swapped_mu, _, _, _ = network({"obs": swap_hands(obs)})
        torch.testing.assert_close(swapped_mu[:, 0:2], mu[:, 2:4])
        torch.testing.assert_close(swapped_mu[:, 2:4], mu[:, 0:2])

    def test_partner_cannot_leak_around_zero_message_gate(self):
        network = make_network()
        obs = make_obs(message_gate=0.0)
        changed = obs.clone()
        changed[:, 4:8] += 20.0
        changed[:, 12:14] -= 10.0
        mu, _, _, _ = network({"obs": obs})
        changed_mu, _, _, _ = network({"obs": changed})
        # Left action is isolated from right-hand state while communication is
        # masked.  The right action can still respond to its own local state.
        torch.testing.assert_close(changed_mu[:, 0:2], mu[:, 0:2])
        self.assertFalse(torch.allclose(changed_mu[:, 2:4], mu[:, 2:4]))

    def test_partner_message_changes_recipient_action_in_stage3(self):
        network = make_network()
        obs = make_obs(message_gate=1.0)
        changed = obs.clone()
        changed[:, 5:8] += 20.0
        changed[:, 12:14] -= 10.0
        mu, _, _, _ = network({"obs": obs})
        changed_mu, _, _, _ = network({"obs": changed})
        self.assertFalse(torch.allclose(changed_mu[:, 0:2], mu[:, 0:2]))

    def test_no_message_ablation_is_partner_isolated(self):
        network = make_network(message_enabled=False)
        obs = make_obs(message_gate=1.0)
        changed = obs.clone()
        changed[:, 4:8] += 20.0
        changed[:, 12:14] -= 10.0
        mu, _, _, _ = network({"obs": obs})
        changed_mu, _, _, _ = network({"obs": changed})
        torch.testing.assert_close(changed_mu[:, 0:2], mu[:, 0:2])

    def test_stage_gate_blocks_partner_encoder_gradients(self):
        network = make_network()
        stage2_obs = make_obs(message_gate=0.0)
        mu, _, _, _ = network({"obs": stage2_obs})
        mu[:, 0:2].sum().backward()
        stage2_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in network.partner_encoder.parameters()
        )
        self.assertEqual(stage2_grad, 0.0)

        network.zero_grad(set_to_none=True)
        stage3_obs = make_obs(message_gate=1.0)
        mu, _, _, _ = network({"obs": stage3_obs})
        mu[:, 0:2].sum().backward()
        stage3_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in network.partner_encoder.parameters()
        )
        self.assertGreater(stage3_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
