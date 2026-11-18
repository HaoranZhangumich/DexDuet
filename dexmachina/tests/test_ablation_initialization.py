import unittest

import torch

from dexmachina.rl.ablation_initialization import parameter_hash
from dexmachina.rl.coordinate_network_builder import CoordinateActorCriticBuilder
from dexmachina.tests.test_coordinate_network_builder import make_params, make_obs


def build(values, message=True, seed=42):
    params = make_params(message)
    params['coordinate'].update(initialization_seed=seed, per_hand_value=values == 2)
    builder = CoordinateActorCriticBuilder()
    builder.load(params)
    return builder.build('ablation', actions_num=4, input_shape=(19,), value_size=values)


class AblationInitializationTests(unittest.TestCase):
    def test_actor_and_critic_trunk_match_across_value_layout_and_message_gate(self):
        scalar = build(1)
        torch.randn(137)
        dual = build(2)
        no_message = build(2, message=False)
        for other in (dual, no_message):
            self.assertEqual(parameter_hash(scalar), parameter_hash(other))
            self.assertEqual(parameter_hash(scalar, 'critic_trunk'), parameter_hash(other, 'critic_trunk'))
        obs = make_obs()
        torch.testing.assert_close(scalar({'obs': obs})[0], dual({'obs': obs})[0], rtol=0, atol=0)
        torch.testing.assert_close(scalar.value.weight[0], dual.value.weight[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(dual.value.weight[0], dual.value.weight[1]))

    def test_seed_changes_actor_weights(self):
        self.assertNotEqual(parameter_hash(build(1, seed=42)), parameter_hash(build(1, seed=43)))

    def test_no_message_control_has_same_weights_and_blocks_partner_information(self):
        network = build(2, message=False)
        obs = make_obs()
        changed = obs.clone()
        changed[:, 4:8] += 100
        changed[:, 12:14] -= 100
        torch.testing.assert_close(network({'obs': obs})[0][:, :2],
                                   network({'obs': changed})[0][:, :2], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
