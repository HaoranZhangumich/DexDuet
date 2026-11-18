"""Factorized two-hand actor with a permutation-invariant partner message.

The released DexMachina actor remains the default.  Importing this module
registers an additional RL-Games network named ``coordinate_actor_critic``;
training selects it only when ``--coordinate_actor`` is passed.

The actor is decentralized by construction: a shared local encoder and action
head are applied once per hand.  The only path from another hand into a local
actor is the shared ``phi`` partner encoder followed by a Deep-Sets ``rho``
aggregator.  A gate channel of the observation can mask that message; the
environment always keeps it open.
The value function remains centralized and consumes the complete observation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence

import torch
from torch import nn

from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder


NETWORK_NAME = "coordinate_actor_critic"


def _as_slice(bounds: Sequence[int]) -> slice:
    if len(bounds) != 2:
        raise ValueError(f"Expected [start, stop] slice bounds, got {bounds}")
    return slice(int(bounds[0]), int(bounds[1]))


class CoordinateActorCriticBuilder(NetworkBuilder):
    """RL-Games builder for DexDuet's factorized per-hand policy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = None

    def load(self, params):
        self.params = deepcopy(params)

    def build(self, name, **kwargs):
        if self.params is None:
            raise RuntimeError("CoordinateActorCriticBuilder.load must run first")
        return self.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        """Shared per-hand actor and centralized critic.

        For each recipient hand ``i`` the partner message is

        ``rho([sum(phi(x_j)), number_of_present_partners])`` for ``j != i``.

        This implementation currently runs on DexMachina's bimanual
        environment, but the sum/count implementation itself is independent of
        partner ordering and supports more than two layout entries.
        """

        def __init__(self, params, **kwargs):
            actions_num = int(kwargs.pop("actions_num"))
            input_shape = kwargs.pop("input_shape")
            self.value_size = int(kwargs.pop("value_size", 1))
            self.num_seqs = int(kwargs.pop("num_seqs", 1))
            super().__init__()

            if len(input_shape) != 1:
                raise ValueError(
                    "Coordinate actor requires a flat observation, got "
                    f"input_shape={input_shape}"
                )
            self.obs_dim = int(input_shape[0])
            self._load_config(params, actions_num)
            self._build_modules(params)

        def _load_config(self, params, actions_num):
            if "coordinate" not in params:
                raise ValueError(
                    "coordinate_actor_critic requires network.coordinate config"
                )
            config = params["coordinate"]
            self.hand_obs_slices = tuple(
                _as_slice(bounds) for bounds in config["hand_obs_slices"]
            )
            self.hand_feature_indices = tuple(
                tuple(int(index) for index in indices)
                for indices in config["hand_feature_indices"]
            )
            self.shared_feature_indices = tuple(
                int(index) for index in config["shared_feature_indices"]
            )
            self.previous_action_slices = tuple(
                _as_slice(bounds) for bounds in config["previous_action_slices"]
            )
            self.active_mask_slice = _as_slice(config["active_mask_slice"])
            self.presence_mask_slice = _as_slice(config["presence_mask_slice"])
            self.message_gate_slice = _as_slice(config["message_gate_slice"])

            self.num_hands = len(self.hand_obs_slices)
            if self.num_hands < 2:
                raise ValueError("Coordinate actor requires at least two hands")
            if not (
                len(self.hand_feature_indices)
                == len(self.previous_action_slices)
                == self.num_hands
            ):
                raise ValueError("Coordinate actor layout has inconsistent hand count")

            self.action_dims = tuple(
                bounds.stop - bounds.start for bounds in self.previous_action_slices
            )
            if len(set(self.action_dims)) != 1:
                raise ValueError(
                    "Shared action head requires equal per-hand action dimensions, "
                    f"got {self.action_dims}"
                )
            if sum(self.action_dims) != actions_num:
                raise ValueError(
                    f"Layout action dimensions {self.action_dims} do not sum to "
                    f"RL-Games actions_num={actions_num}"
                )

            hand_obs_dims = tuple(
                bounds.stop - bounds.start for bounds in self.hand_obs_slices
            )
            if len(set(hand_obs_dims)) != 1:
                raise ValueError(
                    "Shared local encoder requires equal hand observation dimensions, "
                    f"got {hand_obs_dims}"
                )
            partner_dims = tuple(
                len(indices) + action_dim + 1
                for indices, action_dim in zip(
                    self.hand_feature_indices, self.action_dims
                )
            )
            if len(set(partner_dims)) != 1:
                raise ValueError(
                    "Shared partner encoder requires equal descriptors, got "
                    f"{partner_dims}"
                )

            self.hand_obs_dim = hand_obs_dims[0]
            self.shared_feature_dim = len(self.shared_feature_indices)
            self.partner_feature_dim = partner_dims[0]
            self.action_dim_per_hand = self.action_dims[0]
            self.per_hand_value = bool(config.get("per_hand_value", False))
            if self.per_hand_value and self.value_size != self.num_hands:
                raise ValueError(
                    "Per-hand PPO requires env_info.value_size equal to the "
                    f"number of hands ({self.num_hands}), got {self.value_size}"
                )
            self.message_enabled = bool(config.get("message_enabled", True))
            # Ablation (IPPO): "local" evaluates one critic per hand on exactly the
            # actor's local input (own hand, shared object/phase, own previous
            # action); "central" keeps the full-state critic.
            self.critic_mode = str(config.get("critic_mode", "central"))
            if self.critic_mode not in ("central", "local"):
                raise ValueError(
                    f"critic_mode must be central or local, got {self.critic_mode!r}"
                )
            if self.critic_mode == "local" and (
                not self.per_hand_value or self.message_enabled
            ):
                raise ValueError(
                    "Local critics require per-hand values and the partner message off"
                )
            self.encoder_dim = int(config.get("encoder_dim", 128))
            self.message_hidden_dim = int(config.get("message_hidden_dim", 64))
            self.message_dim = int(config.get("message_dim", 64))
            self.actor_units = tuple(
                int(value) for value in config.get("actor_units", [256, 128])
            )
            self.critic_units = tuple(
                int(value)
                for value in config.get("critic_units", [512, 512, 256, 128])
            )
            self.activation_name = config.get("activation", "elu")

            space = params.get("space", {}).get("continuous")
            if space is None:
                raise ValueError("Coordinate actor supports continuous actions only")
            self.space_config = space
            self.fixed_sigma = bool(space["fixed_sigma"])
            if not self.fixed_sigma:
                raise ValueError(
                    "Coordinate actor currently requires fixed_sigma=True"
                )
            self.value_activation_name = params.get("value_activation", "None")

        def _activation(self):
            return self.activations_factory.create(self.activation_name)

        def _mlp(self, input_dim: int, units: Iterable[int]):
            modules = []
            previous = int(input_dim)
            for width in units:
                width = int(width)
                modules.extend((nn.Linear(previous, width), self._activation()))
                previous = width
            return nn.Sequential(*modules), previous

        def _build_modules(self, params):
            local_input_dim = (
                self.hand_obs_dim
                + self.shared_feature_dim
                + self.action_dim_per_hand
            )
            self.local_encoder, local_output_dim = self._mlp(
                local_input_dim, (self.encoder_dim,)
            )
            self.partner_encoder, partner_output_dim = self._mlp(
                self.partner_feature_dim, (self.message_hidden_dim,)
            )
            self.message_aggregator, message_output_dim = self._mlp(
                partner_output_dim + 1, (self.message_dim,)
            )
            self.actor_trunk, actor_output_dim = self._mlp(
                local_output_dim + message_output_dim, self.actor_units
            )
            self.action_head = nn.Linear(
                actor_output_dim, self.action_dim_per_hand
            )

            local_critic = self.critic_mode == "local"
            self.critic, critic_output_dim = self._mlp(
                local_input_dim if local_critic else self.obs_dim, self.critic_units
            )
            # A local critic is shared across hands and emits one value per hand call.
            self.value = nn.Linear(
                critic_output_dim, 1 if local_critic else self.value_size
            )
            # With per_hand_value=True, each output is a separate conditioned
            # estimate V_i(full_state). Both heads use the same central state;
            # the output index identifies the hand. The default remains V(s).
            self.value_act = self.activations_factory.create(
                self.value_activation_name
            )

            self.mu_act = self.activations_factory.create(
                self.space_config["mu_activation"]
            )
            self.sigma_act = self.activations_factory.create(
                self.space_config["sigma_activation"]
            )
            self.sigma = nn.Parameter(
                torch.zeros(sum(self.action_dims), dtype=torch.float32)
            )

            initializer = self.init_factory.create(
                **params["mlp"]["initializer"]
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    initializer(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            mu_initializer = self.init_factory.create(
                **self.space_config["mu_init"]
            )
            mu_initializer(self.action_head.weight)
            sigma_initializer = self.init_factory.create(
                **self.space_config["sigma_init"]
            )
            sigma_initializer(self.sigma)
            if params["coordinate"].get("initialization_seed") is not None:
                from dexmachina.rl.ablation_initialization import initialize_named_layers
                initialize_named_layers(
                    self, params["coordinate"]["initialization_seed"],
                    initializer, mu_initializer, sigma_initializer,
                )

        @staticmethod
        def _select(obs, indices):
            if not indices:
                return obs.new_zeros((obs.shape[0], 0))
            return obs[:, indices]

        def _partner_descriptor(self, obs, hand_index, active_mask):
            return torch.cat(
                (
                    self._select(obs, self.hand_feature_indices[hand_index]),
                    obs[:, self.previous_action_slices[hand_index]],
                    active_mask[:, hand_index : hand_index + 1],
                ),
                dim=-1,
            )

        def _partner_message(
            self, obs, recipient_index, active_mask, presence_mask, message_gate
        ):
            encoded_sum = obs.new_zeros(
                (obs.shape[0], self.message_hidden_dim)
            )
            present_count = obs.new_zeros((obs.shape[0], 1))
            for partner_index in range(self.num_hands):
                if partner_index == recipient_index:
                    continue
                present = presence_mask[
                    :, partner_index : partner_index + 1
                ]
                descriptor = self._partner_descriptor(
                    obs, partner_index, active_mask
                )
                encoded_sum = (
                    encoded_sum + self.partner_encoder(descriptor) * present
                )
                present_count = present_count + present
            message = self.message_aggregator(
                torch.cat((encoded_sum, present_count), dim=-1)
            )
            if not self.message_enabled:
                return torch.zeros_like(message)
            return message * message_gate

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            shared = self._select(obs, self.shared_feature_indices)
            active_mask = obs[:, self.active_mask_slice]
            presence_mask = obs[:, self.presence_mask_slice]
            message_gate = obs[:, self.message_gate_slice]

            hand_actions = []
            local_inputs = []
            for hand_index in range(self.num_hands):
                local_input = torch.cat(
                    (
                        obs[:, self.hand_obs_slices[hand_index]],
                        shared,
                        obs[:, self.previous_action_slices[hand_index]],
                    ),
                    dim=-1,
                )
                local_inputs.append(local_input)
                local_latent = self.local_encoder(local_input)
                message = self._partner_message(
                    obs,
                    recipient_index=hand_index,
                    active_mask=active_mask,
                    presence_mask=presence_mask,
                    message_gate=message_gate,
                )
                actor_latent = self.actor_trunk(
                    torch.cat((local_latent, message), dim=-1)
                )
                hand_actions.append(self.action_head(actor_latent))

            mu = self.mu_act(torch.cat(hand_actions, dim=-1))
            logstd = mu * 0.0 + self.sigma_act(self.sigma)
            if self.critic_mode == "local":
                value = torch.cat(
                    [self.value_act(self.value(self.critic(x))) for x in local_inputs],
                    dim=-1,
                )
            else:
                value = self.value_act(self.value(self.critic(obs)))
            return mu, logstd, value, None

        def hand_actor_latent(self, obs, hand_index):
            """Auxiliary access to the same local/message path as the actor.

            This deliberately takes only the supplied auxiliary observation;
            clean actor activations and centralized critic features cannot leak.
            """
            local_input = torch.cat((
                obs[:, self.hand_obs_slices[hand_index]],
                self._select(obs, self.shared_feature_indices),
                obs[:, self.previous_action_slices[hand_index]],
            ), dim=-1)
            local_latent = self.local_encoder(local_input)
            message = self._partner_message(
                obs, recipient_index=hand_index,
                active_mask=obs[:, self.active_mask_slice],
                presence_mask=obs[:, self.presence_mask_slice],
                message_gate=obs[:, self.message_gate_slice],
            )
            return self.actor_trunk(torch.cat((local_latent, message), dim=-1))

        def is_separate_critic(self):
            return True

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def get_value_layer(self):
            return self.value


def register_coordinate_network():
    """Register the opt-in builder with RL-Games' process-global registry."""

    model_builder.register_network(NETWORK_NAME, CoordinateActorCriticBuilder)


def configure_coordinate_network(agent_cfg, env, actor_cfg):
    """Select the factorized builder and inject the environment layout."""

    network = agent_cfg["params"]["network"]
    network["name"] = NETWORK_NAME
    network["separate"] = True
    network["coordinate"] = {
        **env.get_coordinate_actor_layout(),
        "message_enabled": bool(actor_cfg.get("message_enabled", True)),
        "critic_mode": str(actor_cfg.get("critic_mode", "central")),
        "per_hand_value": bool(actor_cfg.get("per_hand_value", False)),
        "encoder_dim": int(actor_cfg.get("encoder_dim", 128)),
        "message_hidden_dim": int(actor_cfg.get("message_hidden_dim", 64)),
        "message_dim": int(actor_cfg.get("message_dim", 64)),
        "actor_units": list(actor_cfg.get("actor_units", [256, 128])),
        "critic_units": list(
            actor_cfg.get("critic_units", [512, 512, 256, 128])
        ),
        "activation": actor_cfg.get("activation", "elu"),
    }
    return agent_cfg


register_coordinate_network()
