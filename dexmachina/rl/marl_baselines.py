"""Standard MAPPO and IPPO baselines on rl_games' native multi-agent PPO.

MAPPO follows Yu et al. 2022 ("The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"); IPPO follows
de Witt et al. 2020 ("Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?").

Two agents, one per hand (agent 0 = left, agent 1 = right). Each agent acts from its local observation only: its own
hand block, the shared object and phase features, its own previous action and a one-hot agent id. Actor parameters are
shared across the two agents (the usual MAPPO/IPPO choice for agents with identical spaces); there is no
communication. Both agents receive the shared team reward. MAPPO trains one centralized critic V(s) on the full
observation (rl_games ``central_value_config``); IPPO gives each agent a critic on its own local observation.
The PPO update is rl_games' stock ``a2c_continuous`` agent with the same hyperparameters, curriculum, stopping rule
and checkpoint selection as every other run in these studies. Nothing from DexDuet's per-hand learner is used: no
partner message, no per-hand ratio or value heads, no temporal penalty.

rl_games stacks agents env-major: row ``2 * e + i`` is agent ``i`` of environment ``e``.
"""
from __future__ import annotations

from copy import deepcopy

import gym.spaces  # rl_games checks gym (not gymnasium) spaces
import numpy as np
import torch
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.common.ivecenv import IVecEnv
from torch import nn

from dexmachina.rl.training_observer import TrainingObserver

MODES = ("mappo", "ippo")
NUM_AGENTS = 2
SPLIT_NETWORK_NAME = "marl_split_actor_critic"


class SplitWidthA2CBuilder(A2CBuilder):
    """rl_games' ``actor_critic`` with ``separate: True``, except that the critic MLP takes its own ``critic_mlp``
    widths (stock rl_games gives both towers the actor widths). Used by IPPO so its local critic has the same size as
    the MAPPO and per-hand policy critics while the actor keeps the per-hand policy's hand-actor size."""

    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs["input_shape"]
            super().__init__(params, **kwargs)
            units = list(params["critic_mlp"]["units"])
            if not self.separate or self.has_cnn or self.has_rnn:
                raise ValueError("split-width critic needs separate feed-forward towers")
            self.critic_mlp = self._build_mlp(
                input_size=self._calc_input_size(input_shape, self.actor_cnn), units=units,
                activation=self.activation, norm_func_name=self.normalization, dense_func=torch.nn.Linear,
                d2rl=self.is_d2rl, norm_only_first_layer=self.norm_only_first_layer)
            self.value = self._build_value_layer(units[-1], self.value_size)
            mlp_init = self.init_factory.create(**self.initializer)
            for module in list(self.critic_mlp.modules()) + [self.value]:
                if isinstance(module, nn.Linear):   # same initialisation the parent applies to every Linear
                    mlp_init(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def build(self, name, **kwargs):
        return SplitWidthA2CBuilder.Network(self.params, **kwargs)


model_builder.register_network(SPLIT_NETWORK_NAME, SplitWidthA2CBuilder)


class MultiAgentHandsEnv(IVecEnv):
    """Presents the bimanual BaseEnv to rl_games as two hand agents.

    ``with_states=False`` drops the critic state from observations; evaluation uses it because the player only runs
    the actor.
    """

    num_agents = NUM_AGENTS

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float, mode: str, with_states: bool = True):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if list(env.robot_names) != ["left", "right"]:
            raise ValueError(f"expected robots ['left', 'right'], got {env.robot_names}")
        if getattr(env, "per_hand_ppo", False) or getattr(env, "per_hand_value", False):
            raise ValueError("MARL baselines use the scalar team reward; drop --per_hand_ppo and --value_mode per_hand")
        self.env, self.mode, self.with_states = env, mode, with_states
        self._rl_device, self._clip_obs, self._clip_actions = rl_device, clip_obs, clip_actions
        self.obs_dict = None
        layout = env.get_coordinate_actor_layout()
        self.hand_slices = [slice(*b) for b in layout["hand_obs_slices"]]
        self.prev_slices = [slice(*b) for b in layout["previous_action_slices"]]
        self.shared_idx = torch.as_tensor(layout["shared_feature_indices"], dtype=torch.long, device=env.device)
        self.action_idx = [torch.as_tensor(env.action_idxs_to_robot[name], dtype=torch.long, device=env.device)
                           for name in env.robot_names]
        hand_dims = {s.stop - s.start for s in self.hand_slices}
        prev_dims = {s.stop - s.start for s in self.prev_slices}
        act_dims = {len(a) for a in self.action_idx}
        if len(hand_dims) != 1 or len(act_dims) != 1 or prev_dims != act_dims:
            raise ValueError(f"hands must have equal observation/action sizes, got {hand_dims} / {act_dims}")
        self.hand_dim, self.act_dim = hand_dims.pop(), act_dims.pop()
        self.joint_act_dim = int(env.num_actions)
        covered = torch.cat(self.action_idx).sort().values
        if not torch.equal(covered, torch.arange(self.joint_act_dim, device=env.device)):
            raise ValueError("per-hand action indices must partition the joint action")
        self.state_dim = int(env.obs_dim)
        self.local_dim = self.hand_dim + len(self.shared_idx) + self.act_dim + NUM_AGENTS
        self._agent_id = torch.eye(NUM_AGENTS, device=env.device)

    # ---------------------------------------------------------------- rl_games interface
    @property
    def unwrapped(self):
        return self.env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def observation_space(self):
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, shape=(self.local_dim,))

    @property
    def action_space(self):
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, shape=(self.act_dim,))

    @property
    def state_space(self):
        if self.mode != "mappo":
            return None
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, shape=(self.state_dim,))

    @property
    def value_size(self):
        return 1

    def get_obs_dict(self):
        return self.obs_dict

    def get_number_of_agents(self) -> int:
        return NUM_AGENTS

    def get_env_info(self) -> dict:
        return {"observation_space": self.observation_space, "action_space": self.action_space,
                "state_space": self.state_space, "agents": NUM_AGENTS, "value_size": 1}

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._process(obs_dict)

    def step(self, actions):
        actions = torch.clamp(actions.detach().to(self.env.device), -self._clip_actions, self._clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(self.joint_actions(actions))
        if rew.dim() != 1:
            raise RuntimeError("MARL baselines need the scalar team reward")
        if not self.env.is_finite_horizon:
            extras["time_outs"] = self.per_agent(truncated).to(self._rl_device)
        dones = self.per_agent(terminated | truncated).to(self._rl_device)
        rew = self.per_agent(rew).to(self._rl_device)
        extras = {k: v.to(self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()}
        if "log" in extras:
            extras["episode"] = extras.pop("log")
            if "task_rew" in extras["episode"]:
                # The rl_games fork scores episodes (and so selects the best checkpoint) on this per-env task reward;
                # give each agent its env's value, like the team reward. Other logged means are unaffected.
                extras["episode"] = dict(extras["episode"], task_rew=self.per_agent(extras["episode"]["task_rew"]))
        return self._process(obs_dict), rew, dones, extras

    def close(self):
        return self.env.close()

    # ---------------------------------------------------------------- per-agent views
    @staticmethod
    def per_agent(x):
        return x.repeat_interleave(NUM_AGENTS, dim=0)

    def joint_actions(self, actions):
        """(N*2, act_dim) env-major agent actions -> (N, joint_act_dim) simulator action."""
        per_agent = actions.reshape(-1, NUM_AGENTS, self.act_dim)
        joint = actions.new_empty((per_agent.shape[0], self.joint_act_dim))
        for i, idx in enumerate(self.action_idx):
            joint[:, idx] = per_agent[:, i]
        return joint

    def local_observations(self, obs):
        """(N, obs_dim) full observation -> (N*2, local_dim) env-major local observations."""
        parts = [torch.cat((obs[:, self.hand_slices[i]], obs[:, self.shared_idx], obs[:, self.prev_slices[i]],
                            self._agent_id[i].expand(obs.shape[0], -1)), dim=-1) for i in range(NUM_AGENTS)]
        return torch.stack(parts, dim=1).reshape(obs.shape[0] * NUM_AGENTS, self.local_dim)

    def _process(self, obs_dict):
        self.obs_dict = obs_dict
        obs = torch.clamp(obs_dict["policy"], -self._clip_obs, self._clip_obs)
        local = self.local_observations(obs).to(self._rl_device).clone()
        if self.mode == "mappo" and self.with_states:
            return {"obs": local, "states": obs.to(self._rl_device).clone()}
        return local


def configure_marl_agent(agent_cfg: dict, mode: str, actor_units=(128, 256, 128),
                         critic_units=(512, 512, 256, 128)) -> dict:
    """Turn the shared base PPO config into stock rl_games MAPPO or IPPO.

    PPO hyperparameters (learning rate and schedule, clip, GAE, epochs, value normalisation, reward scale) are left
    as the caller set them. ``actor_units`` matches the depth and width of one hand actor of the per-hand policy without
    its message branch (local encoder 128, trunk 256-128); ``critic_units`` matches the per-hand policy's critic.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    p = agent_cfg["params"]
    p["algo"] = {"name": "a2c_continuous"}
    p["model"] = {"name": "continuous_a2c_logstd"}
    net = p["network"]
    net["name"] = "actor_critic"
    net.pop("coordinate", None)
    activation = net["mlp"].get("activation", "elu")
    cfg = p["config"]
    cfg.pop("central_value_config", None)
    # The multi-agent batch holds num_envs * 2 rows; doubling the minibatch keeps the number of gradient steps per
    # epoch equal to the single-agent runs.
    single_minibatch = int(cfg["minibatch_size"])
    cfg["minibatch_size"] = single_minibatch * NUM_AGENTS
    if mode == "ippo":
        # Separate actor and critic MLPs on each agent's local observation; the critic keeps the size of the per-hand
        # policy's critic.
        net["name"] = SPLIT_NETWORK_NAME
        net["separate"] = True
        net["mlp"]["units"] = list(actor_units)
        net["critic_mlp"] = {"units": list(critic_units)}
    else:
        # Policy-only actor network: the value head inside the actor model is not trained (use_experimental_cv off),
        # the team value comes from the centralized critic on the full observation.
        net["separate"] = False
        net["mlp"]["units"] = list(actor_units)
        net.pop("critic_mlp", None)
        cfg["use_experimental_cv"] = False
        cfg["central_value_config"] = {
            "minibatch_size": single_minibatch, "mini_epochs": int(cfg["mini_epochs"]),
            "learning_rate": float(cfg["learning_rate"]), "lr_schedule": "identity",
            "clip_value": bool(cfg.get("clip_value", True)), "normalize_input": bool(cfg.get("normalize_input", False)),
            "truncate_grads": bool(cfg.get("truncate_grads", True)), "grad_norm": float(cfg.get("grad_norm", 1.0)),
            "e_clip": float(cfg.get("e_clip", 0.2)),
            "network": {"name": "actor_critic", "central_value": True,
                        "mlp": {"units": list(critic_units), "activation": activation, "d2rl": False,
                                "initializer": deepcopy(net["mlp"].get("initializer", {"name": "default"})),
                                "regularizer": {"name": "None"}}},
        }
    cfg["marl_baseline"] = {"mode": mode, "agents": NUM_AGENTS, "actor_units": list(actor_units),
                            "critic_units": list(critic_units),
                            "critic_input": "full_observation" if mode == "mappo" else "local_observation",
                            "parameter_sharing": True, "agent_id": "one_hot", "communication": False,
                            "reward": "shared_team_reward", "single_agent_minibatch": single_minibatch}
    return agent_cfg


def native_stop_reason(algo) -> str:
    """Describe why the stock rl_games loop returned (same rules as the per-hand policy's agent records)."""
    uenv = algo.vec_env.env.unwrapped
    if algo.max_epochs != -1 and algo.epoch_num >= algo.max_epochs:
        return "max_epochs"
    if getattr(uenv, "use_curriculum", False) and uenv.curriculum.num_epoch_since_zero > 1500:
        return "native_zero_gain_patience"
    if algo.epoch_num > 2500 and not getattr(uenv, "use_curriculum", False) and (
        algo.game_rewards.current_size == 0 or algo.game_rewards.get_mean()[0] < 35
    ):
        return "native_low_reward"
    if algo.max_frames != -1 and algo.frame >= algo.max_frames:
        return "max_frames"
    if "score_to_win" in algo.config and algo.last_mean_rewards > algo.config["score_to_win"]:
        return "score_to_win"
    return "native_train_return"


class MarlAlgoObserver(TrainingObserver):
    """Same W&B diagnostics as the other studies, plus the centralized critic loss that rl_games does not log."""

    def after_init(self, algo):
        super().after_init(algo)
        if algo.num_agents != NUM_AGENTS:
            raise RuntimeError(f"expected {NUM_AGENTS} rl_games agents, got {algo.num_agents}")
        self._cv_loss = None
        if algo.has_central_value:
            train_net = algo.central_value_net.train_net

            def logged_train_net():
                self._cv_loss = float(train_net())
                return self._cv_loss

            algo.central_value_net.train_net = logged_train_net

    def wandb_after_print_stats(self, frame, epoch_num, total_time):
        metrics = super().wandb_after_print_stats(frame, epoch_num, total_time)
        if self._cv_loss is not None:
            metrics["marl/central_value_loss"] = self._cv_loss
        return metrics
