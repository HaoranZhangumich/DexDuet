"""Explicitly selected PPO learner with per-hand options (DexDuet's per-hand PPO).

Ratio grouping (joint or per hand) and value outputs (scalar or one per hand)
are independent options. The v2 protocol keeps its fixed learning rate and exact
epoch-budget loop. The explicit dexmachina protocol uses the pinned learner's
adaptive scheduler and native training loop. The registered legacy algorithms
are never changed.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import wandb
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.common.a2c_common import print_statistics

from dexmachina.rl.per_hand_ppo import per_hand_entropy, per_hand_kl, per_hand_neglogp


ALGORITHM_NAME = "ablation_ppo_v2"
DEXMACHINA_ALGORITHM_NAME = "ablation_ppo_dexmachina"
SCHEMA_VERSION = 2
NUM_HANDS = 2


def caps_temporal_options(config=None, actions_num=None):
    """Validate the opt-in temporal-only CAPS configuration (no spatial noise)."""
    config = dict(config or {})
    unknown = set(config) - {"coefficient", "wrist_action_indices", "gate", "ref_lin_scale", "ref_rot_scale"}
    if unknown:
        raise ValueError(f"Unknown CAPS temporal options: {sorted(unknown)}")
    gate = config.get("gate", "none")
    if gate not in CAPS_GATES:
        raise ValueError(f"CAPS gate must be one of {CAPS_GATES}")
    raw_coefficient = config.get("coefficient", 0.0)
    if isinstance(raw_coefficient, bool):
        raise ValueError("CAPS temporal coefficient must be a finite nonnegative number")
    coefficient = float(raw_coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("CAPS temporal coefficient must be finite and nonnegative")
    indices = list(config.get("wrist_action_indices", []))
    if (any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("CAPS wrist indices must be unique nonnegative integers")
    if actions_num is not None and any(index >= actions_num for index in indices):
        raise ValueError("CAPS wrist index is outside the action vector")
    if coefficient > 0 and len(indices) != NUM_HANDS * 6:
        raise ValueError("CAPS temporal requires six wrist actions for each of two hands")
    if gate != "none" and coefficient == 0:
        raise ValueError("A CAPS gate needs a positive temporal coefficient")
    option = dict(coefficient=coefficient, wrist_action_indices=indices, gate=gate)
    if gate in CAPS_REF_GATES:
        for key in ("ref_lin_scale", "ref_rot_scale"):
            value = config.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"The reference-motion CAPS gate needs a positive finite {key}")
            option[key] = float(value)
    elif "ref_lin_scale" in config or "ref_rot_scale" in config:
        raise ValueError("Reference-motion scales are only valid with a reference-motion CAPS gate")
    return option


CAPS_GATES = ("none", "task_reward", "ref_motion", "task_reward_ref_motion")
CAPS_REF_GATES = ("ref_motion", "task_reward_ref_motion")


def caps_ref_motion_weight(step_lin, step_rot, lin_scale, rot_scale):
    """Smoothness weight from how fast the demo object moves into the current frame.

    1 when the reference is still, 0.5 when its faster component equals the scale, towards 0 when the reference
    moves fast: quick wrist corrections are needed exactly there, so the penalty steps aside.
    """
    speed = torch.maximum(step_lin.float() / lin_scale, step_rot.float() / rot_scale)
    return 1.0 / (1.0 + torch.nan_to_num(speed, nan=0.0).clamp_min(0.0) ** 2)


def caps_gate_weights(step_task_rewards, horizon_length):
    """Per-pair CAPS weights from the task reward recorded at each rollout step.

    step_task_rewards is (horizon, num_envs): the task reward returned after the action at
    each step, i.e. how well the object tracks its reference in the next state. The pair
    (t-1, t) compares the actions chosen in states t-1 and t, so it is weighted by the
    tracking quality of state t, the reward returned after action t-1. Output is flattened
    environment-major, matching caps_temporal_pairs; the first pair of a horizon is invalid.
    """
    if step_task_rewards.ndim != 2 or step_task_rewards.shape[0] != horizon_length:
        raise ValueError("CAPS gate needs one task reward per environment and rollout step")
    gate = torch.nan_to_num(step_task_rewards.float(), nan=0.0).clamp(0.0, 1.0).transpose(0, 1)
    shifted = torch.roll(gate, shifts=1, dims=1)
    shifted[:, 0] = 1.0
    return shifted.reshape(-1)


def caps_temporal_metadata(option):
    """Persist the exact auxiliary objective with each enabled checkpoint."""
    return dict(schema_version=1, **option,
                variant="CAPS_temporal_only", observations="adjacent_same_episode",
                rollout_layout="environment_major", horizon_boundary="excluded",
                policy="current_unclipped_means_both_branches_differentiable",
                reduction="mean_wrist_dimensions_mean_valid_pairs")


def caps_temporal_pairs(obs, dones, horizon_length):
    """Pair env-major rollout observations; stored dones mark arrival at obs[t]."""
    if (obs.ndim != 2 or type(horizon_length) is not int or horizon_length < 2
            or obs.shape[0] % horizon_length or dones.numel() != obs.shape[0]):
        raise ValueError("CAPS pairing requires flat env-major observations and dones")
    sequence = obs.reshape(-1, horizon_length, obs.shape[-1])
    previous = torch.roll(sequence, shifts=1, dims=1)
    # A horizon's first observation has no previous sample in this buffer.
    previous[:, 0] = sequence[:, 0]
    valid = ~dones.reshape(-1, horizon_length).bool()
    valid[:, 0] = False
    return previous.reshape_as(obs), valid.reshape(-1)


def caps_actor_means(network, obs):
    """Actor-only forward: no sampling, critic, normalization updates, or RNG."""
    actions = [network.action_head(network.hand_actor_latent(obs, hand))
               for hand in range(network.num_hands)]
    return network.mu_act(torch.cat(actions, dim=-1))


def caps_temporal_loss(mu, previous_mu, valid, wrist_action_indices, weight=None):
    if mu.shape != previous_mu.shape or mu.ndim != 2 or valid.shape != mu.shape[:1]:
        raise ValueError("CAPS temporal means and validity mask have incompatible shapes")
    difference = mu[:, wrist_action_indices] - previous_mu[:, wrist_action_indices]
    per_pair = difference.square().mean(-1)
    mask = valid.to(dtype=per_pair.dtype)
    scaled = mask
    if weight is not None:
        if weight.shape != per_pair.shape:
            raise ValueError("CAPS gate weights must have one value per action pair")
        # Constant weights in [0, 1]; normalised by the valid count so that perfect
        # tracking (weight 1 everywhere) reproduces the ungated loss exactly.
        scaled = mask * weight.detach().to(dtype=per_pair.dtype).clamp(0.0, 1.0)
    # Even an empty valid set retains a differentiable zero loss.
    return (per_pair * scaled).sum() / mask.sum().clamp_min(1.0)


def _options(ratio_mode, value_mode, credit_mode="none", credit_coefficient=0.0):
    """Validated learner options. credit_mode and credit_coefficient are fixed fields of the
    saved schema (agent.yaml and checkpoints): 'none' and 0."""
    if ratio_mode not in {"joint", "per_hand"}:
        raise ValueError("ratio_mode must be joint or per_hand")
    if value_mode not in {"scalar", "per_hand"}:
        raise ValueError("value_mode must be scalar or per_hand")
    coefficient = float(credit_coefficient)
    if credit_mode != "none" or not math.isfinite(coefficient) or coefficient != 0:
        raise ValueError("per-hand reward credit is not supported; saved configs must have "
                         "credit_mode none and credit_coefficient 0")
    return dict(ratio_mode=ratio_mode, value_mode=value_mode,
                credit_mode=credit_mode, credit_coefficient=coefficient)


def expand_shared_reward(shared_reward, value_mode):
    """The detached scalar team reward, repeated once per value output."""
    if value_mode not in {"scalar", "per_hand"}:
        raise ValueError("value_mode must be scalar or per_hand")
    if shared_reward.ndim == 2 and shared_reward.shape[-1] == 1:
        shared_reward = shared_reward[:, 0]
    if shared_reward.ndim != 1 or not torch.isfinite(shared_reward).all():
        raise ValueError("Expected finite scalar shared rewards [B] or [B,1]")
    columns = NUM_HANDS if value_mode == "per_hand" else 1
    return shared_reward.detach()[:, None].expand(-1, columns).clone()


def actor_advantages(raw_advantages, ratio_mode, normalize=True):
    """Per-hand ratios with a scalar value duplicate the exact normalized joint advantage."""
    if raw_advantages.ndim != 2 or raw_advantages.shape[-1] not in (1, 2):
        raise ValueError("Value advantages must be [B,1] or [B,2]")
    advantages = raw_advantages.detach()
    if ratio_mode == "joint":
        advantages = advantages.mean(-1, keepdim=True)
    elif ratio_mode != "per_hand":
        raise ValueError("Unknown ratio_mode")
    if normalize:
        # Also valid for a single column, with the same sample-std convention.
        advantages = (advantages - advantages.mean(0, keepdim=True)) / (
            advantages.std(0, keepdim=True, unbiased=advantages.shape[0] > 1) + 1e-8
        )
    if ratio_mode == "per_hand" and advantages.shape[-1] == 1:
        advantages = advantages.expand(-1, NUM_HANDS).clone()
    return advantages


def clipped_actor_loss(old_hand_nll, hand_nll, advantages, ratio_mode, clip):
    """Sum hand surrogates to match joint first-order gradient at rollout policy."""
    if ratio_mode == "joint":
        old_nll = old_hand_nll.sum(-1, keepdim=True)
        nll = hand_nll.sum(-1, keepdim=True)
    elif ratio_mode == "per_hand":
        old_nll, nll = old_hand_nll, hand_nll
    else:
        raise ValueError("Unknown ratio_mode")
    if old_nll.shape != advantages.shape:
        raise ValueError("Ratio and advantage axes must match")
    per_group = common_losses.actor_loss(old_nll, nll, advantages, True, clip)
    return per_group.sum(-1).mean(), per_group, (old_nll - nll).exp()


def configure_ablation_ppo(agent_cfg, *, ratio_mode, value_mode):
    option = _options(ratio_mode, value_mode)
    params = agent_cfg["params"]
    network, config = params["network"], params["config"]
    if network.get("name") != "coordinate_actor_critic":
        raise ValueError("Ablation PPO requires the Coordinate actor")
    for key in ("multi_gpu", "normalize_rms_advantage", "normalize_input",
                "use_action_masks", "use_smooth_clamp"):
        if config.get(key, False):
            raise ValueError(f"Ablation PPO does not support {key}=True")
    if config.get("central_value_config") or network.get("rnn"):
        raise ValueError("Ablation PPO requires its feed-forward centralized critic")
    if not config.get("ppo", True):
        raise ValueError("Ablation PPO requires clipped PPO")
    native_protocol = config.get("ablation_protocol") == "dexmachina"
    if native_protocol:
        for key, expected in (("lr_schedule", "adaptive"), ("schedule_type", "standard"),
                              ("learning_rate", 3e-4), ("kl_threshold", 0.008)):
            if config.get(key) != expected:
                raise ValueError(f"DexMachina PPO requires {key}={expected!r}")
    elif config.get("lr_schedule") not in {"constant", "fixed", "none", "identity"}:
        raise ValueError("Ablation PPO requires an explicit fixed learning-rate schedule")
    if int(config.get("max_epochs", -1)) <= 0:
        raise ValueError("Ablation PPO requires a positive exact max_epochs budget")
    if int(config.get("max_frames", -1)) != -1:
        raise ValueError("Ablation PPO uses max_epochs only; max_frames must be -1")
    network["coordinate"]["per_hand_value"] = value_mode == "per_hand"
    params["algo"]["name"] = DEXMACHINA_ALGORITHM_NAME if native_protocol else ALGORITHM_NAME
    config["ablation_ppo"] = dict(
        **option, schema_version=SCHEMA_VERSION,
        actor_reduction="sum_groups_mean_samples", critic_reduction="mean_outputs_samples",
        regularizer_reduction="sum_actions_mean_samples",
        kl="frozen_rollout_to_current_sum_actions_mean_samples",
        stopping="exact_max_epochs", curriculum_index="completed_updates",
    )
    if native_protocol:
        config["ablation_ppo"].update(
            training_protocol="dexmachina",
            kl="pinned_current_to_mutable_reference_sum_actions_mean_samples",
            stopping="native_dexmachina", curriculum_index="native_upcoming_epoch",
        )
    return agent_cfg


def register_ablation_ppo(runner):
    for name in (ALGORITHM_NAME, DEXMACHINA_ALGORITHM_NAME):
        runner.algo_factory.register_builder(name, lambda **kw: AblationPPOAgent(**kw))
        runner.player_factory.register_builder(name, lambda **kw: PpoPlayerContinuous(**kw))


class AblationPPOAgent(A2CAgent):
    def __init__(self, base_name, params):
        self._caps_gate_steps = []
        self.caps_temporal_option = caps_temporal_options(
            params["config"].get("caps_temporal"))
        self.ablation_option = dict(params["config"].get("ablation_ppo", {}))
        self.training_protocol = params["config"].get("ablation_protocol", "v2")
        self.training_stop_reason = "running"
        if self.training_protocol == "dexmachina" and (
            self.ablation_option.get("training_protocol") != "dexmachina"
        ):
            raise ValueError("DexMachina PPO requires matching training-protocol metadata")
        if self.ablation_option.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Explicit ablation PPO schema 2 is required")
        option = _options(**{key: self.ablation_option[key] for key in (
            "ratio_mode", "value_mode", "credit_mode", "credit_coefficient")})
        for key, value in option.items():
            setattr(self, key, value)
        super().__init__(base_name, params)
        expected_values = NUM_HANDS if self.value_mode == "per_hand" else 1
        if self.value_size != expected_values or self.num_agents != 1:
            raise ValueError("Environment value_size does not match the selected value_mode")
        if self.actions_num % NUM_HANDS or self.model.a2c_network.num_hands != NUM_HANDS:
            raise ValueError("Ablation PPO requires two equal hand action blocks")
        if self.is_rnn or self.has_central_value or self.multi_gpu:
            raise ValueError("Ablation PPO requires feed-forward single-GPU operation")
        uenv = self.vec_env.env.unwrapped
        if self.caps_temporal_option["coefficient"] > 0:
            self.caps_temporal_option = caps_temporal_options(
                self.caps_temporal_option, self.actions_num)
            if (self.training_protocol != "dexmachina" or self.ratio_mode != "per_hand"
                    or self.value_mode != "per_hand" or self.credit_mode != "none"):
                raise ValueError("The CAPS temporal loss requires native per-hand PPO/values and no credit")
            expected_indices = []
            for name in uenv.robot_names:
                robot = uenv.robots[name]
                if robot.action_mode != "hybrid":
                    raise ValueError("The CAPS temporal loss requires hybrid wrist residual actions")
                expected_indices.extend(int(uenv.action_idxs_to_robot[name][
                    robot.action_from_idxs[dof]]) for dof in robot.wrist_dof_idxs)
            if self.caps_temporal_option["wrist_action_indices"] != expected_indices:
                raise ValueError("CAPS configured wrist indices differ from the robot action mapping")

    def _caps_gated(self):
        option = self.caps_temporal_option
        return option["coefficient"] > 0 and option.get("gate", "none") != "none"

    def play_steps(self):
        self._caps_gate_steps = []
        return super().play_steps()

    def env_step(self, actions):
        obs, shared, dones, infos = super().env_step(actions)
        if self._caps_gated():
            episode = infos.get("episode") or {}
            gate = self.caps_temporal_option["gate"]
            weight = None
            if gate in ("task_reward", "task_reward_ref_motion"):
                task_rew = episode.get("task_rew")
                if task_rew is None:
                    raise RuntimeError("Gated CAPS needs per-step task rewards in infos['episode']['task_rew']")
                weight = torch.nan_to_num(torch.as_tensor(task_rew, device=shared.device).detach().float().reshape(-1),
                                          nan=0.0).clamp(0.0, 1.0)
            if gate in CAPS_REF_GATES:
                lin, rot = episode.get("ref_obj_step_lin"), episode.get("ref_obj_step_rot")
                if lin is None or rot is None:
                    raise RuntimeError("The reference-motion CAPS gate needs infos['episode']['ref_obj_step_lin'/'ref_obj_step_rot']")
                ref = caps_ref_motion_weight(
                    torch.as_tensor(lin, device=shared.device).detach().reshape(-1),
                    torch.as_tensor(rot, device=shared.device).detach().reshape(-1),
                    self.caps_temporal_option["ref_lin_scale"], self.caps_temporal_option["ref_rot_scale"])
                weight = ref if weight is None else weight * ref
            self._caps_gate_steps.append(weight)
        rewards = expand_shared_reward(shared, self.value_mode)
        return obs, rewards, dones, infos

    def prepare_dataset(self, batch_dict):
        if self.training_protocol == "dexmachina" and (
            self.ratio_mode == "joint" and self.value_mode == "scalar"
        ):
            # Use the actual pinned implementation, including its original
            # scalar log-probability and advantage arithmetic. Extra snapshots
            # are diagnostics only; its mutable mu/sigma still drive scheduling.
            old_mu = batch_dict["mus"].detach().clone()
            old_sigma = batch_dict["sigmas"].detach().clone()
            super().prepare_dataset(batch_dict)
            self.dataset.values_dict.update(ablation_old_mu=old_mu,
                                            ablation_old_sigma=old_sigma)
            return
        values, returns = batch_dict["values"], batch_dict["returns"]
        if values.shape != returns.shape or values.shape[-1] != self.value_size:
            raise ValueError("GAE returns and values must preserve configured output columns")
        advantages = actor_advantages(returns - values, self.ratio_mode,
                                      self.normalize_advantage)
        if self.normalize_value:
            self.value_mean_std.train(not self.config.get("freeze_critic", False))
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        old_mu, old_sigma = batch_dict["mus"].detach().clone(), batch_dict["sigmas"].detach().clone()
        actions = batch_dict["actions"]
        self.dataset.update_values_dict({
            "old_values": values, "returns": returns, "advantages": advantages,
            "actions": actions, "obs": batch_dict["obses"], "dones": batch_dict["dones"],
            "mu": batch_dict["mus"], "sigma": batch_dict["sigmas"],
            "old_logp_actions": per_hand_neglogp(actions, old_mu, old_sigma),
            "ablation_old_mu": old_mu, "ablation_old_sigma": old_sigma,
            "rnn_states": None, "rnn_masks": None,
        })
        if self.training_protocol == "dexmachina":
            self.dataset.values_dict["baseline_old_logp_actions"] = batch_dict["neglogpacs"]
        if self.caps_temporal_option["coefficient"] > 0:
            previous, valid = caps_temporal_pairs(
                batch_dict["obses"], batch_dict["dones"], self.horizon_length)
            self.dataset.values_dict.update(caps_previous_obs=previous, caps_valid=valid)
            if self._caps_gated():
                if len(self._caps_gate_steps) != self.horizon_length:
                    raise RuntimeError("Gated CAPS recorded a different number of steps than the horizon")
                weight = caps_gate_weights(torch.stack(self._caps_gate_steps), self.horizon_length)
                if weight.shape != valid.shape:
                    raise RuntimeError("Gated CAPS weights do not match the rollout layout")
                self.dataset.values_dict["caps_weight"] = weight.to(valid.device)

    def calc_gradients(self, input_dict):
        if self.training_protocol == "dexmachina" and (
            self.ratio_mode == "joint" and self.value_mode == "scalar"
        ):
            # Exact baseline optimizer path (joint ratio, scalar value). Computing additional
            # diagnostics afterward does not modify gradients or scheduler input.
            super().calc_gradients(input_dict)
            self._native_joint_diagnostics(input_dict)
            return
        caps_metrics = {}
        with torch.autocast(device_type=torch.device(self.ppo_device).type,
                            enabled=self.mixed_precision):
            model_input = {"is_train": True, "prev_actions": input_dict["actions"],
                           "obs": self._preproc_obs(input_dict["obs"])}
            result = self.model(model_input)
            mu, sigma = result["mus"].float(), result["sigmas"].float()
            hand_nll = per_hand_neglogp(input_dict["actions"], mu, sigma)
            if self.training_protocol == "dexmachina" and self.ratio_mode == "joint":
                old_nll = input_dict["baseline_old_logp_actions"]
                nll = result["prev_neglogp"]
                losses = self.actor_loss_func(old_nll, nll, input_dict["advantages"][:, 0],
                                               self.ppo, self.e_clip)
                actor, group_actor = losses.mean(), losses[:, None]
                ratios = (old_nll - nll).exp()[:, None]
            else:
                actor, group_actor, ratios = clipped_actor_loss(
                    input_dict["old_logp_actions"], hand_nll, input_dict["advantages"],
                    self.ratio_mode, self.e_clip,
                )
            per_critic = common_losses.default_critic_loss(
                input_dict["old_values"], result["values"], self.e_clip,
                input_dict["returns"], self.clip_value,
            )
            critic = per_critic.mean()
            entropy = per_hand_entropy(sigma).sum(-1).mean()
            if self.bound_loss_type == "regularisation":
                bound_dimensions = mu.square()
            elif self.bound_loss_type == "bound":
                bound_dimensions = (mu.abs() - 1.1).clamp_min(0).square()
            else:
                bound_dimensions = torch.zeros_like(mu)
            bounds = bound_dimensions.sum(-1).mean()
            loss = (actor + 0.5 * self.critic_coef * critic - self.entropy_coef * entropy
                    + float(self.bounds_loss_coef or 0.0) * bounds)
            if self.caps_temporal_option["coefficient"] > 0:
                previous_mu = caps_actor_means(
                    self.model.a2c_network,
                    self._preproc_obs(input_dict["caps_previous_obs"])).float()
                gate_weight = input_dict.get("caps_weight") if self._caps_gated() else None
                if self._caps_gated() and gate_weight is None:
                    raise RuntimeError("Gated CAPS minibatch is missing its weights")
                temporal_loss = caps_temporal_loss(
                    mu, previous_mu, input_dict["caps_valid"],
                    self.caps_temporal_option["wrist_action_indices"], weight=gate_weight)
                temporal_coefficient = self.caps_temporal_option["coefficient"]
                loss = loss + temporal_coefficient * temporal_loss
                caps_metrics = dict(loss=temporal_loss,
                                    weighted_loss=temporal_coefficient * temporal_loss,
                                    coefficient=loss.new_tensor(temporal_coefficient),
                                    valid_fraction=input_dict["caps_valid"].float().mean())
                if gate_weight is not None:
                    valid_mask = input_dict["caps_valid"].bool()
                    caps_metrics["gate_mean"] = (gate_weight[valid_mask].mean() if valid_mask.any()
                                                 else loss.new_tensor(1.0))
            if self.model.get_aux_loss():
                raise ValueError("Ablation auxiliary losses must be explicitly matched")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ablation PPO loss")
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()
        with torch.no_grad():
            hand_kl = per_hand_kl(input_dict["ablation_old_mu"],
                                  input_dict["ablation_old_sigma"], mu, sigma)
            kl = hand_kl.sum(-1).mean()
            self.aux_loss_dict = {}
            self.aux_loss_dict.update({f"caps_temporal/{key}": [value.detach()]
                                      for key, value in caps_metrics.items()})
            if self.training_protocol == "dexmachina":
                self.aux_loss_dict["ablation_ppo/frozen_rollout_forward_kl"] = [kl]
                # Preserve the pinned epsilon formula, direction and mutable
                # reference. train_epoch updates this cache every minibatch and
                # invokes its scheduler once per mini-epoch, unchanged.
                kl = torch_ext.policy_kl(mu.detach(), sigma.detach(), input_dict["mu"],
                                          input_dict["sigma"], True)
            group_names = ("joint",) if self.ratio_mode == "joint" else ("left", "right")
            for index, name in enumerate(group_names):
                for key, value in (("actor", group_actor), ("ratio", ratios),
                                   ("advantage", input_dict["advantages"])):
                    self.aux_loss_dict[f"ablation_ppo/{name}_{key}"] = [value[:, index].mean().detach()]
                self.aux_loss_dict[f"ablation_ppo/{name}_clip_fraction"] = [
                    ((ratios[:, index] - 1).abs() > self.e_clip).float().mean()]
            for index, name in enumerate(("left", "right")):
                self.aux_loss_dict[f"ablation_ppo/{name}_kl"] = [hand_kl[:, index].mean()]
            for index in range(self.value_size):
                self.aux_loss_dict[f"ablation_ppo/value{index}_loss"] = [per_critic[:, index].mean().detach()]
        # Keep diagnostic value and ratio arrays independently shaped: RL-Games
        # diagnostics operate on each array separately (no hand/value coupling).
        old_nll = input_dict["old_logp_actions"]
        if self.ratio_mode == "joint":
            old_nll, hand_nll = old_nll.sum(-1), hand_nll.sum(-1)
        self.diagnostics.mini_batch(self, {
            "values": input_dict["old_values"].reshape(-1),
            "returns": input_dict["returns"].reshape(-1),
            "new_neglogp": hand_nll.reshape(-1), "old_neglogp": old_nll.reshape(-1),
            "masks": None,
        }, self.e_clip, 0)
        self.train_result = (actor.detach(), critic.detach(), entropy.detach(), kl.detach(),
                             self.last_lr, 1.0, mu.detach(), sigma.detach(), bounds.detach())

    @torch.no_grad()
    def _native_joint_diagnostics(self, input_dict):
        actor, critic, _, _, _, _, mu, sigma, _ = self.train_result
        nll = per_hand_neglogp(input_dict["actions"], mu, sigma).sum(-1)
        ratio = (input_dict["old_logp_actions"] - nll).exp()
        hand_kl = per_hand_kl(input_dict["ablation_old_mu"],
                              input_dict["ablation_old_sigma"], mu, sigma)
        self.aux_loss_dict.update({
            "ablation_ppo/joint_actor": [actor.detach()],
            "ablation_ppo/joint_ratio": [ratio.mean()],
            "ablation_ppo/joint_advantage": [input_dict["advantages"].mean()],
            "ablation_ppo/joint_clip_fraction": [((ratio - 1).abs() > self.e_clip).float().mean()],
            "ablation_ppo/value0_loss": [critic.detach()],
            "ablation_ppo/left_kl": [hand_kl[:, 0].mean()],
            "ablation_ppo/right_kl": [hand_kl[:, 1].mean()],
            "ablation_ppo/frozen_rollout_forward_kl": [hand_kl.sum(-1).mean()],
        })

    def _record_native_termination(self):
        """Describe the native return without changing any stopping decision."""
        uenv = self.vec_env.env.unwrapped
        if self.max_epochs != -1 and self.epoch_num >= self.max_epochs:
            reason = "max_epochs"
        elif (getattr(uenv, "use_curriculum", False)
              and uenv.curriculum.num_epoch_since_zero > 1500):
            reason = "native_zero_gain_patience"
        elif self.epoch_num > 2500 and not getattr(uenv, "use_curriculum", False) and (
            self.game_rewards.current_size == 0 or self.game_rewards.get_mean()[0] < 35
        ):
            reason = "native_low_reward"
        elif self.max_frames != -1 and self.frame >= self.max_frames:
            reason = "max_frames"
        elif "score_to_win" in self.config and self.last_mean_rewards > self.config["score_to_win"]:
            reason = "score_to_win"
        else:
            reason = "native_train_return"
        self.training_stop_reason = reason

    def train(self):
        """Select native training or the v2 exact-budget loop."""
        if self.training_protocol == "dexmachina":
            result = super().train()
            self._record_native_termination()
            return result
        if self.max_epochs <= 0 or self.max_frames != -1:
            raise ValueError("Ablation training requires a positive epoch budget only")
        self.init_tensors()
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        total_time = 0.0
        uenv = self.vec_env.env.unwrapped
        while self.epoch_num < self.max_epochs:
            completed_updates = self.epoch_num
            epoch_num = self.update_epoch()
            if hasattr(uenv, "set_curriculum") and uenv.set_curriculum(completed_updates):
                self.clear_stats()
            if hasattr(uenv, "start_recording") and (
                epoch_num == 1 or (self.record_freq > 0 and epoch_num % self.record_freq == 0)
            ):
                uenv.start_recording()
            (step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses,
             entropies, kls, last_lr, lr_mul) = self.train_epoch()
            total_time += sum_time
            self.frame += self.curr_frames
            frame = self.frame // self.num_agents
            self.dataset.update_values_dict(None)
            self.diagnostics.epoch(self, current_epoch=epoch_num)
            stats = self.write_stats(
                total_time, epoch_num, step_time, play_time, update_time, a_losses,
                c_losses, entropies, kls, last_lr, lr_mul, frame, sum_time,
                play_time, self.curr_frames,
            )
            stats.update(step=frame, iter=epoch_num, time=total_time,
                         **{"ablation/completed_updates": epoch_num,
                            "ablation/environment_transitions": self.frame,
                            "ablation/rollout_assistance_update": completed_updates})
            if b_losses:
                stats["losses/bounds_loss"] = torch_ext.mean_list(b_losses).item()
            if self.game_rewards.current_size > 0:
                means = self.game_rewards.get_mean()
                shaped = self.game_shaped_rewards.get_mean()
                self.mean_rewards = means[0]
                for index in range(self.value_size):
                    name = "rewards" if index == 0 else f"rewards{index}"
                    stats[name], stats[f"shaped_{name}"] = means[index], shaped[index]
                stats["episode_lengths"] = self.game_lengths.get_mean()
                if means[0] > self.last_mean_rewards + 1 and epoch_num >= self.save_best_after:
                    self.last_mean_rewards = means[0]
                    self.save(os.path.join(self.nn_dir, self.config["name"]))
            if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, f"ablation_ep_{epoch_num}"))
            if epoch_num == 1 or epoch_num % 10 == 0 or epoch_num == self.max_epochs:
                print_statistics(self.print_stats, self.curr_frames, step_time, play_time,
                                 sum_time, epoch_num, self.max_epochs, frame, -1)
                if hasattr(uenv, "get_recorded_frames"):
                    frames = uenv.get_recorded_frames(wait_for_max=True)
                    if frames is not None and len(frames):
                        stats[f"videos/epoch{epoch_num}"] = wandb.Video(
                            np.asarray(frames).transpose(0, 3, 1, 2), fps=int(1 / uenv.dt / 4))
            if wandb.run is not None:
                wandb.log(stats)
        self.save(os.path.join(self.nn_dir, "ablation_final"))
        return self.last_mean_rewards, self.epoch_num

    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        state["ablation_ppo"] = dict(self.ablation_option, actions_num=self.actions_num)
        if self.caps_temporal_option["coefficient"] > 0:
            state["caps_temporal"] = caps_temporal_metadata(self.caps_temporal_option)
        if self.training_protocol == "dexmachina":
            state["training_termination"] = {"protocol": "dexmachina",
                                              "stop_reason": self.training_stop_reason}
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        expected = (caps_temporal_metadata(self.caps_temporal_option)
                    if self.caps_temporal_option["coefficient"] > 0 else None)
        if weights.get("caps_temporal") != expected:
            raise ValueError("Checkpoint CAPS temporal objective does not match the training config")
        if weights.get("ablation_ppo") != dict(self.ablation_option, actions_num=self.actions_num):
            raise ValueError("Incompatible/missing ablation PPO resume metadata")
        if weights.get("hand_reconstruction") is not None:
            raise ValueError("Checkpoint was trained with hand reconstruction, which this code no longer supports")
        super().set_full_state_weights(weights, set_epoch=set_epoch)
