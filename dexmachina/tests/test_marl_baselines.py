"""CPU checks for the MAPPO/IPPO adapter: observation split, action merge, reward/done fan-out and the rl_games
networks each baseline builds. Run with the patched Genesis and rl_games on PYTHONPATH."""
import copy
import sys
from pathlib import Path

import torch
import yaml

RUNTIME = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(RUNTIME))

from rl_games.algos_torch import model_builder  # noqa: E402

from dexmachina.rl import marl_baselines as mb  # noqa: E402

HAND, SHARED, ACT = 5, 3, 4          # toy sizes; left/right action blocks are interleaved on purpose
OBS = 2 * HAND + SHARED + 2 * ACT + 2


class FakeEnv:
    """Minimal BaseEnv surface used by the adapter, with a layout like get_coordinate_actor_layout()."""
    robot_names = ["left", "right"]
    num_envs, device, obs_dim, num_actions = 3, "cpu", OBS, 2 * ACT
    is_finite_horizon, render_mode = True, None
    per_hand_ppo = False
    action_idxs_to_robot = {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]}

    def __init__(self):
        self.last_action = None

    def get_coordinate_actor_layout(self):
        prev = 2 * HAND + SHARED
        return {"hand_obs_slices": [[0, HAND], [HAND, 2 * HAND]],
                "shared_feature_indices": list(range(2 * HAND, 2 * HAND + SHARED)),
                "previous_action_slices": [[prev, prev + ACT], [prev + ACT, prev + 2 * ACT]]}

    def _obs(self):
        return {"policy": torch.arange(self.num_envs * OBS, dtype=torch.float32).reshape(self.num_envs, OBS) / 100}

    def reset(self):
        return self._obs(), {}

    def step(self, action):
        self.last_action = action.clone()
        rew = torch.tensor([1.0, 2.0, 3.0])
        term = torch.tensor([False, True, False])
        trunc = torch.tensor([False, False, True])
        return self._obs(), rew, term, trunc, {"log": {"x": torch.ones(1), "task_rew": torch.tensor([0.1, 0.2, 0.3])}}


def test_local_observations_and_state():
    env = FakeEnv()
    wrapped = mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="mappo")
    out = wrapped.reset()
    full = env._obs()["policy"]
    assert out["states"].shape == (3, OBS) and torch.equal(out["states"], full)
    local = out["obs"]
    assert local.shape == (6, wrapped.local_dim) == (6, HAND + SHARED + ACT + 2)
    prev = 2 * HAND + SHARED
    for e in range(3):
        for i in range(2):
            row = local[2 * e + i]
            assert torch.equal(row[:HAND], full[e, i * HAND:(i + 1) * HAND])
            assert torch.equal(row[HAND:HAND + SHARED], full[e, 2 * HAND:2 * HAND + SHARED])
            assert torch.equal(row[HAND + SHARED:HAND + SHARED + ACT], full[e, prev + i * ACT:prev + (i + 1) * ACT])
            assert torch.equal(row[-2:], torch.eye(2)[i])
    ippo = mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="ippo").reset()
    assert torch.is_tensor(ippo) and torch.equal(ippo, local)
    assert torch.equal(mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="mappo", with_states=False).reset(), local)


def test_action_merge_and_fan_out():
    env = FakeEnv()
    env.action_idxs_to_robot = {"left": [0, 2, 4, 6], "right": [1, 3, 5, 7]}   # interleaved joint layout
    wrapped = mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="ippo")
    actions = torch.stack([torch.full((ACT,), 0.1 * (r + 1)) for r in range(6)])  # row r = 2e+i
    actions[0, 0] = 7.0                                                            # clipped to 1
    obs, rew, dones, extras = wrapped.step(actions)
    joint = env.last_action
    for e in range(3):
        assert torch.allclose(joint[e, [0, 2, 4, 6]][1:], actions[2 * e, 1:])
        assert torch.allclose(joint[e, [1, 3, 5, 7]], actions[2 * e + 1])
    assert joint[0, 0] == 1.0
    assert torch.equal(rew, torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))
    assert torch.equal(dones, torch.tensor([False, False, True, True, True, True]))
    assert "episode" in extras and "time_outs" not in extras   # finite-horizon env: same as the single-agent wrapper
    # The fork's play_steps adds infos['episode']['task_rew'] to the (N*2, 1) per-agent return tracker.
    assert torch.equal(extras["episode"]["task_rew"], torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3, 0.3]))
    assert torch.equal(extras["episode"]["x"], torch.ones(1))
    info = wrapped.get_env_info()
    assert info["agents"] == 2 and info["value_size"] == 1 and info["state_space"] is None
    assert info["action_space"].shape == (ACT,)


def test_rejects_bad_layouts():
    env = FakeEnv()
    env.action_idxs_to_robot = {"left": [0, 1, 2, 3], "right": [3, 4, 5, 6]}
    try:
        mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="mappo")
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping action blocks must be rejected")
    env = FakeEnv()
    env.per_hand_ppo = True
    try:
        mb.MultiAgentHandsEnv(env, "cpu", 5.0, 1.0, mode="mappo")
    except ValueError:
        pass
    else:
        raise AssertionError("the per-hand learner's reward must be rejected")


def _base_cfg():
    cfg = yaml.safe_load((RUNTIME / "dexmachina/rl/configs/rl_games_ppo_cfg.yaml").read_text())
    cfg["params"]["config"].update(minibatch_size=96000, mini_epochs=14, num_actors=12000)
    return cfg


def _build(net_cfg, input_dim, actions, **extra):
    builder = model_builder.ModelBuilder()
    model = builder.load({"model": {"name": "continuous_a2c_logstd"}, "network": net_cfg})
    return model.build({"actions_num": actions, "input_shape": (input_dim,), "num_seqs": 2,
                        "value_size": 1, "normalize_value": True, "normalize_input": False, **extra})


def test_configs_and_networks():
    local_dim, full_dim, act = 90, 499, 12
    for mode in mb.MODES:
        base = _base_cfg()
        before = copy.deepcopy(base["params"]["config"])
        cfg = mb.configure_marl_agent(base, mode)
        c = cfg["params"]["config"]
        # PPO hyperparameters are untouched apart from the doubled multi-agent minibatch.
        for key in ("learning_rate", "lr_schedule", "kl_threshold", "e_clip", "gamma", "tau", "critic_coef",
                    "horizon_length", "mini_epochs", "normalize_value", "reward_shaper", "bounds_loss_coef",
                    "entropy_coef", "grad_norm", "value_bootstrap", "clip_value"):
            assert c[key] == before[key], key
        assert c["minibatch_size"] == 2 * before["minibatch_size"]
        assert cfg["params"]["algo"]["name"] == "a2c_continuous"
        net = _build(cfg["params"]["network"], local_dim, act).a2c_network
        linear = [m for m in net.actor_mlp if isinstance(m, torch.nn.Linear)]
        assert [m.out_features for m in linear] == [128, 256, 128] and linear[0].in_features == local_dim
        assert net.mu.out_features == act and net.sigma.shape == (act,)
        if mode == "mappo":
            assert c["use_experimental_cv"] is False and not net.separate
            cv = c["central_value_config"]
            assert cv["minibatch_size"] == before["minibatch_size"] and cv["mini_epochs"] == before["mini_epochs"]
            critic = _build(cv["network"], full_dim, act, num_agents=2).a2c_network
            widths = [m.out_features for m in critic.actor_mlp if isinstance(m, torch.nn.Linear)]
            assert widths == [512, 512, 256, 128] and critic.actor_mlp[0].in_features == full_dim
        else:
            assert "central_value_config" not in c and net.separate
            widths = [m.out_features for m in net.critic_mlp if isinstance(m, torch.nn.Linear)]
            assert widths == [512, 512, 256, 128] and net.critic_mlp[0].in_features == local_dim
            assert net.value.in_features == 128
            mu, logstd, value, _ = net({"obs": torch.randn(6, local_dim)})
            assert mu.shape == (6, act) and value.shape == (6, 1)
            # The critic tower is independent of the actor tower.
            value.sum().backward()
            assert all(p.grad is None for p in net.actor_mlp.parameters())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
