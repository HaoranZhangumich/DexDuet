"""CPU tests for the task-reward-gated CAPS temporal loss."""
import math, sys
from pathlib import Path
import torch

RUNTIME = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(RUNTIME))
from dexmachina.rl.ablation_ppo import (  # noqa: E402
    caps_gate_weights, caps_temporal_loss, caps_temporal_options, caps_temporal_pairs)


def test_gate_weights_align_with_env_major_pairs():
    H, N = 4, 3
    rew = torch.arange(H * N, dtype=torch.float32).reshape(H, N) / 20.0     # rew[t, i]
    w = caps_gate_weights(rew, H).reshape(N, H)
    for i in range(N):
        assert w[i, 0] == 1.0
        for t in range(1, H):
            assert math.isclose(float(w[i, t]), float(rew[t - 1, i]), rel_tol=1e-6)
    obs = torch.arange(H * N, dtype=torch.float32).reshape(N * H, 1)        # env-major
    prev, valid = caps_temporal_pairs(obs, torch.zeros(N * H), H)
    assert w.reshape(-1).shape == valid.shape
    bad = caps_gate_weights(torch.tensor([[float("nan"), -1.0, 2.0]] * H), H)
    assert torch.all((bad >= 0) & (bad <= 1))


def test_weighted_loss_matches_ungated_at_perfect_tracking_and_ignores_weight_grads():
    torch.manual_seed(0)
    mu = torch.randn(10, 24, requires_grad=True)
    prev = torch.randn(10, 24)
    valid = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    idx = list(range(6)) + list(range(12, 18))
    base = caps_temporal_loss(mu, prev, valid, idx)
    assert torch.allclose(base, caps_temporal_loss(mu, prev, valid, idx, weight=torch.ones(10)))
    w = torch.rand(10, requires_grad=True)
    got = caps_temporal_loss(mu, prev, valid, idx, weight=w)
    per = (mu[:, idx] - prev[:, idx]).square().mean(-1)
    want = (per * valid.float() * w.detach()).sum() / valid.sum()
    assert torch.allclose(got, want)
    got.backward()
    assert w.grad is None and mu.grad is not None


def test_gate_option_validation():
    idx = list(range(6)) + list(range(12, 18))
    assert caps_temporal_options(dict(coefficient=0.1, wrist_action_indices=idx, gate="task_reward"))["gate"] == "task_reward"
    assert caps_temporal_options(dict(coefficient=0.1, wrist_action_indices=idx))["gate"] == "none"
    for bad in (dict(coefficient=0.0, wrist_action_indices=[], gate="task_reward"),
                dict(coefficient=0.1, wrist_action_indices=idx, gate="velocity")):
        try:
            caps_temporal_options(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
