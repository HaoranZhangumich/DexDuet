"""CPU tests for RSI adaptive-start assistance."""
import sys
from pathlib import Path
import numpy as np
import torch

RUNTIME = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(RUNTIME))
from dexmachina.envs.curriculum import Curriculum, get_curriculum_cfg  # noqa: E402

T_MAX = 300


class FakeObject:
    device = "cpu"

    def __init__(self, num_envs):
        self.kp = np.full(num_envs, np.nan)
        self.fr = np.full(num_envs, np.nan)
        self.calls = []

    def set_joint_gains(self, kp=None, kv=None, force_range=None, env_idxs=None, zero_velocity=True):
        assert env_idxs is not None, "RSI must always pass explicit env indices"
        n = len(env_idxs)
        for t in (kp, kv, force_range):
            assert tuple(t.shape) == (n, 7), f"gain tensor shape {tuple(t.shape)} for {n} envs"
        assert torch.allclose(kv, kp * (5.0 / 80.0))
        idx = env_idxs.cpu().numpy()
        self.kp[idx] = kp[:, 0].cpu().numpy()
        self.fr[idx] = force_range[:, 0].cpu().numpy()
        self.calls.append(n)


def make(num_envs=2000, **over):
    base = dict(
        kp_init=80.0, kv_init=5.0, force_range_init=50.0, schedule="rsi", fixed_mode="uniform",
        uniform_mode="slow", gain_mode="all", wait_epochs=100, deque_len=40,
        rew_thresholds=dict(task=0.6, con=0.0, imi=0.0, bc=0.0), rsi_step=0.9, rsi_band=2.0)
    base.update(over)
    cfg = get_curriculum_cfg(base)
    obj = FakeObject(num_envs)
    cur = Curriculum(cfg, obj, ["task"], num_envs, 0, T_MAX)
    cur.post_scene_build_setup()
    return cur, obj


def simulate(cur, obj, fail_prob, updates, seed=0, finish_share=1 / 12):
    """fail_prob(bin_index, kp) -> per-episode failure probability."""
    rng = np.random.default_rng(seed)
    n = cur.num_envs
    none = torch.zeros(n, dtype=torch.bool)
    starts = cur.rsi_on_reset(torch.arange(n), none, none)
    trace = []
    for t in range(1, updates + 1):
        done = np.flatnonzero(rng.random(n) < finish_share)
        if len(done):
            bins = cur.rsi_env_bin[torch.as_tensor(done)].numpy()
            failed = rng.random(len(done)) < fail_prob(bins, obj.kp[done])
            ended = torch.ones(len(done), dtype=torch.bool)
            starts = cur.rsi_on_reset(torch.as_tensor(done), ended, torch.as_tensor(failed))
            assert starts.min() >= 0 and starts.max() <= T_MAX - cur.rsi_min_horizon
        zero, moved, _ = cur.set_curriculum(t)
        trace.append((t, float(cur.rsi_top.max()), float(cur.rsi_top.min()), moved, zero))
    return trace


def test_initial_assignment():
    cur, obj = make(num_envs=500)
    # every env draws from the initial band [kp0/band, kp0] = [40, 80]
    assert (obj.kp >= 40.0 - 1e-6).all() and (obj.kp <= 80.0 + 1e-6).all()
    assert not cur.rsi_env_active.any()
    assert np.allclose(obj.fr, 50.0)


def test_starts_cover_the_clip_and_respect_min_horizon():
    cur, _ = make(num_envs=4000)
    none = torch.zeros(4000, dtype=torch.bool)
    starts = cur.rsi_on_reset(torch.arange(4000), none, none)
    assert starts.min() == 0 or starts.min() < 15   # early bin is sampled
    assert starts.max() <= T_MAX - cur.rsi_min_horizon
    counts = torch.bincount(cur.rsi_env_bin, minlength=cur.rsi_bins)
    assert (counts > 0).all(), "uniform prior should populate every bin"
    lo, hi = cur.rsi_bin_lo[cur.rsi_env_bin], cur.rsi_bin_hi[cur.rsi_env_bin]
    assert (starts >= lo).all() and (starts < torch.maximum(hi, lo + 1)).all()


def test_gains_track_the_bin_top_and_band():
    cur, obj = make(num_envs=1000)
    cur.rsi_top[:] = 0.0
    cur.rsi_top[3] = 8.0
    none = torch.zeros(1000, dtype=torch.bool)
    cur.rsi_on_reset(torch.arange(1000), none, none)
    bins = cur.rsi_env_bin.numpy()
    in3 = bins == 3
    assert (obj.kp[~in3] == 0).all() and (obj.fr[~in3] == 0).all()
    if in3.any():
        assert (obj.kp[in3] >= 4.0 - 1e-6).all() and (obj.kp[in3] <= 8.0 + 1e-6).all()
        assert (obj.fr[in3] == 50.0).all()


def test_always_succeeding_policy_zeroes_every_bin_before_deadline():
    cur, obj = make()
    trace = simulate(cur, obj, lambda b, kp: np.zeros(len(b)), updates=7000)
    zero_at = next(t for t, mx, mn, m, z in trace if z)
    assert zero_at < 9000, zero_at
    assert cur.rsi_counts["forced"] == 0 and cur.rsi_counts["up"] == 0
    assert cur.num_epoch_since_zero == 7000 - zero_at + 1


def test_hard_bin_keeps_assistance_longer_and_gets_sampled_more():
    cur, obj = make()
    hard = 8
    trace = simulate(cur, obj, lambda b, kp: np.where(b == hard, 0.9, 0.0), updates=4000, seed=1)
    tops = cur.rsi_top.numpy()
    others = np.delete(np.arange(cur.rsi_bins), hard)
    assert tops[hard] > tops[others].max(), (tops[hard], tops[others].max())
    assert float(cur.rsi_rate[hard]) > 0.5
    probs = cur._rsi_start_probs().numpy()
    uniform = 1.0 / cur.rsi_bins
    assert probs[hard] > 2 * uniform, probs[hard]
    # backward credit: the bin just before the hard one is boosted above uniform too
    assert probs[hard - 1] > uniform
    # bins well before keep at least the uniform mixture floor
    assert probs.min() >= cur.rsi_uniform_mix / cur.rsi_bins - 1e-12


def test_deadline_floor_forces_zero_even_when_failing():
    cur, obj = make(rsi_deadline=400, rsi_zero_ramp=50)
    trace = simulate(cur, obj, lambda b, kp: np.ones(len(b)) * 0.95, updates=600, seed=2)
    assert (cur.rsi_top == 0).all()
    zero_at = next(t for t, mx, mn, m, z in trace if z)
    assert zero_at <= 400 + 50 + 1
    assert cur.rsi_counts["forced"] > 0


def test_raise_is_blocked_above_the_floor():
    cur, obj = make(rsi_deadline=300, rsi_zero_ramp=50, wait_epochs=0)
    simulate(cur, obj, lambda b, kp: np.ones(len(b)) * 0.95, updates=299, seed=3)
    floor = cur._rsi_floor(299)
    # the floor forces a bin only once it is a full step behind, so one step of slack remains
    assert float(cur.rsi_top.max()) <= floor / cur.rsi_step + 1e-9
    assert cur.rsi_counts["up"] == 0, "raises above the floor must be blocked"


def test_current_gains_are_floats():
    cur, obj = make(num_envs=100)
    cur.set_curriculum(1)
    gains = cur.get_current_gains()
    assert all(isinstance(v, float) for v in gains.values()), gains
    assert gains["kp"] == 80.0 and gains["fr"] == 50.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            fails += 1
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    sys.exit(1 if fails else 0)


def test_frame0_share_anchors_starts_at_the_evaluation_start():
    """--rsi_frame0_share: off leaves the draws untouched; on puts that share of starts at frame 0 (bin 0) on top of the
    bins' own sampling."""
    base_cur, _ = make(num_envs=4000)
    off_cur, _ = make(num_envs=4000, rsi_frame0_share=0.0)
    b0, s0 = base_cur._rsi_sample_starts(4000)
    b1, s1 = off_cur._rsi_sample_starts(4000)
    assert torch.equal(b0, b1) and torch.equal(s0, s1)            # default: identical random stream
    cur, _ = make(num_envs=20000, rsi_frame0_share=0.25)
    bins, starts = cur._rsi_sample_starts(20000)
    share = float((starts == 0).double().mean())
    natural = 1.0 / cur.rsi_bins / cur.rsi_bin_hi[0].item()        # frame 0 drawn by bin 0 alone (no keyframes)
    assert abs(share - (0.25 + 0.75 * natural)) < 0.015
    assert bool((bins[starts == 0] == 0).all())
    import pytest
    with pytest.raises(ValueError):
        make(rsi_frame0_share=1.0)
