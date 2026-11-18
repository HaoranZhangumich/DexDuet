"""Tests for confirmed keyframe capture (envs/keyframes.py, confirm > 0).

Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q dexmachina/tests/test_kf_confirm.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(ROOT))
from dexmachina.envs.keyframes import KeyframeBuffer  # noqa: E402

T, NB = 300, 20
EDGES = torch.linspace(0, T, NB + 1)
BIN_HI = EDGES[1:].floor().long().clamp(max=T - 10).tolist()
BIN_LO = torch.minimum(EDGES[:-1].floor().long(), torch.tensor(BIN_HI) - 1).clamp(min=0).tolist()


class Entity:
    def __init__(self, n, nq, nd):
        self.q = torch.zeros(n, nq); self.v = torch.zeros(n, nd)
    def get_qpos(self, envs_idx=None): return self.q if envs_idx is None else self.q[envs_idx]
    def get_dofs_velocity(self, envs_idx=None): return self.v if envs_idx is None else self.v[envs_idx]


class Robot:
    def __init__(self, n): self.entity = Entity(n, 4, 4); self.curr_targets = torch.zeros(n, 4)


class Obj:
    def __init__(self, n): self.entity = Entity(n, 8, 7)


class Env:
    """Just the fields KeyframeBuffer reads; step() mirrors the order of BaseEnv.step (count, reward, capture, reset)."""
    def __init__(self, n):
        self.device, self.num_envs, self.action_dim, self.max_episode_length = torch.device('cpu'), n, 3, T
        self.robots = dict(left=Robot(n)); self.objects = dict(obj=Obj(n))
        self.episode_length_buf = torch.zeros(n, dtype=torch.int32); self.episode_start_buf = torch.zeros(n, dtype=torch.int32)
        self.reset_buf = torch.zeros(n, dtype=torch.bool); self.reset_time_outs = torch.zeros(n, dtype=torch.bool)
        self.nan_envs = torch.zeros(n, dtype=torch.bool)
        self.cumulative_task_rew = torch.zeros(n); self.actions = torch.zeros(n, 3)

    def step(self, kf, rew, failed=None, clean=None):
        self.episode_length_buf += 1
        self.objects['obj'].entity.q[:] = self.episode_length_buf[:, None].float()     # the state encodes its own frame
        self.cumulative_task_rew += rew
        failed = torch.zeros(self.num_envs, dtype=torch.bool) if failed is None else failed
        self.reset_time_outs[:] = self.episode_length_buf >= T
        self.reset_buf[:] = failed | self.reset_time_outs
        if clean is not None: kf.clean[:] = clean
        kf.capture(rew)
        done = self.reset_buf.clone()
        self.episode_length_buf[done] = 0; self.cumulative_task_rew[done] = 0.0


def make(n, confirm, prob=1.0):
    env = Env(n)
    kf = KeyframeBuffer(env, BIN_LO, BIN_HI, capacity=256, capture_prob=prob, min_fill=32, min_task_rew=0.5, confirm=confirm, seed=0)
    kf.clean[:] = True
    return env, kf


def stored_frames(kf):
    return sorted(int(f) for b in range(NB) for f in kf.frame[b, :int(kf.count[b])].tolist())


def run_trace(trace, confirm):
    """One candidate per frame: lane t may only be picked at frame t + 1, every lane follows the same reward trace."""
    n = len(trace); env, kf = make(n, confirm)
    for t, r in enumerate(trace):
        clean = torch.zeros(n, dtype=torch.bool); clean[t] = True
        env.step(kf, torch.full((n,), float(r)), clean=clean)
    return stored_frames(kf)


def test_without_confirmation_states_are_stored_at_once():
    env, kf = make(4, confirm=0)
    for _ in range(15): env.step(kf, torch.ones(4))
    assert stored_frames(kf) == [15] * 4 and 'kf_dropped_frac' not in kf.stats()


def test_confirmed_state_is_the_picked_one_and_waits_30_frames():
    env, kf = make(1, confirm=30)
    for _ in range(44): env.step(kf, torch.ones(1))
    assert stored_frames(kf) == []                                   # picked at frame 15, not yet confirmed at frame 44
    env.step(kf, torch.ones(1))
    assert stored_frames(kf) == [15]
    b = kf.bin_of(torch.tensor([15]))[0]
    assert float(kf.buf['object_obj/qpos'][b, 0, 0]) == 15.0         # the state from frame 15, not the one from frame 45


def test_lane_that_fails_or_drifts_stores_nothing():
    env, kf = make(2, confirm=30)
    for t in range(60):
        rew = torch.tensor([1.0, 1.0]) if t < 20 else torch.tensor([1.0, 0.2])       # lane 1 loses the object after frame 20
        failed = torch.tensor([t == 30, False])                                      # lane 0 is reset as failed at frame 31
        env.step(kf, rew, failed=failed)
    assert stored_frames(kf) == [] and kf.dropped_total >= 2
    assert kf.stats()['kf_dropped_frac'] == 1.0


def test_lane_that_reaches_the_clip_end_confirms_early():
    env, kf = make(1, confirm=30)
    env.episode_length_buf[:] = T - 28; env.episode_start_buf[:] = T - 28; kf.min_stepped = 15
    for _ in range(28): env.step(kf, torch.ones(1))                  # picked at frame T - 13, the clip ends 13 frames later
    assert stored_frames(kf) == [T - 13]


class Cur:
    """The curriculum fields the to-end option reads: per-bin gain tops and each lane's starting help."""
    def __init__(self, n, help_=1.0): self.rsi_top = torch.full((NB,), 1.0, dtype=torch.float64); self.rsi_env_help = torch.full((n,), help_, dtype=torch.float64)


def make_to_end(n, help_=1.0, prob=1.0):
    env = Env(n); env.curriculum = Cur(n, help_)
    kf = KeyframeBuffer(env, BIN_LO, BIN_HI, capacity=256, capture_prob=prob, min_fill=1, min_task_rew=0.5, to_end=True, seed=0)
    kf.clean[:] = True
    return env, kf


def test_to_end_stores_a_state_only_when_its_lane_finishes_the_clip():
    env, kf = make_to_end(1)
    for _ in range(T - 1): env.step(kf, torch.ones(1))
    assert stored_frames(kf) == []                                   # picked at frame 15, the lane has not finished yet
    env.step(kf, torch.ones(1))                                       # frame T: the clip ends
    assert stored_frames(kf) == [15]
    assert float(kf.buf['help'][kf.bin_of(torch.tensor([15]))[0], 0, 0]) == 1.0


def test_to_end_drops_a_state_whose_lane_fails_after_30_frames():
    """A 30-frame confirmation would store this state (the lane fails 60 frames later); the to-end rule does not."""
    for to_end in (False, True):
        env = Env(1)
        if to_end:
            env.curriculum = Cur(1)
            kf = KeyframeBuffer(env, BIN_LO, BIN_HI, capacity=256, capture_prob=1.0, min_fill=1, min_task_rew=0.5, to_end=True, seed=0)
        else:
            kf = KeyframeBuffer(env, BIN_LO, BIN_HI, capacity=256, capture_prob=1.0, min_fill=1, min_task_rew=0.5, confirm=30, seed=0)
        kf.clean[:] = True
        for t in range(80):
            env.step(kf, torch.ones(1), failed=torch.tensor([t == 74]))   # picked at frame 15, the lane fails at frame 75
        assert stored_frames(kf) == ([] if to_end else [15])


def test_to_end_serves_a_state_only_with_at_least_the_help_it_had():
    env, kf = make_to_end(4, help_=10.0)
    for _ in range(T): env.step(kf, torch.ones(4))
    b = int(kf.bin_of(torch.tensor([15]))[0]); bins = torch.full((4,), b)
    env.curriculum.rsi_top[:] = 5.0                                  # the bin now gives less help than the states needed
    have, *_ = kf.sample(bins)
    assert not bool(have.any())
    env.curriculum.rsi_top[:] = 10.0
    have, frames, _, _ = kf.sample(bins)
    assert bool(have.all()) and set(frames.tolist()) == {15}
