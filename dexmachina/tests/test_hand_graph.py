"""Tests for the hand-graph reward option (--hand_graph_weight): palm and finger directions in the object frame plus fingertip
gaps, against the human (MANO) hand.

Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q dexmachina/tests/test_hand_graph.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(ROOT))
from dexmachina.envs.rewards import HAND_GRAPH_NODES, RewardModule, get_reward_cfg  # noqa: E402

DEV = torch.device('cpu')
T, K = 5, 18
KPT_NAMES = {s: [f'{p}_{f}_Tip' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + ['base_link'] + [f'{p}_{f}_L1' for f in ('TH', 'FF', 'MF', 'RF', 'LF')]
             + [f'{p}_{f}_L2' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + [f'{p}_TH_L3', f'{p}_TH_L4'] for s, p in (('left', 'L'), ('right', 'R'))}
# a flat human hand in the xy plane (palm normal along z): wrist, finger bases, fingertips
HUMAN = {'wrist': (0, 0, 0), 'TH_L1': (0.03, 0.03, 0), 'FF_L1': (0.08, 0.02, 0), 'MF_L1': (0.085, 0, 0), 'RF_L1': (0.08, -0.02, 0),
         'LF_L1': (0.075, -0.04, 0), 'TH_Tip': (0.06, 0.07, 0), 'FF_Tip': (0.16, 0.025, 0), 'MF_Tip': (0.17, 0, 0),
         'RF_Tip': (0.16, -0.025, 0), 'LF_Tip': (0.14, -0.05, 0)}


def human_joints():
    j = np.zeros((T, 21, 3), np.float32)
    for n, p in HUMAN.items():
        j[:, HAND_GRAPH_NODES[n]] = p
    return j


def robot_kpts(side, rot=np.eye(3), offset=(0, 0, 0), scale=1.2, n=1):
    """Robot keypoints that match the human hand (scaled), optionally rotated about the wrist and moved."""
    k = np.full((n, K, 3), 9.0, np.float32)
    for name, p in HUMAN.items():
        key = 'base_link' if name == 'wrist' else f"{'L' if side == 'left' else 'R'}_{name}"
        k[:, KPT_NAMES[side].index(key)] = rot @ (scale * np.asarray(p, np.float32)) + np.asarray(offset, np.float32)
    return torch.as_tensor(k)


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def quat_z(deg):
    h = math.radians(deg) / 2
    return torch.tensor([[math.cos(h), 0, 0, math.sin(h)]], dtype=torch.float32)


def make_module(weight, with_joints=True, **extra):
    demo = dict(obj_pos=np.zeros((T, 3), np.float32), obj_quat=np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32),
                obj_arti=np.zeros(T, np.float32))
    if with_joints:
        demo.update(hand_joints_left=human_joints(), hand_joints_right=human_joints())
    rng = np.random.default_rng(0)
    ret = {s: dict(kpts_data=dict(kpt_pos=rng.normal(size=(T, K, 3)).astype(np.float32), kpt_names=KPT_NAMES[s]),
                   wrist_pose=np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float32)) for s in ('left', 'right')}
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.3, contact_rew_weight=0.0, bc_rew_weight=0.3, hand_graph_weight=weight, **extra)
    return RewardModule(cfg, demo, ret, DEV)


def score(m, rot=np.eye(3), offset=(0, 0, 0), obj_quat=None):
    q = quat_z(0) if obj_quat is None else obj_quat
    return m.compute_hand_graph_reward(robot_kpts('left', rot, offset), robot_kpts('right', rot, offset), q, torch.zeros(1, dtype=torch.long))


def test_matching_hand_scores_full_weight_anywhere():
    m = make_module(0.5)
    rew, d = score(m)
    assert rew[0] == pytest.approx(0.5, rel=1e-5)
    assert d['hand_graph_dir_err_right'][0] == pytest.approx(0.0, abs=1e-6)
    assert d['hand_graph_gap_err_right'][0] == pytest.approx(0.0, abs=1e-6)
    assert float(score(m, offset=(0.4, -0.2, 1.1))[0][0]) == pytest.approx(0.5, rel=1e-5)   # position does not matter


def test_rotating_the_hand_lowers_the_reward_and_the_palm_angle_shows_it():
    m = make_module(0.5)
    rew, d = score(m, rot=rot_x(45))                                     # tilt the palm by 45 degrees
    assert d['hand_graph_palm_deg_right'][0] == pytest.approx(45.0, abs=0.1)
    assert d['hand_graph_gap_err_right'][0] == pytest.approx(0.0, abs=1e-6)   # gaps are blind to rotation
    assert d['hand_graph_dir_err_right'][0] > 0.05
    assert rew[0] < 0.5 * math.exp(-5 * 0.05)
    assert float(score(m, rot=rot_x(90))[0][0]) < float(rew[0])          # more rotation, lower reward


def test_turning_hand_and_object_together_changes_nothing():
    m = make_module(0.5)
    rew, d = score(m, rot=rot_z(30), obj_quat=quat_z(30))               # the object frame follows the simulated object
    assert rew[0] == pytest.approx(0.5, rel=1e-5)
    assert d['hand_graph_palm_deg_right'][0] == pytest.approx(0.0, abs=0.1)
    assert float(score(m, rot=rot_z(30))[0][0]) < 0.5 * 0.99             # the same hand turn without the object is penalised


def test_off_by_default_and_option_checks():
    m = make_module(0.0, with_joints=False)
    assert 'hand_graph_dirs_left' not in m.demo_tensors and get_reward_cfg()['hand_graph_weight'] == 0.0
    with pytest.raises(ValueError):
        make_module(0.5, with_joints=False)
    with pytest.raises(ValueError):
        make_module(-0.1)
    with pytest.raises(ValueError):
        make_module(0.5, hand_graph_beta_dir=0.0)
