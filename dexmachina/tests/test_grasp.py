"""Tests for the contact-graph grasp gate (--cg_gate).

Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q dexmachina/tests/test_grasp.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT   # repository root (holds the dexmachina package)
sys.path.insert(0, str(RUNTIME))
from dexmachina.envs.rewards import RewardModule, get_reward_cfg  # noqa: E402

DEV = torch.device('cpu')
T, K, L = 12, 18, 13
KPT_NAMES = {s: [f'{p}_{f}_Tip' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + ['base_link'] + [f'{p}_{f}_L1' for f in ('TH', 'FF', 'MF', 'RF', 'LF')]
             + [f'{p}_{f}_L2' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + [f'{p}_TH_L3', f'{p}_TH_L4'] for s, p in (('left', 'L'), ('right', 'R'))}


def make_module(cg=1.0, contacts=None, kpts=None, obj_pos=None, obj_quat=None):
    rng = np.random.default_rng(0)
    demo = dict(obj_pos=obj_pos if obj_pos is not None else rng.normal(size=(T, 3)).astype(np.float32),
                obj_quat=obj_quat if obj_quat is not None else np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32),
                obj_arti=np.zeros(T, np.float32))
    for s in ('left', 'right'):
        c = np.zeros((T, 2, L, 4), np.float32)
        if contacts is not None: c[:] = contacts[s]
        demo[f'contact_links_{s}'] = c
    ret = {s: dict(kpts_data=dict(kpt_pos=(kpts[s] if kpts is not None else rng.normal(size=(T, K, 3)).astype(np.float32)), kpt_names=KPT_NAMES[s]),
                   wrist_pose=np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float32)) for s in ('left', 'right')}
    cfg = get_reward_cfg(); cfg.update(imi_rew_weight=0.3, contact_rew_weight=3.0, bc_rew_weight=0.3, use_retarget_contact=True,
                                        cg_gate=cg, cg_extra=0.25)
    return RewardModule(cfg, demo, ret, DEV)


def test_contact_graph_values():
    # demo (top, bottom) order on disk; the module flips it to the sim order (bottom, top)
    contacts = {s: np.zeros((T, 2, L, 4), np.float32) for s in ('left', 'right')}
    contacts['right'][:, 1, :4, 3] = 2.0        # demo: right hand, 4 links on the bottom part (demo row 1 = bottom)
    m = make_module(contacts=contacts)
    elb = torch.zeros(4, dtype=torch.long)
    zeros = torch.zeros(4, 2, L, dtype=torch.bool)
    sim_r = zeros.clone()
    sim_r[0, 0, :4] = True                      # lane 0: exact match (sim row 0 = bottom)
    sim_r[1, 0, :2] = True                      # lane 1: two of four missing
    sim_r[2, 0, :4] = True; sim_r[2, 0, 4:8] = True   # lane 2: match plus four extra links
    sim_r[3, 1, :4] = True                      # lane 3: right links, wrong part -> all missing, all extra
    gate, out = m.compute_grasp_gate(zeros, sim_r, elb)
    cg = out['cg_rew'].tolist()
    assert cg[0] == pytest.approx(1.0)
    assert cg[1] == pytest.approx(0.5)
    assert cg[2] == pytest.approx(1.0 - 0.25 * 4 / 4)
    assert cg[3] == pytest.approx(0.0)
    assert torch.allclose(gate, out['cg_rew'])
    # frames where the demo has no contact cost nothing unless the policy touches
    elb2 = torch.full((2,), 5, dtype=torch.long)
    contacts0 = {s: np.zeros((T, 2, L, 4), np.float32) for s in ('left', 'right')}
    m0 = make_module(contacts=contacts0)
    sim = torch.zeros(2, 2, L, dtype=torch.bool); sim[1, 0, 0] = True
    _, out0 = m0.compute_grasp_gate(sim, torch.zeros(2, 2, L, dtype=torch.bool), elb2)
    assert out0['cg_rew'].tolist() == pytest.approx([1.0, 0.75])
