"""Tests for the reference-motion smoothness gate.

Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q dexmachina/tests/test_ref_gate.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root (holds the dexmachina package)
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / 'dexmachina/assets'   # generated assets (docs/SETUP_NEW_MACHINE.md); the real-clip check skips without them
from dexmachina.envs.rewards import RewardModule, get_reward_cfg  # noqa: E402
from dexmachina.rl.ablation_ppo import (caps_gate_weights, caps_ref_motion_weight, caps_temporal_loss,  # noqa: E402
                                        caps_temporal_options)

WRIST = list(range(6)) + list(range(12, 18))
LIN, ROT = 0.00357, 0.035


def test_options():
    base = dict(coefficient=0.1, wrist_action_indices=WRIST)
    assert caps_temporal_options(dict(base, gate='task_reward'), 24) == dict(base, gate='task_reward')
    assert caps_temporal_options(base, 24) == dict(base, gate='none')
    opt = caps_temporal_options(dict(base, gate='task_reward_ref_motion', ref_lin_scale=LIN, ref_rot_scale=ROT), 24)
    assert opt == dict(base, gate='task_reward_ref_motion', ref_lin_scale=LIN, ref_rot_scale=ROT)
    with pytest.raises(ValueError): caps_temporal_options(dict(base, gate='ref_motion'), 24)                      # scales required
    with pytest.raises(ValueError): caps_temporal_options(dict(base, gate='ref_motion', ref_lin_scale=0.0, ref_rot_scale=ROT), 24)
    with pytest.raises(ValueError): caps_temporal_options(dict(base, gate='task_reward', ref_lin_scale=LIN, ref_rot_scale=ROT), 24)
    with pytest.raises(ValueError): caps_temporal_options(dict(base, gate='speed'), 24)


def test_weight_values():
    lin = torch.tensor([0.0, LIN, 0.0, 3 * LIN, float('nan')]); rot = torch.tensor([0.0, 0.0, 3 * ROT, ROT, 0.0])
    w = caps_ref_motion_weight(lin, rot, LIN, ROT)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.1, 0.1, 1.0])


def test_reward_module_reports_demo_motion():
    from scipy.spatial.transform import Rotation as R
    T = 8
    pos = np.cumsum(np.tile([0.002, 0.0, 0.001], (T, 1)), axis=0).astype(np.float32)
    rots = R.from_euler('z', np.arange(T) * 5.0, degrees=True)                      # 5 deg per frame
    quat = np.roll(rots.as_quat(), 1, axis=1).astype(np.float32)                    # wxyz
    demo = dict(obj_pos=pos, obj_quat=quat, obj_arti=np.zeros(T, np.float32))
    cfg = get_reward_cfg(); cfg.update(ref_motion_info=True)
    m = RewardModule(cfg, demo, dict(left={}, right={}), torch.device('cpu'))
    elb = torch.tensor([0, 1, 5, 7, 20])                                            # past the end clamps to the last frame
    lin = m.match_demo_state('obj_step_lin', elb).numpy(); rot = m.match_demo_state('obj_step_rot', elb).numpy()
    assert lin == pytest.approx(np.full(5, np.linalg.norm([0.002, 0.0, 0.001])), rel=1e-4)
    assert np.degrees(rot) == pytest.approx(np.full(5, 5.0), rel=1e-3)
    cfg2 = get_reward_cfg()
    m2 = RewardModule(cfg2, demo, dict(left={}, right={}), torch.device('cpu'))
    assert 'obj_step_lin' not in m2.demo_tensors


def test_weights_through_the_pairing_and_loss():
    horizon, envs = 4, 3
    steps = torch.tensor([[1.0, 0.5, 0.0]] * horizon)                               # (horizon, envs) weights recorded per step
    w = caps_gate_weights(steps, horizon)                                           # env-major, shifted by one
    assert w.reshape(envs, horizon)[:, 1:].tolist() == [[1.0] * 3, [0.5] * 3, [0.0] * 3]
    torch.manual_seed(0)
    mu = torch.randn(envs * horizon, 24); prev = torch.randn(envs * horizon, 24)
    valid = torch.ones(envs * horizon, dtype=torch.bool); valid.reshape(envs, horizon)[:, 0] = False
    full = caps_temporal_loss(mu, prev, valid, WRIST)
    assert caps_temporal_loss(mu, prev, valid, WRIST, weight=torch.ones(envs * horizon)) == pytest.approx(float(full))
    assert float(caps_temporal_loss(mu, prev, valid, WRIST, weight=torch.zeros(envs * horizon))) == 0.0
    gated = caps_temporal_loss(mu, prev, valid, WRIST, weight=w)
    per_pair = (mu[:, WRIST] - prev[:, WRIST]).square().mean(-1).reshape(envs, horizon)
    expected = (per_pair[0, 1:].sum() + 0.5 * per_pair[1, 1:].sum()) / valid.sum()
    assert float(gated) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.skipif(not (ASSETS / 'retargeted/inspire_hand/s01/box_use_01_vector_para.pt').exists(),
                    reason='generated Inspire assets not present')
def test_recorded_demo_gate_separates_tasks():
    """On the real clips the gate keeps the penalty on slow objects and lifts it on Ketchup-100."""
    out = {}
    for name, obj, a, b in (('ketchup-100', 'ketchup', 30, 130), ('box-200', 'box', 30, 230)):
        raw = torch.load(ASSETS / f'retargeted/inspire_hand/s01/{obj}_use_01_vector_para.pt',
                         weights_only=False, map_location='cpu')['demo_data']       # the file stores CUDA tensors
        demo = {k: (v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v))[a:b] for k, v in raw.items() if k.startswith('obj_')}
        demo['obj_arti'] = demo['obj_arti'].reshape(b - a, -1)[:, 0]
        cfg = get_reward_cfg(); cfg.update(ref_motion_info=True)
        m = RewardModule(cfg, demo, dict(left={}, right={}), torch.device('cpu'))
        w = caps_ref_motion_weight(m.demo_tensors['obj_step_lin'], m.demo_tensors['obj_step_rot'], LIN, ROT)
        out[name] = float(w.mean())
    print('\nmean reference-motion weight:', {k: round(v, 2) for k, v in out.items()})
    assert out['ketchup-100'] < 0.4 < out['box-200']
