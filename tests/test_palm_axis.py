"""The calibrated palm axis of hands with fewer than four fingers (finger_spec.palm_normal_mode 'calibrated_axis'; Dex3).

Their robot palm normal is R_wrist(t) a, a fixed axis a of the wrist link, against the human index x little-base normal n(t);
a = normalize(sum_t R_wrist(t)^T n(t)) over the clip's retargeted reference (rewards.calibrate_palm_axis), computed once when
the reward module is built and stored in reward_cfg['hand_graph_palm_axis'], which the trainer saves in params/env.pkl.
  - the calibration recovers a known axis (synthetic random wrist rotations, n = R a*), in the module too;
  - Dex3's reward on its own retargeted Box-200 reference gives the reference check's palm angles (8 / 15 degrees median,
    left / right; 65 / 46 with the finger rule);
  - the rule per hand: Dex3 'calibrated_axis', every hand with four or more fingers 'fingers' (no axis, reward_cfg untouched);
  - the axis is written into reward_cfg, survives the env.pkl round trip, and evaluation uses the stored axis.
Inspire stays bit-identical to the Inspire-only reward code: tests/test_inspire_golden.py.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_palm_axis.py
"""
import copy
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'golden'))
sys.path.insert(0, str(ROOT / 'scripts'))
from dexmachina.asset_utils import get_asset_path  # noqa: E402
from dexmachina.envs import finger_spec as fs  # noqa: E402
from dexmachina.envs.math_utils import matrix_from_quat  # noqa: E402
from dexmachina.envs.rewards import RewardModule, calibrate_palm_axis, get_reward_cfg  # noqa: E402

DEV = torch.device('cpu')
HANDS = ('inspire_hand', 'xhand', 'schunk_hand', 'allegro_hand', 'dex3_hand', 'ability_hand', 'dexrobot_hand')
SIDES = ('left', 'right')
# a flat human hand in the xy plane (MANO joints): wrist, finger bases (13/1/4/10/7), fingertips (16-20); its palm normal,
# (index base - wrist) x (little base - wrist), is -z
HUMAN = {0: (0, 0, 0), 13: (0.03, 0.03, 0), 1: (0.08, 0.02, 0), 4: (0.085, 0, 0), 10: (0.08, -0.02, 0), 7: (0.075, -0.04, 0),
         16: (0.06, 0.07, 0), 17: (0.16, 0.025, 0), 18: (0.17, 0, 0), 19: (0.16, -0.025, 0), 20: (0.14, -0.05, 0)}


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


def angle_deg(u, v):
    return torch.rad2deg(torch.acos((unit(u.double()) * unit(v.double())).sum(-1).clamp(-1.0, 1.0)))


def random_quats(n, seed):
    q = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def qmul(a, b):
    """Hamilton product of wxyz quaternions: matrix_from_quat(qmul(a, b)) = matrix_from_quat(a) @ matrix_from_quat(b)."""
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=-1)


def axis_angle_quat(axis, deg):
    h = math.radians(deg) / 2
    return torch.cat([torch.tensor([math.cos(h)], dtype=torch.float64), math.sin(h) * unit(axis.double())])[None]


# ------------------------------------------------------------------ (1) the calibration recovers a known axis
def test_calibration_recovers_a_known_axis():
    q = random_quats(300, 0)                                         # wrist-link orientations, wxyz
    a_true = unit(torch.tensor([0.3, -0.8, 0.5], dtype=torch.float64))
    n = torch.einsum('tij,j->ti', matrix_from_quat(q), a_true)      # human normal fixed in the wrist frame: n = R a*
    a, length = calibrate_palm_axis(q, n)
    assert a.dtype == torch.float64 and torch.allclose(a, a_true, atol=1e-12) and length == pytest.approx(1.0, abs=1e-12)
    # float32 inputs (as the module passes the human normal) and unnormalised quaternions
    a32, _ = calibrate_palm_axis((3.0 * q).float(), n.float())
    assert angle_deg(a32, a_true) < 1e-3
    # the quaternion order matters: read as xyzw the same data has no fixed axis
    assert calibrate_palm_axis(q[:, [1, 2, 3, 0]], n)[1] < 0.5
    # noisy normals: the axis is the maximiser of the summed cosine, close to a*
    noisy = unit(n + 0.15 * torch.randn(n.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
    a_noisy, length_noisy = calibrate_palm_axis(q, noisy)
    assert angle_deg(a_noisy, a_true) < 1.5 and 0.9 < length_noisy < 1.0
    local = torch.einsum('tji,tj->ti', matrix_from_quat(q), noisy)                 # R^T n
    best = (local @ a_noisy).sum()
    for b in unit(a_noisy + 0.05 * torch.randn(20, 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)):
        assert (local @ b).sum() < best
    assert torch.equal(calibrate_palm_axis(q, noisy)[0], a_noisy)                  # deterministic
    assert torch.allclose(calibrate_palm_axis(q, -n)[0], -a_true, atol=1e-12)      # a flipped human normal flips the axis
    with pytest.raises(ValueError, match='cancel'):
        calibrate_palm_axis(torch.tensor([[1.0, 0, 0, 0]] * 2), torch.tensor([[0, 0, 1.0], [0, 0, -1.0]]))
    with pytest.raises(ValueError, match=r'\(T, 4\)'):
        calibrate_palm_axis(q[:, :3], n)


def synthetic_reference(hand, frames=40, seed=2, a_seed=3):
    """A human hand and a robot hand that follows it, frame t turned by H_t = R_t A: R_t random wrist-link orientations, A a
    fixed rotation between the wrist-link frame and the flat human hand. The human normal is then H_t (-z) = R_t a* with
    a* = A (-z). Robot nodes are the scaled human nodes, so a perfect follower matches every hand-graph term."""
    q = random_quats(frames, seed)
    A = matrix_from_quat(random_quats(1, a_seed))[0]
    H = matrix_from_quat(q) @ A                                                     # (T, 3, 3)
    joints = torch.zeros(frames, 21, 3, dtype=torch.float64)
    for j, p in HUMAN.items():
        joints[:, j] = H @ torch.tensor(p, dtype=torch.float64)
    specs = {s: fs.load_finger_spec(hand, s) for s in SIDES}
    demo = dict(obj_pos=np.zeros((frames, 3), np.float32), obj_quat=np.tile([1, 0, 0, 0], (frames, 1)).astype(np.float32),
                obj_arti=np.zeros(frames, np.float32))
    ret, robot = dict(), dict()
    for s in SIDES:
        spec = specs[s]
        demo[f'hand_joints_{s}'] = joints.float().numpy()
        nodes = {spec.wrist: 0, **dict(zip(spec.bases, spec.human_bases)), **dict(zip(spec.tips, spec.human_tips))}
        names = ['filler'] + list(nodes)
        k = torch.full((frames, len(names), 3), 9.0, dtype=torch.float64)
        for i, name in enumerate(names[1:], start=1):
            k[:, i] = spec.scale * joints[:, nodes[name]]
        pose = torch.cat([torch.zeros(frames, 3, dtype=torch.float64), q], dim=1)
        ret[s] = dict(kpts_data=dict(kpt_pos=k.float().numpy(), kpt_names=names), wrist_pose=pose.float().numpy())
        robot[s] = dict(kpts=k.float(), wrist_pose=pose.float())
    a_true = A @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    return demo, ret, robot, a_true


def module(hand, demo, ret, **cfg_update):
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.0, contact_rew_weight=0.0, bc_rew_weight=0.0, hand_graph_weight=1.0, hand_name=hand, **cfg_update)
    return RewardModule(cfg, demo, ret, DEV), cfg


def score(m, robot, frames):
    t = torch.arange(frames)
    q = torch.tensor([[1.0, 0, 0, 0]]).repeat(frames, 1)
    return m.compute_hand_graph_reward(robot['left']['kpts'], robot['right']['kpts'], q, t, None, None,
                                       robot['left']['wrist_pose'], robot['right']['wrist_pose'])


def test_module_calibrates_the_axis_and_a_perfect_follower_scores_full_weight():
    demo, ret, robot, a_true = synthetic_reference('dex3_hand')
    m, cfg = module('dex3_hand', demo, ret)
    for s in SIDES:
        axis = m.hand_graph_palm_axis[s]
        assert axis.dtype == torch.float32 and angle_deg(axis, a_true) < 1e-3
        assert cfg['hand_graph_palm_axis'][s] == axis.tolist()                    # written into reward_cfg
    rew, out = score(m, robot, 40)
    assert torch.allclose(rew, torch.ones(40), atol=1e-5)
    for s in SIDES:
        assert out[f'hand_graph_palm_deg_{s}'].max() < 0.1
    # the robot normal is the wrist link's: moving the robot's outermost (middle) base changes the forward, not the normal
    moved = {s: dict(robot[s]) for s in SIDES}
    for s in SIDES:
        k = robot[s]['kpts'].clone()
        k[:, int(m.hand_graph_index[s]['palm'][3])] += torch.tensor([0.0, 0.0, 0.02])
        moved[s]['kpts'] = k
    frame = m.hand_graph_palm_frame('left', moved['left']['kpts'], moved['left']['wrist_pose'])
    same = m.hand_graph_palm_frame('left', robot['left']['kpts'], robot['left']['wrist_pose'])
    assert torch.equal(frame[:, 0], same[:, 0]) and not torch.allclose(frame[:, 1], same[:, 1])
    # turning the wrist link 30 degrees about an axis perpendicular to a turns the normal by 30 degrees
    b = unit(torch.linalg.cross(m.hand_graph_palm_axis['left'].double(), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)))
    turned = torch.as_tensor(robot['left']['wrist_pose'], dtype=torch.float64).clone()
    turned[:, 3:] = qmul(turned[:, 3:], axis_angle_quat(b, 30.0).expand(40, 4))
    frame = m.hand_graph_palm_frame('left', robot['left']['kpts'], turned.float())
    assert torch.allclose(angle_deg(frame[:, 0], same[:, 0]), torch.full((40,), 30.0, dtype=torch.float64), atol=1e-3)
    with pytest.raises(ValueError, match='wrist pose'):
        m.compute_hand_graph_reward(robot['left']['kpts'], robot['right']['kpts'], torch.tensor([[1.0, 0, 0, 0]]).repeat(40, 1),
                                    torch.arange(40))


# ------------------------------------------------------------------ (2) Dex3 on its own retargeted reference
def have_clip(hand):
    return (Path(get_asset_path(f'retargeted/{hand}/s01/box_use_01_vector_para.pt')).exists()
            and Path(get_asset_path('arctic/processed/s01/box_use_01.npy')).exists())


@pytest.mark.skipif(not have_clip('dex3_hand'), reason='no retargeted Dex3 Box-200 clip')
def test_dex3_reward_on_its_reference_gives_the_reference_check_angles():
    import make_inspire_golden as mg
    import reference_check
    mg.cpu_torch_load()
    m, demo, ret = reference_check.load('dex3_hand')
    assert all(m.hand_graph_palm_axis[s] is not None for s in SIDES) and m.hand_graph_extra == {'left': [], 'right': []}
    frames = len(demo['obj_quat'])
    kpts = {s: torch.as_tensor(np.asarray(ret[s]['kpts_data']['kpt_pos']), dtype=torch.float32) for s in SIDES}
    wrist = {s: torch.as_tensor(np.asarray(ret[s]['wrist_pose']), dtype=torch.float32) for s in SIDES}
    q = torch.as_tensor(np.asarray(demo['obj_quat']), dtype=torch.float32)
    _, out = m.compute_hand_graph_reward(kpts['left'], kpts['right'], q, torch.arange(frames), None, None,
                                         wrist['left'], wrist['right'])
    medians = dict()
    for s in SIDES:
        ref = reference_check.reference_angles(m, demo, ret, s)['palm']
        got = out[f'hand_graph_palm_deg_{s}']
        assert got.shape == ref.shape == (frames,)
        assert torch.allclose(torch.cos(torch.deg2rad(got)), torch.cos(torch.deg2rad(ref)), atol=1e-6)
        assert abs(got.median().item() - ref.median().item()) < 0.01
        medians[s] = got.median().item()
    # 7.9 / 15.2 degrees median (Inspire's own reference: 12 / 21; the finger rule on Dex3: 65 / 46)
    assert medians['left'] < 12.0 and medians['right'] < 20.0, medians


# ------------------------------------------------------------------ (3) which hands use the calibrated axis
def test_the_threshold_is_four_fingers():
    assert fs.PALM_NORMAL_MIN_FINGERS == 4
    assert [fs.palm_normal_mode(n) for n in (3, 4, 5)] == ['calibrated_axis', 'fingers', 'fingers']


@pytest.mark.parametrize('hand', HANDS)
def test_palm_normal_mode_per_hand(hand):
    want = 'calibrated_axis' if hand == 'dex3_hand' else 'fingers'
    for s in SIDES:
        spec = fs.load_finger_spec(hand, s)
        assert spec.palm_normal_mode == want, (hand, s, spec.fingers)
        if want == 'fingers':
            assert len(spec.fingers) >= 4 and spec.human_palm_nodes == (0, 1, 4, fs.MANO_BASES[spec.outermost])
        else:
            assert spec.fingers == ('thumb', 'index', 'middle') and spec.human_palm_nodes == (0, 1, 4, 7)
    demo, ret, robot, _ = synthetic_reference(hand, frames=6)
    m, cfg = module(hand, demo, ret)
    if want == 'fingers':
        assert m.hand_graph_palm_axis == {'left': None, 'right': None} and 'hand_graph_palm_axis' not in cfg
        # the wrist pose is not used: none, or a wrong one, gives the same reward
        t, q = torch.arange(6), torch.tensor([[1.0, 0, 0, 0]]).repeat(6, 1)
        a = m.compute_hand_graph_reward(robot['left']['kpts'], robot['right']['kpts'], q, t)
        wrong = torch.zeros(6, 7)
        wrong[:, 4] = 1.0
        b = m.compute_hand_graph_reward(robot['left']['kpts'], robot['right']['kpts'], q, t, None, None, wrong, wrong)
        assert torch.equal(a[0], b[0]) and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    else:
        assert all(m.hand_graph_palm_axis[s] is not None for s in SIDES) and set(cfg['hand_graph_palm_axis']) == set(SIDES)


def test_a_stored_axis_must_fit_the_hand():
    demo, ret, _, _ = synthetic_reference('xhand', frames=6)
    with pytest.raises(ValueError, match='comes from its fingers'):
        module('xhand', demo, ret, hand_graph_palm_axis={'left': [0.0, 0.0, 1.0]})
    demo, ret, _, _ = synthetic_reference('dex3_hand', frames=6)
    with pytest.raises(ValueError, match='unit 3-vector'):
        module('dex3_hand', demo, ret, hand_graph_palm_axis={'left': [0.0, 0.0, 2.0]})
    with pytest.raises(ValueError, match='unit 3-vector'):
        module('dex3_hand', demo, ret, hand_graph_palm_axis={'left': [0.0, 1.0]})
    no_pose = {s: {k: v for k, v in ret[s].items() if k != 'wrist_pose'} for s in SIDES}
    with pytest.raises(ValueError, match='needs the retargeted wrist_pose'):
        module('dex3_hand', demo, no_pose)
    m, _ = module('dex3_hand', demo, no_pose, hand_graph_palm_axis={'left': [0.0, 0.0, 1.0], 'right': [0.0, 1.0, 0.0]})
    assert m.hand_graph_palm_axis['right'].tolist() == [0.0, 1.0, 0.0]          # a stored axis needs no reference


# ------------------------------------------------------------------ (5) the axis travels with the run to evaluation
def test_axis_is_saved_in_env_pkl_and_evaluation_reproduces_it(tmp_path):
    demo, ret, robot, _ = synthetic_reference('dex3_hand')
    # training: train_rl_games.py builds env_kwargs, then BaseEnv(**env_kwargs), whose RewardModule gets
    # env_kwargs['reward_cfg'] itself and writes the axis into it, then dumps env_kwargs to params/env.pkl
    m_train, cfg = module('dex3_hand', demo, ret)
    env_kwargs = dict(env_cfg=dict(use_rl_games=True), reward_cfg=cfg)
    pkl = tmp_path / 'params' / 'env.pkl'
    pkl.parent.mkdir()
    pickle.dump(env_kwargs, open(pkl, 'wb'))
    loaded = pickle.load(open(pkl, 'rb'))                                          # eval_rl_games.py reads it back
    assert loaded['reward_cfg']['hand_graph_palm_axis'] == cfg['hand_graph_palm_axis']
    assert yaml.safe_load(yaml.safe_dump(cfg['hand_graph_palm_axis'])) == cfg['hand_graph_palm_axis']   # params/env.yaml
    # evaluation with a reference whose wrist links are turned by 20 degrees in their own frame: it would calibrate another
    # axis, but uses the stored one
    ret_eval = copy.deepcopy(ret)
    turn = axis_angle_quat(torch.tensor([1.0, 0.3, -0.2]), 20.0)
    for s in SIDES:
        pose = ret_eval[s]['wrist_pose']
        q = torch.as_tensor(pose[:, 3:], dtype=torch.float64)
        assert torch.allclose(matrix_from_quat(qmul(q, turn.expand_as(q))), matrix_from_quat(q) @ matrix_from_quat(turn), atol=1e-6)
        pose[:, 3:] = qmul(q, turn.expand_as(q)).float().numpy()
    m_eval = RewardModule(loaded['reward_cfg'], demo, ret_eval, DEV)
    m_fresh, _ = module('dex3_hand', demo, ret_eval)
    for s in SIDES:
        assert torch.equal(m_eval.hand_graph_palm_axis[s], m_train.hand_graph_palm_axis[s])
        assert angle_deg(m_fresh.hand_graph_palm_axis[s], m_train.hand_graph_palm_axis[s]) > 1.0
    a, b = score(m_train, robot, 40), score(m_eval, robot, 40)
    assert torch.equal(a[0], b[0]) and all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    assert loaded['reward_cfg']['hand_graph_palm_axis'] == cfg['hand_graph_palm_axis']   # evaluation does not rewrite it


def test_trainer_dumps_env_pkl_after_building_the_env():
    """The axis reaches params/env.pkl only because the trainer dumps env_kwargs after BaseEnv(**env_kwargs) has built the
    reward module, and BaseEnv hands reward_cfg itself (not a copy) to RewardModule."""
    train = (ROOT / 'dexmachina' / 'rl' / 'train_rl_games.py').read_text()
    assert -1 < train.index('env = BaseEnv(') < train.index('pickle.dump(env_kwargs')
    base = (ROOT / 'dexmachina' / 'envs' / 'base_env.py').read_text()
    call = base[base.index('self.reward_module = RewardModule('):]
    assert call[:call.index(')')].split('(')[1].split(',')[0].strip() == 'reward_cfg'
