"""The hand-graph and hand-object contact rewards on every hand (finger correspondence from dexmachina/envs/finger_spec.py):
synthetic hands built from a flat human hand, dropped fingers, extra (non-keypoint) nodes, region selection of the human
contact targets, and the robot / env wiring of the extra nodes (with stand-ins for the simulator).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_general_fingers.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.envs.finger_spec import FINGERS, load_finger_spec  # noqa: E402
from dexmachina.envs.ho_contact import ANCHORS, PARTS, grid_path  # noqa: E402
from dexmachina.envs.rewards import RewardModule, get_reward_cfg  # noqa: E402

DEV = torch.device('cpu')
T = 5
HANDS = ('inspire_hand', 'xhand', 'schunk_hand', 'allegro_hand', 'dex3_hand', 'ability_hand', 'dexrobot_hand')
# a flat human hand in the xy plane (MANO joints): wrist, finger bases (13/1/4/10/7), fingertips (16-20)
HUMAN = {0: (0, 0, 0), 13: (0.03, 0.03, 0), 1: (0.08, 0.02, 0), 4: (0.085, 0, 0), 10: (0.08, -0.02, 0), 7: (0.075, -0.04, 0),
         16: (0.06, 0.07, 0), 17: (0.16, 0.025, 0), 18: (0.17, 0, 0), 19: (0.16, -0.025, 0), 20: (0.14, -0.05, 0)}


def human_joints(move=None):
    j = np.zeros((T, 21, 3), np.float32)
    for n, p in HUMAN.items():
        j[:, n] = p
    for n, p in (move or {}).items():
        j[:, n] = p
    return j


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)


def quat_z(deg):
    h = math.radians(deg) / 2
    return torch.tensor([[math.cos(h), 0, 0, math.sin(h)]], dtype=torch.float32)


def quat_of(rot):
    """(w, x, y, z) of a rotation matrix by less than 180 degrees."""
    r = np.asarray(rot, np.float64)
    w = math.sqrt(1.0 + np.trace(r)) / 2.0
    return np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])


def wrist_pose(rot, offset=(0, 0, 0)):
    """robot.wrist_pose (1, 7) of a wrist link turned by rot (the synthetic robot hands' wrist frame is the world frame)."""
    return torch.as_tensor(np.concatenate([np.asarray(offset, np.float64), quat_of(rot)])[None], dtype=torch.float32)


def node_joints(spec):
    """Robot link -> MANO joint for the hand-graph nodes."""
    out = {spec.wrist: 0}
    out.update(zip(spec.bases, spec.human_bases))
    out.update(zip(spec.tips, spec.human_tips))
    return out


def kpt_names(spec, leave_out=()):
    """Keypoint names as retargeting would record them: some other links first, then the nodes (minus leave_out)."""
    return ['filler_a', 'filler_b'] + [n for n in node_joints(spec) if n not in leave_out]


def robot_kpts(spec, names, rot=np.eye(3, dtype=np.float32), offset=(0, 0, 0), n=1, scale=None):
    """Keypoints of a robot hand that matches the (scaled) human hand; non-node keypoints far away."""
    scale = spec.scale if scale is None else scale
    k = np.full((n, len(names), 3), 9.0, np.float32)
    joint_of = node_joints(spec)
    for i, name in enumerate(names):
        if name in joint_of:
            k[:, i] = rot @ (np.float32(scale) * np.asarray(HUMAN[joint_of[name]], np.float32)) + np.asarray(offset, np.float32)
    return torch.as_tensor(k)


def node_positions(spec, links, **kw):
    return robot_kpts(spec, list(links), **kw)


def make_module(hand, weight=0.5, leave_out=None, move=None, **extra):
    specs = {s: load_finger_spec(hand, s) for s in ('left', 'right')}
    demo = dict(obj_pos=np.zeros((T, 3), np.float32), obj_quat=np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32),
                obj_arti=np.zeros(T, np.float32), hand_joints_left=human_joints(move), hand_joints_right=human_joints(move))
    ret = dict()
    for s in ('left', 'right'):
        names = kpt_names(specs[s], (leave_out or {}).get(s, ()))
        ret[s] = dict(kpts_data=dict(kpt_pos=np.zeros((T, len(names), 3), np.float32), kpt_names=names),
                      wrist_pose=np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float32))
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.3, contact_rew_weight=0.0, bc_rew_weight=0.3, hand_graph_weight=weight, hand_name=hand, **extra)
    return RewardModule(cfg, demo, ret, DEV), specs, ret


def score(m, specs, ret, rot=np.eye(3, dtype=np.float32), offset=(0, 0, 0), obj_quat=None, extra=None):
    q = quat_z(0) if obj_quat is None else obj_quat
    k = {s: robot_kpts(specs[s], ret[s]['kpts_data']['kpt_names'], rot, offset) for s in ('left', 'right')}
    e = {s: (None if not (extra or {}).get(s) else node_positions(specs[s], extra[s], rot=rot, offset=offset)) for s in ('left', 'right')}
    w = wrist_pose(rot, offset)                     # used only by a 'calibrated_axis' hand (Dex3)
    return m.compute_hand_graph_reward(k['left'], k['right'], q, torch.zeros(1, dtype=torch.long), e['left'], e['right'], w, w)


@pytest.mark.parametrize('hand', HANDS)
def test_matching_hand_scores_full_weight_and_shapes(hand):
    m, specs, ret = make_module(hand)
    spec = specs['left']
    assert m.hand_graph_scale == spec.scale                                  # the hand's retargeting scale by default
    assert m.demo_tensors['hand_graph_fingers_left'].shape == (T, len(spec.fingers), 3)
    assert m.demo_tensors['hand_graph_gaps_left'].shape == (T, len(spec.gap_pairs))
    assert m.demo_tensors['hand_graph_palm_left'].shape == (T, 2, 3)
    rew, d = score(m, specs, ret)
    assert rew[0] == pytest.approx(0.5, rel=1e-5)
    for side in ('left', 'right'):
        assert d[f'hand_graph_dir_err_{side}'][0] == pytest.approx(0.0, abs=1e-6)
        assert d[f'hand_graph_gap_err_{side}'][0] == pytest.approx(0.0, abs=1e-6)
    assert float(score(m, specs, ret, offset=(0.4, -0.2, 1.1))[0][0]) == pytest.approx(0.5, rel=1e-5)


@pytest.mark.parametrize('hand', HANDS)
def test_tilting_turning_and_the_palm_frame(hand):
    m, specs, ret = make_module(hand)
    rew, d = score(m, specs, ret, rot=rot_x(45))
    assert d['hand_graph_palm_deg_right'][0] == pytest.approx(45.0, abs=0.1)
    assert d['hand_graph_gap_err_right'][0] == pytest.approx(0.0, abs=1e-6)
    assert rew[0] < 0.5 * math.exp(-5 * 0.05)
    rew, d = score(m, specs, ret, rot=rot_z(30), obj_quat=quat_z(30))
    assert rew[0] == pytest.approx(0.5, rel=1e-5) and d['hand_graph_palm_deg_left'][0] == pytest.approx(0.0, abs=0.1)
    # the palm normal of the flat hand is -z for every finger subset (index x outermost base; Dex3 index x little base and
    # the calibrated wrist-link axis, -z for the identity wrist of make_module), on both sides
    palm = m.demo_tensors['hand_graph_palm_left'][0]
    assert torch.allclose(palm[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    index, middle = np.asarray(HUMAN[1]), np.asarray(HUMAN[4])
    fwd = middle if specs['left'].forward_to_middle else 0.5 * (index + middle)          # Dex3: midpoint of index and middle
    assert torch.allclose(palm[1], torch.as_tensor(fwd / np.linalg.norm(fwd), dtype=torch.float32), atol=1e-6)
    assert specs['left'].forward_to_middle == (hand != 'dex3_hand')


def test_fingers_the_robot_lacks_do_not_count():
    """Moving the human pinky changes the five-finger reward, not Allegro's; moving the ring finger and the pinky tip leaves
    Dex3's alone. Dex3's human palm normal does use the little-finger base ('calibrated_axis': index x little base)."""
    moved = {20: (0.1, -0.09, 0.03), 7: (0.07, -0.05, 0.01)}
    for hand, drop in (('allegro_hand', moved), ('dex3_hand', {20: moved[20], 19: (0.12, -0.08, 0.05), 10: (0.07, -0.03, 0.01)})):
        m, specs, ret = make_module(hand)
        m2, _, _ = make_module(hand, move=drop)
        assert torch.equal(score(m, specs, ret, rot=rot_x(20))[0], score(m2, specs, ret, rot=rot_x(20))[0])
    m, specs, ret = make_module('xhand')
    m2, _, _ = make_module('xhand', move=moved)
    assert float(score(m2, specs, ret)[0][0]) < 0.5 * 0.99
    m, _, _ = make_module('dex3_hand')
    m2, _, _ = make_module('dex3_hand', move={7: moved[7]})
    assert not torch.allclose(m.demo_tensors['hand_graph_palm_left'][:, 0], m2.demo_tensors['hand_graph_palm_left'][:, 0])


@pytest.mark.parametrize('hand,side_links', [('allegro_hand', lambda s: [s.wrist]), ('dex3_hand', lambda s: list(s.tips)),
                                             ('xhand', lambda s: [s.wrist, s.tips[2]])])
def test_extra_nodes_give_the_same_reward_as_keypoints(hand, side_links):
    m, specs, ret = make_module(hand)
    leave = {s: side_links(specs[s]) for s in ('left', 'right')}
    m2, _, ret2 = make_module(hand, leave_out=leave)
    assert m.hand_graph_extra == {'left': [], 'right': []}
    assert m2.hand_graph_extra == {s: leave[s] for s in ('left', 'right')}             # the wrist first when it is extra
    for rot, q in ((np.eye(3, dtype=np.float32), None), (rot_x(30) @ rot_z(50), quat_z(10))):
        a = score(m, specs, ret, rot=rot, obj_quat=q)
        b = score(m2, specs, ret2, rot=rot, obj_quat=q, extra=leave)
        assert torch.equal(a[0], b[0])
        for k in a[1]:
            assert torch.equal(a[1][k], b[1][k]), k
    with pytest.raises(ValueError, match='not keypoints'):
        score(m2, specs, ret2)                                                    # extra nodes not passed


# ------------------------------------------------------------------ hand-object contact
BOX = dict(bottom=(0.10, 0.06, 0.04), top=(0.04, 0.04, 0.02))


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    from dexmachina.hand_proc.ho_contact_assets import build_grid
    root = tmp_path_factory.mktemp('assets')
    for part, ext in BOX.items():
        path = Path(grid_path(str(root), 'toy', part))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **build_grid(trimesh.creation.box(extents=ext), spacing=0.003, margin=0.06, samples=100000))
    return root


def contact_module(assets, hand, dist, spot, frames=4):
    demo = dict(obj_pos=np.zeros((frames, 3), np.float32), obj_quat=np.tile([1, 0, 0, 0], (frames, 1)).astype(np.float32),
                obj_arti=np.zeros(frames, np.float32))
    for side in ('left', 'right'):
        demo[f'ho_dist_{side}'] = dist.copy()
        demo[f'ho_spot_{side}'] = spot.copy()
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.0, contact_rew_weight=0.0, bc_rew_weight=0.0, ho_contact_weight=1.0, ho_contact_object='toy',
               ho_contact_assets=str(assets), hand_name=hand)
    m = RewardModule(cfg, demo, {}, DEV)
    n = len(m.ho_contact_regions['left'])
    for side in ('left', 'right'):
        m.setup_ho_contact(side, [(i, i + 1) for i in range(n)], [[i] for i in range(n)])
    return m, demo


def test_regions_and_human_targets_per_hand(assets):
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 0.1, (4, len(ANCHORS), len(PARTS))).astype(np.float32)
    spot = rng.uniform(-0.05, 0.05, (4, len(ANCHORS), len(PARTS), 3)).astype(np.float32)
    for hand, regions in (('inspire_hand', ANCHORS), ('xhand', ANCHORS), ('schunk_hand', ANCHORS),
                          ('allegro_hand', ANCHORS[:5]), ('dex3_hand', ANCHORS[:4])):
        m, demo = contact_module(assets, hand, dist, spot)
        assert m.ho_contact_regions['left'] == regions == ('palm',) + load_finger_spec(hand, 'left').fingers
        pick = [ANCHORS.index(r) for r in regions]
        for side in ('left', 'right'):
            assert torch.equal(m.demo_tensors[f'ho_dist_{side}'], torch.tensor(dist[:, pick]))
            assert torch.equal(m.demo_tensors[f'ho_spot_{side}'], torch.tensor(spot[:, pick]))
        with pytest.raises(ValueError, match='contact regions'):
            m.setup_ho_contact('left', [(i, i + 1) for i in range(len(regions) + 1)], [[0]] * (len(regions) + 1))


def contact_step(m, touch_region=None, spot=(0.0, 0.0, 0.02)):
    """All robot regions far from the object (5 m), optionally one region touching the bottom part at spot (simulator)."""
    n = len(m.ho_contact_regions['left'])
    pts = torch.full((1, n, 3), -5.0)
    valid = torch.zeros(1, 2, n, dtype=torch.bool)
    pos = torch.zeros(1, 2, n, 3)
    if touch_region is not None:
        k = m.ho_contact_regions['left'].index(touch_region)
        valid[0, 0, k] = True
        pos[0, 0, k] = torch.tensor(spot)
    inputs = dict(points_left=pts, points_right=pts.clone(), part_pos=torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]]),
                  part_quat=torch.tensor([[[1.0, 0, 0, 0]] * 2]))
    return m.compute_ho_contact_reward(inputs, pos, valid, pos.clone(), valid.clone(), torch.zeros(1, dtype=torch.long))


def human_targets(touching):
    """Human regions far (20 cm) from both parts, except `touching`, which touch the bottom part on its top face."""
    dist = np.full((4, len(ANCHORS), len(PARTS)), 0.2, np.float32)
    spot = np.zeros((4, len(ANCHORS), len(PARTS), 3), np.float32)
    for r in touching:
        dist[:, ANCHORS.index(r), 0] = 0.0
        spot[:, ANCHORS.index(r), 0] = (0.0, 0.0, 0.02)
    return dist, spot


def test_a_dropped_finger_leaves_the_sum_and_the_normalisation(assets):
    # only the human pinky touches; no other region (human or robot) is within the 5 cm cap
    dist, spot = human_targets(['pinky'])
    five, _ = contact_module(assets, 'xhand', dist, spot)
    rew5, out5 = contact_step(five)
    assert float(out5['ho_hand_left'][0]) < 0.2                      # the robot pinky is far from where the human touches
    for hand in ('allegro_hand', 'dex3_hand'):
        m, _ = contact_module(assets, hand, dist, spot)
        rew, out = contact_step(m)
        assert float(out['ho_hand_left'][0]) == 1.0 and float(rew[0]) == 1.0     # nothing left to match: score 1
    # the pinky and the index touch; the robot index touches the same spot
    dist, spot = human_targets(['pinky', 'index'])
    five, _ = contact_module(assets, 'xhand', dist, spot)
    _, out5 = contact_step(five, touch_region='index')
    s_pinky = float(out5['ho_hand_left'][0]) * 2.0 - 1.0              # (index 1 + pinky s) / 2
    assert 0.0 < s_pinky < 0.2
    for hand in ('allegro_hand', 'dex3_hand'):
        m, _ = contact_module(assets, hand, dist, spot)
        _, out = contact_step(m, touch_region='index')
        hand_score = float(out['ho_hand_left'][0])
        # equals the score over the remaining regions only (computed directly), i.e. 1 for the matched index
        n = len(m.ho_contact_regions['left'])
        d_r = torch.full((1, n, 2), 1.0)
        d_r[0, 2, 0] = 0.0
        s_r = torch.zeros(1, n, 2, 3)
        s_r[0, 2, 0] = torch.tensor([0.0, 0.0, 0.02])
        direct, _ = m.ho_contact.score(d_r, s_r, torch.tensor(dist[:1, :n]), torch.tensor(spot[:1, :n]))
        assert hand_score == pytest.approx(float(direct[0]), abs=1e-6)
        assert hand_score == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ robot / env wiring of the extra nodes (stand-ins)
class _Link:
    def __init__(self, name):
        self.name = name


class _Entity:
    def __init__(self, names, n):
        self.links = [_Link(x) for x in names]
        self.pos = torch.randn(n, len(names), 3, generator=torch.Generator().manual_seed(0))
        self.quat = torch.nn.functional.normalize(torch.randn(n, len(names), 4, generator=torch.Generator().manual_seed(1)), dim=-1)
        self.n = n

    def get_links_pos(self):
        return self.pos.clone()

    def get_links_quat(self):
        return self.quat.clone()

    def get_dofs_position(self, idx):
        return torch.zeros(self.n, len(idx))

    get_dofs_velocity = get_dofs_control_force = get_dofs_position

    def set_dofs_position(self, **kw):
        pass

    def zero_all_dofs_velocity(self, **kw):
        pass


def fake_robot(names, wrist, n=3):
    from dexmachina.envs.robot import BaseRobot
    r = BaseRobot.__new__(BaseRobot)
    r.name, r.cfg, r.num_envs, r.device, r.initialized = 'fake', dict(urdf_path='fake.urdf'), n, DEV, True
    r.entity = _Entity(names, n)
    r.actuated_dof_idxs, r.ndof, r.kpt_markers = [0], 1, []
    r.set_kpt_links([x for x in names if x.startswith('kpt')])
    r.wrist_link_name, r.wrist_link_idx = wrist, names.index(wrist)
    r.extra_link_names, r.extra_link_idxs, r.extra_kpt_pos = [], [], None
    r.init_qpos = torch.zeros(n, 1)
    r.action_mode, r.residual_qpos, r.is_eval = 'residual', None, False
    r.initialize_value_buffers()
    r.episode_length_buf = torch.zeros(n, dtype=torch.int32)
    return r


def test_robot_tracks_extra_links_and_reports_missing_keypoints():
    names = ['root', 'base_dummy_link', 'kpt_base', 'kpt_a', 'tip_x', 'tip_y']
    r = fake_robot(names, 'base_dummy_link')
    assert r.missing_kpt_links == [] and r.kpt_link_names == ['kpt_base', 'kpt_a']
    r.set_kpt_links(['kpt_a', 'nope', 'kpt_base'])
    assert r.missing_kpt_links == ['nope'] and r.kpt_link_names == ['kpt_a', 'kpt_base']
    r.set_kpt_links(['kpt_base', 'kpt_a'])
    with pytest.raises(ValueError, match='tip_z'):
        r.track_extra_links(['tip_x', 'tip_z'])
    r.track_extra_links(['tip_y', 'tip_x'])
    r.update_value_buffers()
    assert torch.equal(r.extra_kpt_pos, r.entity.pos[:, [5, 4]]) and torch.equal(r.kpt_pos, r.entity.pos[:, [2, 3]])
    assert torch.equal(r.wrist_pose[:, :3], r.entity.pos[:, 1])
    before = r.extra_kpt_pos.clone()
    r.entity.pos = r.entity.pos + 1.0
    r.reset_idx(torch.tensor([1]))
    assert torch.equal(r.extra_kpt_pos[1], r.entity.pos[1, [5, 4]]) and torch.equal(r.extra_kpt_pos[0], before[0])


def test_env_passes_the_wrist_pose_and_the_extra_buffer():
    from dexmachina.envs.base_env import BaseEnv
    left = fake_robot(['root', 'base_dummy_link', 'kpt_base', 'tip_x'], 'base_dummy_link')
    right = fake_robot(['root', 'base_dummy_link', 'kpt_base', 'tip_x'], 'base_dummy_link')
    left.track_extra_links(['tip_x'])
    left.update_value_buffers()
    right.update_value_buffers()
    env = BaseEnv.__new__(BaseEnv)
    env.robots = dict(left=left, right=right)
    env.reward_module = SimpleNamespace(hand_graph_extra=dict(left=['base_dummy_link', 'tip_x'], right=[]))
    out = env.get_hand_graph_extra('left')
    assert out.shape == (3, 2, 3)
    assert torch.equal(out[:, 0], left.wrist_pose[:, :3]) and torch.equal(out[:, 1], left.entity.pos[:, 3])
    assert env.get_hand_graph_extra('right') is None
