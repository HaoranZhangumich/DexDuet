"""Tests for the hand-object contact option (--ho_contact_weight): distance + nearest-spot matching against the human hand,
per anchor and object part, weighted by touch / near / far; added on top of the unchanged DexMachina contact reward.

Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q dexmachina/tests/test_ho_contact.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dexmachina.envs.ho_contact import (ANCHOR_POINTS, ANCHORS, PARTS, HandObjectContact, PartDistanceGrid,  # noqa: E402
                                        anchor_contacts, grid_path, human_target_path, load_human_targets,
                                        robot_anchor_points)
from dexmachina.envs.rewards import RewardModule, get_reward_cfg  # noqa: E402
from dexmachina.hand_proc.ho_contact_assets import build_grid, human_targets  # noqa: E402

DEV = torch.device('cpu')
BOX = dict(bottom=(0.10, 0.06, 0.04), top=(0.04, 0.04, 0.02))      # part extents (m), centred on the part frame


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    root = tmp_path_factory.mktemp('assets')
    for part, ext in BOX.items():
        path = Path(grid_path(str(root), 'toy', part))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **build_grid(trimesh.creation.box(extents=ext), spacing=0.003, margin=0.06, samples=100000))
    return root


def contact(assets, **cfg):
    return HandObjectContact(cfg, [grid_path(str(assets), 'toy', p) for p in PARTS], DEV)


def t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_grid_matches_exact_distances(assets):
    half = np.asarray(BOX['bottom']) / 2
    grid = PartDistanceGrid(grid_path(str(assets), 'toy', 'bottom'), DEV)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.09, 0.09, size=(400, 3))
    pts = pts[(np.abs(pts) > half).any(-1)]                            # outside the box
    near = np.clip(pts, -half, half)                                   # exact nearest surface point and distance of a box
    exact = np.linalg.norm(pts - near, axis=-1)
    d, s = grid.query(t(pts))
    keep = exact < 0.05
    assert np.abs(d.numpy()[keep] - exact[keep]).max() < 0.002
    assert np.linalg.norm(s.numpy()[keep] - near[keep], axis=-1).max() < 0.004
    far, _ = grid.query(t([[1.0, 0.0, 0.0]]))                         # outside the grid: border value, beyond the cap
    assert float(far[0]) > 0.05


def score_one(c, d_r, s_r, d_h, s_h):
    """Single anchor, single part (others far on both sides)."""
    shape = (1, len(ANCHORS), len(PARTS))
    dr, dh = torch.full(shape, 1.0), torch.full(shape, 1.0)
    sr, sh = torch.zeros(shape + (3,)), torch.zeros(shape + (3,))
    dr[0, 1, 0], dh[0, 1, 0], sr[0, 1, 0], sh[0, 1, 0] = d_r, d_h, t(s_r), t(s_h)
    hand, det = c.score(dr, sr, dh, sh)
    return float(hand[0]), float(det['weight'][0, 1, 0])


def test_score_cases(assets):
    c = contact(assets)
    assert score_one(c, 0.0, (0.05, 0, 0.02), 0.0, (0.05, 0, 0.02)) == (pytest.approx(1.0), 1.0)          # same spot, both touch
    assert score_one(c, 0.0, (0.05, 0, 0.02), 0.004, (0.045, 0, 0.02))[0] == pytest.approx(1.0)          # within the 1 cm tolerance
    hover = score_one(c, 0.0, (0.0, 0, 0.02), 0.05, (0.0, 0, 0.02))                                     # robot touches, human 5 cm above
    assert hover == (pytest.approx(math.exp(-(0.05 - 0.01) / 0.02), rel=1e-4), 1.0)
    side = score_one(c, 0.0, (0.0, 0.03, 0.0), 0.0, (0.0, -0.03, 0.0))                                 # opposite sides, 6 cm apart
    assert side[0] == pytest.approx(math.exp(-(0.06 - 0.01) / 0.02), rel=1e-4)
    approach = score_one(c, 0.03, (0, 0, 0.02), 0.08, (0, 0, 0.02))                                    # human far: distance only
    assert approach == (pytest.approx(math.exp(-(0.02 - 0.01) / 0.02), rel=1e-4), 0.5)
    shape = (2, len(ANCHORS), len(PARTS))
    far = c.score(torch.full(shape, 0.2), torch.zeros(shape + (3,)), torch.full(shape, 0.2), torch.zeros(shape + (3,)))
    assert torch.allclose(far[0], torch.ones(2))                                                     # nothing near: nothing to match


def test_touch_and_near_entries_are_weighted(assets):
    c = contact(assets)
    shape = (1, len(ANCHORS), len(PARTS))
    dr, dh = torch.full(shape, 1.0), torch.full(shape, 1.0)
    sr, sh = torch.zeros(shape + (3,)), torch.zeros(shape + (3,))
    dr[0, 0, 0], dh[0, 0, 0] = 0.0, 0.05                           # palm: robot touches where the human hovers -> weight 1
    dr[0, 2, 0], dh[0, 2, 0] = 0.03, 0.03                          # index: both 3 cm away, same spot -> weight 0.5, score 1
    hand, det = c.score(dr, sr, dh, sh)
    s_touch = math.exp(-(0.05 - 0.01) / 0.02)
    assert float(hand[0]) == pytest.approx((s_touch + 0.5 * 1.0) / 1.5, rel=1e-4)


def test_robot_distances_use_part_frames_and_simulator_contacts(assets):
    c = contact(assets)
    n, a = 1, len(ANCHORS)
    slices = [(i, i + 1) for i in range(a)]
    pts = torch.zeros(n, a, 3)
    part_pos = t([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    h = math.radians(90) / 2
    part_quat = t([[[1, 0, 0, 0], [math.cos(h), 0, 0, math.sin(h)]]])   # top turned 90 deg about z
    pts[0, 0] = t([1.0, 2.0, 3.0 + 0.02 + 0.03])                        # palm 3 cm above the bottom part's top face (z = 0.02)
    pts[0, 1] = t([1.0 + 0.05, 2.0, 3.0])                              # thumb 5 cm along world +x (checked against the top part)
    touching = torch.zeros(n, 2, a, dtype=torch.bool)
    touch_pos = torch.zeros(n, 2, a, 3)
    touching[0, 0, 2] = True
    touch_pos[0, 0, 2] = t([1.0 + 0.05, 2.0, 3.0])                   # index touching the bottom part at its +x face
    d, s, grid_d = c.robot_distances(pts, slices, part_pos, part_quat, touching, touch_pos)
    assert float(d[0, 0, 0]) == pytest.approx(0.03, abs=0.0015)
    assert np.allclose(s[0, 0, 0].numpy(), [0.0, 0.0, 0.02], atol=0.003)
    # the top part (4 x 4 x 2 cm) turned 90 deg: a point 5 cm along world +x is 5 cm along its local -y, 3 cm from its face
    assert float(d[0, 1, 1]) == pytest.approx(0.03, abs=0.0015)
    assert np.allclose(s[0, 1, 1].numpy(), [0.0, -0.02, 0.0], atol=0.003)
    assert float(d[0, 2, 0]) == 0.0 and np.allclose(s[0, 2, 0].numpy(), [0.05, 0.0, 0.0], atol=1e-6)
    assert float(grid_d[0, 2, 0]) > 0.0                                  # the grid value before the contact override is kept


def test_anchor_contacts_take_the_most_distal_touching_link():
    valid = torch.zeros(1, 2, 13, dtype=torch.bool)
    pos = torch.zeros(1, 2, 13, 3)
    valid[0, 1, 11] = valid[0, 1, 12] = True
    pos[0, 1, 11], pos[0, 1, 12] = t([1, 0, 0]), t([2, 0, 0])
    touching, points = anchor_contacts(valid, pos, [[0], [11, 12]])
    assert touching[0, 1].tolist() == [False, True] and touching[0, 0].tolist() == [False, False]
    assert points[0, 1, 1].tolist() == [2.0, 0.0, 0.0]


def test_inspire_anchor_points_follow_the_urdf():
    for side, prefix in (('left', 'L'), ('right', 'R')):
        urdf = ROOT / 'dexmachina' / 'assets' / 'inspire_hand' / f'{side}_xyz_copy.urdf'
        spec = robot_anchor_points(str(urdf), 'inspire_hand', side)
        assert spec['links'] == ['base_link', f'{prefix}_TH_L3', f'{prefix}_TH_L4', f'{prefix}_FF_L2', f'{prefix}_MF_L2',
                                 f'{prefix}_RF_L2', f'{prefix}_LF_L2']
        assert [e - s for s, e in spec['slices']] == [ANCHOR_POINTS[a] for a in ANCHORS]
        assert spec['slices'][0][0] == 0 and all(spec['slices'][i][1] == spec['slices'][i + 1][0] for i in range(len(ANCHORS) - 1))
        assert spec['link_ranges'][-1][1] == len(spec['points']) == sum(ANCHOR_POINTS[a] for a in ANCHORS)
        assert np.abs(spec['points']).max() < 0.12                       # link frames, metres
        with pytest.raises(ValueError, match='no_such_hand'):              # other hands: tests/test_contact_sampler.py
            robot_anchor_points(str(urdf), 'no_such_hand', side)


def test_human_targets_from_mesh_vertices():
    meshes = [trimesh.creation.box(extents=BOX['bottom']), trimesh.creation.box(extents=BOX['top'])]
    frames = 2
    vertex_joint = np.zeros(778, np.int16)
    vertex_joint[:10] = 3                                                # index distal phalanx
    vertex_joint[10:20] = 15                                             # thumb distal phalanx
    vertex_joint[20:30], vertex_joint[30:40], vertex_joint[40:50] = 6, 12, 9   # middle, ring, little: far away
    vertex_joint[50:] = 0                                                # palm
    verts = np.zeros((frames, 778, 3))
    verts[:, :10] = [0.0, 0.0, 0.02]                                     # index touches the bottom part's top face
    verts[:, 10:20] = [0.0, 0.03 + 0.01, 0.0]                            # thumb 1 cm off its +y face
    verts[:, 20:50] = [0.0, 0.0, 1.0]
    verts[:, 50:] = [0.0, 0.0, 0.02 + 0.04]                              # palm 4 cm above
    obj_trans = np.zeros((frames, 3))
    obj_trans[1] = [0.5, 0.0, 0.0]                                       # the object (and the hand) move: same targets
    verts[1] += [0.5, 0.0, 0.0]
    obj_quat = np.tile([1.0, 0, 0, 0], (frames, 1))
    dist, spot = human_targets({'left': verts, 'right': verts}, {'left': vertex_joint, 'right': vertex_joint}, obj_trans, obj_quat,
                               np.zeros(frames), meshes, samples=50000)
    i, th, pa = ANCHORS.index('index'), ANCHORS.index('thumb'), ANCHORS.index('palm')
    for f in range(frames):
        assert dist[f, 1, i, 0] == pytest.approx(0.0, abs=0.002)
        assert dist[f, 1, th, 0] == pytest.approx(0.01, abs=0.002)
        assert dist[f, 1, pa, 0] == pytest.approx(0.04, abs=0.002)
        assert np.allclose(spot[f, 1, th, 0], [0.0, 0.03, 0.0], atol=0.003)


def make_module(assets, weight, frames=4):
    demo = dict(obj_pos=np.zeros((frames, 3), np.float32), obj_quat=np.tile([1, 0, 0, 0], (frames, 1)).astype(np.float32),
                obj_arti=np.zeros(frames, np.float32))
    if weight > 0:
        for side in ('left', 'right'):
            demo[f'ho_dist_{side}'] = np.full((frames, len(ANCHORS), len(PARTS)), 0.2, np.float32)
            demo[f'ho_spot_{side}'] = np.zeros((frames, len(ANCHORS), len(PARTS), 3), np.float32)
            demo[f'ho_dist_{side}'][:, 0, 0] = 0.05                     # human palm hovers 5 cm above the bottom part
            demo[f'ho_spot_{side}'][:, 0, 0] = [0.0, 0.0, 0.02]
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.0, contact_rew_weight=0.0, bc_rew_weight=0.0, ho_contact_weight=weight, ho_contact_object='toy',
               ho_contact_assets=str(assets))
    m = RewardModule(cfg, demo, {}, DEV)
    if weight > 0:
        for side in ('left', 'right'):
            m.setup_ho_contact(side, [(i, i + 1) for i in range(len(ANCHORS))], [[0], [11, 12], [7], [8], [9], [10]])
    return m


def reward(m, palm_touch):
    n = 1
    pts = torch.full((n, len(ANCHORS), 3), -5.0)                          # anchors far away ...
    pts[0, 0] = t([0.0, 0.0, 0.02 + (0.0 if palm_touch else 0.05)])      # ... except the palm
    valid = torch.zeros(n, 2, 13, dtype=torch.bool)
    pos = torch.zeros(n, 2, 13, 3)
    if palm_touch:
        valid[0, 0, 0] = True
        pos[0, 0, 0] = t([0.0, 0.0, 0.02])
    part_pos = t([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])                   # top part out of the way
    inputs = dict(points_left=pts, points_right=pts.clone(), part_pos=part_pos, part_quat=t([[[1, 0, 0, 0]] * 2]))
    kw = dict(actions=torch.zeros(n, 2), bc_dist=torch.zeros(n, 2), obj_pos=torch.zeros(n, 3), obj_quat=t([[1, 0, 0, 0]]),
              obj_arti=torch.zeros(n, 1), kpts_left=None, kpts_right=None, contact_link_pos_left=pos, contact_link_valid_left=valid,
              contact_link_pos_right=pos.clone(), contact_link_valid_right=valid.clone(), wrist_pose_left=None, wrist_pose_right=None,
              contact_forces=None, episode_length_buf=torch.zeros(n, dtype=torch.long))
    if m.ho_contact_weight > 0:
        kw['ho_contact_inputs'] = inputs
    return m.compute_reward(**kw)


def test_reward_module_adds_the_term_only_when_on(assets):
    base, _ = reward(make_module(assets, 0.0), palm_touch=True)
    on = make_module(assets, 1.0)
    hover, d_hover = reward(on, palm_touch=False)                        # robot palm hovers 5 cm above, like the human
    touch, d_touch = reward(on, palm_touch=True)                         # robot palm rests on the part
    assert float(hover[0] - base[0]) == pytest.approx(1.0, abs=0.02)
    assert float(touch[0] - base[0]) == pytest.approx(math.exp(-(0.05 - 0.01) / 0.02), rel=0.05)
    assert float(d_touch['ho_unwanted_left'][0]) == 1.0 and float(d_hover['ho_unwanted_left'][0]) == 0.0
    assert float(d_touch['ho_contact_check'][0]) < 0.003                 # simulator contact agrees with the grid
    assert 'ho_contact_rew' not in reward(make_module(assets, 0.0), palm_touch=True)[1]
    with pytest.raises(ValueError):
        make_module(assets, -1.0)


def test_human_target_file_checks(tmp_path):
    path = Path(human_target_path(str(tmp_path), 's01', 'toy', '01'))
    path.parent.mkdir(parents=True)
    dist = np.zeros((10, 2, len(ANCHORS), len(PARTS)), np.float32)
    np.savez(path, dist=dist, spot=np.zeros(dist.shape + (3,), np.float32), anchors=np.array(ANCHORS), parts=np.array(PARTS))
    out = load_human_targets(str(tmp_path), 's01', 'toy', '01', 2, 7)
    assert out['ho_dist_left'].shape == (5, len(ANCHORS), len(PARTS)) and out['ho_spot_right'].shape == (5, len(ANCHORS), len(PARTS), 3)
    with pytest.raises(ValueError):
        load_human_targets(str(tmp_path), 's01', 'toy', '01', 2, 11)
    with pytest.raises(FileNotFoundError):
        load_human_targets(str(tmp_path), 's01', 'toy', '02', 0, 5)
