"""Collision-surface sampling of the hand-object contact regions (ho_contact.robot_anchor_points): every <collision> element of a
link, box / cylinder (z axis, centred) / sphere primitives, origin xyz + fixed-axis rpy, mesh scale; and the regions of the
real hands.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_contact_sampler.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.envs.finger_spec import load_finger_spec  # noqa: E402
from dexmachina.envs.ho_contact import (ANCHOR_POINTS, collision_elements, is_plain_mesh, robot_anchor_points,  # noqa: E402
                                        rpy_matrix, sample_box, sample_cylinder, sample_sphere)

N = 4000


def on_box(p, size, tol=1e-6):
    h = np.asarray(size) / 2.0
    inside = (np.abs(p) <= h + tol).all(-1)
    face = (np.abs(np.abs(p) - h) < tol).any(-1)
    return inside & face


def on_cylinder(p, radius, length, tol=1e-6):
    r = np.linalg.norm(p[:, :2], axis=-1)
    side = (np.abs(r - radius) < tol) & (np.abs(p[:, 2]) <= length / 2 + tol)
    cap = (np.abs(np.abs(p[:, 2]) - length / 2) < tol) & (r <= radius + tol)
    return side | cap


def test_box_points_lie_on_the_faces_in_proportion_to_area():
    size = (0.04, 0.02, 0.01)
    p = sample_box(size, N, np.random.default_rng(0))
    assert p.shape == (N, 3) and on_box(p, size).all()
    a, b, c = size
    on_z = (np.abs(np.abs(p[:, 2]) - c / 2) < 1e-9).mean()
    assert on_z == pytest.approx(a * b / (a * b + b * c + a * c), abs=0.03)


def test_cylinder_points_lie_on_the_side_or_the_caps_about_z():
    radius, length = 0.045, 0.03
    p = sample_cylinder(radius, length, N, np.random.default_rng(1))
    assert on_cylinder(p, radius, length).all()
    assert np.abs(p[:, 2]).max() == pytest.approx(length / 2)               # centred on the origin, along z
    side = np.abs(np.linalg.norm(p[:, :2], axis=-1) - radius) < 1e-9
    assert side.mean() == pytest.approx(length / (length + radius), abs=0.03)   # 2 pi r L / (2 pi r L + 2 pi r^2)


def test_sphere_points_lie_at_the_radius():
    p = sample_sphere(0.01, N, np.random.default_rng(2))
    assert np.allclose(np.linalg.norm(p, axis=-1), 0.01, atol=1e-9)
    assert np.abs(p.mean(0)).max() < 0.001


def test_rpy_is_fixed_axis_roll_pitch_yaw():
    for rpy in ((0.3, 0.0, 0.0), (0.0, -0.7, 0.0), (0.0, 0.0, 1.1), (4.450589592585541, 0.0, 0.0), (3.141592, 0.0, 0.34383),
                (0.2, -0.4, 2.5)):
        assert np.allclose(rpy_matrix(rpy), Rotation.from_euler('xyz', rpy).as_matrix(), atol=1e-12)   # extrinsic x, y, z


URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="palm">
    <collision><origin xyz="0.01 -0.02 0.03" rpy="0.3 -0.2 1.0"/><geometry><box size="0.04 0.03 0.02"/></geometry></collision>
  </link>
  <link name="cyl">
    <collision><origin xyz="0 0 -0.009" rpy="0 0 0"/><geometry><cylinder radius="0.045" length="0.045"/></geometry></collision>
  </link>
  <link name="ball">
    <collision><origin xyz="0 0 0.00121"/><geometry><sphere radius="0.01"/></geometry></collision>
  </link>
  <link name="multi">
    <collision><origin xyz="0 0 0.01" rpy="0 0 0"/><geometry><box size="0.02 0.02 0.02"/></geometry></collision>
    <collision><origin xyz="0.03 0 0" rpy="0 0 0"/><geometry><sphere radius="0.005"/></geometry></collision>
  </link>
  <link name="scaled">
    <collision><origin xyz="0.0 0.01 0.0" rpy="0 1.5707963267948966 0"/><geometry><mesh filename="box.stl" scale="2 1 0.5"/></geometry></collision>
  </link>
  <link name="plain">
    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="box.stl" scale="1 1 1"/></geometry></collision>
  </link>
  <link name="noorigin"><collision><geometry><mesh filename="box.stl"/></geometry></collision></link>
  <link name="visual_only"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
</robot>
"""
MESH_EXTENTS = (0.03, 0.02, 0.01)


@pytest.fixture()
def toy(tmp_path):
    trimesh.creation.box(extents=MESH_EXTENTS).export(tmp_path / 'box.stl')
    path = tmp_path / 'toy.urdf'
    path.write_text(URDF)
    return str(path)


def to_link_origin_frame(points, xyz, rpy):
    """Undo a collision origin: link frame -> geometry frame."""
    return (np.asarray(points, np.float64) - np.asarray(xyz)) @ rpy_matrix(rpy)


def test_collision_elements_and_the_plain_mesh_rule(toy):
    el = collision_elements(toy)
    assert set(el) == {'palm', 'cyl', 'ball', 'multi', 'scaled', 'plain', 'noorigin'}
    assert [e['kind'] for e in el['multi']] == ['box', 'sphere']
    assert el['ball'][0]['rpy'] == (0.0, 0.0, 0.0) and el['ball'][0]['xyz'] == (0.0, 0.0, 0.00121)
    assert el['scaled'][0]['scale'] == (2.0, 1.0, 0.5)
    assert is_plain_mesh(el['plain']) and is_plain_mesh(el['noorigin'])
    assert not any(is_plain_mesh(el[n]) for n in ('palm', 'cyl', 'ball', 'multi', 'scaled'))


def test_primitive_regions_follow_origin_rpy_and_scale(toy):
    el = collision_elements(toy)
    regions = [('palm', ['palm']), ('thumb', ['cyl', 'ball']), ('index', ['multi']), ('middle', ['scaled']),
               ('ring', ['plain', 'ball']), ('pinky', ['noorigin'])]
    out = robot_anchor_points(toy, 'toy', 'left', regions=regions)
    assert [e - s for s, e in out['slices']] == [ANCHOR_POINTS[r] for r, _ in regions]
    assert out['links'] == ['palm', 'cyl', 'ball', 'multi', 'scaled', 'plain', 'ball', 'noorigin']
    assert out['regions'] == [r for r, _ in regions]
    pts = {name: out['points'][s:e].astype(np.float64) for name, (s, e) in list(zip(out['links'], out['link_ranges']))[:6]}
    tol = 2e-6                                                                       # float32 points
    e = el['palm'][0]
    assert on_box(to_link_origin_frame(pts['palm'], e['xyz'], e['rpy']), e['size'], tol).all()
    assert not on_box(pts['palm'], e['size'], tol).all()                             # the offset is applied
    e = el['cyl'][0]
    assert on_cylinder(to_link_origin_frame(pts['cyl'], e['xyz'], e['rpy']), e['radius'], e['length'], tol).all()
    e = el['ball'][0]
    assert np.allclose(np.linalg.norm(to_link_origin_frame(pts['ball'], e['xyz'], e['rpy']), axis=-1), e['radius'], atol=tol)
    # the thumb's points split over its two links by area, the remainder going to the larger one
    area_cyl, area_ball = 2 * np.pi * 0.045 * 0.045 + 2 * np.pi * 0.045 ** 2, 4 * np.pi * 0.01 ** 2
    assert len(pts['ball']) == int(np.floor(64 * area_ball / (area_cyl + area_ball))) and len(pts['cyl']) + len(pts['ball']) == 64
    # two collision elements in one link: every point lies on one of them, both are used
    box, ball = el['multi']
    on_b = on_box(to_link_origin_frame(pts['multi'], box['xyz'], box['rpy']), box['size'], tol)
    on_s = np.abs(np.linalg.norm(to_link_origin_frame(pts['multi'], ball['xyz'], ball['rpy']), axis=-1) - ball['radius']) < tol
    assert (on_b | on_s).all() and on_b.sum() > 0 and on_s.sum() > 0
    # scaled mesh with an offset and a rotation: undo the origin, then the scale -> on the mesh's own box
    e = el['scaled'][0]
    local = to_link_origin_frame(pts['scaled'], e['xyz'], e['rpy']) / np.asarray(e['scale'])
    assert on_box(local, MESH_EXTENTS, 1e-5).all()
    # plain meshes are sampled as in the Inspire-only code: trimesh.sample_surface(seed=0) of the loaded mesh, no transform,
    # also next to a primitive link in the same region (ring: plain + ball)
    mesh = trimesh.load(el['plain'][0]['path'], force='mesh')
    s5, e5 = out['link_ranges'][5]
    assert 0 < e5 - s5 < 48 and out['link_ranges'][6][1] - s5 == 48
    assert np.array_equal(out['points'][s5:e5], trimesh.sample.sample_surface(mesh, e5 - s5, seed=0)[0].astype(np.float32))
    expect = trimesh.sample.sample_surface(mesh, 48, seed=0)[0].astype(np.float32)
    assert np.array_equal(out['points'][slice(*out['link_ranges'][7])], expect)


def test_sampling_touches_no_global_rng(toy):
    torch.manual_seed(3); np.random.seed(3); random.seed(3)
    states = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    robot_anchor_points(toy, 'toy', 'left', regions=[('palm', ['palm', 'multi']), ('thumb', ['scaled', 'cyl'])])
    assert torch.equal(torch.get_rng_state(), states[0]) and np.array_equal(np.random.get_state()[1], states[1])
    assert random.getstate() == states[2]


def test_missing_collision_link_is_named(toy):
    with pytest.raises(ValueError, match='visual_only'):
        robot_anchor_points(toy, 'toy', 'left', regions=[('palm', ['visual_only'])])
    with pytest.raises(ValueError, match='no_such_hand'):
        robot_anchor_points(toy, 'no_such_hand', 'left')                 # no regions given and no finger spec for the hand


HANDS = ('inspire_hand', 'xhand', 'schunk_hand', 'allegro_hand', 'dex3_hand', 'ability_hand', 'dexrobot_hand')


@pytest.mark.parametrize('side', ('left', 'right'))
@pytest.mark.parametrize('hand', HANDS)
def test_real_hand_regions(hand, side):
    spec = load_finger_spec(hand, side)
    out = robot_anchor_points(spec.urdf_path, hand, side)
    assert out['regions'] == list(spec.regions)
    assert [e - s for s, e in out['slices']] == [ANCHOR_POINTS[r] for r in spec.regions]
    assert out['anchor_links'] == [list(spec.contact_links[r]) for r in spec.regions]
    assert np.isfinite(out['points']).all() and np.abs(out['points']).max() < 0.25      # link frames, metres
    assert out['link_ranges'][-1][1] == len(out['points'])
    el = collision_elements(spec.urdf_path)
    for name, (s, e) in zip(out['links'], out['link_ranges']):
        items = el[name]
        if all(i['kind'] == 'box' for i in items):                    # e.g. Allegro segments and palm, Schunk palm, Dex3 thumb_1
            p = out['points'][s:e]
            on_any = np.zeros(len(p), bool)
            for i in items:
                on_any |= on_box(to_link_origin_frame(p, i['xyz'], i['rpy']), i['size'], 2e-6)
            assert on_any.all(), name


def test_allegro_palm_uses_all_three_boxes_and_the_tip_mesh_offset():
    for side in ('left', 'right'):
        spec = load_finger_spec('allegro_hand', side)
        out = robot_anchor_points(spec.urdf_path, 'allegro_hand', side)
        el = collision_elements(spec.urdf_path)
        s, e = out['link_ranges'][out['links'].index('base_link')]
        p = out['points'][s:e]
        hits = [on_box(to_link_origin_frame(p, b['xyz'], b['rpy']), b['size'], 2e-6).sum() for b in el['base_link']]
        assert len(hits) == 3 and min(hits) > 0
        tip = spec.contact_links['thumb'][-1]
        assert tip == 'link_15.0_tip' and el[tip][0]['xyz'] == (0.0, 0.0, -0.012)
        s, e = out['link_ranges'][out['links'].index(tip)]
        lo, hi = trimesh.load(el[tip][0]['path'], force='mesh').bounds
        local = out['points'][s:e] - np.array([0.0, 0.0, -0.012])                 # undo the collision origin
        assert e > s and ((local >= lo - 1e-6) & (local <= hi + 1e-6)).all()
        assert not ((out['points'][s:e] >= lo - 1e-6) & (out['points'][s:e] <= hi + 1e-6)).all()   # the offset was applied
