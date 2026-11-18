"""The Inspire hand-graph and hand-object contact rewards must be bit-identical to a golden file
(tests/golden/inspire_golden.npz, made by tests/golden/make_inspire_golden.py in a clean checkout): index tensors, finger
weights, human targets, reward outputs on the retargeted Box-200 reference and on seeded perturbations, anchor points, the
contact score, and the global RNG state after the construction. Comparisons are torch.equal / np.array_equal. For the
paper, the golden file came from the Inspire-only reward code that the paper's Inspire runs used. The file holds data
derived from ARCTIC, so it is not included; without it (or without the Inspire Box-200 data) every test here skips.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_inspire_golden.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'golden'))
import make_inspire_golden as mg  # noqa: E402
from dexmachina.asset_utils import get_asset_path  # noqa: E402

GOLDEN = ROOT / 'tests' / 'golden' / 'inspire_golden.npz'
SIDES = ('left', 'right')
DATA = [get_asset_path('retargeted/inspire_hand/s01/box_use_01_vector_para.pt'), get_asset_path('arctic/processed/s01/box_use_01.npy'),
        get_asset_path('ho_contact/human/s01/box_use_01.npz'), get_asset_path('ho_contact/grids/box_bottom.npz')]
pytestmark = pytest.mark.skipif(not GOLDEN.exists() or not all(Path(p).exists() for p in DATA),
                                reason='golden file or the Inspire Box-200 data (retargeted, processed ARCTIC, ho_contact) missing')
DEV = torch.device('cpu')


@pytest.fixture(scope='module')
def golden():
    return dict(np.load(GOLDEN))


def build(golden, pass_specs=True, cfg_update=None):
    """The CPU part of the env construction with the current code, from the golden's global RNG state: data, finger specs
    (as base_env builds them from the robot cfgs), reward module, anchor points."""
    from dexmachina.envs.finger_spec import load_finger_spec, load_robot_cfg
    from dexmachina.envs.ho_contact import robot_anchor_points
    from dexmachina.envs.rewards import RewardModule
    cfg = json.loads(str(golden['cfg']))
    cfg.update(cfg_update or {})
    mg.seed_globals()
    before = mg.rng_digest()
    demo, ret = mg.load_clip()
    specs = {side: load_finger_spec('inspire_hand', side, robot_cfg=load_robot_cfg('inspire_hand', side)) for side in SIDES}
    m = RewardModule(cfg, demo, ret, DEV, finger_specs=specs if pass_specs else None)
    anchors = {side: robot_anchor_points(mg.urdf_path(side), 'inspire_hand', side,
                                         regions=specs[side].contact_regions() if pass_specs else None) for side in SIDES}
    after = mg.rng_digest()
    return m, anchors, before, after


@pytest.fixture(scope='module')
def built(golden):
    return build(golden)


def t(golden, key, dtype=torch.float32):
    return torch.as_tensor(golden[key]) if dtype is None else torch.as_tensor(golden[key], dtype=dtype)


def test_golden_comes_from_a_clean_checkout(golden):
    assert str(golden['dirty']) == ''


def test_construction_leaves_the_global_rng_state_as_the_golden_code_did(golden, built):
    _, _, before, after = built
    assert before == str(golden['rng_before'])
    assert after == str(golden['rng_after'])            # the golden code drew nothing either: rng_before == rng_after


def test_hand_graph_tables_and_targets_are_bit_identical(golden, built):
    m = built[0]
    for side in SIDES:
        for k in ('src', 'dst', 'tips', 'palm'):
            assert torch.equal(m.hand_graph_index[side][k], t(golden, f'index_{k}_{side}', None)), (side, k)
        assert m.hand_graph_extra[side] == []
        for k in ('palm', 'fingers', 'gaps'):
            assert torch.equal(m.demo_tensors[f'hand_graph_{k}_{side}'], t(golden, f'demo_hand_graph_{k}_{side}')), (side, k)
        for k in ('ho_dist', 'ho_spot'):
            assert torch.equal(m.demo_tensors[f'{k}_{side}'], t(golden, f'demo_{k}_{side}')), (side, k)
        assert torch.equal(m.hand_graph_finger_weights[side], t(golden, 'finger_weight'))
    assert torch.equal(m.hand_graph_finger_weight, t(golden, 'finger_weight'))
    assert m.hand_graph_scale == 1.2


@pytest.mark.parametrize('case', ['ref', 'pert'])
def test_hand_graph_reward_is_bit_identical(golden, built, case):
    m = built[0]
    rew, out = m.compute_hand_graph_reward(t(golden, f'{case}_kpts_left'), t(golden, f'{case}_kpts_right'),
                                           t(golden, f'{case}_obj_quat'), t(golden, f'{case}_t', None))
    assert torch.equal(rew, t(golden, f'hg_{case}_rew'))
    keys = sorted(k[len(f'hg_{case}_out_'):] for k in golden if k.startswith(f'hg_{case}_out_'))
    assert sorted(out) == keys
    for k in keys:
        assert torch.equal(out[k], t(golden, f'hg_{case}_out_{k}')), k


def test_anchor_points_are_bit_identical(golden, built):
    anchors = built[1]
    for side in SIDES:
        a = anchors[side]
        assert np.array_equal(a['points'], golden[f'anchor_points_{side}']) and a['points'].dtype == golden[f'anchor_points_{side}'].dtype
        assert a['links'] == json.loads(str(golden[f'anchor_links_{side}']))
        assert np.array_equal(np.asarray(a['link_ranges'], np.int64), golden[f'anchor_link_ranges_{side}'])
        assert np.array_equal(np.asarray(a['slices'], np.int64), golden[f'anchor_slices_{side}'])
        assert a['anchor_links'] == json.loads(str(golden[f'anchor_anchor_links_{side}']))


def contact_reward(m, golden, anchors):
    coll = json.loads(str(golden['coll_links']))
    for side in SIDES:
        a = anchors[side]
        m.setup_ho_contact(side, a['slices'], [[coll[side].index(n) for n in links] for links in a['anchor_links']])
    inputs = dict(points_left=t(golden, 'hoc_points_left'), points_right=t(golden, 'hoc_points_right'),
                  part_pos=t(golden, 'hoc_part_pos'), part_quat=t(golden, 'hoc_part_quat'))
    return m.compute_ho_contact_reward(inputs, t(golden, 'hoc_pos_left'), t(golden, 'hoc_valid_left', None),
                                       t(golden, 'hoc_pos_right'), t(golden, 'hoc_valid_right', None), t(golden, 'hoc_t', None))


def test_contact_reward_and_score_are_bit_identical(golden, built):
    m, anchors = built[0], built[1]
    rew, out = contact_reward(m, golden, anchors)
    assert torch.equal(rew, t(golden, 'hoc_rew'))
    keys = sorted(k[len('hoc_out_'):] for k in golden if k.startswith('hoc_out_'))
    assert sorted(out) == keys
    for k in keys:
        assert torch.equal(out[k], t(golden, f'hoc_out_{k}')), k
    for side in SIDES:
        tt = t(golden, 'hoc_t', None)
        hand, det = m.ho_contact.score(t(golden, f'score_dr_{side}'), t(golden, f'score_sr_{side}'),
                                       m.match_demo_state(f'ho_dist_{side}', tt), m.match_demo_state(f'ho_spot_{side}', tt))
        assert torch.equal(hand, t(golden, f'score_hand_{side}'))
        for k, v in det.items():
            assert torch.equal(v, t(golden, f'score_{k}_{side}')), (side, k)


def test_without_specs_or_scale_the_module_resolves_the_same(golden):
    """RewardModule built without the env's specs (eval tools) and with hand_graph_scale None (taken from the retargeting
    config) gives the same bits; so does the hand name under its new key."""
    m, anchors, _, after = build(golden, pass_specs=False, cfg_update=dict(hand_graph_scale=None, hand_name='inspire_hand'))
    assert after == str(golden['rng_after'])
    assert m.hand_graph_scale == 1.2
    for side in SIDES:
        assert torch.equal(m.demo_tensors[f'hand_graph_gaps_{side}'], t(golden, f'demo_hand_graph_gaps_{side}'))
        assert np.array_equal(anchors[side]['points'], golden[f'anchor_points_{side}'])
    rew, _ = m.compute_hand_graph_reward(t(golden, 'pert_kpts_left'), t(golden, 'pert_kpts_right'), t(golden, 'pert_obj_quat'),
                                         t(golden, 'pert_t', None))
    assert torch.equal(rew, t(golden, 'hg_pert_rew'))
    rew, _ = contact_reward(m, golden, anchors)
    assert torch.equal(rew, t(golden, 'hoc_rew'))
