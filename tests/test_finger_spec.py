"""Finger correspondence per hand (dexmachina/envs/finger_spec.py), resolved from each hand's retarget_config.yaml, hand cfg and
URDF, for all seven hand cfgs. The Inspire spec must equal the tables that the Inspire-only reward code of the paper's
Inspire runs hard-coded.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_finger_spec.py
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.envs import finger_spec as fs  # noqa: E402
from dexmachina.envs.finger_spec import FINGERS, hand_graph_nodes, load_finger_spec, load_robot_cfg  # noqa: E402
from dexmachina.envs.rewards import FINGER_PAIRS, HAND_GRAPH_FINGERS, HAND_GRAPH_NODES, MANO_TIP_JOINTS  # noqa: E402

SIDES = ('left', 'right')
HANDS = ('inspire_hand', 'xhand', 'schunk_hand', 'allegro_hand', 'dex3_hand', 'ability_hand', 'dexrobot_hand')
# ho_contact.ROBOT_ANCHOR_LINKS of the Inspire-only code ({p} = L / R)
LEGACY_INSPIRE_ANCHOR_LINKS = {'palm': ('base_link',), 'thumb': ('{p}_TH_L3', '{p}_TH_L4'), 'index': ('{p}_FF_L2',),
                               'middle': ('{p}_MF_L2',), 'ring': ('{p}_RF_L2',), 'pinky': ('{p}_LF_L2',)}
LEGACY_ANCHORS = ('palm', 'thumb', 'index', 'middle', 'ring', 'pinky')
FIVE_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4))


def expected(hand, side):
    """The resolved tables, written out by hand from the URDFs and retarget configs (design tables, with these
    corrections: Allegro wrist base_dummy_link, Schunk palm e1 + e2, Dex3 thumb thumb_1 + thumb_2)."""
    s = side
    if hand == 'inspire_hand':
        p = 'L' if side == 'left' else 'R'
        f = ('TH', 'FF', 'MF', 'RF', 'LF')
        return dict(fingers=FINGERS, wrist='base_link', bases=tuple(f'{p}_{x}_L1' for x in f), tips=tuple(f'{p}_{x}_Tip' for x in f),
                    scale=1.2, palm=('base_link',), contact=((f'{p}_TH_L3', f'{p}_TH_L4'),) + tuple((f'{p}_{x}_L2',) for x in f[1:]))
    if hand == 'xhand':
        h = f'{s}_hand'
        return dict(fingers=FINGERS, wrist=f'{h}_link', scale=1.4, palm=(f'{h}_link',),
                    bases=(f'{h}_thumb_bend_link', f'{h}_index_bend_link', f'{h}_mid_link1', f'{h}_ring_link1', f'{h}_pinky_link1'),
                    tips=(f'{h}_thumb_rota_tip', f'{h}_index_rota_tip', f'{h}_mid_tip', f'{h}_ring_tip', f'{h}_pinky_tip'),
                    contact=((f'{h}_thumb_rota_link1', f'{h}_thumb_rota_link2'), (f'{h}_index_rota_link2',), (f'{h}_mid_link2',),
                             (f'{h}_ring_link2',), (f'{h}_pinky_link2',)))
    if hand == 'schunk_hand':
        h = f'{s}_hand'
        return dict(fingers=FINGERS, wrist=f'{h}_base_link', scale=1.3, palm=(f'{h}_e1', f'{h}_e2'),
                    bases=(f'{h}_z', f'{h}_virtual_l', f'{h}_virtual_k', f'{h}_virtual_j', f'{h}_virtual_i'),
                    tips=('thtip', 'fftip', 'mftip', 'rftip', 'lftip'),
                    contact=tuple((f'{h}_{a}', f'{h}_{b}') for a, b in (('b', 'c'), ('p', 't'), ('o', 's'), ('n', 'r'), ('m', 'q'))))
    if hand == 'allegro_hand':
        index, ring = (8, 0) if side == 'left' else (0, 8)          # index and ring are mirrored between the sides
        fingers = (12, index, 4, ring)
        return dict(fingers=('thumb', 'index', 'middle', 'ring'), wrist='base_dummy_link', scale=1.7, palm=('base_link',),
                    bases=tuple(f'link_{b}.0' for b in fingers), tips=tuple(f'link_{b + 3}.0_tip' for b in fingers),
                    contact=tuple((f'link_{b + 2}.0', f'link_{b + 3}.0', f'link_{b + 3}.0_tip') for b in fingers))
    if hand == 'dex3_hand':
        h = f'{s}_hand'
        return dict(fingers=('thumb', 'index', 'middle'), wrist=f'{h}_palm_link', scale=1.5, palm=(f'{h}_palm_link',),
                    bases=(f'{h}_thumb_0_link', f'{h}_index_0_link', f'{h}_middle_0_link'),
                    tips=(f'{s}_thumb_tip', f'{s}_index_tip', f'{s}_middle_tip'),
                    contact=((f'{h}_thumb_1_link', f'{h}_thumb_2_link'), (f'{h}_index_1_link',), (f'{h}_middle_1_link',)))
    if hand == 'ability_hand':
        n = ('thumb', 'index', 'middle', 'ring', 'pinky')
        return dict(fingers=FINGERS, wrist='base', scale=1.0, palm=('thumb_base',), bases=tuple(f'{x}_L1' for x in n),
                    tips=tuple(f'{x}_tip' for x in n), contact=(('thumb_L1', 'thumb_L2'),) + tuple((f'{x}_L2',) for x in n[1:]))
    if hand == 'dexrobot_hand':
        x = 'l' if side == 'left' else 'r'
        return dict(fingers=FINGERS, wrist='base_dummy', scale=1.2, palm=(f'{s}_hand_base',),
                    bases=tuple(f'{x}_f_link{k}_1' for k in range(1, 6)), tips=tuple(f'{x}_f_link{k}_tip' for k in range(1, 6)),
                    contact=tuple((f'{x}_f_link{k}_3', f'{x}_f_link{k}_4', f'{x}_f_link{k}_tip') for k in range(1, 6)))
    raise KeyError(hand)


@pytest.mark.parametrize('side', SIDES)
@pytest.mark.parametrize('hand', HANDS)
def test_spec_matches_the_tables(hand, side):
    spec = load_finger_spec(hand, side)
    e = expected(hand, side)
    assert spec.fingers == e['fingers']
    assert spec.missing == tuple(f for f in FINGERS if f not in e['fingers'])
    assert spec.wrist == e['wrist']
    assert spec.bases == e['bases']
    assert spec.tips == e['tips']
    assert spec.scale == e['scale']
    assert spec.contact_links['palm'] == e['palm']
    assert tuple(spec.contact_links[f] for f in spec.fingers) == e['contact']
    assert spec.regions == ('palm',) + e['fingers']
    for links in spec.contact_links.values():                         # base_env indexes them in robot.coll_link_names
        assert set(links) <= set(spec.collision_links)
    n = len(spec.fingers)
    assert spec.gap_pairs == tuple((0, j) for j in range(1, n)) + tuple((j, j + 1) for j in range(1, n - 1))
    assert spec.outermost == e['fingers'][-1]
    assert spec.forward_to_middle == (n > 3)
    assert spec.human_bases == tuple({'thumb': 13, 'index': 1, 'middle': 4, 'ring': 10, 'pinky': 7}[f] for f in spec.fingers)
    assert spec.human_tips == tuple({'thumb': 16, 'index': 17, 'middle': 18, 'ring': 19, 'pinky': 20}[f] for f in spec.fingers)
    assert all(spec.chains[f][0] == b and spec.chains[f][-1] == t for f, b, t in zip(spec.fingers, spec.bases, spec.tips))


def test_counts_pairs_outermost_and_weights_per_hand():
    table = {'inspire_hand': (5, 7, 'pinky', 1.2), 'xhand': (5, 7, 'pinky', 1.4), 'schunk_hand': (5, 7, 'pinky', 1.3),
             'allegro_hand': (4, 5, 'ring', 1.7), 'dex3_hand': (3, 3, 'middle', 1.5)}
    for hand, (fingers, pairs, outer, scale) in table.items():
        spec = load_finger_spec(hand, 'right')
        assert (len(spec.fingers), len(spec.gap_pairs), spec.outermost, spec.scale) == (fingers, pairs, outer, scale)
        w = torch.tensor(spec.finger_weights(0.5), dtype=torch.float32)
        w = w / w.sum()
        assert float(w[0]) == pytest.approx(0.5 / (fingers - 0.5)) and float(w[1]) == pytest.approx(1.0 / (fingers - 0.5))
    assert load_finger_spec('allegro_hand', 'left').gap_pairs == ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3))
    assert load_finger_spec('dex3_hand', 'left').gap_pairs == ((0, 1), (0, 2), (1, 2))


@pytest.mark.parametrize('side', SIDES)
def test_inspire_spec_equals_the_legacy_constants(side):
    """The tables of rewards.py / ho_contact.py in the Inspire-only code: the same names in the same order."""
    p = 'L' if side == 'left' else 'R'
    spec = load_finger_spec('inspire_hand', side)
    src, dst = zip(*HAND_GRAPH_FINGERS)
    assert spec.bases == tuple(f'{p}_{n}' for n in src) and spec.tips == tuple(f'{p}_{n}' for n in dst)
    assert spec.palm_nodes == ('base_link', f'{p}_FF_L1', f'{p}_MF_L1', f'{p}_LF_L1')
    assert spec.human_palm_nodes == tuple(HAND_GRAPH_NODES[n] for n in ('wrist', 'FF_L1', 'MF_L1', 'LF_L1'))
    assert spec.human_bases == tuple(HAND_GRAPH_NODES[n] for n in src)
    assert spec.human_tips == tuple(HAND_GRAPH_NODES[n] for n in dst) == MANO_TIP_JOINTS
    assert spec.gap_pairs == FINGER_PAIRS == FIVE_PAIRS
    assert spec.forward_to_middle
    assert spec.regions == LEGACY_ANCHORS
    assert [(r, l) for r, l in spec.contact_regions()] == [
        (a, tuple(n.format(p=p) for n in LEGACY_INSPIRE_ANCHOR_LINKS[a])) for a in LEGACY_ANCHORS]
    # finger weights: the float32 tensor the Inspire-only code builds from HAND_GRAPH_FINGERS, bit for bit
    legacy = torch.tensor([0.5 if e[0] == 'TH_L1' else 1.0 for e in HAND_GRAPH_FINGERS], dtype=torch.float32)
    new = torch.tensor(spec.finger_weights(0.5), dtype=torch.float32)
    assert torch.equal(new / new.sum(), legacy / legacy.sum())
    assert spec.scale == 1.2                                           # the same double as the old CLI default


def test_inspire_node_lookup_equals_the_suffix_lookup_on_the_retargeted_names():
    """The Inspire-only code looked nodes up by the literal 'base_link' and by name suffix; the spec looks them up by full
    name."""
    for side, p in (('left', 'L'), ('right', 'R')):
        names = [f'{p}_{f}_Tip' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + ['base_link'] + \
                [f'{p}_{f}_L1' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + [f'{p}_{f}_L2' for f in ('TH', 'FF', 'MF', 'RF', 'LF')] + \
                [f'{p}_TH_L3', f'{p}_TH_L4']
        find = lambda n: names.index('base_link') if n == 'wrist' else next(i for i, m in enumerate(names) if m.endswith(f'_{n}'))
        src, dst = zip(*HAND_GRAPH_FINGERS)
        index, extra = hand_graph_nodes(load_finger_spec('inspire_hand', side), names)
        assert extra == []
        assert index == dict(src=[find(n) for n in src], dst=[find(n) for n in dst],
                             tips=[find(f'{f}_Tip') for f in ('TH', 'FF', 'MF', 'RF', 'LF')],
                             palm=[find(n) for n in ('wrist', 'FF_L1', 'MF_L1', 'LF_L1')])


def retargeted_names(hand, side, kpt_link_names=None):
    """What retargeting records as kpt_names: the cfg's kpt_link_names, then every collision link (robot.py)."""
    spec = load_finger_spec(hand, side)
    names = list(load_robot_cfg(hand, side)['kpt_link_names'] if kpt_link_names is None else kpt_link_names)
    return names + list(spec.collision_links)


def test_extra_nodes_per_hand():
    for hand in ('inspire_hand', 'xhand', 'schunk_hand', 'ability_hand'):
        for side in SIDES:
            assert hand_graph_nodes(load_finger_spec(hand, side), retargeted_names(hand, side))[1] == []
    for side in SIDES:
        # Allegro: the wrist (base_dummy_link, the retargeting origin) has no collision and is no keypoint
        index, extra = hand_graph_nodes(load_finger_spec('allegro_hand', side), retargeted_names('allegro_hand', side))
        n = len(retargeted_names('allegro_hand', side))
        assert extra == ['base_dummy_link'] and index['palm'][0] == n
        # DexRobot likewise (base_dummy)
        assert hand_graph_nodes(load_finger_spec('dexrobot_hand', side), retargeted_names('dexrobot_hand', side))[1] == ['base_dummy']
        # Dex3: with the fingertips now in kpt_link_names nothing is extra; with the old list the three tips are
        spec = load_finger_spec('dex3_hand', side)
        assert hand_graph_nodes(spec, retargeted_names('dex3_hand', side))[1] == []
        old = [n for n in load_robot_cfg('dex3_hand', side)['kpt_link_names'] if not n.endswith('_tip')]
        assert len(old) == 8
        index, extra = hand_graph_nodes(spec, retargeted_names('dex3_hand', side, old))
        assert extra == list(spec.tips)
        k = len(retargeted_names('dex3_hand', side, old))
        assert index['tips'] == [k, k + 1, k + 2] and index['dst'] == index['tips']


def test_dex3_cfg_lists_the_fingertips_last():
    for side in SIDES:
        names = load_robot_cfg('dex3_hand', side)['kpt_link_names']
        assert names[-3:] == [f'{side}_thumb_tip', f'{side}_index_tip', f'{side}_middle_tip'] and len(names) == 11


def test_errors_name_the_hand_side_and_link(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='no_such_hand'):
        load_finger_spec('no_such_hand', 'left')
    # a fingertip missing from the URDF
    cfg = dict(load_robot_cfg('inspire_hand', 'left'))
    text = Path(cfg['urdf_path']).read_text()
    start = text.index('<link name="L_FF_Tip"')
    end = text.index('</link>', start) + len('</link>')
    broken = tmp_path / 'left.urdf'
    broken.write_text(text[:start] + text[end:])
    cfg['urdf_path'] = str(broken)
    with pytest.raises(ValueError, match=r"inspire_hand.*left.*L_FF_Tip"):
        load_finger_spec('inspire_hand', 'left', robot_cfg=cfg)
    # a wrist that is not the retargeting origin
    cfg = dict(load_robot_cfg('allegro_hand', 'right'), wrist_link_name='base_link')
    with pytest.raises(ValueError, match=r"allegro_hand.*right.*base_link"):
        load_finger_spec('allegro_hand', 'right', robot_cfg=cfg)
    # a hand without a middle finger
    original = fs.load_retarget_config

    def no_middle(hand):
        rc = original(hand)
        for side in SIDES:
            keep = [i for i, idx in enumerate(rc[side]['human_finger_idxs']) if idx != 18]
            rc[side]['human_finger_idxs'] = [rc[side]['human_finger_idxs'][i] for i in keep]
            rc[side]['target_finger_links'] = [rc[side]['target_finger_links'][i] for i in keep]
        return rc
    monkeypatch.setattr(fs, 'load_retarget_config', no_middle)
    with pytest.raises(ValueError, match=r"dex3_hand.*left.*middle"):
        load_finger_spec('dex3_hand', 'left')
    monkeypatch.setattr(fs, 'load_retarget_config', original)
    # a contact region deeper than the finger's collision links
    monkeypatch.setitem(fs.CONTACT_DEPTH, 'inspire_hand', (2, 3))
    with pytest.raises(ValueError, match=r"inspire_hand.*right.*index"):
        load_finger_spec('inspire_hand', 'right')


def test_hand_graph_scale_resolution():
    assert fs.resolve_hand_graph_scale('inspire_hand') == 1.2 and fs.resolve_hand_graph_scale('allegro_hand') == 1.7
    assert fs.resolve_hand_graph_scale('dex3_hand') == 1.5 and fs.resolve_hand_graph_scale('xhand') == 1.4
    assert fs.resolve_hand_graph_scale('schunk_hand') == 1.3
    assert fs.resolve_hand_graph_scale('allegro_hand', 1.25) == 1.25        # an explicit value wins
    assert isinstance(fs.resolve_hand_graph_scale('inspire_hand', None), float)
    assert fs.resolve_hand_graph_scale('no_such_hand', None, needed=False) == 1.2
    with pytest.raises(ValueError, match='no_such_hand'):
        fs.resolve_hand_graph_scale('no_such_hand', None, needed=True)
