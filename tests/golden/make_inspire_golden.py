"""Golden values of the Inspire hand-graph and hand-object contact rewards, written from the code of the checkout it runs
in; tests/test_inspire_golden.py then checks the current code against the file bit for bit (torch.equal / np.array_equal).
For the paper, it ran once, on CPU, in a clean checkout of the Inspire-only reward code that the paper's Inspire runs used
(before the general finger correspondence), which is not part of this repository. Run in a clean checkout of this
repository, it gives a reference for checking later changes.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=<checkout> python tests/golden/make_inspire_golden.py

Real data: the retargeted Inspire Box-200 clip (box-30-230-s01-u01: reference keypoints, human MANO joints, human contact
targets, the box distance grids) and the Inspire URDFs. The file therefore holds data derived from ARCTIC and is not
included in this repository; keep it local. Seeded inputs: perturbed reference keypoints with random object orientations
(hand graph), and random anchor points, object part poses and simulator contacts around the box (contact). Only APIs
that the Inspire-only reward code already had are used here.
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CLIP = dict(obj_name='box', frame_start=30, frame_end=230, subject_name='s01', use_clip='01')
SIDES = ('left', 'right')
GLOBAL_SEED = 1234          # global RNG state before the construction (RNG check)
INPUT_SEED = 20261019       # local generator for the seeded inputs
N_GRAPH, N_CONTACT = 256, 48


def reward_cfg():
    """DexDuet's reward settings of scripts/train_coordinates_hand.py (Box-200) as constructors.get_all_env_cfg writes them."""
    from dexmachina.envs.rewards import get_reward_cfg
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.2, imi_wrist_weight=0.5, contact_rew_weight=2.0, contact_rew_function='exp', contact_beta=10.0,
               wrist_frame_contact=True, mask_zero_contact=True, use_retarget_contact=True, retarget_objframe=True,
               action_penalty=0.01, ref_motion_info=False, cg_gate=0.0, cg_extra=0.25, hand_graph_weight=0.5,
               hand_graph_beta_dir=5.0, hand_graph_beta_gap=20.0, hand_graph_thumb=0.5, hand_graph_scale=1.2,
               ho_contact_weight=1.0, ho_contact_cap=0.05, ho_contact_tol=0.01, ho_contact_scale=0.02, ho_contact_touch=0.01,
               ho_contact_near=0.5, ho_contact_hand='inspire_hand', obj_pos_beta=10.0, obj_rot_beta=1.0, obj_arti_beta=5.0,
               bc_rew_weight=0.2, bc_beta=500.0, objdex_baseline=False, multiply_all_rew=False, ho_contact_object='box')
    return cfg


def urdf_path(side):
    from dexmachina.envs.hand_cfgs.inspire import INSPIRE_CFGs
    return str(INSPIRE_CFGs[side]['urdf_path'])


def collision_links(path):
    """Links with a collision element, in URDF order (the order of BaseRobot.coll_link_names for this hand)."""
    return [l.get('name') for l in ET.parse(path).getroot().findall('link') if l.find('collision') is not None]


def rng_digest():
    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    st = np.random.get_state()
    h.update(st[0].encode())
    h.update(np.asarray(st[1]).tobytes())
    h.update(repr(st[2:]).encode())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def seed_globals():
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)


def cpu_torch_load():
    """The retargeted .pt files hold CUDA tensors; load them on the CPU (same values)."""
    if getattr(torch.load, '_cpu_only', False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return original(*args, **kwargs)
    load._cpu_only = True
    torch.load = load


def load_clip():
    from dexmachina.asset_utils import get_asset_path
    cpu_torch_load()
    from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
    from dexmachina.envs.ho_contact import load_human_targets
    c = CLIP
    _, ret = load_genesis_retarget_data(obj_name=c['obj_name'], hand_name='inspire_hand', frame_start=c['frame_start'],
                                        frame_end=c['frame_end'], save_name='para', use_clip=c['use_clip'],
                                        subject_name=c['subject_name'])
    demo = get_demo_data(obj_name=c['obj_name'], hand_name='inspire_hand', frame_start=c['frame_start'], frame_end=c['frame_end'],
                         use_clip=c['use_clip'], subject_name=c['subject_name'], load_retarget_contact=True, load_hand_joints=True)
    demo.update(load_human_targets(str(get_asset_path('')), c['subject_name'], c['obj_name'], c['use_clip'], c['frame_start'],
                                   c['frame_end']))
    return demo, ret


def as_t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def graph_inputs(rng, demo, ret, kpt_names):
    """Reference keypoints at every frame, then N_GRAPH seeded perturbations (hand rotated about its wrist, moved, noised;
    object orientation either the demo's or random)."""
    T = len(demo['obj_quat'])
    out = dict()
    for side in SIDES:
        out[f'ref_kpts_{side}'] = np.asarray(ret[side]['kpts_data']['kpt_pos'], np.float32)
    out['ref_obj_quat'] = np.asarray(demo['obj_quat'], np.float32)
    out['ref_t'] = np.arange(T, dtype=np.int64)
    t = rng.integers(0, T, N_GRAPH)
    out['pert_t'] = t.astype(np.int64)
    rq = unit_quats(rng, N_GRAPH)
    angle = rng.uniform(0.0, 1.0, N_GRAPH)                          # mix small and large rotations
    rq[:, 1:] *= angle[:, None]
    rq /= np.linalg.norm(rq, axis=-1, keepdims=True)
    R = quat_to_mat(rq).astype(np.float32)
    for side in SIDES:
        ref = out[f'ref_kpts_{side}'][t]                             # (N, K, 3)
        w = ref[:, kpt_names[side].index('base_link')][:, None]
        k = np.einsum('nij,nkj->nki', R, ref - w) * rng.uniform(0.9, 1.1, (N_GRAPH, 1, 1)) + w
        k = k + rng.normal(scale=0.003, size=k.shape) + rng.uniform(-0.2, 0.2, (N_GRAPH, 1, 3))
        out[f'pert_kpts_{side}'] = k.astype(np.float32)
    q = np.asarray(demo['obj_quat'], np.float32)[t]
    rand = unit_quats(rng, N_GRAPH)
    out['pert_obj_quat'] = np.where((rng.uniform(size=N_GRAPH) < 0.5)[:, None], q, rand).astype(np.float32)
    return out


def contact_inputs(rng, demo, n_points, n_coll):
    T = len(demo['obj_pos'])
    t = rng.integers(0, T, N_CONTACT)
    out = dict(hoc_t=t.astype(np.int64))
    centre = np.asarray(demo['obj_pos'], np.float32)[t]                                     # (N, 3)
    part_pos = centre[:, None] + rng.normal(scale=0.01, size=(N_CONTACT, 2, 3))
    base = np.asarray(demo['obj_quat'], np.float32)[t]
    part_quat = np.stack([base, base + rng.normal(scale=0.2, size=base.shape)], 1)
    part_quat /= np.linalg.norm(part_quat, axis=-1, keepdims=True)
    out['hoc_part_pos'] = part_pos.astype(np.float32)
    out['hoc_part_quat'] = part_quat.astype(np.float32)
    for side in SIDES:
        out[f'hoc_points_{side}'] = (centre[:, None] + rng.uniform(-0.12, 0.12, (N_CONTACT, n_points[side], 3))).astype(np.float32)
        out[f'hoc_valid_{side}'] = rng.uniform(size=(N_CONTACT, 2, n_coll[side])) < 0.08
        out[f'hoc_pos_{side}'] = (centre[:, None, None] + rng.uniform(-0.06, 0.06, (N_CONTACT, 2, n_coll[side], 3))).astype(np.float32)
        # score(): robot distances / spots against the human targets of the same frames
        out[f'score_dr_{side}'] = rng.uniform(0.0, 0.08, (N_CONTACT, 6, 2)).astype(np.float32)
        out[f'score_sr_{side}'] = rng.uniform(-0.08, 0.08, (N_CONTACT, 6, 2, 3)).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'inspire_golden.npz'))
    args = ap.parse_args()
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    from dexmachina.envs.ho_contact import HandObjectContact, robot_anchor_points
    from dexmachina.envs.rewards import RewardModule
    dev = torch.device('cpu')
    cfg = reward_cfg()

    # construction (data, reward module, anchor points) from a seeded global RNG state
    seed_globals()
    rng_before = rng_digest()
    demo, ret = load_clip()
    m = RewardModule(cfg, demo, ret, dev)
    anchors = {side: robot_anchor_points(urdf_path(side), 'inspire_hand', side) for side in SIDES}
    rng_after = rng_digest()

    g = dict(commit=np.array(head), dirty=np.array(dirty), cfg=np.array(json.dumps(cfg, sort_keys=True)),
             rng_before=np.array(rng_before), rng_after=np.array(rng_after), clip=np.array(json.dumps(CLIP)))
    kpt_names = {s: [str(n) for n in ret[s]['kpts_data']['kpt_names']] for s in SIDES}
    g['kpt_names'] = np.array(json.dumps(kpt_names))
    for side in SIDES:
        for k, v in m.hand_graph_index[side].items():
            g[f'index_{k}_{side}'] = v.numpy()
        for k in ('palm', 'fingers', 'gaps'):
            g[f'demo_hand_graph_{k}_{side}'] = m.demo_tensors[f'hand_graph_{k}_{side}'].numpy()
        for k in ('ho_dist', 'ho_spot'):
            g[f'demo_{k}_{side}'] = m.demo_tensors[f'{k}_{side}'].numpy()
    g['finger_weight'] = m.hand_graph_finger_weight.numpy()

    # hand graph on the reference and on seeded perturbations
    rng = np.random.default_rng(INPUT_SEED)
    gi = graph_inputs(rng, demo, ret, kpt_names)
    g.update(gi)
    for case in ('ref', 'pert'):
        rew, out = m.compute_hand_graph_reward(as_t(gi[f'{case}_kpts_left']), as_t(gi[f'{case}_kpts_right']),
                                               as_t(gi[f'{case}_obj_quat']), torch.as_tensor(gi[f'{case}_t']))
        g[f'hg_{case}_rew'] = rew.numpy()
        for k, v in out.items():
            g[f'hg_{case}_out_{k}'] = v.numpy()

    # contact: anchor points, the reward module's contact term and the score on seeded inputs
    coll = {side: collision_links(urdf_path(side)) for side in SIDES}
    g['coll_links'] = np.array(json.dumps(coll))
    for side in SIDES:
        a = anchors[side]
        g[f'anchor_points_{side}'] = a['points']
        g[f'anchor_links_{side}'] = np.array(json.dumps(a['links']))
        g[f'anchor_link_ranges_{side}'] = np.asarray(a['link_ranges'], np.int64)
        g[f'anchor_slices_{side}'] = np.asarray(a['slices'], np.int64)
        g[f'anchor_anchor_links_{side}'] = np.array(json.dumps(a['anchor_links']))
        m.setup_ho_contact(side, a['slices'], [[coll[side].index(n) for n in links] for links in a['anchor_links']])
    ci = contact_inputs(rng, demo, {s: len(anchors[s]['points']) for s in SIDES}, {s: len(coll[s]) for s in SIDES})
    g.update(ci)
    inputs = dict(points_left=as_t(ci['hoc_points_left']), points_right=as_t(ci['hoc_points_right']),
                  part_pos=as_t(ci['hoc_part_pos']), part_quat=as_t(ci['hoc_part_quat']))
    rew, out = m.compute_ho_contact_reward(inputs, as_t(ci['hoc_pos_left']), torch.as_tensor(ci['hoc_valid_left']),
                                           as_t(ci['hoc_pos_right']), torch.as_tensor(ci['hoc_valid_right']),
                                           torch.as_tensor(ci['hoc_t']))
    g['hoc_rew'] = rew.numpy()
    for k, v in out.items():
        g[f'hoc_out_{k}'] = v.numpy()
    hoc = m.ho_contact
    assert isinstance(hoc, HandObjectContact)
    for side in SIDES:
        d_h = m.match_demo_state(f'ho_dist_{side}', torch.as_tensor(ci['hoc_t']))
        s_h = m.match_demo_state(f'ho_spot_{side}', torch.as_tensor(ci['hoc_t']))
        hand, det = hoc.score(as_t(ci[f'score_dr_{side}']), as_t(ci[f'score_sr_{side}']), d_h, s_h)
        g[f'score_hand_{side}'] = hand.numpy()
        for k, v in det.items():
            g[f'score_{k}_{side}'] = v.numpy()
    np.savez_compressed(args.out, **g)
    print(f'wrote {args.out}: commit {head} dirty={bool(dirty)} rng unchanged by construction: {rng_before == rng_after}; '
          f'hg_ref_rew mean {g["hg_ref_rew"].mean():.6f} hg_pert_rew mean {g["hg_pert_rew"].mean():.6f} '
          f'hoc_rew mean {g["hoc_rew"].mean():.6f}')


if __name__ == '__main__':
    main()
