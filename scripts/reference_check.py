"""Hand-graph reference check of a retargeted clip (CPU): how well the retargeted robot reference follows the human hand in the
terms of the hand-graph reward, with the reward's own finger correspondence and palm frame (RewardModule +
dexmachina/envs/finger_spec.py).

Per frame and side: the angle between the robot's and the human's palm normal, palm forward direction and each finger edge
(base to tip), and the fingertip-gap error |robot gap - scale * human gap| per pair. Prints median / p90 / max per side and
flags an implausible map: a median angle above 90 degrees (a mirrored finger map or a flipped normal sits near 180 degrees)
or a mean gap error above 3 cm. The retargeting matches fingertip positions, not directions, so angles of 10-40 degrees are
normal (Inspire's own reference: palm 12 / 21 degrees median, finger edges up to ~43 degrees, gap error ~2 cm).

The palm normal follows finger_spec.palm_normal_mode: on hands with four or more fingers the finger-base normal on both hands;
on hands with fewer (Dex3) the human index x little-base normal against R_wrist a, with the wrist-link axis a calibrated on
this same reference (rewards.calibrate_palm_axis), as the reward computes it in training.

When the wrist is not a retargeted keypoint (Allegro: base_dummy_link) it is taken from the retargeted wrist_pose, as the env
takes it from robot.wrist_pose; the distance from that wrist to each keypoint is printed as a rigid-offset sanity check.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/reference_check.py --hand allegro_hand [--clip box-30-230-s01-u01] [--json f]
Exit 0 if no side is flagged, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def stats(x):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    return dict(median=round(x.median().item(), 3), p90=round(torch.quantile(x, 0.9).item(), 3), max=round(x.max().item(), 3))


def load(hand, clip='box-30-230-s01-u01', save_name='para'):
    """The clip's retargeted reference and human demo (CPU), and a RewardModule with only the hand graph on."""
    from dexmachina.envs.constructors import parse_clip_string
    from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
    from dexmachina.envs.rewards import RewardModule, get_reward_cfg
    original_load = torch.load
    torch.load = lambda *x, **k: original_load(*x, **{'map_location': 'cpu', **k})   # the .pt files hold CUDA tensors
    try:
        obj, start, end, subject, use_clip = parse_clip_string(clip)
        kw = dict(obj_name=obj, frame_start=start, frame_end=end, subject_name=subject, use_clip=use_clip)
        _, ret = load_genesis_retarget_data(hand_name=hand, save_name=save_name, **kw)
        demo = get_demo_data(hand_name=hand, load_hand_joints=True, **kw)
    finally:
        torch.load = original_load
    cfg = get_reward_cfg()
    cfg.update(imi_rew_weight=0.0, contact_rew_weight=0.0, bc_rew_weight=0.0, hand_graph_weight=1.0, hand_name=hand)
    return RewardModule(cfg, demo, ret, torch.device('cpu')), demo, ret


def reference_angles(m, demo, ret, side):
    """Per frame, the retargeted reference against the human hand with the reward module's own nodes and palm frame:
    palm normal / forward and finger-edge angles (degrees), fingertip-gap errors (cm). Also the wrist-to-keypoint distances
    when the wrist comes from wrist_pose (None otherwise)."""
    from dexmachina.envs.math_utils import matrix_from_quat
    q = torch.as_tensor(np.asarray(demo['obj_quat']), dtype=torch.float32)
    rot = matrix_from_quat(q / q.norm(dim=-1, keepdim=True))
    unit = lambda v: v / v.norm(dim=-1, keepdim=True)
    angle = lambda u, v: torch.rad2deg(torch.acos((u * v).sum(-1).clamp(-1.0, 1.0)))
    to_world = lambda key: torch.einsum('nij,nej->nei', rot, m.demo_tensors[key])
    spec = m.finger_specs[side]
    names = [str(n) for n in ret[side]['kpts_data']['kpt_names']]
    k = torch.as_tensor(np.asarray(ret[side]['kpts_data']['kpt_pos']), dtype=torch.float32)
    wrist_pose = torch.as_tensor(np.asarray(ret[side]['wrist_pose']), dtype=torch.float32)
    extra = m.hand_graph_extra[side]
    wrist_info = None
    if extra:
        if extra != [spec.wrist]:
            raise ValueError(f'{spec.hand} {side}: hand-graph nodes {extra} were not recorded by the retargeting')
        wrist = wrist_pose[:, :3]
        d = (k - wrist[:, None]).norm(dim=-1)                                   # (T, K) wrist to each keypoint
        wrist_info = {n: dict(mean_m=round(d[:, i].mean().item(), 4), std_m=round(d[:, i].std().item(), 5))
                      for i, n in enumerate(names) if n in (spec.bases + ('base_link',))}
        k = torch.cat([k, wrist[:, None]], dim=1)
    idx = m.hand_graph_index[side]
    palm = m.hand_graph_palm_frame(side, k, wrist_pose)
    human = to_world(f'hand_graph_palm_{side}')
    pa, pb = m.hand_graph_pairs[side]
    tips = k[:, idx['tips']]
    return dict(palm=angle(palm[:, 0], human[:, 0]), forward=angle(palm[:, 1], human[:, 1]),
                fingers=angle(unit(k[:, idx['dst']] - k[:, idx['src']]), to_world(f'hand_graph_fingers_{side}')),   # (T, F)
                gap_cm=((tips[:, pa] - tips[:, pb]).norm(dim=-1) - m.demo_tensors[f'hand_graph_gaps_{side}']).abs() * 100.0,
                wrist_from_wrist_pose=bool(extra), wrist_info=wrist_info)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hand', required=True)
    ap.add_argument('--clip', default='box-30-230-s01-u01')
    ap.add_argument('--save-name', default='para')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    m, demo, ret = load(a.hand, a.clip, a.save_name)
    report, flagged = dict(hand=a.hand, clip=a.clip, scale=m.hand_graph_scale), []
    for side in ('left', 'right'):
        spec = m.finger_specs[side]
        try:
            x = reference_angles(m, demo, ret, side)
        except ValueError as e:
            sys.exit(str(e))
        axis = m.hand_graph_palm_axis.get(side)
        mode = spec.palm_normal_mode
        normal = ('robot R_wrist a (calibrated wrist-link axis) vs human index x little base' if mode == 'calibrated_axis'
                  else f'index x {spec.outermost} base on both hands')
        r = dict(fingers=list(spec.fingers), missing=list(spec.missing), wrist=spec.wrist,
                 wrist_from_wrist_pose=x['wrist_from_wrist_pose'], bases=list(spec.bases), tips=list(spec.tips),
                 outermost=spec.outermost, palm_normal_mode=mode, palm_normal=normal,
                 palm_axis=None if axis is None else [round(v, 6) for v in axis.tolist()],
                 forward='wrist->middle base' if spec.forward_to_middle else 'wrist->mid(index, middle bases)',
                 frames=int(x['palm'].shape[0]), palm_normal_deg=stats(x['palm']), palm_forward_deg=stats(x['forward']),
                 finger_edge_deg={f: stats(x['fingers'][:, j]) for j, f in enumerate(spec.fingers)},
                 gap_err_cm={f'{spec.fingers[i]}-{spec.fingers[j]}': stats(x['gap_cm'][:, n])
                             for n, (i, j) in enumerate(spec.gap_pairs)},
                 gap_err_mean_cm=round(x['gap_cm'].mean().item(), 3), wrist_to_keypoint=x['wrist_info'])
        meds = [r['palm_normal_deg']['median'], r['palm_forward_deg']['median']] + [v['median'] for v in r['finger_edge_deg'].values()]
        if max(meds) > 90.0 or r['gap_err_mean_cm'] > 3.0:
            flagged.append(side)
        report[side] = r
        print(f'== {a.hand} {side}: fingers {r["fingers"]} (missing {r["missing"]}), wrist {r["wrist"]}'
              f'{" (from wrist_pose)" if r["wrist_from_wrist_pose"] else ""}, outermost {r["outermost"]}, forward {r["forward"]}, '
              f'{r["frames"]} frames')
        print(f'   palm normal: {mode}, {normal}' + (f', axis {r["palm_axis"]}' if axis is not None else ''))
        print(f'   palm normal deg {r["palm_normal_deg"]}   palm forward deg {r["palm_forward_deg"]}')
        for f, s in r['finger_edge_deg'].items():
            print(f'   edge {f:6s} deg {s}')
        for p, s in r['gap_err_cm'].items():
            print(f'   gap {p:13s} err cm {s}')
        print(f'   mean gap error {r["gap_err_mean_cm"]} cm' + (f'; wrist to keypoint {r["wrist_to_keypoint"]}'
                                                               if r['wrist_to_keypoint'] else ''))
    report['flagged'] = flagged
    print(f'FLAGGED {flagged}' if flagged else 'no side flagged')
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    sys.exit(1 if flagged else 0)


if __name__ == '__main__':
    main()
