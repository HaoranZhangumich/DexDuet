"""Hand and object tracking metrics of one evaluation recording, offline on the CPU: MPJPE, the ManipTrans errors and success
rate, and the SPIDER success rate.

  CUDA_VISIBLE_DEVICES= python -m dexmachina.eval.tracking_metrics <recording dir> [<recording dir> ...]
      [--env_pkl <run>/params/env.pkl] [--out <file.json>]

Input: a recording dir written by dexmachina/eval/slip_record.py (recording.npz + record.json) and the run's params/env.pkl
(found through record.json inputs.run_dir, else the checkpoint's grandparent, else the recording dir's parent). env.pkl gives
the retargeted robot reference (retarget_data[side]: kpts_data kpt_pos / kpt_names, wrist_pose) and, for runs whose recording
has none, the human MANO joints (demo_data hand_joints_<side>; older runs: re-read from the ARCTIC processed file exactly as
dexmachina/envs/demo_data.py:36-39 get_demo_data(load_hand_joints=True) slices them). Output: <recording dir>/
tracking_metrics.json, or --out. Nothing else is written.

COMMON CONVENTIONS
  Alignment: recording row i (the state after control step i+1, i = 0..S-1, S = T-1) is compared with demo / reference
    frame i+1 (slip_record.py:14-15). This is the training reward's pairing: base_env.py:738 increments episode_length_buf
    before rewards.py:357-362 match_demo_state indexes the reference. dexmachina/eval/slip_auc.py:320 uses the same frames 1..S.
  Positions are Genesis link origins in the world frame, plain (unsquared) L2 distances, no root or Procrustes alignment.
    Units: cm and degrees, except the SPIDER block (m and rad, as the paper).
  Lanes: the evaluation has 100 lanes (seed 1042, deterministic actions). Each lane is scored on its own, and every success
    rate is the mean of the per-lane 0/1 flags. A lane with a non-finite state fails every success test.
  Rotation angles: the geodesic angle 2 atan2(|Im r|, |Re r|) of r = q_a q_b^-1. It is the same number as ManipTrans's
    |rotvec(R_a R_b^T)| with min(theta, 2 pi - theta) (HO-Tracker eval_score.py:46-56) and SPIDER's
    arccos(2 <q_a, q_b>^2 - 1) (SPIDER v2 Sec. 3.1), and it stays exact near 0, where arccos loses precision.
  Robot-to-MANO correspondence (hand-agnostic): from dexmachina/envs/finger_spec.py (load_finger_spec). It gives the wrist link,
    which is the retargeting origin and is paired with MANO joint 0; each present finger's fingertip, with MANO 16..20; and each
    finger's own chain of links. The MANO joint of every chain link follows ManipTrans's hand2dex_mapping (dexhands/<hand>.py),
    by the number n of non-tip links in the chain:
    n=2 -> (proximal, intermediate); n=3 -> (proximal, proximal, intermediate); n>=4 -> (proximal x (n-2), intermediate,
    distal). With this rule, the official ManipTrans tables for Allegro (allegro.py:84-106, left 201-223) and XHand
    (xhand.py:49-71) come out link for link, and so does Inspire (inspire.py:48-70, matched by link name). See
    tests/test_tracking_metrics.py.
    MANO joints (21, smplx order; finger_spec.py:36-38): 0 wrist; proximal/intermediate/distal: index 1-3, middle 4-6, pinky
    7-9, ring 10-12, thumb 13-15; tips: 16 thumb, 17 index, 18 middle, 19 ring, 20 pinky.

1. MPJPE: hand keypoint tracking error. DexMachina (arXiv 2505.24853) evaluates the object only (ADD-AUC, App. B.4) and
   defines no hand metric. The definition is the standard MPJPE (Human3.6M, Ionescu et al., TPAMI 2014: the mean unsquared
   Euclidean joint distance), here without root alignment.
   mpjpe_ref = mean over steps and the unique keypoint links (first occurrence; Allegro records its four tips twice,
     robot.py:191-199) of ||x - r||. x is the recorded keypoint and r the retargeted robot reference, kpts_data kpt_pos, the
     policy's imitation target (rewards.py:221-227, 423-447; DexMachina Sec. 4.2 r_imi). It is computed per hand. The
     headline 'MPJPE' is the mean of the two hands over the 100 lanes.
   tip_ref: the same, over the finger_spec fingertips only. wrist_ref: ||p_wrist - wrist_pose[:3]||, and wrist_rot_ref: the
     angle between the wrist-link quaternions. mpjpe_ref_allK: the training reward's average, duplicates included.
     mpjpe_ref_wristframe: the keypoints in each hand's own wrist frame, which leaves finger articulation only.
   vs the human MANO hand (unscaled), where a mapping exists:
     tip_mano: finger_spec tips vs MANO tips. wrist_mano: wrist link vs MANO 0. wrist_tips_mano: their mean over the
     (1 + #fingers) points. tip_mano_scaled = ||(x_tip - p_w) - s (m_tip - m_0)||, s = retarget scaling_factor; this is the
     retargeting objective (retarget_utils.py:70-76). joint_mano equals the ManipTrans E_j of that hand (section 2).
     Each MANO error comes with reference_floor: the same error for the retargeted reference itself, i.e. what a perfect
     tracker of our reference would score.

2. ManipTrans (Li et al., CVPR 2025, arXiv 2503.21860, Sec. 4.1 "Metrics", sec/4_experiments.tex:14-18).
   Headline SR (summary maniptrans_sr; block SR_v2d) = NVIDIA video_to_data's (V2D, the CHORD release) object-only
   maniptrans_success, robotic_grounding/flash_chord/src/flash_chord/evaluation/metrics.py:384-404 (commit 8c778f9):
   per lane, each object body's position error and geodesic rotation error are averaged over the rollout, the bodies of
   one object are averaged, and the lane succeeds when that position < 3 cm and rotation < 30 deg (strict). No hand term,
   no per-frame test, no grace rows. Bodies: the bottom and top parts (V2D binds both links of an ARCTIC object,
   assets/registry.py:220-228). In all our ARCTIC URDFs the hinge 'rotation' sits at the bottom origin (axis -z), so both
   parts share the root position and the top rotation is root x Rz(-hinge). Pairing: V2D compares the state after step
   k+1 with reference frame k, k = 0..T-1 (evaluation/rollout.py record_object_pose_rollout, reference_object_pose), i.e.
   our rows 0..S-1 plus final_state vs demo frames 0..T-1. SR_v2d_reward_pairing: the same test, row i vs frame i+1.
   The ManipTrans authors' own test, which also requires both hands (SR_hand_object below; summary
   maniptrans_sr_hand_object): the procedure is the first author's (GitHub ManipTrans issue #18, comment 2826735174) and
   his released scorer KailinLi/HO-Tracker-Baseline
   main/rl/eval_score.py, with the eval-mode failure test maniptrans_envs/lib/envs/tasks/dexhandmanip_bih.py:1977-1991
   ("for evaluation, use the same standard as in the paper"). Where the paper text and the code differ, the code is followed.
   E_t  = mean_t ||p_obj - p_demo|| (cm; eval_score.py:38-43; the paper prints ||.||_2^2 but gives cm).
   E_r  = mean_t geodesic angle of R_demo R_sim^T (deg; eval_score.py:46-56).
   E_j  = mean over t and bodies f of ||p_robot_f - p_MANO(f)|| (cm; eval_score.py:59-64, with the wrist prepended to the
          targets at :143-168). The body list is the wrist plus every finger-chain link, each compared with the first MANO
          joint mapped to it (to_hand()[0]).
   E_ft = the same over the 5 tip slots thumb/index/middle/ring/pinky (weight_idx *_tip; eval_score.py:67-75). On a hand
          without a pinky the ring tip fills the pinky slot and is counted twice, as in ManipTrans allegro.py:134-142.
   Bimanual: E = (E_left + E_right) / 2 per rollout (issue #18; eval_score.py:277-365). E_* are reported over the successful
          lanes (Table 1: "computed only on successfully rolled-out sequences") and, labelled, over all lanes.
   SR_hand_object: a lane FAILS at the first row i >= 8 where any of the following holds, for either hand:
          - object position error > 3 cm;
          - object rotation error > 30 deg;
          - any single fingertip slot > 6 cm from its MANO tip;
          - the mean distance of the level_1 body group > 8 cm, or of the level_2 group > 8 cm. Groups (weight_idx): bodies
            mapped to a proximal joint / to an intermediate or distal joint. That is official for Allegro and XHand; for
            Inspire the official odd grouping of inspire.py:99-107 is used, by link name.
        The 8-row grace comes from running_progress_buf >= 8: compute_reward runs before the increment (bih.py:1759-1766), so
        rows 0..7 are exempt. A lane also fails at ANY row where the sanity check error_buf fires (bih.py:1950-1957; not
        graced). It SUCCEEDS when it reaches the last frame without failing (bih.py:2015-2017). Bimanual success is right AND
        left (bih.py:978). SR = mean over the lanes.
   Also reported:
     SR_paper_text: sec/4_experiments.tex:18 read literally, i.e. the trajectory means E_r, E_t, E_j, E_ft under 30 deg,
       3 cm, 8 cm, 6 cm for both hands.
     SR_object_only and SR_hand_only: the two halves of the per-frame test.
     SR_robot_reference: NOT ManipTrans's definition. The same per-frame test, with the hand distances measured against the
       retargeted robot reference (same links; the wrist against wrist_pose) instead of the human hand.

3. SPIDER (Pan et al., arXiv 2511.09484; v2 Sec. 3.1 "Evaluation Metrics", v3 App. A.1. Code facebookresearch/spider
   spider/postprocess/get_success_rate.py:119-286 compute_object_tracking_error, :358-359 thresholds, :481-484 success,
   :584-588 overall rate).
   E_pos = mean_t ||p_obj - p_ref|| (m) and E_rot = mean_t arccos(2 <q, q_ref>^2 - 1) (rad), on the object root (bottom part)
   pose, over every step. Success = E_pos < 0.1 m and E_rot < 0.5 rad: the object only, with no hand term. The hinge is
   reported as d_arti (v3 App. B.3, Table 19) and is not part of success. SR = mean over the lanes, every trajectory weighted
   equally, as get_success_rate.py:584-588 does.
   Also reported:
     the code's "<=" (since commit f40e64c);
     the official evaluator's pairing: the state after step i+1 vs demo[i], which SPIDER's evaluate_rl_eval reads from
       eval_ep0.npy (evaluate_dexmachina.py:181-204);
     per-trajectory mean-centred positions (get_success_rate.py:174-186; only its two-object branch does this);
     the superseded v1 rule E_pos < 3 cm;
     lane 0 alone, SPIDER's literal one-rollout protocol.

Each run's JSON lists, under 'deviations', every way it departs from the papers (hand, keypoints, reference, object,
lanes). Things that cannot be reproduced are said plainly there.
"""
import argparse
import io
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from dexmachina.envs.finger_spec import FINGERS, MANO_TIPS, load_finger_spec

TOOL_VERSION = 2                                                            # 2: headline ManipTrans SR = V2D object-only
SIDES = ('left', 'right')
MANO_JOINT_NAMES = ('wrist', 'index_proximal', 'index_intermediate', 'index_distal', 'middle_proximal', 'middle_intermediate',
                    'middle_distal', 'pinky_proximal', 'pinky_intermediate', 'pinky_distal', 'ring_proximal',
                    'ring_intermediate', 'ring_distal', 'thumb_proximal', 'thumb_intermediate', 'thumb_distal', 'thumb_tip',
                    'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip')
MANO_FINGER_JOINTS = {'thumb': (13, 14, 15), 'index': (1, 2, 3), 'middle': (4, 5, 6), 'ring': (10, 11, 12),
                      'pinky': (7, 8, 9)}                                    # proximal, intermediate, distal
ROLE_COLUMN = {'proximal': 0, 'intermediate': 1, 'distal': 2}

# ManipTrans eval-mode failure test (HO-Tracker dexhandmanip_bih.py:1977-1991 == issue #18 step 3), metres / degrees
MT_OBJ_POS_M, MT_OBJ_ROT_DEG, MT_TIP_M, MT_LEVEL_M, MT_GRACE = 0.03, 30.0, 0.06, 0.08, 8
MT_PAPER = dict(E_r_deg=30.0, E_t_cm=3.0, E_j_cm=8.0, E_ft_cm=6.0)            # sec/4_experiments.tex:18, trajectory means
MT_SANITY_LIN = 100.0                                                       # error_buf: wrist / mean joint / object, m/s
# SPIDER success (get_success_rate.py:358-359; paper v2 Sec. 3.1), superseded v1 position rule
SPIDER_POS_M, SPIDER_ROT_RAD, SPIDER_V1_POS_M = 0.1, 0.5, 0.03

# ManipTrans's own tables per hand (dexhands/<hand>.py). Level groups are given as (finger, chain position) where the
# official grouping is not the proximal / intermediate+distal rule; only Inspire's is (inspire.py:99-107).
MANIPTRANS_HANDS = {
    'allegro_hand': 'official ManipTrans dexhands/allegro.py tables (same link names as our URDF)',
    'xhand': "official ManipTrans dexhands/xhand.py tables (our links carry a 'left_'/'right_' prefix)",
    'inspire_hand': ('ManipTrans dexhands/inspire.py tables, for a different Inspire URDF; our links are matched by name '
                     '(FF/MF/RF/LF = index/middle/ring/pinky, L1/L2 = proximal/intermediate, TH_L1..TH_L4 = thumb '
                     'proximal_base/proximal/intermediate/distal)'),
}
OFFICIAL_LEVEL_GROUPS = {
    'inspire_hand': dict(level_1=(('index', 0), ('middle', 0), ('thumb', 1)),
                         level_2=(('index', 1), ('middle', 1), ('pinky', 0), ('pinky', 1), ('ring', 0), ('ring', 1),
                                  ('thumb', 0), ('thumb', 2), ('thumb', 3))),
}


# --------------------------------------------------------------------------------------------------------------------------
# geometry (quaternions wxyz)
def _qnorm(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _qz(angle):
    h = np.asarray(angle, np.float64) / 2
    z = np.zeros_like(h)
    return np.stack([np.cos(h), z, z, np.sin(h)], -1)


def quat_angle(a, b):
    """Geodesic angle in rad, in [0, pi], between quaternions a and b (wxyz; normalised here; sign-invariant)."""
    r = _qmul(_qnorm(a), _qconj(_qnorm(b)))
    return 2.0 * np.arctan2(np.linalg.norm(r[..., 1:], axis=-1), np.abs(r[..., 0]))


def rotate(q, v):
    """Rotate v (..., 3) by the unit quaternion q (..., 4) wxyz."""
    q = _qnorm(q)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


# --------------------------------------------------------------------------------------------------------------------------
# hand correspondence
def chain_roles(n):
    """MANO role of each of the n non-tip links of a finger chain, palm to tip (ManipTrans hand2dex_mapping convention)."""
    if n <= 0:
        return ()
    if n == 1:
        return ('proximal',)
    if n == 2:
        return ('proximal', 'intermediate')
    if n == 3:
        return ('proximal', 'proximal', 'intermediate')
    return ('proximal',) * (n - 2) + ('intermediate', 'distal')


def maniptrans_hand(spec):
    """The ManipTrans body list of one hand side from its finger_spec: [wrist] + every finger-chain link, each with its MANO
    joint; the 5 tip slots (a missing finger uses the nearest present one, ties toward the thumb; ManipTrans Allegro:
    pinky = ring); and the level_1 / level_2 groups."""
    bodies = [dict(link=spec.wrist, mano=0, role='wrist', finger=None, pos=None)]
    at = {}
    for f in spec.fingers:
        chain = spec.chains[f]
        nontip = chain[:-1]
        for k, (link, role) in enumerate(zip(nontip, chain_roles(len(nontip)))):
            bodies.append(dict(link=link, mano=MANO_FINGER_JOINTS[f][ROLE_COLUMN[role]], role=role, finger=f, pos=k))
            at[(f, k)] = link
        bodies.append(dict(link=chain[-1], mano=MANO_TIPS[f], role='tip', finger=f, pos=len(nontip)))
    present = list(spec.fingers)
    tip_slots = {}
    for i, f in enumerate(FINGERS):
        g = f if f in present else min(present, key=lambda p: (abs(FINGERS.index(p) - i), FINGERS.index(p)))
        tip_slots[f] = spec.tips[present.index(g)]
    official = OFFICIAL_LEVEL_GROUPS.get(spec.hand)
    if official is not None:
        level_1 = [at[x] for x in official['level_1']]
        level_2 = [at[x] for x in official['level_2']]
    else:
        level_1 = [b['link'] for b in bodies if b['role'] == 'proximal']
        level_2 = [b['link'] for b in bodies if b['role'] in ('intermediate', 'distal')]
    source = MANIPTRANS_HANDS.get(spec.hand, f'{spec.hand}: no ManipTrans config; tables derived by the rule above (unofficial)')
    return dict(bodies=bodies, tip_slots=tip_slots, level_1=level_1, level_2=level_2, source=source)


def wrist_child_offset(urdf, wrist, names):
    """(keypoint link, xyz in the wrist frame) of a keypoint link fixed to the wrist link, for hands whose wrist link is not a
    keypoint (Allegro: base_link, 0.095 m along the base_dummy_link z axis, allegro_hand_*_6dof.urdf base_dummy_joint)."""
    root = ET.parse(urdf).getroot()
    for j in root.findall('joint'):
        if j.get('type') == 'fixed' and j.find('parent').get('link') == wrist and j.find('child').get('link') in names:
            o = j.find('origin')
            xyz = np.array([float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
            return j.find('child').get('link'), xyz
    raise ValueError(f'{urdf}: the wrist link {wrist!r} is not a keypoint and no keypoint link is fixed to it')


def wrist_position(kp, names, quat, spec):
    """Wrist-link positions (..., 3) from keypoints (..., K, 3) and the wrist-link quaternion (..., 4); and how."""
    if spec.wrist in names:
        return kp[..., names.index(spec.wrist), :], 'keypoint'
    child, t = wrist_child_offset(spec.urdf_path, spec.wrist, names)
    return kp[..., names.index(child), :] - rotate(quat, t), f'{child} - R_wrist @ {t.tolist()} (URDF fixed joint)'


def wrist_body_point(names, spec):
    """Index of the keypoint that moves with the wrist link (the wrist itself, or the link fixed to it)."""
    if spec.wrist in names:
        return names.index(spec.wrist)
    return names.index(wrist_child_offset(spec.urdf_path, spec.wrist, names)[0])


# --------------------------------------------------------------------------------------------------------------------------
# inputs
def _np(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, np.float64)


def load_env_pkl(path):
    """params/env.pkl on the CPU (the file stores CUDA tensors): torch's storage loader is mapped to the CPU while unpickling,
    then restored."""
    import torch
    import torch.storage
    orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def find_env_pkl(meta, rec_dir):
    inputs = meta.get('inputs', {})
    cands = []
    if inputs.get('run_dir'):
        cands.append(Path(inputs['run_dir']) / 'params' / 'env.pkl')
    if inputs.get('checkpoint'):
        cands.append(Path(inputs['checkpoint']).parents[1] / 'params' / 'env.pkl')
    cands.append(Path(rec_dir).resolve().parent / 'params' / 'env.pkl')
    for c in cands:
        if c.is_file():
            return c
    return None


def hand_of(meta, side, env=None):
    r = str((meta.get('robots') or {}).get(side, ''))
    if r.endswith('_' + side):
        return r[:-len(side) - 1]
    if env is not None and side in env.get('robot_cfgs', {}) and env['robot_cfgs'][side].get('name'):
        n = str(env['robot_cfgs'][side]['name'])
        return n[:-len(side) - 1] if n.endswith('_' + side) else n
    raise ValueError(f'cannot tell the {side} hand from record.json robots={meta.get("robots")}')


def reference_of(env, side):
    rd = env['retarget_data'][side]
    kd = rd['kpts_data']
    return dict(kpt_pos=_np(kd['kpt_pos']), kpt_names=[str(n) for n in kd['kpt_names']], wrist_pose=_np(rd['wrist_pose']))


def mano_of(rec, env, meta, side, T):
    """(T, 21, 3) MANO joints and a note on where they came from. Sources, in order: the recording, env.pkl demo_data, the
    ARCTIC processed file (demo_data.get_demo_data(load_hand_joints=True), parsed from the clip string). When two sources
    exist they must agree (1e-5 m)."""
    found = []
    r = rec.get(f'demo_hand_joints_{side}')
    if r is not None and len(r):
        found.append(('recording demo_hand_joints', np.asarray(r, np.float64)))
    if env is not None and f'hand_joints_{side}' in env.get('demo_data', {}):
        found.append(('env.pkl demo_data hand_joints', _np(env['demo_data'][f'hand_joints_{side}'])))
    if not found:
        from dexmachina.envs.demo_data import get_demo_data
        obj, f0, f1, subject, use = str(meta['clip']).split('-')
        d = get_demo_data(obj_name=obj, frame_start=int(f0), frame_end=int(f1), subject_name=subject,
                          use_clip=use.lstrip('u'), load_hand_joints=True)
        found.append((f'ARCTIC processed file via get_demo_data({meta["clip"]})', np.asarray(d[f'hand_joints_{side}'], np.float64)))
    name, m = found[0]
    diffs = {n: float(np.abs(v - m).max()) for n, v in found[1:]}
    for n, v in diffs.items():
        if v > 1e-5:
            raise ValueError(f'{side} MANO joints differ between {name} and {n} by {v:.2e} m')
    if m.shape[0] < T:
        raise ValueError(f'{side} MANO joints cover {m.shape[0]} frames, the demo has {T}')
    return m[:T], dict(source=name, cross_checked_max_abs_diff_m=diffs)


# --------------------------------------------------------------------------------------------------------------------------
# helpers for the report
def _r(x, nd=5):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _lanes(v, nd=4):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in np.asarray(v, np.float64)]


def _lane_mean(v, mask=None):
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
    return float(v[ok].mean()) if ok.any() else None


def _stats(v):
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return dict(mean=None, median=None, min=None, max=None, nonfinite_lanes=int((~ok).sum()))
    return dict(mean=_r(v[ok].mean()), median=_r(np.median(v[ok])), min=_r(v[ok].min()), max=_r(v[ok].max()),
                nonfinite_lanes=int((~ok).sum()))


def _inf_where(v, bad):
    v = np.array(v, np.float64)
    v[bad] = np.inf
    return v


# --------------------------------------------------------------------------------------------------------------------------
def compute(rec, meta, ref=None, mano=None, hands=None, input_notes=None):
    """All metrics of one recording.

    rec: recording.npz arrays (dict); meta: record.json; ref: {side: dict(kpt_pos (T, K, 3), kpt_names, wrist_pose (T, 7))}
    or None; mano: {side: (T, 21, 3)} or None; hands: {side: hand name} (default: from meta['robots'])."""
    obj = np.asarray(rec['obj_state'], np.float64)                       # (S, L, 8) [pos, wxyz, hinge]
    S, L = obj.shape[:2]
    demo = np.asarray(rec['demo'], np.float64)                           # (T, 8)
    T = demo.shape[0]
    if S > T - 1:
        raise ValueError(f'{S} recorded steps but only {T} demo frames')
    dt = float(rec['dt']) if 'dt' in rec else 1.0 / 60
    fr = np.arange(1, S + 1)                                             # row i <-> frame i+1
    dref = demo[fr]
    hands = hands or {s: hand_of(meta, s) for s in SIDES}
    specs = {s: load_finger_spec(hands[s], s) for s in SIDES}
    graced = (np.arange(S) >= MT_GRACE)[:, None]                         # (S, 1)
    deviations = dict(general=[], mpjpe=[], maniptrans=[], spider=[])
    missing = dict(input_notes or {})

    # ---------------- object
    obj_ok = np.isfinite(obj).all(-1) & (np.linalg.norm(np.nan_to_num(obj[..., 3:7]), axis=-1) > 1e-6)   # (S, L)
    o = np.where(obj_ok[..., None], obj, np.array([0, 0, 0, 1, 0, 0, 0, 0.0]))
    pos_err = _dist(o[..., :3], dref[:, None, :3])                       # (S, L) m
    q_s, q_d = o[..., 3:7], np.broadcast_to(dref[:, None, 3:7], o[..., 3:7].shape)
    rot_bottom = quat_angle(q_s, q_d)                                    # rad
    rot_top = quat_angle(_qmul(_qnorm(q_s), _qz(-o[..., 7])), _qmul(_qnorm(q_d), _qz(-np.broadcast_to(dref[:, None, 7], o[..., 7].shape))))
    hinge_err = np.abs(o[..., 7] - dref[:, None, 7])
    for a in (pos_err, rot_bottom, rot_top, hinge_err):
        a[~obj_ok] = np.inf
    obj_lane_ok = obj_ok.all(0)                                          # (L,)

    # ---------------- hands
    sides = {}
    finite_all = obj_ok.copy()                                           # (S, L)
    sanity = np.zeros((S, L), bool)
    hand_fail_mano = np.zeros((S, L), bool)
    hand_fail_ref = np.zeros((S, L), bool)
    for s in SIDES:
        spec = specs[s]
        kp = np.asarray(rec[f'kpts_{s}'], np.float64)                    # (S, L, K, 3)
        names = [str(n) for n in rec[f'kpt_names_{s}']]
        first = {}
        for i, n in enumerate(names):
            first.setdefault(n, i)
        uniq = sorted(first.values())
        wq = np.asarray(rec[f'wrist_quat_{s}'], np.float64)              # (S, L, 4)
        ok = np.isfinite(kp).all((-1, -2)) & np.isfinite(wq).all(-1) & (np.linalg.norm(np.nan_to_num(wq), axis=-1) > 1e-6)
        finite_all &= ok
        kp = np.where(ok[..., None, None], kp, 0.0)
        wq = np.where(ok[..., None], wq, np.array([1.0, 0, 0, 0]))
        pw, wsrc = wrist_position(kp, names, wq, spec)                  # (S, L, 3)
        tips = [first[t] for t in spec.tips]
        lane_bad = ~ok.all(0)
        out = dict(hand=spec.hand, urdf=spec.urdf_path, fingers=list(spec.fingers), missing_fingers=list(spec.missing),
                   n_keypoints=len(names), n_unique_keypoints=len(uniq), wrist_link=spec.wrist, wrist_source=wsrc,
                   tips=list(spec.tips), retarget_scale=spec.scale, nonfinite_lanes=int(lane_bad.sum()))
        per_lane = {}

        # (a) vs the retargeted robot reference
        R = W = None
        if ref is not None:
            rf = ref[s]
            if rf['kpt_names'] != names:
                raise ValueError(f'{s}: retarget kpt_names differ from the recorded keypoint names')
            if rf['kpt_pos'].shape[0] < S + 1:
                raise ValueError(f'{s}: the reference has {rf["kpt_pos"].shape[0]} frames, need {S + 1}')
            R, W = rf['kpt_pos'][fr], rf['wrist_pose'][fr]               # (S, K, 3), (S, 7)
            rw = wrist_position(rf['kpt_pos'], names, rf['wrist_pose'][:, 3:], spec)[0]
            out['reference_wrist_vs_wrist_pose_max_cm'] = _r(np.abs(rw - rf['wrist_pose'][:, :3]).max() * 100, 6)
            d = _dist(kp, R[:, None]) * 100                              # (S, L, K) cm
            per_lane['mpjpe_ref'] = d[..., uniq].mean((0, 2))
            per_lane['mpjpe_ref_allK'] = d.mean((0, 2))
            per_lane['tip_ref'] = d[..., tips].mean((0, 2))
            per_lane['wrist_ref'] = (_dist(pw, W[:, None, :3]) * 100).mean(0)
            per_lane['wrist_rot_ref_deg'] = np.degrees(quat_angle(wq, np.broadcast_to(W[:, None, 3:], wq.shape))).mean(0)
            local = rotate(_qconj(_qnorm(wq))[..., None, :], kp[..., uniq, :] - pw[..., None, :])
            local_ref = rotate(_qconj(_qnorm(W[:, 3:]))[:, None, :], R[:, uniq] - W[:, None, :3])
            per_lane['mpjpe_ref_wristframe'] = (_dist(local, local_ref[:, None]) * 100).mean((0, 2))
        else:
            missing['retarget_reference'] = 'no env.pkl retarget_data: the robot-reference metrics are null'

        # (b) vs the human MANO hand, and the ManipTrans body metrics
        mt = maniptrans_hand(spec)
        absent = [b['link'] for b in mt['bodies'][1:] if b['link'] not in first]
        if absent:
            deviations['maniptrans'].append(f'{s}: ManipTrans bodies not recorded as keypoints, left out: {absent}')
        bodies = [mt['bodies'][0]] + [b for b in mt['bodies'][1:] if b['link'] in first]
        col = {b['link']: i for i, b in enumerate(bodies)}
        P = np.stack([pw] + [kp[..., first[b['link']], :] for b in bodies[1:]], -2)          # (S, L, F, 3)
        tip_cols = [col[mt['tip_slots'][f]] for f in FINGERS]
        lv1 = [col[x] for x in mt['level_1'] if x in col]
        lv2 = [col[x] for x in mt['level_2'] if x in col]
        out['maniptrans_bodies'] = {b['link']: f"{MANO_JOINT_NAMES[b['mano']]} ({b['mano']})" for b in bodies}
        out['maniptrans_tip_slots'] = dict(mt['tip_slots'])
        out['maniptrans_level_1'] = [bodies[i]['link'] for i in lv1]
        out['maniptrans_level_2'] = [bodies[i]['link'] for i in lv2]
        out['maniptrans_tables'] = mt['source']

        def hand_test(dist):                                             # dist (S, L, F) m -> per-frame failure (S, L)
            return ((dist[..., tip_cols] > MT_TIP_M).any(-1) | (dist[..., lv1].mean(-1) > MT_LEVEL_M)
                    | (dist[..., lv2].mean(-1) > MT_LEVEL_M))

        if R is not None:
            Pref = np.stack([W[:, :3]] + [R[:, first[b['link']]] for b in bodies[1:]], -2)    # (S, F, 3)
            dref_h = _dist(P, Pref[:, None])
            hf = hand_test(dref_h)
            hand_fail_ref |= hf
            per_lane['mt_E_j_robot_ref'] = dref_h.mean((0, 2)) * 100
            per_lane['mt_E_ft_robot_ref'] = dref_h[..., tip_cols].mean((0, 2)) * 100
            out['maniptrans_robot_ref_frac_checked_frames_failing'] = _r(hf[MT_GRACE:].mean())
        if mano is not None:
            M = mano[s][fr]                                              # (S, 21, 3)
            mano_idx = [b['mano'] for b in bodies]
            dm = _dist(P, M[:, None, mano_idx])                          # (S, L, F) m
            hf = hand_test(dm)
            hand_fail_mano |= hf
            per_lane['mt_E_j'] = dm.mean((0, 2)) * 100
            per_lane['mt_E_ft'] = dm[..., tip_cols].mean((0, 2)) * 100
            tipd = dm[..., tip_cols]
            out['maniptrans_frac_checked_frames_failing'] = dict(
                any=_r(hf[MT_GRACE:].mean()), tip_over_6cm=_r((tipd > MT_TIP_M).any(-1)[MT_GRACE:].mean()),
                level_over_8cm=_r(((dm[..., lv1].mean(-1) > MT_LEVEL_M) | (dm[..., lv2].mean(-1) > MT_LEVEL_M))[MT_GRACE:].mean()))
            out['maniptrans_tip_slot_cm'] = {f: _r(tipd[..., i].mean() * 100) for i, f in enumerate(FINGERS)}
            out['maniptrans_level_cm'] = dict(level_1=_r(dm[..., lv1].mean() * 100), level_2=_r(dm[..., lv2].mean() * 100))
            ht = list(spec.human_tips)
            tip_m = _dist(kp[..., tips, :], M[:, None, ht]) * 100        # (S, L, n) cm
            wr_m = _dist(pw, M[:, None, 0]) * 100                        # (S, L)
            per_lane['tip_mano'] = tip_m.mean((0, 2))
            per_lane['wrist_mano'] = wr_m.mean(0)
            per_lane['wrist_tips_mano'] = np.concatenate([wr_m[..., None], tip_m], -1).mean((0, 2))
            per_lane['tip_mano_scaled'] = (_dist(kp[..., tips, :] - pw[..., None, :],
                                                 spec.scale * (M[:, None, ht] - M[:, None, :1])) * 100).mean((0, 2))
            if R is not None:                                            # the retargeted reference itself vs MANO
                tip_f = _dist(R[:, tips], M[:, ht]) * 100
                wr_f = _dist(W[:, :3], M[:, 0]) * 100
                dmf = _dist(Pref, M[:, mano_idx])                        # (S, F) m
                hff = hand_test(dmf[:, None])[:, 0]
                out['reference_floor'] = dict(
                    tip_mano=_r(tip_f.mean()), wrist_mano=_r(wr_f.mean()),
                    wrist_tips_mano=_r(np.concatenate([wr_f[:, None], tip_f], -1).mean()),
                    tip_mano_scaled=_r((_dist(R[:, tips] - W[:, None, :3], spec.scale * (M[:, ht] - M[:, :1])) * 100).mean()),
                    tip_mano_per_finger={f: _r(v) for f, v in zip(spec.fingers, tip_f.mean(0))},
                    maniptrans_E_j=_r(dmf.mean() * 100), maniptrans_E_ft=_r(dmf[:, tip_cols].mean() * 100),
                    maniptrans_frac_checked_frames_failing_hand_test=_r(hff[MT_GRACE:].mean()),
                    maniptrans_hand_test_passes=bool(not hff[MT_GRACE:].any()))
        else:
            missing['mano'] = 'no MANO hand joints: the human-hand metrics and the ManipTrans SR are null'

        # sanity check (error_buf), from finite differences of recorded positions; the start state is row -1
        start = rec.get(f'start_kpts_{s}')
        if start is not None and len(start):
            seq = np.concatenate([np.asarray(start, np.float64)[None], kp])
            vel = _dist(seq[1:], seq[:-1]) / dt                          # (S, L, K) m/s
            vel[0][~np.isfinite(vel[0])] = 0.0
        else:
            vel = np.concatenate([np.zeros((1, L, kp.shape[2])), _dist(kp[1:], kp[:-1]) / dt])
        wb = wrist_body_point(names, spec)
        chain_k = [first[b['link']] for b in bodies[1:]]
        sanity |= (vel[..., wb] > MT_SANITY_LIN) | (vel[..., chain_k].mean(-1) > MT_SANITY_LIN)

        for k, v in per_lane.items():
            per_lane[k] = _inf_where(v, lane_bad)
        out['mean_over_lanes'] = {k: _r(_lane_mean(v)) for k, v in per_lane.items()}
        out['median_over_lanes'] = {k: _r(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else None
                                    for k, v in per_lane.items()}
        out['_per_lane'] = per_lane
        sides[s] = out

    start_obj = rec.get('start_obj')
    oseq = np.concatenate([np.asarray(start_obj, np.float64)[None, :, :3], o[..., :3]]) if start_obj is not None and len(start_obj) \
        else np.concatenate([o[:1, :, :3], o[..., :3]])
    sanity |= _dist(oseq[1:], oseq[:-1]) / dt > MT_SANITY_LIN
    nonfinite = ~finite_all
    lane_ok = finite_all.all(0)

    def both(key):
        if key not in sides['left']['_per_lane']:
            return None
        return 0.5 * (sides['left']['_per_lane'][key] + sides['right']['_per_lane'][key])

    # ---------------- MPJPE block
    mp = dict(
        definition=('vs the retargeted robot reference (retarget_data kpts_data, the imitation target): mean over steps i '
                    '(row i vs frame i+1), unique keypoint links, lanes; both hands = mean of the two hands; cm'),
        MPJPE_cm=_stats(both('mpjpe_ref')) if both('mpjpe_ref') is not None else None,
        tip_cm=_stats(both('tip_ref')) if both('tip_ref') is not None else None,
        wrist_cm=_stats(both('wrist_ref')) if both('wrist_ref') is not None else None,
        wrist_rot_deg=_stats(both('wrist_rot_ref_deg')) if both('wrist_rot_ref_deg') is not None else None,
        MPJPE_allK_cm=_stats(both('mpjpe_ref_allK')) if both('mpjpe_ref_allK') is not None else None,
        MPJPE_wristframe_cm=_stats(both('mpjpe_ref_wristframe')) if both('mpjpe_ref_wristframe') is not None else None,
        vs_mano=None,
    )
    if mano is not None:
        floors = [sides[s].get('reference_floor') for s in SIDES]
        mp['vs_mano'] = dict(
            definition=('vs the human MANO hand, unscaled (ARCTIC MANO joints; tips 16-20, wrist 0); joint = ManipTrans E_j; '
                        'reference_floor = the retargeted reference itself'),
            wrist_tips_cm=_stats(both('wrist_tips_mano')), tip_cm=_stats(both('tip_mano')),
            wrist_cm=_stats(both('wrist_mano')), joint_cm=_stats(both('mt_E_j')),
            tip_scaled_cm=_stats(both('tip_mano_scaled')),
            reference_floor=({k: _r(0.5 * (floors[0][k] + floors[1][k])) for k in
                              ('wrist_tips_mano', 'tip_mano', 'wrist_mano', 'tip_mano_scaled', 'maniptrans_E_j')}
                             if all(floors) else None))
    mp['per_hand'] = {s: {k: v for k, v in sides[s].items() if k in ('mean_over_lanes', 'median_over_lanes', 'reference_floor')}
                      for s in SIDES}
    mp['per_lane'] = {k: _lanes(both(k)) for k in ('mpjpe_ref', 'tip_ref', 'wrist_ref', 'wrist_tips_mano') if both(k) is not None}

    # rows 0..S-1 plus the final state (after step T) vs demo frames 0..T-1: V2D's and SPIDER's official pairing
    fs = rec.get('final_state')
    rows = np.concatenate([obj, np.asarray(fs, np.float64)[None]]) if fs is not None and len(fs) else obj
    rows_ok = np.isfinite(rows).all(-1).all(0) & (np.linalg.norm(np.nan_to_num(rows[..., 3:7]), axis=-1) > 1e-6).all(0)
    rr = np.where(np.isfinite(rows), rows, 0.0)
    dpair = demo[:len(rows)]

    def v2d(o_, d_, ok):                                                 # (N, L, 8), (N, 8), (L,) -> success, E_t cm, E_r deg
        d_ = np.broadcast_to(d_[:, None], o_.shape)
        pe = _dist(o_[..., :3], d_[..., :3]).mean(0) * 100               # both parts share the root position
        rb = quat_angle(o_[..., 3:7], d_[..., 3:7]).mean(0)
        rt = quat_angle(_qmul(_qnorm(o_[..., 3:7]), _qz(-o_[..., 7])), _qmul(_qnorm(d_[..., 3:7]), _qz(-d_[..., 7]))).mean(0)
        pe, re_ = _inf_where(pe, ~ok), _inf_where(np.degrees(0.5 * (rb + rt)), ~ok)
        return (pe < MT_OBJ_POS_M * 100) & (re_ < MT_OBJ_ROT_DEG), pe, re_

    # ---------------- ManipTrans block
    v2d_ok, v2d_E_t, v2d_E_r = v2d(rr, dpair, rows_ok)
    E_t = _inf_where(pos_err.mean(0) * 100, ~lane_ok)
    E_r_bottom = np.degrees(rot_bottom).mean(0)
    E_r_top = np.degrees(rot_top).mean(0)
    E_r = _inf_where(0.5 * (E_r_bottom + E_r_top), ~lane_ok)
    obj_fail = (pos_err > MT_OBJ_POS_M) | (np.degrees(rot_bottom) > MT_OBJ_ROT_DEG) | (np.degrees(rot_top) > MT_OBJ_ROT_DEG)
    mtb = dict(
        definition_v2d=('SR_v2d (headline): V2D flash_chord maniptrans_success (evaluation/metrics.py:384-404), object only: '
                        'per lane, the rollout-mean position and rotation errors of the bottom and top parts, averaged over '
                        'the two parts, < 3 cm and < 30 deg; state after step k+1 vs demo frame k, k = 0..T-1'),
        SR_v2d=_r(v2d_ok.mean()), n_success_v2d=int(v2d_ok.sum()),
        E_t_cm_v2d_all=_r(_lane_mean(v2d_E_t)), E_r_deg_v2d_all=_r(_lane_mean(v2d_E_r)),
        SR_v2d_reward_pairing=_r(v2d(o, dref, obj_lane_ok)[0].mean()),
        definition=('SR_hand_object: per-frame eval test of the ManipTrans authors (HO-Tracker dexhandmanip_bih.py:1977-1991): '
                    'fail at the first row >= 8 with object > 3 cm or > 30 deg, any fingertip > 6 cm or a level-group mean '
                    '> 8 cm (either hand, vs MANO), or at any row on the sanity check; success = right AND left; SR = mean '
                    'over lanes'),
        lanes=L, steps=S, grace_rows=MT_GRACE, thresholds=dict(obj_pos_cm=3.0, obj_rot_deg=30.0, tip_cm=6.0, level_cm=8.0),
        E_t_cm_all=_r(_lane_mean(E_t)), E_r_deg_all=_r(_lane_mean(E_r)),
        E_r_bottom_deg_all=_r(_lane_mean(_inf_where(E_r_bottom, ~lane_ok))),
        E_r_top_deg_all=_r(_lane_mean(_inf_where(E_r_top, ~lane_ok))),
        hinge_err_deg_all=_r(_lane_mean(_inf_where(np.degrees(hinge_err).mean(0), ~lane_ok))),
    )
    obj_only_fail = ((obj_fail & graced) | nonfinite).any(0)
    mtb['SR_object_only'] = _r(1.0 - obj_only_fail.mean())
    per_lane_mt = dict(E_t=_lanes(E_t), E_r=_lanes(E_r), success_v2d=[int(x) for x in v2d_ok], E_t_v2d=_lanes(v2d_E_t),
                       E_r_v2d=_lanes(v2d_E_r))
    if mano is not None:
        E_j, E_ft = both('mt_E_j'), both('mt_E_ft')
        fail = ((obj_fail | hand_fail_mano) & graced) | sanity | nonfinite
        success = ~fail.any(0)
        first_fail = np.where(fail.any(0), fail.argmax(0), -1)
        paper_ok = lane_ok & (E_r < MT_PAPER['E_r_deg']) & (E_t < MT_PAPER['E_t_cm'])
        for s in SIDES:
            paper_ok &= (sides[s]['_per_lane']['mt_E_j'] < MT_PAPER['E_j_cm']) & (sides[s]['_per_lane']['mt_E_ft'] < MT_PAPER['E_ft_cm'])
        ever = lambda m: _r(m.any(0).mean())                              # noqa: E731
        mtb.update(
            SR_hand_object=_r(success.mean()), n_success=int(success.sum()),
            E_r_deg=_r(_lane_mean(E_r, success)), E_t_cm=_r(_lane_mean(E_t, success)),
            E_j_cm=_r(_lane_mean(E_j, success)), E_ft_cm=_r(_lane_mean(E_ft, success)),
            E_note='E_* = ManipTrans convention, successful lanes only (null when none succeeded); E_*_all = every lane',
            E_j_cm_all=_r(_lane_mean(E_j)), E_ft_cm_all=_r(_lane_mean(E_ft)),
            SR_paper_text=_r(paper_ok.mean()),
            SR_hand_only=_r(1.0 - ((hand_fail_mano & graced) | nonfinite).any(0).mean()),
            first_fail_row_median=_r(np.median(first_fail[first_fail >= 0])) if (first_fail >= 0).any() else None,
            frac_lanes_ever_failing=dict(
                object_pos=ever((pos_err > MT_OBJ_POS_M) & graced),
                object_rot=ever(((np.degrees(rot_bottom) > MT_OBJ_ROT_DEG) | (np.degrees(rot_top) > MT_OBJ_ROT_DEG)) & graced),
                hand=ever(hand_fail_mano & graced), sanity=ever(sanity), nonfinite=ever(nonfinite)),
        )
        per_lane_mt.update(success_hand_object=[int(x) for x in success], first_fail_row=[int(x) for x in first_fail],
                           E_j=_lanes(E_j), E_ft=_lanes(E_ft))
    else:
        mtb.update(SR_hand_object=None, note='no MANO hand joints')
    if ref is not None:
        fail_r = ((obj_fail | hand_fail_ref) & graced) | sanity | nonfinite
        mtb['SR_robot_reference'] = _r(1.0 - fail_r.any(0).mean())
        mtb['SR_robot_reference_note'] = ('NOT ManipTrans: the same per-frame test with the hand distances measured against '
                                          'the retargeted robot reference instead of MANO')
        mtb['E_j_robot_ref_cm_all'] = _r(_lane_mean(both('mt_E_j_robot_ref')))
        mtb['E_ft_robot_ref_cm_all'] = _r(_lane_mean(both('mt_E_ft_robot_ref')))
        per_lane_mt['success_robot_reference'] = [int(x) for x in ~fail_r.any(0)]
    mtb['per_hand'] = {s: {k: v for k, v in sides[s].items() if k.startswith('maniptrans_') or k == 'reference_floor'}
                       for s in SIDES}
    mtb['per_lane'] = per_lane_mt

    # ---------------- SPIDER block
    def spider(pe, re_, ae):                                             # object only: a lane fails on a non-finite object
        pe, re_, ae = (_inf_where(x, ~obj_lane_ok) for x in (pe, re_, ae))
        ok = (pe < SPIDER_POS_M) & (re_ < SPIDER_ROT_RAD)
        return pe, re_, ae, ok
    e_pos, e_rot, e_arti, sp_ok = spider(pos_err.mean(0), rot_bottom.mean(0), hinge_err.mean(0))
    spb = dict(
        definition=('object root pose only: E_pos = mean_t ||p - p_ref|| (m), E_rot = mean_t arccos(2<q,q_ref>^2-1) (rad) over '
                    'every step (row i vs demo frame i+1); success = E_pos < 0.1 m and E_rot < 0.5 rad; SR = mean over lanes'),
        thresholds=dict(E_pos_m=SPIDER_POS_M, E_rot_rad=SPIDER_ROT_RAD),
        SR=_r(sp_ok.mean()), n_success=int(sp_ok.sum()),
        E_pos_m=_stats(e_pos), E_rot_rad=_stats(e_rot), E_arti_rad=_stats(e_arti),
        lane0=dict(success=bool(sp_ok[0]), E_pos_m=_r(e_pos[0]), E_rot_rad=_r(e_rot[0]), E_arti_rad=_r(e_arti[0])),
        SR_code_le=_r(((e_pos <= SPIDER_POS_M) & (e_rot <= SPIDER_ROT_RAD)).mean()),
        SR_v1_3cm_superseded=_r(((e_pos < SPIDER_V1_POS_M) & (e_rot < SPIDER_ROT_RAD)).mean()),
    )
    pc = o[..., :3] - o[..., :3].mean(0, keepdims=True)
    rc = dref[:, None, :3] - dref[:, :3].mean(0)[None, None]
    spb['SR_centered'] = _r(spider(_dist(pc, rc).mean(0), rot_bottom.mean(0), hinge_err.mean(0))[3].mean())
    pe_o = _inf_where(_dist(rr[..., :3], dpair[:, None, :3]).mean(0), ~rows_ok)
    re_o = _inf_where(quat_angle(rr[..., 3:7], np.broadcast_to(dpair[:, None, 3:7], rr[..., 3:7].shape)).mean(0), ~rows_ok)
    spb['SR_official_pairing'] = _r(((pe_o < SPIDER_POS_M) & (re_o < SPIDER_ROT_RAD)).mean())
    spb['per_lane'] = dict(success=[int(x) for x in sp_ok], E_pos_m=_lanes(e_pos, 5), E_rot_rad=_lanes(e_rot, 5))

    # ---------------- deviations
    hand_names = sorted({specs[s].hand for s in SIDES})
    g = deviations['general']
    g.append(f'{L} lanes of one deterministic policy from one verified start (slip_record.py); a success rate is the fraction '
             'of lanes, so one run is usually close to 0 or 1. The papers average over sequences and seeds (ManipTrans: '
             'rollouts per sequence, then sequences; SPIDER: task x seed). Aggregate over clips x seeds.')
    g.append('alignment: row i (state after step i+1) vs demo frame i+1 (our reward pairing). ManipTrans compares the state '
             'after j+1 steps with frame b+j (compute_reward runs before progress_buf += 1, HO-Tracker bih.py:1759-1766, '
             '990-991): a one-frame shift (the reference moves about 0.7-1.5 cm per frame).')
    g.append(f'hand(s): {", ".join(hand_names)}; ManipTrans reports Inspire, Allegro, XHand and others on OakInk-V2/GRAB; '
             'SPIDER reports Inspire only on ARCTIC (Allegro only on OakInk/GigaHands in MuJoCo).')
    m_ = deviations['mpjpe']
    m_.append('MPJPE is not a metric of DexMachina, ManipTrans or SPIDER for this setting; the headline is against our '
              'retargeted robot reference (the policy target), not a human hand.')
    for s in SIDES:
        sd = sides[s]
        if sd['n_keypoints'] != sd['n_unique_keypoints']:
            m_.append(f'{s}: {sd["n_keypoints"]} recorded keypoints, {sd["n_unique_keypoints"]} unique (duplicates dropped; '
                      'the training reward averages over all, see MPJPE_allK_cm).')
    m_.append('the keypoint sets differ per hand (links with collision geometry + tips), so MPJPE_cm is not comparable across '
              'hands; across hands use tip_cm / wrist_cm or vs_mano.wrist_tips_cm (fingertips + wrist).')
    scales = ', '.join(sorted({f'{specs[s].hand} {specs[s].scale}' for s in SIDES}))
    m_.append('MANO targets: ARCTIC MANO joints (world_coord joints.<side>), unscaled; our retargeting scales the human hand '
              f'by {scales} about the wrist, '
              'so a perfect tracker of our reference already has the reference_floor error.')
    t = deviations['maniptrans']
    t.append('headline SR = V2D\'s object-only maniptrans_success (rollout means, no hands); the ManipTrans paper and the '
             'authors\' code also require both hands per frame, which is SR_hand_object. The notes below concern SR_hand_object '
             'and the E_j / E_ft errors.')
    for s in SIDES:
        t.append(f'{s} ({specs[s].hand}): {sides[s]["maniptrans_tables"]}')
        if specs[s].missing:
            used = [sides[s]['maniptrans_tip_slots'][f] for f in specs[s].missing]
            t.append(f'{s}: no {"/".join(specs[s].missing)} finger; its tip slot uses {used} '
                     + ('(official: allegro.py:134-142, pinky = ring)' if specs[s].hand == 'allegro_hand'
                        else '(nearest present finger; not official)'))
        if sides[s]['wrist_source'] != 'keypoint':
            t.append(f'{s}: wrist body = our wrist link {specs[s].wrist} (rebuilt as {sides[s]["wrist_source"]}) vs MANO joint 0; '
                     'ManipTrans compares its base body with the SMPL-X wrist shifted by 0.25 x (middle MCP - wrist) + '
                     'relative_translation (oakink2_dataset_dexhand_rh.py:176-179), which cannot be reproduced on ARCTIC.')
        else:
            t.append(f'{s}: wrist body = our wrist link {specs[s].wrist} vs MANO joint 0; ManipTrans uses the SMPL-X wrist '
                     'shifted by 0.25 x (middle MCP - wrist) + relative_translation (oakink2_dataset_dexhand_rh.py:176-179), '
                     'which cannot be reproduced on ARCTIC.')
    t.append('human targets: ARCTIC MANO joints and joint fingertips 16-20; ManipTrans uses SMPL-X joints 40-54 and single tip '
             'vertices 7706/7818/8046/7929/8096 (oakink2_dataset_dexhand_rh.py:153-174), a few mm apart.')
    t.append('embodiment scale: ManipTrans compares with the unscaled human hand; our policies track a reference scaled by '
             f'{specs["left"].scale} ({specs["left"].hand}), see per_hand.reference_floor (the reference itself fails the '
             'per-frame hand test whenever maniptrans_hand_test_passes is false, which makes SR_hand_object 0 by construction).')
    t.append('articulated object: ManipTrans defines no articulated metric (issue #71). The object test is applied to both parts '
             '(root position; bottom rotation and top rotation = root x Rz(-hinge), slip_record.part_points); E_r = mean of the '
             'two parts; the hinge error is reported separately.')
    t.append('one object for both hands: ManipTrans bimanual tasks give each hand its own object; here both hands share the '
             'articulated object, so the object test is the same for both hands.')
    t.append('sanity check (error_buf): approximated by finite differences of recorded positions (wrist, mean finger-body and '
             'object linear speed > 100 m/s, row 0 against the recorded start state) plus a non-finite check. The angular '
             'terms (> 200 rad/s) cannot be resolved at 60 Hz (finite differences top out at pi/dt = 188 rad/s) and dof '
             'velocities are not recorded.')
    t.append('rollout filtering: ManipTrans\'s player drops rollouts shorter than 20 steps (lib/rl/player.py:432-444) and the '
             'HO-Tracker scorer skips sequences without a success; neither applies (no early termination, every lane scored).')
    p = deviations['spider']
    p.append('object root (bottom part) pose, as SPIDER on DexMachina/ARCTIC; the hinge is not part of success (d_arti).')
    p.append(f'control rate 1/dt = {1 / dt:.0f} Hz, not resampled to SPIDER\'s 50 Hz ref_dt (get_success_rate.py:452-460); a '
             'per-step mean over uniform steps is a time average either way.')
    p.append('raw world positions (the paper formula and evaluate_dexmachina.py); the mean-centred variant of the code\'s '
             'two-object branch is SR_centered.')
    p.append('SPIDER scores one rollout per (task, seed) and reads lane 0 of eval_ep0.npy for DexMachina RL '
             '(evaluate_dexmachina.py:181-204); lane 0 is reported separately.')
    p.append('SPIDER v2 Table 3 ARCTIC success numbers cannot be reproduced from the released code (no script computes a '
             'success flag for the DexMachina workflow); v3 dropped them for ADD-AUC.')

    for s in SIDES:
        sides[s].pop('_per_lane')
    summary = dict(
        task=meta.get('task'), clip=meta.get('clip'), hands=hand_names, lanes=L, steps=S,
        add_auc=_r(meta.get('add_auc')),
        mpjpe_cm=mp['MPJPE_cm']['mean'] if mp['MPJPE_cm'] else None,
        mpjpe_tip_cm=mp['tip_cm']['mean'] if mp['tip_cm'] else None,
        mpjpe_wrist_cm=mp['wrist_cm']['mean'] if mp['wrist_cm'] else None,
        wrist_rot_deg=mp['wrist_rot_deg']['mean'] if mp['wrist_rot_deg'] else None,
        mano_wrist_tips_cm=mp['vs_mano']['wrist_tips_cm']['mean'] if mp['vs_mano'] else None,
        mano_wrist_tips_floor_cm=(mp['vs_mano']['reference_floor'] or {}).get('wrist_tips_mano') if mp['vs_mano'] else None,
        maniptrans_sr=mtb['SR_v2d'], maniptrans_E_t_cm_v2d=mtb['E_t_cm_v2d_all'], maniptrans_E_r_deg_v2d=mtb['E_r_deg_v2d_all'],
        maniptrans_sr_hand_object=mtb.get('SR_hand_object'), maniptrans_sr_robot_reference=mtb.get('SR_robot_reference'),
        maniptrans_sr_paper_text=mtb.get('SR_paper_text'), maniptrans_sr_object_only=mtb.get('SR_object_only'),
        maniptrans_E_r_deg_all=mtb['E_r_deg_all'], maniptrans_E_t_cm_all=mtb['E_t_cm_all'],
        maniptrans_E_j_cm_all=mtb.get('E_j_cm_all'), maniptrans_E_ft_cm_all=mtb.get('E_ft_cm_all'),
        spider_sr=spb['SR'], spider_E_pos_m=spb['E_pos_m']['mean'], spider_E_rot_rad=spb['E_rot_rad']['mean'],
        spider_E_arti_rad=spb['E_arti_rad']['mean'],
    )
    return dict(tool='dexmachina/eval/tracking_metrics.py', tool_version=TOOL_VERSION, summary=summary,
                inputs=dict(missing=missing), hands={s: {k: v for k, v in sides[s].items()
                                                         if not (k.startswith(('maniptrans_', 'mean_', 'median_'))
                                                                 or k == 'reference_floor')}
                                                     for s in SIDES},
                mpjpe=mp, maniptrans=mtb, spider=spb, deviations=deviations,
                definitions=__doc__)


def score(rec_dir, env_pkl=None):
    """Load a slip_record.py recording dir (and the run's env.pkl) and compute everything."""
    rec_dir = Path(rec_dir)
    z = np.load(rec_dir / 'recording.npz', allow_pickle=False)
    rec = {k: z[k] for k in z.files}
    meta = json.loads((rec_dir / 'record.json').read_text())
    env_path = Path(env_pkl) if env_pkl else find_env_pkl(meta, rec_dir)
    env = load_env_pkl(env_path) if env_path is not None else None
    T = rec['demo'].shape[0]
    notes = {}
    ref = mano = None
    if env is not None and 'retarget_data' in env:
        ref = {s: reference_of(env, s) for s in SIDES}
    mano, mano_notes = {}, {}
    try:
        for s in SIDES:
            mano[s], mano_notes[s] = mano_of(rec, env, meta, s, T)
    except (OSError, KeyError, ValueError) as e:
        mano, notes['mano_error'] = None, str(e)
    hands = {s: hand_of(meta, s, env) for s in SIDES}
    out = compute(rec, meta, ref, mano, hands, notes)
    if env is not None:
        dd = env.get('demo_data', {})
        if all(k in dd for k in ('obj_pos', 'obj_quat')):
            out['inputs']['env_demo_vs_recording_demo_max_abs'] = _r(max(
                np.abs(_np(dd['obj_pos'])[:T] - rec['demo'][:, :3]).max(), np.abs(_np(dd['obj_quat'])[:T] - rec['demo'][:, 3:7]).max()), 8)
    out['inputs'].update(recording=str(rec_dir.resolve() / 'recording.npz'), record_json=str(rec_dir.resolve() / 'record.json'),
                         env_pkl=str(env_path) if env_path else None, mano=mano_notes or None,
                         checkpoint=meta.get('inputs', {}).get('checkpoint'),
                         checkpoint_sha256=meta.get('inputs', {}).get('checkpoint_sha256'))
    slip = rec_dir / 'slip_auc.json'
    if slip.is_file():
        out['summary']['slip_auc'] = _r(json.loads(slip.read_text()).get('slip_auc'))
    return out


def summary_line(res, name=''):
    s = res['summary']
    f = lambda v, nd=2: '   -' if v is None else f'{v:.{nd}f}'                        # noqa: E731
    return (f"{name or s.get('task') or '?':<28} ADD {f(s['add_auc'], 3)}  MPJPE {f(s['mpjpe_cm'])} cm (tip {f(s['mpjpe_tip_cm'])}, "
            f"wrist {f(s['mpjpe_wrist_cm'])})  MANO wrist+tips {f(s['mano_wrist_tips_cm'])} (floor {f(s['mano_wrist_tips_floor_cm'])})  "
            f"ManipTrans SR (V2D, object) {f(s['maniptrans_sr'])} [with hands {f(s['maniptrans_sr_hand_object'])}, robot-ref "
            f"{f(s['maniptrans_sr_robot_reference'])}, paper-text {f(s['maniptrans_sr_paper_text'])}]  SPIDER SR {f(s['spider_sr'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('recordings', nargs='+', help='slip_record.py output dirs (recording.npz + record.json)')
    ap.add_argument('--env_pkl', default=None, help="the run's params/env.pkl (default: from record.json); one recording only")
    ap.add_argument('--out', default=None, help='output JSON (default <recording dir>/tracking_metrics.json); one recording only')
    a = ap.parse_args(argv)
    if len(a.recordings) > 1 and (a.env_pkl or a.out):
        ap.error('--env_pkl / --out take a single recording')
    for r in a.recordings:
        res = score(r, a.env_pkl)
        out = Path(a.out) if a.out else Path(r) / 'tracking_metrics.json'
        out.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
        print(summary_line(res, Path(r).name), flush=True)


if __name__ == '__main__':
    main()
