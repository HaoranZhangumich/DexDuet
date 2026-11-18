"""Hold test, step 3 (CPU): per-run metrics from a run folder written by run.py and add.py.

The functions auc_overall, quat_angle_deg and lane_metrics are those of the paper's analysis,
unchanged. Per run, for a lane set (all 100 lanes = primary; lanes 0-98 = sensitivity):
  Hold-ADD-AUC  official compute_add_stats AUC on the (h, j) pairs h = 1..H: per part accuracy(tau) = mean(ADD_p < tau),
                tau = 0.01..0.09 m, trapz over linspace(0, 1, 9), mean of the two parts. Non-finite lanes: ADD = +inf.
  Hold-SR@3/5   fraction of lanes with max_h ADD(h, j) <= 0.03 / 0.05 m (ADD = mean of the two part ADDs).
  drift         at h = 60/120/180 (1/2/3 s): median and mean over finite lanes of the root position error (cm), the root
                rotation error (deg), the hinge error (deg), the object ADD (cm) and the displacement since the clip end (cm).
  drop rate     fraction of lanes with z(s_H) < z(g) - 0.10 m, or z(s_H) < 0.60 m while z(g) >= 0.60 m; non-finite lanes
                count as dropped.
  contact       per finite lane, the share of hold steps with >= 1 valid contact between the object and any collision link
                of either hand; finger links only (links on the hand's finger chains, dexmachina/envs/finger_spec.py); the
                left hand; the right hand. Per run: mean over lanes.
Conditional ('c_' prefix): the same over on-track lanes (finite throughout, object ADD(0, j) <= 0.03 m at the clip end);
undefined (NaN) when a run has no on-track lane. Prefix 'u_' = all lanes; 'l98_' = lanes 0-98.

  python -m dexmachina.eval.hold_test.metrics <run folder>      # prints the run's metrics (also written to hold_metrics.json)
"""
import json
import sys
from os.path import join

import numpy as np

TAU = list(np.arange(0.01, 0.1, 0.01))
HOLD_T = (60, 120, 180)
HAND_KEY = {'Inspire': 'inspire_hand', 'Allegro': 'allegro_hand', 'XHand': 'xhand', 'Schunk': 'schunk_hand'}
# finger links per side in the contact masks (left side shown; the paper's analysis asserted these counts)
FINGER_LINK_COUNTS = {'Inspire': (12, 13), 'Allegro': (20, 21), 'XHand': (12, 13), 'Schunk': (20, 23)}
# task classes of the protocol, from the demonstration at its last frame (object resting on the support box, or in hand)
TASK_CLASS = {'box-200': 'resting', 'notebook-300': 'resting', 'ketchup-100': 'in-hand', 'ketchup-300': 'in-hand',
              'mixer-170': 'in-hand', 'mixer-300': 'in-hand', 'waffleiron-300': 'in-hand'}


def trapz(y, x):
    return float(np.trapz(y, x=x))


def auc_overall(addB, addT):
    """compute_add_stats(...)['auc']['overall'] on arrays of (lanes, frames)."""
    x = np.linspace(0, 1, len(TAU))
    aucs = []
    for a in (addB, addT):
        acc = np.array([np.mean(a < t) for t in TAU])
        aucs.append(trapz(acc, x))
    return float(np.mean(aucs))


def quat_angle_deg(qg, qs):
    """geodesic angle (deg) of q_g^-1 q_s, quaternions wxyz: 2 atan2(|v|, |w|)."""
    w1, v1 = qg[..., 0], qg[..., 1:]
    w2, v2 = qs[..., 0], qs[..., 1:]
    # conj(q_g) * q_s
    w = w1 * w2 + np.sum(v1 * v2, -1)
    v = w1[..., None] * v2 - w2[..., None] * v1 - np.cross(v1, v2)
    return np.degrees(2 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w)))


def lane_metrics(addB, addT, phys, g, contact, masks, lanes, L, H):
    """metrics over the lane index array `lanes` (section 6); addB/addT (N, H+1) with +inf on non-finite lanes."""
    out = {}
    n = len(lanes)
    out['n_lanes'] = n
    if n == 0:
        return out
    aB, aT = addB[lanes], addT[lanes]
    aO = 0.5 * (aB + aT)
    fin = np.isfinite(aO).all(1)
    out['n_nonfinite_lanes'] = int((~fin).sum())
    out['hold_add_auc'] = 100 * auc_overall(aB[:, 1:], aT[:, 1:])
    mx = aO[:, 1:].max(1)
    out['hold_sr3'] = 100 * float(np.mean(mx <= 0.03))
    out['hold_sr5'] = 100 * float(np.mean(mx <= 0.05))
    out['mean_hold_add_cm'] = 100 * float(np.mean(aO[fin, 1:])) if fin.any() else np.nan
    zH = phys[L - 1 + H, lanes, 2].astype(np.float64)
    zg = float(g[2])
    drop = (zH < zg - 0.10) | ((zH < 0.60) & (zg >= 0.60)) | ~fin
    out['drop_rate'] = 100 * float(np.mean(drop))
    fl = lanes[fin]
    s0 = phys[L - 1, fl].astype(np.float64)
    for h in HOLD_T:
        sh = phys[L - 1 + h, fl].astype(np.float64)
        d = dict(pos=100 * np.linalg.norm(sh[:, :3] - g[None, :3], axis=-1),
                 rot=quat_angle_deg(np.broadcast_to(g[3:7], sh[:, 3:7].shape), sh[:, 3:7]),
                 hinge=np.degrees(np.abs(sh[:, 7] - g[7])),
                 add=100 * aO[fin, h],
                 disp=100 * np.linalg.norm(sh[:, :3] - s0[:, :3], axis=-1))
        for k, v in d.items():
            out[f'{k}_{h // 60}s_med'] = float(np.median(v)) if len(v) else np.nan
            out[f'{k}_{h // 60}s_mean'] = float(np.mean(v)) if len(v) else np.nan
    c = contact[L:L + H][:, fl]                        # (H, lanes, parts, links): rows after hold steps 1..H
    for k, m in masks.items():
        if len(fl) == 0:
            out[f'contact_{k}'] = np.nan
            continue
        any_c = c[..., m].reshape(H, len(fl), -1).any(-1)   # (H, lanes)
        out[f'contact_{k}'] = 100 * float(np.mean(any_c.mean(0)))
    return out


def finger_links(hand, side):
    from dexmachina.envs import finger_spec as FS
    return {l for ch in FS.load_finger_spec(HAND_KEY[hand], side).chains.values() for l in ch}


def replay_checks(meta, z):
    """Protocol section 4 on hold_eval.json and hold_rollout.npz (the paper's replay checks, without the check that the
    run belongs to the paper's run list): start state, prefix identity, last clip step, frozen references, no reset, overflow.
    A run that fails any of them is flagged ('replay not identical'): kept in the primary analysis, left out in the
    sensitivity analysis."""
    bw = lambda d: bool(isinstance(d, dict) and d.get('bitwise_equal'))
    L = int(meta['L'])
    S = int(z['phys_state'].shape[0])
    t = L - 1
    prefix = bw(meta.get('prefix_0_to_L-2_vs_scored'))
    for key in ('demo_state_vs_scored', 'prefix_vs_slip_recording'):
        if meta.get(key) is not None:
            prefix = prefix and bw(meta[key])
    last, slip_last = meta.get('last_row_pre_reset_vs_scored'), meta.get('last_row_pre_reset_vs_slip_final_state')
    last_needed = meta.get('study_protocol') in ('v2', 'dexmachina')
    last_ok = (bw(last) if last_needed else last is None or bw(last)) and (slip_last is None or bw(slip_last))
    demo_idx = sorted({int(x) for x in meta.get('eval_demo_index_lane0_during_hold', [])})
    ph = meta.get('phase_obs_values_during_patch', [])
    frozen = dict(
        ebuf_env=meta.get('ebuf_env_used_during_patch') == [t], ebuf_robot=meta.get('ebuf_robot_used_during_patch') == [t],
        ebuf_obj=meta.get('ebuf_obj_used_during_patch') == [t], robot_t=meta.get('robot_t_at_action_during_hold') == [t],
        demo_index=demo_idx == [t], demo_rows=meta.get('demo_state_rows_during_hold_equal_final_ref') is True,
        residual_hold=meta.get('residual_target_hold_equals_last_frame') is True,
        residual_last_clip=meta.get('residual_target_last_clip_step_equals_last_frame') is True,
        phase=len(ph) == 1 and ph[0] == meta.get('phase_obs_last_clip_observation_acted_on') and abs(ph[0] - (2.0 * t / L - 1.0)) <= 1e-6,
        state_diff=float(meta.get('state_diff_goal_is_last_frame_maxabs', np.inf)) <= 1e-6,
        state_diff_obs=float(meta.get('state_diff_obs_matches_buffer_maxabs', np.inf)) <= 1e-6,
        reward_pos_dist=float(meta.get('reward_pos_dist_vs_last_frame_maxabs', np.inf)) <= 1e-5)
    nan_any = z['nan'].any(0)
    clip_resets = int(z['dones'][:L - 1][:, ~nan_any].sum())
    ovf = meta.get('overflow') or []
    return dict(
        start_state=(meta.get('reset_verify') or {}).get('verification') == 'exact_arrays_equal_before_first_policy_action',
        prefix=prefix, last_row=last_ok, frozen=all(frozen.values()),
        no_reset=clip_resets == 0 and meta.get('resets_in_patch_steps_non_nan_lanes') == 0
        and meta.get('hold_patch_steps') == [L - 1, S - 1],
        overflow=len(ovf) > 0 and all(o['pair_overflow_total'] == 0 and o['contact_overflow_total'] == 0 for o in ovf))


def run_metrics(folder, hand=None, method=None, task=None, seed=None):
    """One row of per-run metrics for a run folder with hold_eval.json, hold_rollout.npz and hold_add.npz."""
    meta = json.load(open(join(folder, 'hold_eval.json')))
    hand = hand or meta.get('hand')
    a = np.load(join(folder, 'hold_add.npz'))
    z = np.load(join(folder, 'hold_rollout.npz'))
    L, H = int(z['L']), int(z['H'])
    addB, addT, bad = a['bottom'], a['top'], a['bad']
    N = addB.shape[0]
    assert addB.shape == (N, H + 1)
    phys = z['phys_state']
    g = z['final_ref'].astype(np.float64)
    contact = z['contact'].astype(bool)
    nl = int(z['num_left_contact_links'])
    left = [str(x) for x in z['contact_left_links']]
    right = [str(x) for x in z['contact_right_links']]
    assert contact.shape[-1] == len(left) + len(right) == 2 * nl
    links = left + right
    side = ['left'] * len(left) + ['right'] * len(right)
    fingers = {s: finger_links(hand, s) for s in ('left', 'right')}
    fmask = np.array([lk in fingers[s] for lk, s in zip(links, side)])
    masks = dict(any=np.ones(len(links), bool), finger=fmask,
                 left=np.array([s == 'left' for s in side]), right=np.array([s == 'right' for s in side]))
    assert np.array_equal(bad, z['nan'].any(0) | ~np.isfinite(phys).all(axis=(0, 2)))
    if hand in FINGER_LINK_COUNTS:
        assert (int(fmask[:len(left)].sum()), len(left)) == FINGER_LINK_COUNTS[hand], (folder, hand)
    clip_bad = z['nan'][:L].any(0) | ~np.isfinite(phys[:L]).all(axis=(0, 2))
    aO0 = 0.5 * (addB[:, 0] + addT[:, 0])
    ontrack = (~bad) & (aO0 <= 0.03)
    checks = replay_checks(meta, z)
    task = task or meta.get('task')
    row = dict(key=meta.get('name'), hand=hand, method=method or meta.get('method'), task=task,
               task_class=TASK_CLASS.get(task), seed=int(seed if seed is not None else meta.get('seed')),
               flagged=not all(checks.values()), failed_checks=';'.join(k for k, v in checks.items() if not v),
               study_protocol=meta.get('study_protocol'), nan_lanes=';'.join(str(j) for j in np.where(bad)[0]),
               n_clip_nonfinite_lanes=int(clip_bad.sum()), table_add_auc_pct=meta.get('table_add_pct'))
    for lname, lanes in (('', np.arange(N)), ('l98_', np.arange(N - 1))):
        on = lanes[ontrack[lanes]]
        row[lname + 'n_ontrack'] = int(len(on))
        row[lname + 'ontrack_frac'] = 100 * len(on) / len(lanes)
        um = lane_metrics(addB, addT, phys, g, contact, masks, lanes, L, H)
        cm = lane_metrics(addB, addT, phys, g, contact, masks, on, L, H) if len(on) else {}
        for k, v in um.items():
            row[f'{lname}u_{k}'] = v
        for k in um:
            row[f'{lname}c_{k}'] = cm.get(k, np.nan) if k != 'n_lanes' else len(on)
    return row


def main():
    folder = sys.argv[1]
    row = run_metrics(folder)
    json.dump(row, open(join(folder, 'hold_metrics.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: row[k] for k in ('key', 'hand', 'method', 'task', 'seed', 'flagged', 'u_hold_add_auc', 'u_hold_sr5',
                                          'u_drop_rate', 'n_ontrack', 'c_hold_add_auc')}, default=float))


if __name__ == '__main__':
    main()
