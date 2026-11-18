"""Offline tracking metrics (dexmachina/eval/tracking_metrics.py). Covered here:
  - the robot-to-MANO tables against ManipTrans's own;
  - perfect tracking on synthetic recordings (0 error, every success rate 1);
  - known offsets against the ManipTrans and SPIDER thresholds;
  - the 8 grace rows, per-lane success rates, non-finite lanes;
  - the rotation-angle formula;
  - the file round trip (recording dir + params/env.pkl -> tracking_metrics.json).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_tracking_metrics.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.envs.finger_spec import load_finger_spec  # noqa: E402
from dexmachina.eval import tracking_metrics as tm  # noqa: E402

# ManipTrans github.com/ManipTrans/ManipTrans @ a3d08cf, maniptrans_envs/lib/envs/dexhands/<hand>.py: body -> to_hand(body)[0]
# (first human joint in hand2dex_mapping order), the weight_idx tip slots and level groups; the wrist body is left out (we
# compare our wrist link with MANO 0, see the module docstring).
P, I, D = 'proximal', 'intermediate', 'distal'


def _allegro(index, ring):
    """allegro.py:84-106 (right: index = links 0-3, ring = links 8-11) and :201-223 (left: mirrored)."""
    body = {}
    for finger, (a, b, c, d) in (('thumb', (12, 13, 14, 15)), ('index', index), ('middle', (4, 5, 6, 7)), ('ring', ring)):
        body.update({f'link_{a}.0': f'{finger}_{P}', f'link_{b}.0': f'{finger}_{P}', f'link_{c}.0': f'{finger}_{I}',
                     f'link_{d}.0': f'{finger}_{D}', f'link_{d}.0_tip': f'{finger}_tip'})
    tips = dict(thumb='link_15.0_tip', index=f'link_{index[3]}.0_tip', middle='link_7.0_tip', ring=f'link_{ring[3]}.0_tip',
                pinky=f'link_{ring[3]}.0_tip')                              # weight_idx pinky_tip == ring_tip
    level_1 = {f'link_{k}.0' for k in (0, 1, 12, 13, 4, 5, 8, 9)}            # weight_idx level_1_joints [1,2,6,7,11,12,16,17]
    level_2 = {f'link_{k}.0' for k in (2, 3, 14, 15, 6, 7, 10, 11)}          # level_2_joints [3,4,8,9,13,14,18,19]
    return body, tips, level_1, level_2


def _xhand(side):
    """xhand.py:49-71 hand2dex_mapping and :100-108 weight_idx; our URDF prefixes the links with left_/right_."""
    n = lambda x: f'{side}_hand_{x}'                                         # noqa: E731
    body = {n('thumb_bend_link'): f'thumb_{P}', n('thumb_rota_link1'): f'thumb_{P}', n('thumb_rota_link2'): f'thumb_{I}',
            n('thumb_rota_tip'): 'thumb_tip', n('index_bend_link'): f'index_{P}', n('index_rota_link1'): f'index_{P}',
            n('index_rota_link2'): f'index_{I}', n('index_rota_tip'): 'index_tip', n('mid_link1'): f'middle_{P}',
            n('mid_link2'): f'middle_{I}', n('mid_tip'): 'middle_tip', n('ring_link1'): f'ring_{P}', n('ring_link2'): f'ring_{I}',
            n('ring_tip'): 'ring_tip', n('pinky_link1'): f'pinky_{P}', n('pinky_link2'): f'pinky_{I}', n('pinky_tip'): 'pinky_tip'}
    tips = dict(thumb=n('thumb_rota_tip'), index=n('index_rota_tip'), middle=n('mid_tip'), ring=n('ring_tip'),
                pinky=n('pinky_tip'))
    level_1 = {n(x) for x in ('index_bend_link', 'index_rota_link1', 'mid_link1', 'pinky_link1', 'ring_link1', 'thumb_bend_link',
                              'thumb_rota_link1')}                           # [1, 2, 5, 8, 11, 14, 15]
    level_2 = {n(x) for x in ('index_rota_link2', 'mid_link2', 'pinky_link2', 'ring_link2', 'thumb_rota_link2')}  # [3,6,9,12,16]
    return body, tips, level_1, level_2


def _inspire(p):
    """inspire.py:48-70 and :99-107, by link name: FF/MF/RF/LF = index/middle/ring/pinky, L1/L2 = proximal/intermediate,
    TH_L1..L4 = thumb_proximal_base/proximal/intermediate/distal."""
    body = {f'{p}_TH_L1': f'thumb_{P}', f'{p}_TH_L2': f'thumb_{P}', f'{p}_TH_L3': f'thumb_{I}', f'{p}_TH_L4': f'thumb_{D}',
            f'{p}_TH_Tip': 'thumb_tip'}
    for code, finger in (('FF', 'index'), ('MF', 'middle'), ('RF', 'ring'), ('LF', 'pinky')):
        body.update({f'{p}_{code}_L1': f'{finger}_{P}', f'{p}_{code}_L2': f'{finger}_{I}', f'{p}_{code}_Tip': f'{finger}_tip'})
    tips = dict(thumb=f'{p}_TH_Tip', index=f'{p}_FF_Tip', middle=f'{p}_MF_Tip', ring=f'{p}_RF_Tip', pinky=f'{p}_LF_Tip')
    level_1 = {f'{p}_FF_L1', f'{p}_MF_L1', f'{p}_TH_L2'}                     # [1, 4, 14]
    level_2 = {f'{p}_FF_L2', f'{p}_MF_L2', f'{p}_LF_L1', f'{p}_LF_L2', f'{p}_RF_L1', f'{p}_RF_L2', f'{p}_TH_L1', f'{p}_TH_L3',
               f'{p}_TH_L4'}                                                 # [2, 5, 7, 8, 10, 11, 13, 15, 16]
    return body, tips, level_1, level_2


OFFICIAL = {('allegro_hand', 'right'): _allegro((0, 1, 2, 3), (8, 9, 10, 11)),
            ('allegro_hand', 'left'): _allegro((8, 9, 10, 11), (0, 1, 2, 3)),
            ('xhand', 'right'): _xhand('right'), ('xhand', 'left'): _xhand('left'),
            ('inspire_hand', 'right'): _inspire('R'), ('inspire_hand', 'left'): _inspire('L')}


@pytest.mark.parametrize('hand,side', sorted(OFFICIAL))
def test_body_tables_match_maniptrans(hand, side):
    body, tips, level_1, level_2 = OFFICIAL[(hand, side)]
    mt = tm.maniptrans_hand(load_finger_spec(hand, side))
    assert mt['bodies'][0]['role'] == 'wrist' and mt['bodies'][0]['mano'] == 0
    got = {b['link']: tm.MANO_JOINT_NAMES[b['mano']] for b in mt['bodies'][1:]}
    assert got == body
    assert mt['tip_slots'] == tips
    assert set(mt['level_1']) == level_1 and set(mt['level_2']) == level_2


def test_chain_roles():
    assert tm.chain_roles(2) == (P, I)
    assert tm.chain_roles(3) == (P, P, I)
    assert tm.chain_roles(4) == (P, P, I, D)
    assert tm.chain_roles(5) == (P, P, P, I, D)


def test_quat_angle_matches_both_papers():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((500, 4)), rng.standard_normal((500, 4))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    got = tm.quat_angle(a, b)
    spider = np.arccos(np.clip(2 * np.sum(a * b, 1) ** 2 - 1, -1, 1))                       # SPIDER v2 Sec. 3.1
    Ra, Rb = (Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix() for q in (a, b))
    th = np.linalg.norm(Rotation.from_matrix(Ra @ np.transpose(Rb, (0, 2, 1))).as_rotvec(), axis=1)
    maniptrans = np.minimum(th, 2 * np.pi - th)                                             # eval_score.py:46-56
    assert np.allclose(got, spider, atol=1e-6) and np.allclose(got, maniptrans, atol=1e-9)
    assert tm.quat_angle(a, -a).max() < 1e-12                                               # sign-invariant, exact at 0


# ------------------------------------------------------------------------------------------------------------------------
WRIST_POINT = {'allegro_hand': 'base_link'}      # Allegro's wrist base_dummy_link is not a keypoint; base_link is fixed to it


def _quat_traj(rng, T, max_angle):
    ax = rng.standard_normal(3)
    ax /= np.linalg.norm(ax)
    ang = np.linspace(0, max_angle, T)
    q0 = rng.standard_normal(4)
    q0 /= np.linalg.norm(q0)
    return tm._qmul(np.concatenate([np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax], 1), q0[None])


def synthetic(hand, S=30, L=4, seed=0):
    """A perfect tracker: the robot keypoints sit on their MANO joints, the reference equals the recording, the object
    follows the demo exactly; row i = frame i+1."""
    rng = np.random.default_rng(seed)
    T = S + 1
    t = np.linspace(0, 1, T)
    rec, ref, mano = {}, {}, {}
    for s in tm.SIDES:
        spec = load_finger_spec(hand, s)
        mt = tm.maniptrans_hand(spec)
        M = 0.4 + 0.05 * rng.standard_normal((1, 21, 3)) + 0.1 * t[:, None, None] * rng.standard_normal(3)
        q = _quat_traj(rng, T, 0.6)
        pos = {b['link']: M[:, b['mano']] for b in mt['bodies'][1:]}
        wp = WRIST_POINT.get(hand, spec.wrist)
        if wp == spec.wrist:
            pos[wp] = M[:, 0]
        else:
            child, off = tm.wrist_child_offset(spec.urdf_path, spec.wrist, [wp])
            pos[wp] = M[:, 0] + tm.rotate(q, off)
        names = list(spec.tips) + [wp] + [b['link'] for b in mt['bodies'][1:] if b['role'] != 'tip'] + list(spec.tips)
        kp = np.stack([pos[n] for n in names], 1)                                         # (T, K, 3)
        rec[f'kpts_{s}'] = np.repeat(kp[1:, None], L, 1)
        rec[f'start_kpts_{s}'] = np.repeat(kp[:1], L, 0)
        rec[f'wrist_quat_{s}'] = np.repeat(q[1:, None], L, 1)
        rec[f'kpt_names_{s}'] = np.array(names)
        rec[f'demo_hand_joints_{s}'] = M
        ref[s] = dict(kpt_pos=kp, kpt_names=names, wrist_pose=np.concatenate([M[:, 0], q], 1))
        mano[s] = M
    demo = np.concatenate([0.5 + 0.2 * t[:, None] * rng.standard_normal(3), _quat_traj(rng, T, 1.0), 0.4 * t[:, None]], 1)
    rec.update(obj_state=np.repeat(demo[1:, None], L, 1), demo=demo, dt=np.float64(1 / 60), start_obj=np.repeat(demo[:1], L, 0),
               final_state=np.repeat(demo[-1:], L, 0))
    meta = dict(task='synthetic', clip='box-0-31-s01-u01', robots={s: f'{hand}_{s}' for s in tm.SIDES}, add_auc=1.0)
    return rec, meta, ref, mano


def run(hand, rec, meta, ref, mano):
    res = tm.compute(rec, meta, ref, mano, {s: hand for s in tm.SIDES})
    json.dumps(res, allow_nan=False)                                                       # always valid JSON
    return res


@pytest.mark.parametrize('hand', ['inspire_hand', 'allegro_hand', 'xhand'])
def test_perfect_tracking(hand):
    res = run(hand, *synthetic(hand))
    s = res['summary']
    for k in ('mpjpe_cm', 'mpjpe_tip_cm', 'mpjpe_wrist_cm', 'wrist_rot_deg', 'mano_wrist_tips_cm', 'mano_wrist_tips_floor_cm',
              'maniptrans_E_r_deg_all', 'maniptrans_E_t_cm_all', 'maniptrans_E_j_cm_all', 'maniptrans_E_ft_cm_all',
              'spider_E_pos_m', 'spider_E_rot_rad', 'spider_E_arti_rad'):
        assert s[k] == pytest.approx(0.0, abs=1e-6), k
    for k in ('maniptrans_sr', 'maniptrans_sr_hand_object', 'maniptrans_sr_robot_reference', 'maniptrans_sr_paper_text',
              'maniptrans_sr_object_only', 'spider_sr'):
        assert s[k] == 1.0, k
    mt = res['maniptrans']
    assert mt['SR_hand_only'] == 1.0 and mt['E_j_cm'] == pytest.approx(0.0, abs=1e-6)       # success-only errors exist
    assert all(mt['per_lane']['success_hand_object']) and all(mt['per_lane']['success_v2d'])
    assert all(res['spider']['per_lane']['success'])
    assert mt['SR_v2d_reward_pairing'] == 1.0 and mt['E_t_cm_v2d_all'] > 0.1               # V2D pairs row i with frame i
    for side in tm.SIDES:
        assert res['maniptrans']['per_hand'][side]['reference_floor']['maniptrans_hand_test_passes']
    assert res['hands']['left']['n_unique_keypoints'] < res['hands']['left']['n_keypoints']   # duplicated tips dropped


def _shift_hands(rec, dx):
    for s in tm.SIDES:
        rec[f'kpts_{s}'] = rec[f'kpts_{s}'] + np.array([dx, 0, 0])


@pytest.mark.parametrize('hand', ['inspire_hand', 'allegro_hand'])
def test_hand_offsets(hand):
    rec, meta, ref, mano = synthetic(hand)
    _shift_hands(rec, 0.02)                                                                # every link 2 cm off
    s = run(hand, rec, meta, ref, mano)['summary']
    assert s['mpjpe_cm'] == pytest.approx(2.0) and s['mpjpe_tip_cm'] == pytest.approx(2.0)
    assert s['mpjpe_wrist_cm'] == pytest.approx(2.0) and s['maniptrans_E_j_cm_all'] == pytest.approx(2.0)
    assert s['maniptrans_sr'] == 1.0 and s['spider_sr'] == 1.0                             # 2 cm < 6 cm tip, < 8 cm groups
    rec, meta, ref, mano = synthetic(hand)
    _shift_hands(rec, 0.07)                                                                # tips 7 cm > 6 cm
    s = run(hand, rec, meta, ref, mano)['summary']
    assert s['maniptrans_sr_hand_object'] == 0.0 and s['maniptrans_sr_robot_reference'] == 0.0
    assert s['maniptrans_sr_paper_text'] == 0.0                                            # E_ft 7 > 6
    assert s['maniptrans_sr_object_only'] == 1.0 and s['spider_sr'] == 1.0                 # object-only metrics untouched
    assert s['maniptrans_sr'] == 1.0                                                       # V2D: no hand term


def _move_object(rec, dpos=(0, 0, 0), rot_deg=0.0, rows=slice(None), lanes=slice(None)):
    o = rec['obj_state'].copy()
    o[rows, lanes, :3] += np.asarray(dpos)
    if rot_deg:
        qz = np.array([np.cos(np.radians(rot_deg) / 2), 0, 0, np.sin(np.radians(rot_deg) / 2)])
        o[rows, lanes, 3:7] = tm._qmul(qz, o[rows, lanes, 3:7])
    rec['obj_state'] = o


@pytest.mark.parametrize('dpos,rot,mt_sr,sp_sr', [
    ((0.02, 0, 0), 0.0, 1.0, 1.0),      # 2 cm: both pass
    ((0.05, 0, 0), 0.0, 0.0, 1.0),      # 5 cm: ManipTrans (3 cm per frame) fails, SPIDER (10 cm mean) passes
    ((0.12, 0, 0), 0.0, 0.0, 0.0),      # 12 cm: both fail
    ((0, 0, 0), 25.0, 1.0, 1.0),        # 25 deg = 0.436 rad: both pass
    ((0, 0, 0), 35.0, 0.0, 0.0),        # 35 deg > 30 deg; 0.611 rad > 0.5 rad
])
def test_object_thresholds(dpos, rot, mt_sr, sp_sr):
    rec, meta, ref, mano = synthetic('inspire_hand')
    _move_object(rec, dpos, rot)
    res = run('inspire_hand', rec, meta, ref, mano)
    assert res['summary']['maniptrans_sr_hand_object'] == mt_sr and res['summary']['maniptrans_sr_object_only'] == mt_sr
    assert res['summary']['maniptrans_sr'] == mt_sr                                        # V2D: constant offset, same verdict
    assert res['summary']['spider_sr'] == sp_sr
    assert res['summary']['maniptrans_E_t_cm_all'] == pytest.approx(100 * np.linalg.norm(dpos))
    assert res['maniptrans']['E_r_bottom_deg_all'] == pytest.approx(rot) and res['maniptrans']['E_r_top_deg_all'] == pytest.approx(rot)


def test_grace_rows():
    rec, meta, ref, mano = synthetic('inspire_hand')
    _move_object(rec, (0.05, 0, 0), rows=slice(0, tm.MT_GRACE))                           # rows 0..7 only: exempt
    assert run('inspire_hand', rec, meta, ref, mano)['summary']['maniptrans_sr_hand_object'] == 1.0
    rec, meta, ref, mano = synthetic('inspire_hand')
    _move_object(rec, (0.05, 0, 0), rows=slice(tm.MT_GRACE, tm.MT_GRACE + 1))             # row 8: fails
    res = run('inspire_hand', rec, meta, ref, mano)
    assert res['summary']['maniptrans_sr_hand_object'] == 0.0
    assert set(res['maniptrans']['per_lane']['first_fail_row']) == {tm.MT_GRACE}


def test_per_lane_rates_and_nonfinite():
    rec, meta, ref, mano = synthetic('inspire_hand', L=4)
    _move_object(rec, (0.15, 0, 0), lanes=slice(1, 2))                                     # lane 1 loses the object
    o = rec['obj_state'].copy()
    o[5, 2, 0] = np.nan                                                                    # lane 2 blows up
    rec['obj_state'] = o
    res = run('inspire_hand', rec, meta, ref, mano)
    assert res['maniptrans']['per_lane']['success_hand_object'] == [1, 0, 0, 1]
    assert res['maniptrans']['per_lane']['success_v2d'] == [1, 0, 0, 1]
    assert res['spider']['per_lane']['success'] == [1, 0, 0, 1]
    assert res['summary']['maniptrans_sr'] == 0.5 and res['summary']['spider_sr'] == 0.5
    assert res['spider']['E_pos_m']['nonfinite_lanes'] == 1 and res['maniptrans']['frac_lanes_ever_failing']['nonfinite'] == 0.25


def _v2d_aligned(L=4):
    """A perfect tracker in V2D's pairing: row i (state after step i+1) = demo frame i, final state = the last frame."""
    rec, meta, ref, mano = synthetic('inspire_hand', L=L)
    demo = rec['demo']
    rec['obj_state'], rec['final_state'] = np.repeat(demo[:-1, None], L, 1), np.repeat(demo[-1:], L, 0)
    return rec, meta, ref, mano


def _v2d_move(rec, dpos=(0, 0, 0), dhinge_deg=0.0, rows=slice(None)):
    rows_all = np.concatenate([rec['obj_state'], rec['final_state'][None]])                # (S + 1, L, 8)
    rows_all[rows, :, :3] += np.asarray(dpos)
    rows_all[rows, :, 7] += np.radians(dhinge_deg)
    rec['obj_state'], rec['final_state'] = rows_all[:-1], rows_all[-1]


def test_v2d_pairing():
    """V2D compares the state after step k+1 with demo frame k: that recording scores zero error, ours one frame of lag."""
    mt = run('inspire_hand', *_v2d_aligned(L=2))['maniptrans']
    assert mt['E_t_cm_v2d_all'] == pytest.approx(0.0, abs=1e-9) and mt['E_r_deg_v2d_all'] == pytest.approx(0.0, abs=1e-5)
    assert mt['SR_v2d'] == 1.0
    mt = run('inspire_hand', *synthetic('inspire_hand', L=2))['maniptrans']                # row i = frame i + 1
    assert mt['E_t_cm_v2d_all'] > 0.1 and mt['E_t_cm_all'] == pytest.approx(0.0, abs=1e-9)


def test_v2d_rollout_means_and_parts():
    """V2D (flash_chord maniptrans_success): rollout means, the two parts averaged, strict 3 cm / 30 deg, no hands."""
    rec, meta, ref, mano = _v2d_aligned()
    _v2d_move(rec, (0.05, 0, 0), rows=slice(10, 20))                                       # 10 of 31 rows 5 cm off
    mt = run('inspire_hand', rec, meta, ref, mano)['maniptrans']
    assert mt['E_t_cm_v2d_all'] == pytest.approx(5.0 * 10 / 31, abs=1e-5)
    assert mt['SR_v2d'] == 1.0 and mt['SR_hand_object'] == 0.0                            # per-frame test fails at row 10
    rec, meta, ref, mano = _v2d_aligned()
    _v2d_move(rec, dhinge_deg=40.0)                                                        # top part 40 deg off, bottom exact
    mt = run('inspire_hand', rec, meta, ref, mano)['maniptrans']
    assert mt['E_r_deg_v2d_all'] == pytest.approx(20.0)                                    # (0 + 40) / 2
    assert mt['SR_v2d'] == 1.0
    rec, meta, ref, mano = _v2d_aligned()
    _v2d_move(rec, (0.031, 0, 0))
    mt = run('inspire_hand', rec, meta, ref, mano)['maniptrans']
    assert mt['E_t_cm_v2d_all'] == pytest.approx(3.1) and mt['SR_v2d'] == 0.0             # 3.1 cm mean
    rec, meta, ref, mano = _v2d_aligned()
    _v2d_move(rec, dhinge_deg=62.0)                                                        # parts 0 and 62 deg: mean 31 > 30
    assert run('inspire_hand', rec, meta, ref, mano)['maniptrans']['SR_v2d'] == 0.0


def test_file_round_trip(tmp_path):
    rec, meta, ref, mano = synthetic('allegro_hand')
    run_dir = tmp_path / 'run'
    (run_dir / 'params').mkdir(parents=True)
    env = dict(retarget_data={s: dict(kpts_data=dict(kpt_pos=ref[s]['kpt_pos'], kpt_names=ref[s]['kpt_names']),
                                      wrist_pose=ref[s]['wrist_pose']) for s in tm.SIDES},
               demo_data=dict(obj_pos=rec['demo'][:, :3], obj_quat=rec['demo'][:, 3:7],
                              **{f'hand_joints_{s}': mano[s] for s in tm.SIDES}))
    with open(run_dir / 'params' / 'env.pkl', 'wb') as f:
        pickle.dump(env, f)
    rec_dir = tmp_path / 'slip'
    rec_dir.mkdir()
    rec['demo_hand_joints_left'] = np.zeros((0, 21, 3))                                    # older recordings: no MANO joints
    np.savez_compressed(rec_dir / 'recording.npz', **rec)
    (rec_dir / 'record.json').write_text(json.dumps(dict(meta, inputs=dict(run_dir=str(run_dir)))))
    tm.main([str(rec_dir)])
    out = json.loads((rec_dir / 'tracking_metrics.json').read_text())
    assert out['summary']['maniptrans_sr'] == 1.0 and out['summary']['spider_sr'] == 1.0
    assert out['summary']['mpjpe_cm'] == pytest.approx(0.0, abs=1e-6)
    assert out['inputs']['mano']['left']['source'] == 'env.pkl demo_data hand_joints'
    assert out['inputs']['env_pkl'] == str(run_dir / 'params' / 'env.pkl')
