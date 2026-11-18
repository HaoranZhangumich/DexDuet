"""Build-and-step check of one hand with DexDuet's rewards (GPU): builds the env with the task's training flags from
scripts/train_coordinates_hand.py (--hand, hand graph 0.5, hand-object contact 1.0), steps it with zero actions, and reports
  - the finger spec per side (fingers, missing, wrist, bases, tips, palm frame, gap pairs, contact regions and links);
  - the robot keypoints (retargeted kpt_names), the hand-graph extra nodes and where they come from (wrist_pose or the
    robot's extra buffer), with their shapes; every step the extra positions are checked bitwise against the simulator's
    link positions;
  - the contact anchor points (count and slices per region), and how far they lie from the simulator's own collision geometry
    of their link: per link, the distance of each sampled point (link frame) to the nearest surface of Genesis's collision geoms
    of that link (|max over face planes| of the closest geom; exact for points on or inside a convex geom, which Genesis's
    collision geoms are). Points on boxes, cylinders and spheres and on meshes Genesis keeps as they are should be at ~0;
    a non-convex mesh that Genesis convexifies leaves points inside its hull (the old Inspire path has the same);
  - the palm-normal rule per side (finger_spec.palm_normal_mode) and, for a 'calibrated_axis' hand (Dex3), the wrist-link
    axis: it must be in the env kwargs' reward_cfg (what the trainer pickles into params/env.pkl after building the env),
    equal to the reward module's axis, survive a pickle round trip, and rebuild the same axis from that reward_cfg;
  - per reward key: shape, min / max / mean over all steps, and NaN / inf counts; the median palm-normal angle per side
    (zero actions track the retargeted reference, so it should be near the reference check's angle).
Fails (exit 1) on an exception, a NaN / inf reward, a hand_graph_rew outside [0, weight] or ho_contact_rew outside
[0, weight], ho_contact_check >= 3 mm, an extra-node mismatch or a palm axis missing from the saved kwargs.

  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<patched Genesis>:<repo> python scripts/hand_env_check.py --hand dex3_hand --steps 50 \
      --num-envs 64 --out check.json
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))


def quat_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def anchor_geometry(env, side):
    """Per contact-region link: distance (mm) of its sampled anchor points to the link's Genesis collision geoms."""
    robot = env.robots[side]
    pts = env.ho_contact_points[side]
    local = pts['local'].detach().cpu().numpy().astype(np.float64)
    out = dict()
    for k, (s, e) in enumerate(pts['ranges']):
        link = robot.entity.links[int(pts['link_idx'][k])]
        p = local[s:e]
        dists, types = [], []
        for g in link.geoms:
            v = np.asarray(g.init_verts, dtype=np.float64) @ quat_matrix(g.init_quat).T + np.asarray(g.init_pos, dtype=np.float64)
            f = np.asarray(g.init_faces)
            n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            keep = np.linalg.norm(n, axis=1) > 1e-12
            n = n[keep] / np.linalg.norm(n[keep], axis=1, keepdims=True)
            b = (n * v[f[keep, 0]]).sum(1)
            c = v.mean(0)
            flip = (n @ c - b) > 0                                       # outward normals
            n[flip], b[flip] = -n[flip], -b[flip]
            dists.append(np.abs((p @ n.T - b).max(axis=1)))
            types.append(f'{str(g.type).split(".")[-1]}{"" if g.is_convex else "(nonconvex)"}')
        d = np.min(np.stack(dists), axis=0) * 1000.0
        out[link.name] = dict(points=int(e - s), geoms=types, median_mm=round(float(np.median(d)), 4),
                              p95_mm=round(float(np.percentile(d, 95)), 4), max_mm=round(float(d.max()), 4))
    return out


def env_args(task, hand, num_envs, seed):
    import train_coordinates_hand as tch
    from dexmachina.envs.constructors import get_common_argparser, parse_clip_string
    cmd = tch.training_command(task, seed, 'env-check', 'none', sys.executable, hand=hand)
    argv = cmd[2:]
    argv[argv.index('-B') + 1] = str(num_envs)
    args, unknown = get_common_argparser().parse_known_args(argv)
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object, args.frame_start, args.frame_end = obj_name, start, end
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--hand', required=True)
    ap.add_argument('--task', default='Box-200')
    ap.add_argument('--num-envs', type=int, default=64)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t0 = time.time()
    import genesis as gs
    from genesis.engine.solvers.rigid import collider_decomp
    if not hasattr(collider_decomp.Collider, 'check_overflow'):
        sys.exit('ABORT: Genesis lacks the collision-pair overflow fix (fca4168; no Collider.check_overflow)')
    import dexmachina.envs.base_env as base_env
    from dexmachina.envs.constructors import get_all_env_cfg
    args = env_args(a.task, a.hand, a.num_envs, a.seed)
    kw = get_all_env_cfg(args, device='cuda:0')
    kw['env_cfg']['use_rl_games'] = True
    gs.init(backend=gs.gpu, logging_level='warning')
    env = base_env.BaseEnv(**kw)
    rm = env.reward_module
    report = dict(hand=a.hand, task=a.task, num_envs=env.num_envs, steps=a.steps, genesis=gs.__file__,
                  hand_graph_weight=rm.hand_graph_weight, ho_contact_weight=rm.ho_contact_weight,
                  hand_graph_scale=rm.hand_graph_scale, sides=dict())
    for side, robot in env.robots.items():
        spec = rm.finger_specs[side]
        names = [link.name for link in robot.entity.links]
        pts = env.ho_contact_points[side]
        report['sides'][side] = dict(
            fingers=list(spec.fingers), missing=list(spec.missing), wrist=spec.wrist, bases=list(spec.bases),
            tips=list(spec.tips), palm_nodes=list(spec.palm_nodes), outermost=spec.outermost,
            forward_to_middle=spec.forward_to_middle, gap_pairs=[list(p) for p in spec.gap_pairs],
            finger_weights=[round(x, 6) for x in rm.hand_graph_finger_weights[side].tolist()],
            regions=list(spec.regions), contact_links={r: list(l) for r, l in spec.contact_regions()},
            kpt_names=list(robot.kpt_link_names), n_kpts=robot.n_kpts, missing_kpt_links=list(robot.missing_kpt_links),
            hand_graph_index={k: v.tolist() for k, v in rm.hand_graph_index[side].items()},
            extra_nodes=list(rm.hand_graph_extra[side]), robot_extra_links=list(robot.extra_link_names),
            robot_extra_buffer_shape=(list(robot.extra_kpt_pos.shape) if robot.extra_kpt_pos is not None else None),
            anchor_points=int(pts['local'].shape[0]), anchor_slices=[list(s) for s in rm.ho_contact_spec[side]['slices']],
            anchor_links=[[robot.coll_link_names[i] for i in idx] for idx in rm.ho_contact_spec[side]['links']],
            link_names_with_extra=[n for n in rm.hand_graph_extra[side] if n in names],
            palm_normal_mode=spec.palm_normal_mode,
            palm_axis=None if rm.hand_graph_palm_axis.get(side) is None else rm.hand_graph_palm_axis[side].tolist())
    report['anchor_geometry'] = {side: anchor_geometry(env, side) for side in env.robots}
    stats, extra_info, problems = dict(), dict(), []
    # the calibrated palm axis must travel with the run: in kw['reward_cfg'] (pickled to params/env.pkl after the env is built)
    saved = pickle.loads(pickle.dumps(kw))['reward_cfg'].get('hand_graph_palm_axis') or {}
    report['palm_axis_in_saved_reward_cfg'] = saved
    report['reward_cfg_is_the_modules'] = rm.cfg is kw['reward_cfg']
    for side in env.robots:
        axis = rm.hand_graph_palm_axis.get(side)
        if axis is None:
            if side in saved:
                problems.append(f'{side}: palm axis saved for a fingers-rule hand')
            continue
        if side not in saved or not torch.equal(torch.tensor(saved[side], dtype=torch.float32), axis.cpu()):
            problems.append(f'{side}: palm axis {axis.tolist()} not saved in reward_cfg ({saved.get(side)})')
    if saved:
        from dexmachina.envs.rewards import RewardModule
        again = RewardModule(pickle.loads(pickle.dumps(kw['reward_cfg'])), kw['demo_data'], kw['retarget_data'],
                             rm.demo_tensors['obj_pos'].device, finger_specs=rm.finger_specs)
        report['palm_axis_rebuilt_from_saved_equal'] = all(
            torch.equal(again.hand_graph_palm_axis[s], rm.hand_graph_palm_axis[s]) for s in saved)
        if not report['palm_axis_rebuilt_from_saved_equal']:
            problems.append('the palm axis rebuilt from the saved reward_cfg differs')
        # and calibrating again on the same data, without the stored axis, gives the same bits (deterministic)
        fresh_cfg = {k: v for k, v in kw['reward_cfg'].items() if k != 'hand_graph_palm_axis'}
        fresh = RewardModule(fresh_cfg, kw['demo_data'], kw['retarget_data'], rm.demo_tensors['obj_pos'].device,
                             finger_specs=rm.finger_specs)
        report['palm_axis_recalibrated_equal'] = all(
            torch.equal(fresh.hand_graph_palm_axis[s], rm.hand_graph_palm_axis[s]) for s in saved)
        if not report['palm_axis_recalibrated_equal']:
            problems.append('calibrating the palm axis again gives a different axis')
    original = rm.compute_reward
    palm_deg = dict()

    def compute_reward(**kwargs):
        for side in ('left', 'right'):
            x = kwargs.get(f'hand_graph_extra_{side}')
            names = rm.hand_graph_extra[side]
            if names:
                if x is None:
                    problems.append(f'{side}: extra nodes {names} not passed')
                    continue
                robot = env.robots[side]
                link_names = [link.name for link in robot.entity.links]
                want = robot.entity.get_links_pos()[:, [link_names.index(n) for n in names]]
                ok = x.shape == want.shape and torch.equal(x, want)
                extra_info.setdefault(side, dict(shape=list(x.shape), names=names, equal_to_link_pos_steps=0, steps=0))
                extra_info[side]['steps'] += 1
                extra_info[side]['equal_to_link_pos_steps'] += int(ok)
        rew, d = original(**kwargs)
        for side in ('left', 'right'):
            if f'hand_graph_palm_deg_{side}' in d:
                palm_deg.setdefault(side, []).append(d[f'hand_graph_palm_deg_{side}'].detach().float().cpu())
        for k, v in list(d.items()) + [('total_reward', rew)]:
            if not torch.is_tensor(v):
                continue
            v = v.detach().float()
            s = stats.setdefault(k, dict(shape=list(v.shape), min=math.inf, max=-math.inf, sum=0.0, n=0, nan=0, inf=0))
            finite = v[torch.isfinite(v)]
            s['nan'] += int(torch.isnan(v).sum())
            s['inf'] += int(torch.isinf(v).sum())
            if finite.numel():
                s['min'] = min(s['min'], finite.min().item())
                s['max'] = max(s['max'], finite.max().item())
                s['sum'] += finite.sum().item()
                s['n'] += finite.numel()
        return rew, d

    rm.compute_reward = compute_reward
    env.reset()
    zeros = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    resets = 0
    for _ in range(a.steps):
        env.step(zeros)
        resets += int(env.reset_buf.sum().item())
    for s in stats.values():
        s['mean'] = s.pop('sum') / max(s['n'], 1)
    report['palm_deg_median'] = {side: round(torch.cat(v).median().item(), 3) for side, v in palm_deg.items()}
    report.update(reward_stats=stats, extra_node_inputs=extra_info, resets=resets, nan_envs_end=int(env.nan_envs.sum()),
                  seconds=round(time.time() - t0, 1), max_memory_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20))
    bad_values = [k for k, s in stats.items() if s['nan'] or s['inf']]
    if bad_values:
        problems.append(f'NaN/inf in {bad_values}')
    hg, hoc = stats.get('hand_graph_rew'), stats.get('ho_contact_rew')
    if hg is None or not (0.0 <= hg['min'] and hg['max'] <= rm.hand_graph_weight + 1e-6):
        problems.append(f'hand_graph_rew out of [0, {rm.hand_graph_weight}]: {hg}')
    if hoc is None or not (0.0 <= hoc['min'] and hoc['max'] <= rm.ho_contact_weight + 1e-6):
        problems.append(f'ho_contact_rew out of [0, {rm.ho_contact_weight}]: {hoc}')
    chk = stats.get('ho_contact_check')
    if chk is None or chk['max'] >= 0.003:
        problems.append(f'ho_contact_check >= 3 mm: {chk}')
    for side, e in extra_info.items():
        if e['equal_to_link_pos_steps'] != e['steps']:
            problems.append(f'{side}: extra nodes differ from the link positions in {e["steps"] - e["equal_to_link_pos_steps"]} steps')
    report['problems'] = problems
    report['ok'] = not problems
    text = json.dumps(report, indent=1)
    print(text, flush=True)
    if a.out:
        Path(a.out).write_text(text)
    print('PASS' if not problems else f'FAIL {problems}', flush=True)
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
