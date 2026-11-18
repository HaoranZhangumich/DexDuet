"""Hold test, step 2 (GPU): the official part ADD of every hold frame against the frozen final reference.

Stage A of the paper's hold test for one run folder written by run.py, with the method unchanged:
  * hold ADD: the physical states s_0..s_H (hold_rollout.npz phys_state rows L-1 .. L-1+H; s_0 = the pre-reset state after the
    last clip step) against the frozen final reference g (demo frame L-1, the evaluator's demo_state during the hold), with
    the official dexmachina.eval.compute_add functions: create_scene, load_part_verts (500 vertices per part, seed-42
    sampling) and get_all_add (Genesis part poses), in a scene with H+1 environments (one per frame);
  * non-finite lanes (an evaluator NaN flag or a non-finite phys_state at any step of clip or hold): states replaced by g
    before the Genesis call (only to keep the batched kinematics finite), ADD set to +inf afterwards;
  * check: the same functions on the replayed clip (eval100/eval_ep0.npy, a scene with L environments, as compute_add builds
    it); hold row h = 0 must equal the clip's last frame on the finite lanes.
Writes <run>/hold_add.npz (bottom, top: (lanes, H+1) part ADD in m; bad: non-finite lanes) and <run>/hold_add.json.

  CUDA_VISIBLE_DEVICES=0 python -m dexmachina.eval.hold_test.add <hold test output folder> [<folder> ...]
"""
import argparse
import hashlib
import json
import os
import time
from os.path import join

import numpy as np

THR = list(np.arange(0.01, 0.1, 0.01))
t0 = time.time()


def log(*a):
    print(f'[hold add {time.time() - t0:7.1f}s]', *a, flush=True)


def hold_add(out, device, create_scene, load_part_verts, get_all_add, compute_add_stats):
    import torch
    z = np.load(join(out, 'hold_rollout.npz'))
    meta = json.load(open(join(out, 'hold_eval.json')))
    obj_name = meta['task'].split('-')[0]
    L, H = int(z['L']), int(z['H'])
    phys = z['phys_state']
    assert H > 0 and phys.shape[0] == L + H
    g = z['demo_state'][L - 1]
    assert np.array_equal(z['demo_state'][L:], np.repeat(g[None], H, 0)), 'hold reference is not frame L-1'
    assert np.allclose(g.astype(np.float64), z['final_ref'], atol=0, rtol=0), 'final_ref != demo_state[L-1]'
    bad = z['nan'].any(0) | ~np.isfinite(phys).all(axis=(0, 2))
    report = dict(object=obj_name, L=L, H=H)
    scene, obj = create_scene(obj_name, device, num_envs=H + 1)
    pv = load_part_verts(obj, obj_name, device)
    report['part_verts_sha256'] = {p: hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for p, v in pv.items()}
    st = phys[L - 1:L + H].copy()
    st[:, bad] = g
    demo = np.repeat(g[None], H + 1, 0)
    add = get_all_add(obj, pv, torch.tensor(demo, dtype=torch.float32, device=device),
                      torch.tensor(st, dtype=torch.float32, device=device))
    add = {p: np.asarray(v, dtype=np.float64) for p, v in add.items()}
    for p in add:
        add[p][bad] = np.inf
    report['hold_add_auc_all_lanes'] = compute_add_stats({p: add[p][:, 1:] for p in add}, THR)['auc']['overall']
    del scene, obj
    scene, obj = create_scene(obj_name, device, num_envs=L)
    pv = load_part_verts(obj, obj_name, device)
    e = np.load(join(out, 'eval100', 'eval_ep0.npy'), allow_pickle=True).item()
    clip = get_all_add(obj, pv, torch.tensor(e['demo_state'], dtype=torch.float32, device=device),
                       torch.tensor(e['obj_state'], dtype=torch.float32, device=device))
    ok = np.setdiff1d(np.arange(e['obj_state'].shape[1]), np.where(bad)[0])
    for p in ('bottom', 'top'):
        c = np.asarray(clip[p], np.float64)
        report[f'hold_h0_{p}_equals_clip_last_frame'] = bool(np.array_equal(add[p][ok, 0], c[ok, L - 1]))
    report['clip_add_auc_replay'] = compute_add_stats({p: np.asarray(clip[p]) for p in ('bottom', 'top')}, THR)['auc']['overall']
    del scene, obj
    report['n_bad_lanes'] = int(bad.sum())
    report['bad_lanes'] = np.where(bad)[0].tolist()
    np.savez_compressed(join(out, 'hold_add.npz'), bottom=add['bottom'], top=add['top'], bad=bad, L=L, H=H, g=g)
    json.dump(report, open(join(out, 'hold_add.json'), 'w'), indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('outs', nargs='+', help='hold test output folders (run.py --out)')
    args = ap.parse_args()
    for out in args.outs:
        if os.path.exists(join(out, 'hold_add.npz')):
            raise SystemExit(f'{out}/hold_add.npz exists (outputs are never overwritten)')
    import torch
    import genesis as gs
    from dexmachina.eval.compute_add import create_scene, load_part_verts, get_all_add, compute_add_stats
    gs.init(backend=gs.gpu, logging_level='warning')
    device = torch.device('cuda')
    for out in args.outs:
        r = hold_add(out, device, create_scene, load_part_verts, get_all_add, compute_add_stats)
        log(out, f"Hold-ADD-AUC (all lanes) {100 * r['hold_add_auc_all_lanes']:.2f}; clip ADD-AUC {100 * r['clip_add_auc_replay']:.2f};",
            'h0 = clip last frame:', r['hold_h0_bottom_equals_clip_last_frame'] and r['hold_h0_top_equals_clip_last_frame'])
    os._exit(0)


if __name__ == '__main__':
    main()
