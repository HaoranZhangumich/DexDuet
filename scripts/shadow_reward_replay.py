"""Shadow reward replay (GPU): this checkout's reward module against the one of another commit (--commit, required), on the
same simulator states, bit for bit. For the paper, --commit was the Inspire-only reward code that the paper's Inspire runs
used, which is not part of this repository; give a commit of your own clone (for example one before a change to the
rewards).

One Genesis env is built with this checkout's code: Inspire, the task's DexDuet training flags from
scripts/train_coordinates_hand.py (hand graph 0.5, hand-object contact 1.0), --num-envs envs, stepped with a fixed
pseudo-random action sequence (actions resample every --hold steps; resets, RSI starts and keyframes happen as in training).
The env's RewardModule is the new one. A second RewardModule is built from <commit>'s rewards.py and ho_contact.py (taken
with git show into a temporary package, only the 'from dexmachina.envs.ho_contact import' lines renamed), with copies of the
same reward cfg, demo data and retargeted data. At every step it gets a copy of exactly the inputs the env passes to the new
module, and every output (the summed reward and every key of the reward dict) must be bitwise equal (float tensors are
compared as integer views, so NaN payloads and -0.0 count).

TorchScript: dexmachina/envs/math_utils.py and reward_utils.py are @torch.jit.script functions. Their first call(s) in a process
run the unfused profiling path, later calls the fused kernel, which can differ in the last bit (matrix_from_quat: up to
~1.5e-7). Whichever module is built or called first in a process takes those first calls, old code or new (checked: a
module built first equals the other code's module built first, in separate processes). So a third, discarded module
(<commit>'s code) is built before the two compared ones and computes the reward --warm-calls times before them at every
step; old and new then run in the same executor state.

Before stepping it also checks bitwise:
  - every attribute the two modules share (demo tensors, hand-graph index tensors, finger weight, scale, contact grids, ...);
  - the env's contact anchor points (link indices, point ranges, local points) and the region slices / contact-link indices
    given to the reward, against <commit>'s robot_anchor_points with <commit>'s base_env.setup_ho_contact code.

  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=<patched Genesis>:<repo> python scripts/shadow_reward_replay.py --commit <commit> \
      --out replay.json
Exit 0 when everything matches, 1 otherwise.
"""
import argparse
import copy
import hashlib
import importlib
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

INT_VIEW = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}


def same(a, b, _depth=0):
    """Bitwise equality of tensors / arrays / floats, recursively through dicts, lists and tuples."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
            return False
        if a.dtype in INT_VIEW:
            return torch.equal(a.contiguous().view(INT_VIEW[a.dtype]), b.contiguous().view(INT_VIEW[b.dtype]))
        return torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack('<d', a) == struct.pack('<d', b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], _depth) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y, _depth) for x, y in zip(a, b))
    if type(a).__name__ == type(b).__name__ and hasattr(a, '__dict__') and not callable(a) and _depth < 3:   # e.g. grids
        va, vb = vars(a), vars(b)
        return va.keys() == vb.keys() and all(same(va[k], vb[k], _depth + 1) for k in va)
    try:
        return bool(a == b)
    except Exception:
        return a is b


def clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone(v) for v in x)
    return x


def old_package(commit, tmp):
    """<commit>'s rewards.py and ho_contact.py as package rewards_old_<commit> (its ho_contact imports point into the
    package)."""
    name = f'rewards_old_{commit}'
    pkg = Path(tmp) / name
    pkg.mkdir()
    (pkg / '__init__.py').write_text('')
    info = dict()
    for mod in ('rewards', 'ho_contact'):
        src = subprocess.run(['git', '-C', str(REPO), 'show', f'{commit}:dexmachina/envs/{mod}.py'], check=True,
                             capture_output=True, text=True).stdout
        n = src.count('from dexmachina.envs.ho_contact import')
        (pkg / f'{mod}.py').write_text(src.replace('from dexmachina.envs.ho_contact import', f'from {name}.ho_contact import'))
        info[mod] = dict(git_blob_sha256=hashlib.sha256(src.encode()).hexdigest(), renamed_imports=n)
    sys.path.insert(0, str(tmp))
    rewards = importlib.import_module(f'{name}.rewards')
    ho = importlib.import_module(f'{name}.ho_contact')
    return rewards, ho, info


def env_args(task, num_envs, seed):
    """DexDuet's training flags of the task (Inspire), parsed as train_rl_games.py does for the env."""
    import train_coordinates_hand as tch
    from dexmachina.envs.constructors import get_common_argparser, parse_clip_string
    cmd = tch.training_command(task, seed, 'shadow-replay', 'none', sys.executable)
    argv = cmd[2:]
    argv[argv.index('-B') + 1] = str(num_envs)
    args, unknown = get_common_argparser().parse_known_args(argv)
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object, args.frame_start, args.frame_end = obj_name, start, end
    return args, argv, unknown


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--commit', required=True, help='commit whose rewards.py and ho_contact.py are compared (git show)')
    ap.add_argument('--task', default='Box-200')
    ap.add_argument('--num-envs', type=int, default=256)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42, help='training seed in the flags (the env does not draw from it)')
    ap.add_argument('--action-seed', type=int, default=0)
    ap.add_argument('--action-scale', type=float, default=1.0)
    ap.add_argument('--hold', type=int, default=4, help='steps each random action is held')
    ap.add_argument('--warm-calls', type=int, default=2, help='discarded reward calls of the third module before each step')
    ap.add_argument('--out', default=None, help='JSON summary')
    a = ap.parse_args()
    t0 = time.time()

    import genesis as gs
    from genesis.engine.solvers.rigid import collider_decomp
    genesis_info = dict(file=gs.__file__, overflow_guard=hasattr(collider_decomp.Collider, 'check_overflow'))
    print(f'[replay] genesis {genesis_info}', flush=True)
    if not genesis_info['overflow_guard']:
        sys.exit('[replay] ABORT: Genesis lacks the collision-pair overflow fix (fca4168; no Collider.check_overflow)')

    tmp = tempfile.mkdtemp(prefix='rewards_old_')
    old_rewards, old_ho, old_info = old_package(a.commit, tmp)
    OldRM = old_rewards.RewardModule
    print(f'[replay] old modules from {a.commit}: {old_info} ({tmp})', flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    args, argv, unknown = env_args(a.task, a.num_envs, a.seed)
    if args.hand != 'inspire_hand' or not (args.hand_graph_weight > 0 and args.ho_contact_weight > 0):
        sys.exit(f'[replay] ABORT: expected Inspire with both rewards on, got hand {args.hand} hg {args.hand_graph_weight} '
                 f'hoc {args.ho_contact_weight}')

    import dexmachina.envs.base_env as base_env
    from dexmachina.envs.constructors import get_all_env_cfg
    NewRM = base_env.RewardModule
    state = dict(calls=0, keys=None, mismatches=[], setup=dict(), stats=dict(), resets=0, nan_envs=0)

    def factory(reward_cfg, demo_data, retarget_data, device, finger_specs=None):
        copies = copy.deepcopy((reward_cfg, demo_data, retarget_data))
        warm = OldRM(*copy.deepcopy(copies), device)        # takes the TorchScript first calls (see the docstring)
        new = NewRM(reward_cfg, demo_data, retarget_data, device, finger_specs=finger_specs)
        old = OldRM(*copies, device)
        state.update(new=new, old=old, warm=warm)
        new_setup = new.setup_ho_contact

        def setup_ho_contact(side, slices, anchor_link_idx):
            state['setup'][side] = copy.deepcopy((slices, anchor_link_idx))
            warm.setup_ho_contact(side, copy.deepcopy(slices), copy.deepcopy(anchor_link_idx))
            old.setup_ho_contact(side, copy.deepcopy(slices), copy.deepcopy(anchor_link_idx))
            return new_setup(side, slices, anchor_link_idx)

        new_compute = new.compute_reward

        def compute_reward(**kw):
            extra = {k: v for k, v in kw.items() if k.startswith('hand_graph_extra')}
            if extra:
                state['mismatches'].append(f'call {state["calls"]}: the env passed {sorted(extra)} (Inspire has no extra nodes)')
            for _ in range(a.warm_calls):
                warm.compute_reward(**clone({k: v for k, v in kw.items() if k not in extra}))
            kw_old = clone({k: v for k, v in kw.items() if k not in extra})
            rew_old, d_old = old.compute_reward(**kw_old)
            rew_new, d_new = new_compute(**kw)
            step = state['calls']
            state['calls'] += 1
            if state['keys'] is None:
                state['keys'] = sorted(d_new)
            if sorted(d_new) != sorted(d_old):
                state['mismatches'].append(f'call {step}: keys differ: new-only {sorted(set(d_new) - set(d_old))}, '
                                           f'old-only {sorted(set(d_old) - set(d_new))}')
            if not same(rew_new, rew_old):
                state['mismatches'].append(f'call {step}: summed reward differs')
            for k in sorted(set(d_new) & set(d_old)):
                if not same(d_new[k], d_old[k]):
                    state['mismatches'].append(f'call {step}: {k} differs')
            for k in ('hand_graph_rew', 'ho_contact_rew'):
                v = d_new.get(k)
                if torch.is_tensor(v) and v.numel():
                    s = state['stats'].setdefault(k, dict(min=float('inf'), max=float('-inf'), sum=0.0, n=0))
                    v = v.float()
                    s['min'], s['max'] = min(s['min'], v.min().item()), max(s['max'], v.max().item())
                    s['sum'] += v.sum().item()
                    s['n'] += v.numel()
            return rew_new, d_new

        new.setup_ho_contact = setup_ho_contact
        new.compute_reward = compute_reward
        return new

    base_env.RewardModule = factory
    env_kwargs = get_all_env_cfg(args, device='cuda:0')
    env_kwargs['env_cfg']['use_rl_games'] = True
    gs.init(backend=gs.gpu, logging_level='warning')
    env = base_env.BaseEnv(**env_kwargs)
    new, old = state['new'], state['old']
    t_build = time.time() - t0

    # 1. module attributes shared by both
    checks = dict()
    shared = sorted(set(vars(old)) & set(vars(new)))
    attr_diff = []
    for k in shared:
        va, vb = getattr(new, k), getattr(old, k)
        if k in ('setup_ho_contact', 'compute_reward'):
            continue
        if hasattr(va, '__dict__') and not torch.is_tensor(va) and not callable(va) and type(va) is not type(vb):
            if type(va).__name__ == type(vb).__name__:     # e.g. HandObjectContact of the old package vs of this checkout
                sub = sorted(set(vars(va)) & set(vars(vb)))
                bad = [s for s in sub if not same(getattr(va, s), getattr(vb, s))]
                if bad or set(vars(va)) != set(vars(vb)):
                    attr_diff.append(f'{k}: {bad} / attrs {sorted(set(vars(va)) ^ set(vars(vb)))}')
                continue
        if not same(va, vb):
            attr_diff.append(k)
    checks['module_attributes'] = dict(shared=len(shared), new_only=sorted(set(vars(new)) - set(vars(old))),
                                       old_only=sorted(set(vars(old)) - set(vars(new))), differ=attr_diff)
    checks['demo_tensor_keys'] = sorted(old.demo_tensors)
    # 2. contact anchor points as <commit>'s base_env.setup_ho_contact would have built them
    anchor_diff = []
    for side, robot in env.robots.items():
        spec = old_ho.robot_anchor_points(str(robot.cfg['urdf_path']), 'inspire_hand', side)
        names = [link.name for link in robot.entity.links]
        coll = list(robot.coll_link_names)
        want = dict(link_idx=torch.tensor([names.index(n) for n in spec['links']], dtype=torch.long, device=env.device),
                    ranges=spec['link_ranges'],
                    local=torch.tensor(spec['points'], dtype=torch.float32, device=env.device))
        got = env.ho_contact_points[side]
        for k in want:
            if not same(got[k], want[k]):
                anchor_diff.append(f'{side} {k}')
        want_setup = (spec['slices'], [[coll.index(n) for n in links] for links in spec['anchor_links']])
        if not same(list(state['setup'][side][0]), list(want_setup[0])) or not same(state['setup'][side][1], want_setup[1]):
            anchor_diff.append(f'{side} setup_ho_contact args')
        checks.setdefault('anchor_points', dict())[side] = dict(points=int(want['local'].shape[0]),
                                                                slices=[list(s) for s in spec['slices']])
    checks['anchor_differ'] = anchor_diff
    checks['extra_nodes'] = {s: list(v) for s, v in new.hand_graph_extra.items()}
    print(f'[replay] build {t_build:.0f} s; attribute check {checks["module_attributes"]}; anchors differ {anchor_diff}',
          flush=True)

    # 3. step with a fixed action sequence
    env.reset()
    g = torch.Generator(device=env.device)
    g.manual_seed(a.action_seed)
    act = None
    lengths = []
    t1 = time.time()
    for t in range(a.steps):
        if t % a.hold == 0:
            act = (torch.rand((env.num_envs, env.action_dim), generator=g, device=env.device) * 2.0 - 1.0) * a.action_scale
        env.step(act)
        state['resets'] += int(env.reset_buf.sum().item())
        state['nan_envs'] = max(state['nan_envs'], int(env.nan_envs.sum().item()))
        lengths.append(env.episode_length_buf.float().mean().item())
        if (t + 1) % 50 == 0:
            print(f'[replay] step {t + 1}/{a.steps}: calls {state["calls"]}, mismatches {len(state["mismatches"])}, '
                  f'resets so far {state["resets"]}, mean frame {lengths[-1]:.1f}', flush=True)
    t_steps = time.time() - t1
    for s in state['stats'].values():
        s['mean'] = s.pop('sum') / max(s['n'], 1)
    ok = (not state['mismatches'] and not attr_diff and not anchor_diff and state['calls'] == a.steps)
    summary = dict(ok=ok, commit=a.commit, warm_calls=a.warm_calls, repo_head=subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                                                     capture_output=True, text=True).stdout.strip(),
                   task=a.task, num_envs=env.num_envs, steps=a.steps, reward_calls=state['calls'],
                   reward_keys=state['keys'], n_keys=len(state['keys'] or []), mismatches=state['mismatches'][:50],
                   n_mismatches=len(state['mismatches']), env_resets=state['resets'], max_nan_envs=state['nan_envs'],
                   mean_episode_frame=dict(first=lengths[0], last=lengths[-1], max=max(lengths)), reward_stats=state['stats'],
                   checks=checks, old_modules=old_info, genesis=genesis_info, unknown_launcher_flags=unknown,
                   build_s=round(t_build, 1), step_s=round(t_steps, 1),
                   max_memory_allocated_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20))
    print('[replay] ' + json.dumps(summary, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))
    print(f'[replay] {"PASS" if ok else "FAIL"}', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
