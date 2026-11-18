"""Hold test (paper: "Holding after the last frame"): the scored 100-lane evaluation of
one checkpoint, continued for H policy steps after the last clip step with every reference frozen at the last clip frame.

The rollout mechanics are those of the paper's hold test (README.md in this folder). The inputs are the
checkpoint, the start-state manifest of its scored evaluation and that evaluation's recording, given on the command line.

  * Same environment and agent construction as dexmachina.rl.eval_rl_games (env.pkl of the run, is_eval, every lane
    policy-controlled, early_reset_threshold 0, no randomisation, rand_init_ratio 0, object not actuated, no curriculum,
    100 lanes, eval seed 1042, the run's agent config, deterministic actions, corrected first observation, start state
    verified byte for byte against a COPY of the manifest, Genesis overflow guard asserted, overflow counters checked).
  * --hold-steps H > 0 (the paper: 180 = 3 s at 60 Hz): before the LAST clip step (i = L-1) the hold patch is switched on,
    an instance-level wrapper of two BaseEnv methods:
      _compute_intermediate_values: pins env.episode_length_buf, every robot's and the object's episode_length_buf to L-1,
        then runs the original. Every reference is indexed by these buffers (wrist residual target, object goal in the
        observation, phase input, reward/contact/keypoint targets), so all of them stay at the last
        frame for the whole hold, and nothing moves.
      _get_dones: the time-out termination is disabled (the evaluator's reset of non-finite lanes is kept; such lanes count
        as failures in the analysis).
    The physics of step L-1 is unchanged, so the clip part is the scored rollout with the pre-reset state as its last row.
  * --hold-steps 0: the plain scored evaluation (including the automatic reset after the last step).
  * Records, per step and lane: the evaluator's object state, the physical post-step pre-reset object state, hand-object
    contact flags (env.contact_link_valid), reset/NaN flags and the reference diagnostics; lane 0 observations/actions.
  * Mechanics checks only (prefix identity vs the scored rollout, frozen references, no reset); add.py computes the hold
    ADD and metrics.py the metrics. Outputs only in --out (a new folder): hold_rollout.npz, eval100/eval_ep0.npy (clip
    part, evaluator format), manifest copy, hold_eval.json.

  CUDA_VISIBLE_DEVICES=0 python -m dexmachina.eval.hold_test.run --checkpoint <run>/nn/inspire_hand.pth \
      --manifest <evaluation_reset.json of its evaluation> --out hold_test/ketchup-100_seed15 --task ketchup-100 --seed 15
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from os.path import join

import numpy as np

HAND_LABELS = {'inspire_hand': 'Inspire', 'allegro_hand': 'Allegro', 'xhand': 'XHand', 'schunk_hand': 'Schunk'}
ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--checkpoint', required=True, help='<run>/nn/<hand>.pth; <run>/params holds env.pkl and agent.yaml')
ap.add_argument('--manifest', required=True,
                help='evaluation_reset.json of the scored evaluation (its .npz next to it); copied into --out, never written')
ap.add_argument('--reference', default=None,
                help='the scored evaluation recording: eval_ep0.npy (eval_rl_games) or recording.npz (slip_record); '
                     'default <run>/<hand>_eval_all_policy/eval_ep0.npy')
ap.add_argument('--slip-recording', default=None, help='optional second reference (<run>/slip/recording.npz)')
ap.add_argument('--out', required=True, help='new output folder')
ap.add_argument('--hold-steps', type=int, default=180, help='hold steps after the last clip step (paper: 180 = 3 s)')
ap.add_argument('--name', default=None, help='label of this run in the analysis (default: the --out folder name)')
ap.add_argument('--hand', default=None, help='hand label for the analysis (default from the checkpoint file name)')
ap.add_argument('--method', default='DexDuet', help='method label for the analysis')
ap.add_argument('--task', default=None, help='task label, e.g. ketchup-100 (default from the run folder name)')
ap.add_argument('--seed', default=None, help='training seed label')
ap.add_argument('--table-add', type=float, default=None, help="the run's reported ADD-AUC in %% (context only)")
ap.add_argument('--eval-seed', type=int, default=1042)
ap.add_argument('--num-envs', type=int, default=100)
ap.add_argument('--zero-pose-last-lane', action='store_true',
                help="DexMachina's released checkpoints: replay their evaluation with its zero-pose last lane "
                     "(eval_rl_games.py --zero-pose-last-lane)")
args = ap.parse_args()
OUT = os.path.abspath(args.out)


def _run_record():
    ckpt = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    hand_key = os.path.splitext(os.path.basename(ckpt))[0]
    reference = os.path.abspath(args.reference or join(run_dir, f'{hand_key}_eval_all_policy', 'eval_ep0.npy'))
    task = args.task
    if task is None:
        import re
        m = re.search(r'_(box|ketchup|mixer|notebook|waffleiron)(\d+)-(\d+)-', os.path.basename(run_dir))
        task = f'{m.group(1)}-{int(m.group(3)) - int(m.group(2))}' if m else 'unknown'
    manifest = os.path.abspath(args.manifest)
    return dict(name=args.name or os.path.basename(OUT), hand=args.hand or HAND_LABELS.get(hand_key, hand_key),
                hand_key=hand_key, method=args.method, task=task, seed=args.seed, run_id=os.path.basename(run_dir),
                set=None, run_dir=run_dir, ckpt=ckpt, env_pkl=join(run_dir, 'params', 'env.pkl'), manifest=manifest,
                manifest_npz=manifest[:-5] + '.npz', reference=reference, table_add_pct=args.table_add,
                slip_recording=os.path.abspath(args.slip_recording) if args.slip_recording else None)


REC = _run_record()
assert not OUT.startswith(os.path.abspath(REC['run_dir']) + '/'), 'never write into a run dir'
for _key in ('ckpt', 'env_pkl', 'manifest', 'manifest_npz', 'reference'):
    if not os.path.exists(REC[_key]):
        sys.exit(f'missing {_key}: {REC[_key]}')
os.makedirs(OUT, exist_ok=False)
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# --- inputs: recorded by hash ---
REC['ckpt_sha256'] = sha256(REC['ckpt'])
REC['env_pkl_sha256'] = sha256(REC['env_pkl'])
MAN = join(OUT, 'manifest', 'evaluation_reset.json')
os.makedirs(join(OUT, 'manifest'))
shutil.copy2(REC['manifest'], MAN)
shutil.copy2(REC['manifest_npz'], MAN[:-5] + '.npz')
assert sha256(MAN) == sha256(REC['manifest']) and sha256(MAN[:-5] + '.npz') == sha256(REC['manifest_npz'])

import torch  # noqa: E402
import yaml  # noqa: E402
import genesis as gs  # noqa: E402
from genesis.engine.solvers.rigid.collider_decomp import Collider  # noqa: E402

assert hasattr(Collider, 'check_overflow'), f'{gs.__file__}: Genesis without the overflow guard (patch fca4168)'
import dexmachina  # noqa: E402
import rl_games  # noqa: E402

REC['code_repo'] = os.path.dirname(os.path.dirname(os.path.realpath(dexmachina.__file__)))
from dexmachina.envs.base_env import BaseEnv  # noqa: E402
from dexmachina.asset_utils import get_rl_config_path  # noqa: E402
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv  # noqa: E402
from dexmachina.rl.coordinate_network_builder import NETWORK_NAME  # noqa: E402,F401  (registers the network)
from dexmachina.rl import eval_rl_games as ev  # noqa: E402
from dexmachina.eval.protocol import verify_common_reset_state  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

COLLIDERS = []
_orig_init = Collider.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    COLLIDERS.append(self)


Collider.__init__ = _init   # bookkeeping only (list of instances), no behaviour change

SEED, N, HOLD = args.eval_seed, args.num_envs, args.hold_steps
git = lambda repo: subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
log('genesis', gs.__file__, 'dexmachina', dexmachina.__file__, 'rl_games', rl_games.__file__, 'code HEAD', git(REC['code_repo']))
ev.seed_evaluation(SEED)
kw = pickle.load(open(REC['env_pkl'], 'rb'))
assert kw['env_cfg']['use_rl_games']
# --- evaluation settings, identical to eval_rl_games.py --num_envs 100 --all_policy_envs --eval-seed 1042 ---
kw['env_cfg']['early_reset_threshold'] = 0.0
kw['env_cfg']['is_eval'] = True
kw['env_cfg']['eval_reference_env'] = False
if args.zero_pose_last_lane:
    kw['env_cfg']['zero_pose_last_lane'] = True
kw['rand_cfg']['randomize'] = False
kw['env_cfg']['num_envs'] = N
kw['env_cfg']['rand_init_ratio'] = 0.0
for cfg in kw['object_cfgs'].values():
    cfg['actuated'] = False
first_obs, first_obs_rule = ev.first_observation_mode(kw.get('curriculum_cfg'), 'corrected')
kw.pop('curriculum_cfg', None)
gs.init(backend=gs.gpu, logging_level='warning', seed=SEED)
env = BaseEnv(**kw)
study_protocol = env.ablation_protocol
REC['study_protocol'] = study_protocol
# where the scored outputs hold the pre-reset state of the last clip step (protocol section 4, check 3)
REC['pre_reset_final_ref'] = ('eval_ep0.npy obj_state[-1] (eval_object_state_before_reset)'
                              if study_protocol in ('v2', 'dexmachina') and REC['reference'].endswith('.npy') else None)
if study_protocol in ('v2', 'dexmachina') and any(r.is_eval for r in env.robots.values()):
    raise RuntimeError('Ablation robots must not reserve a reference environment during reset')

# --- agent, identical to eval_rl_games.py ---
agent_cfg_fname = get_rl_config_path('rl_games_ppo_cfg')
if env.coordinate_actor_cfg.get('enabled', False):
    agent_cfg_fname = join(REC['run_dir'], 'params', 'agent.yaml')
agent_cfg = yaml.full_load(open(agent_cfg_fname, encoding='utf-8'))
saved_training_config = dict(agent_cfg['params']['config'])
if saved_training_config.get('marl_baseline') is not None:
    sys.exit('MARL checkpoints not supported here')
wenv = RlGamesVecEnvWrapper(env, agent_cfg['params']['config']['device'], agent_cfg['params']['env'].get('clip_observations', math.inf),
                            agent_cfg['params']['env'].get('clip_actions', math.inf))
vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **k: RlGamesGpuEnv(config_name, num_actors, **k))
uenv = wenv.unwrapped
env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper', 'env_creator': lambda **k: wenv})
agent_cfg['params']['config']['num_actors'] = uenv.num_envs
agent_cfg['params']['seed'] = SEED
runner = Runner()
if study_protocol in ('v2', 'dexmachina'):
    from dexmachina.rl.ablation_ppo import register_ablation_ppo
    register_ablation_ppo(runner)
runner.load(agent_cfg)
agent = runner.create_player()
ev.restore_trusted_local_checkpoint(agent, os.path.abspath(REC['ckpt']))
agent.reset()
assert [int(i) for i in uenv._step_env_idxs] == list(range(N)), 'every lane must be policy-controlled'
L = int(uenv.max_episode_length)
obj = uenv.objects[uenv.object_names[0]]
assert not obj.actuated
D = int(uenv.demo_length)
R = {s: int(r.residual_num_frames) for s, r in uenv.robots.items()}
assert L == D == obj.num_demo_frames and all(v == L for v in R.values()), (L, D, obj.num_demo_frames, R)
assert uenv.chunk_ep_length <= 0
TPIN = L - 1
log('checkpoint', REC['ckpt'], 'protocol', study_protocol, 'L', L, 'hold', HOLD, 'dt', uenv.dt)

# --- hold patch (instance level) and per-step capture hooks ---
HOLD_ON = {'on': False}
CAP = {}
_orig_civ = uenv._compute_intermediate_values
_orig_dones = uenv._get_dones


def _civ():
    if HOLD_ON['on']:
        for buf in [uenv.episode_length_buf] + [r.episode_length_buf for r in uenv.robots.values()] + \
                   [o.episode_length_buf for o in uenv.objects.values()]:
            buf.clamp_(max=TPIN)
    CAP['ebuf_env'] = uenv.episode_length_buf.clone()
    CAP['ebuf_robot'] = torch.stack([r.episode_length_buf.clone() for r in uenv.robots.values()])
    CAP['ebuf_obj'] = obj.episode_length_buf.clone()
    _orig_civ()


def _dones():
    timeout, timeout2 = _orig_dones()
    # physical post-step state before any reset (the object buffers were refreshed in _compute_intermediate_values)
    CAP['phys'] = torch.cat([obj.root_pos, obj.root_quat, obj.dof_pos], -1).detach().clone()
    CAP['contact'] = uenv.contact_link_valid.detach().clone() if uenv.use_contact_reward else None
    CAP['nan'] = uenv.nan_envs.detach().clone()
    CAP['timeout_orig'] = timeout.detach().clone()
    if HOLD_ON['on']:
        nanr = uenv.nan_envs.clone()
        return nanr, nanr.clone()      # time-out disabled; the evaluator's non-finite reset is kept
    return timeout, timeout2


uenv._compute_intermediate_values = _civ
uenv._get_dones = _dones

# --- one evaluation episode (eval_rl_games.eval_one_episode) + hold ---
ev.seed_evaluation(SEED + 0)
obs_state_tensor = ev.gather_object_state_tensor(kw['demo_data'])
wenv.reset()
verify = verify_common_reset_state(MAN, ev.evaluation_reset_arrays(uenv), seed=SEED, num_envs=uenv.num_envs)
log('reset manifest check:', verify['verification'])
obs = ev.refresh_first_observation(wenv)
if isinstance(obs, dict):
    obs = obs['obs']
_ = agent.get_batch_size(obs, 1)
if agent.is_rnn:
    agent.init_rnn()
phase_sl = slice(*uenv.obs_idxs['episode_length'])
o_lo, o_hi = uenv.obs_idxs[uenv.object_names[0]]
sd_sl = slice(o_lo + 3 * obj.n_links + 4 * obj.n_links + obj.num_joints, o_lo + 3 * obj.n_links + 4 * obj.n_links + obj.num_joints + 8)
rec = defaultdict(list)
obs0 = [obs[0].detach().cpu().numpy().copy()]
torch.cuda.reset_peak_memory_stats()
t_loop = time.time()
S = L + HOLD
for i in range(S):
    if HOLD > 0 and i == L - 1:
        HOLD_ON['on'] = True
    with torch.inference_mode():
        env_step = uenv.episode_length_buf.cpu().numpy()[0]
        robot_t_before = np.stack([r.episode_length_buf.cpu().numpy() for r in uenv.robots.values()])
        actions = agent.get_action(obs, is_deterministic=True)
        demo_state = obs_state_tensor[env_step]
        obs, rew, dones, infos = wenv.step(actions)
        if study_protocol in ('v2', 'dexmachina'):
            obj_state = infos['eval_object_state_before_reset']
        else:
            obj_state = torch.cat([obj.root_pos, obj.root_quat, obj.dof_pos], dim=-1)
        rec['obj_state'].append(obj_state.cpu().numpy())
        rec['demo_state'].append(demo_state.cpu().numpy())
        for key in ['pos_dist', 'rot_dist', 'arti_dist']:
            rec[key].append(uenv.rew_dict[key].cpu().numpy())
        rec['phys_state'].append(CAP['phys'].cpu().numpy())
        if CAP['contact'] is not None:
            rec['contact'].append(CAP['contact'].cpu().numpy().astype(np.uint8))
        rec['nan'].append(CAP['nan'].cpu().numpy())
        rec['dones'].append(dones.cpu().numpy())
        rec['timeout_orig'].append(CAP['timeout_orig'].cpu().numpy())
        rec['ebuf_env_used'].append(CAP['ebuf_env'].cpu().numpy())
        rec['ebuf_robot_used'].append(CAP['ebuf_robot'].cpu().numpy())
        rec['ebuf_obj_used'].append(CAP['ebuf_obj'].cpu().numpy())
        rec['ebuf_env_after'].append(uenv.episode_length_buf.cpu().numpy())
        rec['robot_t_at_action'].append(robot_t_before)
        rec['env_step_lane0'].append(int(env_step))
        rec['res_qpos'].append(np.stack([r.curr_res_qpos.cpu().numpy() for r in uenv.robots.values()]))
        rec['state_diff'].append(obj.state_diff.cpu().numpy())
        rec['phase_obs'].append(obs[:, phase_sl].cpu().numpy())
        rec['state_diff_obs'].append(obs[:, sd_sl].cpu().numpy())
        rec['wrist_pose'].append(np.stack([r.wrist_pose.cpu().numpy() for r in uenv.robots.values()]))
        rec['action0'].append(actions[0].cpu().numpy())
        obs0.append(obs[0].detach().cpu().numpy().copy())
        if len(dones) > 0 and agent.is_rnn and agent.states is not None:
            for s_ in agent.states:
                s_[:, dones, :] = 0.0
loop_s = time.time() - t_loop
peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
log('loop %.1fs (%d steps), torch peak allocated %.0f MiB' % (loop_s, S, peak_alloc))
A = {k: np.stack(v) for k, v in rec.items()}
A['obs_lane0'] = np.stack(obs0)

# --- outputs: clip part in the evaluator's format, full arrays, manifest copy ---
os.makedirs(join(OUT, 'eval100'))
clip = {k: A[k][:L] for k in ('obj_state', 'demo_state', 'pos_dist', 'rot_dist', 'arti_dist')}
np.save(join(OUT, 'eval100', 'eval_ep0.npy'), clip)
# --- mechanics checks (no hold metric) ---
res = dict(name=REC['name'], hand=REC['hand'], hand_key=REC['hand_key'], table_add_pct=REC['table_add_pct'],
           zero_pose_last_lane=bool(args.zero_pose_last_lane), reference=REC['reference'],
           task=REC['task'], method=REC['method'], seed=REC['seed'], run_id=REC['run_id'], set=REC['set'],
           hold_steps=HOLD, L=L, dt=float(uenv.dt), study_protocol=study_protocol, checkpoint=REC['ckpt'],
           checkpoint_sha256=REC['ckpt_sha256'], code_repo=REC['code_repo'], code_head=git(REC['code_repo']),
           genesis=gs.__file__, genesis_head=git(os.path.dirname(os.path.dirname(gs.__file__))), dexmachina=dexmachina.__file__,
           rl_games=rl_games.__file__, eval_seed=SEED, num_envs=N, first_obs=first_obs, reset_verify=verify,
           manifest_source=REC['manifest'], loop_s=round(loop_s, 1), torch_peak_alloc_mib=round(peak_alloc),
           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), mps_vars=[k for k in os.environ if k.startswith('CUDA_MPS')])


def diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return dict(bitwise_equal=False, shape=[list(a.shape), list(b.shape)])
    bad = np.where(np.any((a != b) & ~(np.isnan(a) & np.isnan(b)), axis=tuple(range(1, a.ndim))))[0] if a.ndim > 1 else \
        np.where((a != b) & ~(np.isnan(a) & np.isnan(b)))[0]
    return dict(bitwise_equal=bool(len(bad) == 0), first_diff_row=int(bad[0]) if len(bad) else None, n_diff_rows=int(len(bad)),
                max_abs=float(np.nanmax(np.abs(a - b))) if a.size else 0.0)


ref = REC['reference']
mine = A['obj_state'][:L]
if ref.endswith('.npy'):
    R0 = np.load(ref, allow_pickle=True).item()['obj_state']
    res['ref_rows'] = int(len(R0))
    res['prefix_0_to_L-2_vs_scored'] = diff(mine[:L - 1], R0[:L - 1])
    res['demo_state_vs_scored'] = diff(A['demo_state'][:L], np.load(ref, allow_pickle=True).item()['demo_state'])
    if HOLD == 0:
        res['full_clip_vs_scored_incl_last_row'] = diff(mine, R0)
    if REC.get('pre_reset_final_ref', '') and REC['pre_reset_final_ref'].startswith('eval_ep0'):
        res['last_row_pre_reset_vs_scored'] = diff(A['phys_state'][L - 1], R0[L - 1])
else:
    z = np.load(ref)
    res['ref_rows'] = int(len(z['obj_state'])) + int(len(z['final_state']) > 0)
    res['prefix_0_to_L-2_vs_scored'] = diff(mine[:L - 1], z['obj_state'])
    if len(z['final_state']):
        res['last_row_pre_reset_vs_scored'] = diff(A['phys_state'][L - 1], z['final_state'])
        if HOLD == 0:
            res['full_clip_vs_scored_incl_last_row'] = diff(mine, np.concatenate([z['obj_state'], z['final_state'][None]]))
if REC.get('slip_recording') and REC['slip_recording'] != ref:
    zs = np.load(REC['slip_recording'])
    res['prefix_vs_slip_recording'] = diff(mine[:L - 1], zs['obj_state'])
    if len(zs['final_state']):
        res['last_row_pre_reset_vs_slip_final_state'] = diff(A['phys_state'][L - 1], zs['final_state'])
res['eval_obj_state_equals_phys_state_rows_0_to_L-2'] = diff(A['obj_state'][:L - 1], A['phys_state'][:L - 1])['bitwise_equal']
nan_any = A['nan'].any(0)
res['nan_lanes_at_end'] = [int(j) for j in np.where(A['nan'][-1])[0]]
res['nan_lanes_first_step'] = {int(j): int(np.argmax(A['nan'][:, j])) for j in np.where(nan_any)[0]}
res['nonfinite_states'] = int((~np.isfinite(A['phys_state'])).any(-1).sum())
res['lanes_eq_lane0_clip'] = int(sum(np.array_equal(A['phys_state'][:L, j], A['phys_state'][:L, 0]) for j in range(N)))
res['resets_during_clip_steps_0_to_L-2'] = int(A['dones'][:L - 1].sum())
if HOLD > 0:
    hs = slice(L - 1, S)            # the steps run with the patch on: the last clip step and the H hold steps
    ok = ~nan_any
    res['hold_patch_steps'] = [L - 1, S - 1]
    res['resets_in_patch_steps_non_nan_lanes'] = int(A['dones'][hs][:, ok].sum())
    res['timeouts_suppressed_last_clip_step'] = int(A['timeout_orig'][L - 1].sum())
    res['ebuf_env_used_during_patch'] = sorted({int(v) for v in np.unique(A['ebuf_env_used'][hs][:, ok])})
    res['ebuf_robot_used_during_patch'] = sorted({int(v) for v in np.unique(A['ebuf_robot_used'][hs][:, :, ok])})
    res['ebuf_obj_used_during_patch'] = sorted({int(v) for v in np.unique(A['ebuf_obj_used'][hs][:, ok])})
    res['robot_t_at_action_during_hold'] = sorted({int(v) for v in np.unique(A['robot_t_at_action'][L:][:, :, ok])})
    res['eval_demo_index_lane0_during_hold'] = sorted(set(A['env_step_lane0'][L:]))
    res['demo_state_rows_during_hold_equal_final_ref'] = bool(np.array_equal(A['demo_state'][L:], np.repeat(
        obs_state_tensor[L - 1].numpy()[None], HOLD, 0)))
    rq = A['res_qpos'][L:][:, :, ok]                         # (H, 2, lanes, ndof): wrist residual target used in the hold
    last_res = np.stack([r.residual_qpos[R[s] - 1].cpu().numpy() for s, r in uenv.robots.items()])
    res['residual_target_hold_equals_last_frame'] = bool(np.array_equal(rq, np.broadcast_to(last_res[None, :, None], rq.shape)))
    res['residual_target_last_clip_step_equals_last_frame'] = bool(np.array_equal(
        A['res_qpos'][L - 1][:, ok], np.broadcast_to(last_res[:, None], A['res_qpos'][L - 1][:, ok].shape)))
    ph = A['phase_obs'][hs][:, ok]
    res['phase_obs_values_during_patch'] = sorted({float(v) for v in np.unique(ph)})
    res['phase_obs_expected'] = float(np.float32(2.0 * TPIN / L - 1.0))
    res['phase_obs_last_clip_observation_acted_on'] = float(A['phase_obs'][L - 2][0, 0])
    # object goal in the observation: state_diff = demo_states[D-1] - state (scaled by the object's obs_scale)
    goal = obj.demo_states[D - 1].cpu().numpy().astype(np.float64)
    sd = A['state_diff'][hs][:, ok].astype(np.float64)
    st = A['phys_state'][hs][:, ok].astype(np.float64)
    res['state_diff_goal_is_last_frame_maxabs'] = float(np.abs(sd - (goal - st)).max())
    scale = float(getattr(obj, 'obs_scale', {}).get('state_diff', 1.0)) if isinstance(getattr(obj, 'obs_scale', None), dict) else 1.0
    res['state_diff_obs_matches_buffer_maxabs'] = float(np.abs(np.clip(A['state_diff_obs'][hs][:, ok], -5, 5) -
                                                            np.clip(A['state_diff'][hs][:, ok] * scale, -5, 5)).max())
    pdist = A['pos_dist'][L:][:, ok].astype(np.float64)
    res['reward_pos_dist_vs_last_frame_maxabs'] = float(np.abs(pdist - np.linalg.norm(
        A['phys_state'][L:][:, ok, :3].astype(np.float64) - goal[:3], axis=-1)).max())
    res['hold_lanes_eq_lane0'] = int(sum(np.array_equal(A['phys_state'][L:, j], A['phys_state'][L:, 0]) for j in range(N)))
link_names = {s: list(r.coll_link_names) if hasattr(r, 'coll_link_names') else None for s, r in uenv.robots.items()}
n_left = int(getattr(uenv, 'num_left_contact_links', 0) or 0)
A['res_qpos_lane0'] = A.pop('res_qpos')[:, :, 0]   # (S, 2, ndof); the all-lane check is above
np.savez_compressed(
    join(OUT, 'hold_rollout.npz'), L=L, H=HOLD, dt=float(uenv.dt), final_ref=obs_state_tensor[L - 1].numpy().astype(np.float64),
    demo=obs_state_tensor.numpy(), **A, contact_obj_links=np.array(list(obj.link_names)),
    contact_left_links=np.array(link_names.get('left') or []), contact_right_links=np.array(link_names.get('right') or []),
    num_left_contact_links=n_left)

ovf = []
for c in COLLIDERS:
    n_pair, n_con = c.n_pair_overflow.to_numpy(), c.n_contact_overflow.to_numpy()
    ovf.append(dict(B=int(len(n_pair)), pair_capacity=int(c._max_broad_pairs), detection_calls=int(c._n_detection_calls),
                    pair_overflow_total=int(n_pair.sum()), contact_overflow_total=int(n_con.sum())))
res['overflow'] = ovf
for c in COLLIDERS:
    c.check_overflow()
res['elapsed_s'] = round(time.time() - t0, 1)
json.dump(res, open(join(OUT, 'hold_eval.json'), 'w'), indent=1, default=str)
log('mechanics:', json.dumps({k: v for k, v in res.items() if k.startswith(('prefix', 'last_row', 'full_clip', 'resets', 'ebuf',
                                                                            'phase', 'residual', 'state_diff', 'nan', 'demo_state'))},
                             default=str)[:3000])
log('DONE')
os._exit(0)
