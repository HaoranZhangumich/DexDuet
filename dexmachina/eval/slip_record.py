"""Record an evaluation rollout for Slip-AUC (evaluation only; GPU).

Replays ONE checkpoint with the official evaluator's settings (100 lanes, seed 1042, deterministic actions, object not
actuated, no curriculum, no randomisation, no early reset), from the verified common start state of an existing
evaluation_reset.json (copied into the output folder and checked bit for bit before the first action), with the first
corrected first observation (re-read from the simulator, as eval_rl_games.py always does). Saves, for every lane and
control step, the object state and both hands' keypoints, plus the demo object track, the ADD points and the ADD-AUC of
this rollout, so dexmachina/eval/slip_auc.py can score it offline:

  python -m dexmachina.eval.slip_record --checkpoint <run>/nn/inspire_hand.pth --clip ketchup-40-340-s01-u02 \
      --manifest resets/ketchup-300/evaluation_reset.json --out <new dir>
  python -m dexmachina.eval.slip_auc <new dir> --mano <MANO export dir>

Saved per step i (state after the i-th physics step; replay step i ~ demo frame i+1), all lanes:
  obj_state (S, L, 8) [pos, quat wxyz, hinge angle] as the evaluator records it; kpts_<side> (S, L, n_kpts, 3); wrist_quat_<side>.
Recording tool, version 1.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

TOOL_VERSION = 1
SEED, N_LANES = 1042, 100                      # official evaluator: 100 policy-controlled lanes, seed 1042
ADD_THRESHOLDS = np.arange(0.01, 0.1, 0.01)    # 1-9 cm, as the official compute_add
N_SURFACE = 500                                # surface samples per part for the surface-sample convention (trimesh seed 0)

p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
p.add_argument('--checkpoint', required=True, help='<run dir>/nn/<checkpoint>.pth; the run dir has params/env.pkl and agent.yaml')
p.add_argument('--manifest', required=True, help="the study's evaluation_reset.json (verified start state); copied, never written")
p.add_argument('--clip', required=True, help='clip string, e.g. ketchup-40-340-s01-u02 (checked against env.pkl)')
p.add_argument('--out', required=True, help='new output directory')
p.add_argument('--task', default='', help='label only, e.g. Ketchup-300')
# deprecated no-op, hidden: kept so older commands parse; the first observation is always corrected
p.add_argument('--first_obs', choices=('corrected', 'auto'), default='corrected', help=argparse.SUPPRESS)
p.add_argument('--official_eval', default='', help="the study's eval_ep0.npy (cross-check of the ADD code)")
p.add_argument('--official_auc', type=float, default=None, help="the study's official ADD-AUC for this checkpoint")
p.add_argument('--zero-pose-last-lane', action='store_true',
               help="DexMachina's released checkpoints, as their evaluation (eval_rl_games.py --zero-pose-last-lane): the "
                    "last lane starts from a zero hand pose")
p.add_argument('--allow-last-lane-reset', action='store_true',
               help="DexMachina's released checkpoints only: if the last lane (DexMachina's evaluator starts it from a zero hand "
                    "pose) resets early, record its states from that step on as NaN, which Contact-AUC scores as lost, instead "
                    "of stopping. An early reset of any other lane still stops the recording.")
args = p.parse_args()

T_START = time.time()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=False)
CKPT = Path(args.checkpoint)
RUN_DIR = CKPT.resolve().parents[1]
ADAPT = []
BROKEN_FROM = {}   # --allow-last-lane-reset: lane -> first step after its early reset


def log(msg):
    print(f'[record {time.time() - T_START:7.1f}s] {msg}', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------
# geometry (identical to hold_test.py)
from scipy.spatial.transform import Rotation  # noqa: E402

IDENTITY_STATE = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)


def rotmats(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    return Rotation.from_quat(np.roll(q, -1, axis=1)).as_matrix().reshape(*np.shape(quat)[:-1], 3, 3)


def part_points(pts, pos, quat, arti, top):
    rot = rotmats(quat)
    if top:
        a = np.asarray(arti, dtype=np.float64)
        rot = rot @ Rotation.from_rotvec(np.outer(-a.reshape(-1), [0, 0, 1])).as_matrix().reshape(*a.shape, 3, 3)
    return np.einsum('...ij,pj->...pi', rot, pts) + np.asarray(pos)[..., None, :]


def part_errors(states, demo, surf):
    S, L = states.shape[:2]
    good = np.isfinite(states).all(-1) & (np.linalg.norm(np.nan_to_num(states[..., 3:7]), axis=-1) > 1e-6)
    st = np.where(good[..., None], states, IDENTITY_STATE)
    err = np.zeros((2, S, L))
    for k, part in enumerate(('bottom', 'top')):
        dp = part_points(surf[part], demo[:, :3], demo[:, 3:7], demo[:, 7], part == 'top')
        for lane in range(L):
            sp = part_points(surf[part], st[:, lane, :3], st[:, lane, 3:7], st[:, lane, 7], part == 'top')
            err[k, :, lane] = np.linalg.norm(sp - dp, axis=-1).mean(-1)
    err[:, ~good] = np.inf
    return err


def add_auc(err):
    aucs = [float(np.trapz([np.mean(e < th) for th in ADD_THRESHOLDS], x=np.linspace(0, 1, len(ADD_THRESHOLDS)))) for e in err]
    return float(np.mean(aucs))


# ------------------------------------------------------------------------------------------------------------------------------
# environment + policy, exactly as hold_test.py (and the study evaluator dexmachina/rl/eval_rl_games.py) builds them
import torch  # noqa: E402
import yaml  # noqa: E402
import trimesh  # noqa: E402
import dexmachina  # noqa: E402
import genesis as gs  # noqa: E402
from dexmachina.envs.base_env import BaseEnv  # noqa: E402
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from dexmachina.rl import eval_rl_games as ev  # noqa: E402
from dexmachina.eval.protocol import verify_common_reset_state  # noqa: E402

log(f'runtime {Path(dexmachina.__file__).parent.parent}  checkpoint {CKPT}')
ev.seed_evaluation(SEED)
kw = pickle.load(open(RUN_DIR / 'params' / 'env.pkl', 'rb'))

FIRST_OBS, first_obs_reason = ev.first_observation_mode(kw.get('curriculum_cfg'), args.first_obs)

kw['env_cfg'].update(early_reset_threshold=0.0, is_eval=True, eval_reference_env=False, num_envs=N_LANES, rand_init_ratio=0.0)
if args.zero_pose_last_lane:
    kw['env_cfg']['zero_pose_last_lane'] = True
    ADAPT.append('--zero-pose-last-lane: the last lane starts from a zero hand pose')
kw['rand_cfg']['randomize'] = False
for cfg in kw['object_cfgs'].values():
    cfg['actuated'] = False
kw.pop('curriculum_cfg', None)

clip_obj, clip_f0, clip_f1 = args.clip.split('-')[0], int(args.clip.split('-')[1]), int(args.clip.split('-')[2])
obj_name = list(kw['object_cfgs'])[0]
if obj_name != clip_obj or int(kw['env_cfg']['episode_length']) != clip_f1 - clip_f0:
    sys.exit(f'clip {args.clip} does not match env.pkl (object {obj_name}, episode length {kw["env_cfg"]["episode_length"]})')

gs.init(backend=gs.gpu, logging_level='warning', seed=SEED)
env = BaseEnv(**kw)
obj = env.objects[obj_name]
robots = env.robots
T = int(env.max_episode_length)
S = T - 1
DT = float(env.dt)

agent_cfg = yaml.full_load(open(RUN_DIR / 'params' / 'agent.yaml'))
if agent_cfg['params']['config'].get('marl_baseline') is not None:
    sys.exit('MARL (MAPPO/IPPO) checkpoints are not supported by this tool yet')
wenv = RlGamesVecEnvWrapper(env, agent_cfg['params']['config']['device'], agent_cfg['params']['env'].get('clip_observations', math.inf),
                            agent_cfg['params']['env'].get('clip_actions', math.inf))
vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **k: RlGamesGpuEnv(config_name, num_actors, **k))
env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper', 'env_creator': lambda **k: wenv})
agent_cfg['params']['config']['num_actors'] = N_LANES
agent_cfg['params']['seed'] = SEED
runner = Runner()
PROTOCOL = getattr(env, 'ablation_protocol', 'legacy')
if PROTOCOL in ('v2', 'dexmachina'):
    from dexmachina.rl.ablation_ppo import register_ablation_ppo
    register_ablation_ppo(runner)
runner.load(agent_cfg)
agent = runner.create_player()
ev.restore_trusted_local_checkpoint(agent, str(CKPT))
agent.reset()

manifest_src = Path(args.manifest)
if not manifest_src.is_file() or not manifest_src.with_suffix('.npz').is_file():
    sys.exit(f'manifest {manifest_src} (+ .npz) missing: refusing to create a new evaluation start')
(OUT / 'common_reset').mkdir()
manifest = OUT / 'common_reset' / manifest_src.name
shutil.copy2(manifest_src, manifest)
shutil.copy2(manifest_src.with_suffix('.npz'), manifest.with_suffix('.npz'))


def verified_start():
    """Reset like the evaluator, verify the physical start against the frozen manifest, return the first observation."""
    ev.seed_evaluation(SEED)
    obs = wenv.reset()
    reset_arrays = ev.evaluation_reset_arrays(env)
    check = verify_common_reset_state(str(manifest), reset_arrays, seed=SEED, num_envs=N_LANES)['verification']
    obs = ev.refresh_first_observation(wenv)   # never the stale observation env.reset() returns
    if isinstance(obs, dict):
        obs = obs['obs']
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    return obs, check


def entity_object_state():
    hinge_dofs = [j.dof_idx_local for j in obj.entity.joints if j.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)]
    return torch.cat([obj.entity.get_pos().float(), obj.entity.get_quat().float(),
                      obj.entity.get_dofs_position(hinge_dofs[:1]).float()], -1)


def recorded_rollout():
    """One evaluator episode (hold_test.py scored_rollout), recording every lane's object state and both hands' keypoints."""
    obs, check = verified_start()
    start = dict(obj=entity_object_state().cpu().numpy(),
                 kpts={s: r.entity.get_links_pos()[:, r.kpt_link_idxs].float().cpu().numpy() for s, r in robots.items()})
    out = dict(check=check, start=start, obj_state=np.zeros((S, N_LANES, 8), np.float32),
               obj_state_entity=np.zeros((S, N_LANES, 8), np.float32), final_state=None,
               kpts={s: np.zeros((S, N_LANES, r.n_kpts, 3), np.float32) for s, r in robots.items()},
               wrist_quat={s: np.zeros((S, N_LANES, 4), np.float32) for s, r in robots.items()})
    has_pre = None
    for i in range(S):
        with torch.inference_mode():
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _rew, dones, infos = wenv.step(actions)
            if isinstance(obs, dict):
                obs = obs['obs']
            if has_pre is None:
                has_pre = 'eval_object_state_before_reset' in infos
                if not has_pre:
                    ADAPT.append('no eval_object_state_before_reset: object state read from the object buffers after step')
            o = infos['eval_object_state_before_reset'] if has_pre else torch.cat([obj.root_pos, obj.root_quat, obj.dof_pos], -1)
            out['obj_state'][i] = o.float().cpu().numpy()
            out['obj_state_entity'][i] = entity_object_state().cpu().numpy()
            for s, r in robots.items():
                out['kpts'][s][i] = r.kpt_pos.float().cpu().numpy()
                out['wrist_quat'][s][i] = r.entity.get_links_quat()[:, r.wrist_link_idx].float().cpu().numpy()
            if dones.any() and agent.is_rnn and agent.states is not None:
                for st_ in agent.states:
                    st_[:, dones, :] = 0.0
        if bool(dones.any()):
            done_lanes = set(torch.nonzero(dones).flatten().tolist())
            if args.allow_last_lane_reset and done_lanes <= {N_LANES - 1}:
                BROKEN_FROM.setdefault(N_LANES - 1, i)
            else:
                raise RuntimeError(f'unexpected reset at step {i} (before the final step)')
    for lane, i0 in BROKEN_FROM.items():   # states from the reset on are broken (NaN): Contact-AUC scores them as lost
        ADAPT.append(f'--allow-last-lane-reset: lane {lane} reset at step {i0}; its states from step {i0} set to NaN')
        out['obj_state'][i0:, lane] = np.nan
        out['obj_state_entity'][i0:, lane] = np.nan
        for s_ in out['kpts']:
            out['kpts'][s_][i0:, lane] = np.nan
            out['wrist_quat'][s_][i0:, lane] = np.nan
    if has_pre:
        with torch.inference_mode():
            _o, _r, _d, infos = wenv.step(agent.get_action(obs, is_deterministic=True))
            out['final_state'] = infos['eval_object_state_before_reset'].float().cpu().numpy()
    return out


t0 = time.time()
P1 = recorded_rollout()
pass1_s = time.time() - t0
obj_state = P1['obj_state']
log(f'pass 1 ({FIRST_OBS} first observation; {first_obs_reason}): {S} steps in {pass1_s:.1f}s; start {P1["check"]}')

# ------------------------------------------------------------------------------------------------------------------------------
# ADD-AUC of this rollout (hold_test.py code), both conventions
assets = Path(dexmachina.__file__).parent / 'assets' / 'arctic' / obj_name
meshes = {part: trimesh.load(assets / f'{part}_watertight_tiny.obj', force='mesh') for part in ('bottom', 'top')}
surf = {part: trimesh.sample.sample_surface(m, N_SURFACE, seed=0)[0] for part, m in meshes.items()}
demo = ev.gather_object_state_tensor(kw['demo_data']).numpy().astype(np.float64)       # (T, 8) [pos, wxyz, arti]
err = part_errors(obj_state.astype(np.float64), demo[np.arange(1, T)], surf)
add_waffle = add_auc(err)


def official_vertices():
    out = {}
    for part in ('bottom', 'top'):
        v = np.asarray(trimesh.load(kw['object_cfgs'][obj_name][f'{part}_mesh_fname']).vertices)
        np.random.seed(42)
        out[part] = v[np.random.choice(len(v), 500, replace=500 > len(v))]
    return out


OFFICIAL_VERTS = official_vertices()
rows = np.concatenate([obj_state, P1['final_state'][None]]) if P1['final_state'] is not None else obj_state
err_off = part_errors(rows.astype(np.float64), demo[:len(rows)], OFFICIAL_VERTS)
add_off = add_auc(err_off)
official = {}
if args.official_eval and Path(args.official_eval).is_file():
    off_all = np.load(args.official_eval, allow_pickle=True).item()['obj_state'].astype(np.float64)
    official = dict(file=args.official_eval, official_add_auc=args.official_auc,
                    our_official_convention_on_official_rollout=add_auc(part_errors(off_all, demo[:len(off_all)], OFFICIAL_VERTS)))
entity_diff = float(np.nanmax(np.abs(P1['obj_state_entity'] - obj_state)))
log(f'ADD-AUC {add_off:.4f} (official convention; surface-sample convention {add_waffle:.4f}); entity vs evaluator object state '
    f'max |diff| {entity_diff:.2e}; official cross-check {official}')

dd = kw['demo_data']
kpt_names = {s: list(r.kpt_link_names) for s, r in robots.items()}
np.savez_compressed(
    OUT / 'recording.npz', obj_state=obj_state, obj_state_entity=P1['obj_state_entity'],
    final_state=P1['final_state'] if P1['final_state'] is not None else np.zeros((0, 8), np.float32),
    start_obj=P1['start']['obj'], demo=demo, dt=DT,
    **{f'demo_hand_joints_{s}': np.asarray(dd[f'hand_joints_{s}'], np.float32) if f'hand_joints_{s}' in dd else np.zeros((0, 21, 3), np.float32)
       for s in ('left', 'right')},   # older runtimes' demo_data has no hand joints
    add_points_bottom=OFFICIAL_VERTS['bottom'], add_points_top=OFFICIAL_VERTS['top'],
    err_official_cm=(err_off * 100).astype(np.float32), err_cm=(err * 100).astype(np.float32),
    **{f'kpts_{s}': v for s, v in P1['kpts'].items()}, **{f'wrist_quat_{s}': v for s, v in P1['wrist_quat'].items()},
    **{f'start_kpts_{s}': v for s, v in P1['start']['kpts'].items()},
    **{f'kpt_names_{s}': np.array(v) for s, v in kpt_names.items()})

meta = dict(
    tool='dexmachina/eval/slip_record.py', tool_version=TOOL_VERSION, task=args.task, clip=args.clip, object=obj_name,
    protocol_source='official evaluator settings (eval_rl_games.py), verified common start',
    inputs=dict(run_dir=str(RUN_DIR), checkpoint=str(CKPT), checkpoint_sha256=sha256(CKPT), manifest=str(manifest_src),
                manifest_sha256=sha256(manifest_src)),
    settings=dict(first_observation=FIRST_OBS, first_observation_rule=first_obs_reason, lanes=N_LANES, seed=SEED),
    mesh_files={part: str(kw['object_cfgs'][obj_name][f'{part}_mesh_fname']) for part in ('bottom', 'top')},
    kpt_names=kpt_names, robots={s: r.name if hasattr(r, 'name') else s for s, r in robots.items()},
    add_auc=add_off, add_auc_score_waffle_convention=add_waffle, official_cross_check=official,
    entity_vs_evaluator_object_state_max_abs_diff=entity_diff,
    nonfinite_object_states=int((~np.isfinite(obj_state).all(-1)).sum()),
    nonfinite_kpts={s: int((~np.isfinite(v).all((-1, -2))).sum()) for s, v in P1['kpts'].items()},
    runtime=dict(runtime_dir=str(Path(dexmachina.__file__).parent.parent), genesis=str(Path(gs.__file__).parent),
                 python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''), host=socket.gethostname(),
                 start_verification=P1['check'], protocol=PROTOCOL, dt=DT, steps=S, adaptations=ADAPT,
                 pass1_s=round(pass1_s, 1), total_s=round(time.time() - T_START, 1),
                 utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
(OUT / 'record.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
log(f'DONE add_auc {add_off:.4f}; saved {OUT / "recording.npz"}')
