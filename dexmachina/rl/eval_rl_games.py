import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
import random
from datetime import datetime 

from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
# Import registers the opt-in factorized network before Runner loads a saved
# agent configuration.  It does not change the released default network.
from dexmachina.rl.coordinate_network_builder import NETWORK_NAME
from dexmachina.eval.protocol import (
    build_evaluation_metadata, evaluation_folder_name, prepare_evaluation_directory,
    validate_evaluation_options,
    verify_common_reset_state,
)

  
from collections import defaultdict
import shutil
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner


def restore_trusted_local_checkpoint(agent, checkpoint_path):
    """Restore a locally trained rl-games checkpoint with PyTorch >=2.6.

    rl-games calls torch.load without an explicit weights_only value. Newer
    PyTorch releases therefore reject the NumPy scalars in these checkpoints.
    Scope the legacy unpickler to this trusted local restore only.
    """
    from rl_games.algos_torch import torch_ext

    original_safe_load = torch_ext.safe_load
    torch_ext.safe_load = lambda filename: torch_ext.safe_filesystem_op(
        torch.load, filename, weights_only=False
    )
    try:
        agent.restore(checkpoint_path)
    finally:
        torch_ext.safe_load = original_safe_load
 

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

def seed_evaluation(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluation_reset_arrays(uenv):
    """Physical state and reset buffers before the first policy action."""
    arrays = {}
    for index, state in enumerate(uenv.scene.get_state().solvers_state):
        if state is None:
            continue
        for name in ("qpos", "dofs_vel", "links_pos", "links_quat", "i_pos_shift",
                     "mass_shift", "friction_ratio"):
            value = getattr(state, name, None)
            if torch.is_tensor(value):
                arrays[f"solver{index}.{name}"] = value.detach().cpu().numpy().copy()
    for name in ("episode_start_buf", "episode_length_buf", "last_actions"):
        arrays[f"environment.{name}"] = getattr(uenv, name).detach().cpu().numpy().copy()
    for side, robot in uenv.robots.items():
        for name in ("dof_pos", "dof_vel", "curr_targets", "prev_targets", "curr_res_qpos"):
            arrays[f"robot.{side}.{name}"] = getattr(robot, name).detach().cpu().numpy().copy()
    if not any(name.endswith(".qpos") for name in arrays):
        raise RuntimeError("Cannot verify evaluation initial states without simulator qpos")
    return arrays


def first_observation_mode(curriculum_cfg=None, requested='corrected'):
    """The first observation is always 'corrected': the robot and object buffers are re-read from the simulator after the
    reset, before the first action, for every method. The original DexMachina evaluator acted on the observation env.reset()
    returns, which is stale: object.reset_idx zeroes the cached part poses and robot.reset_idx indexes the keypoint array with
    a link index (wrist about 11 cm off). The physical start state was always correct. That mode was removed;
    'auto' is accepted only so older commands still parse, and also means 'corrected'."""
    if requested not in ('corrected', 'auto'):
        raise SystemExit(f"--first-obs {requested!r}: the original (stale) first observation was removed; every method is "
                         "evaluated with the corrected first observation")
    return 'corrected', 'always corrected (the stale DexMachina first observation is no longer available)'


def refresh_first_observation(env):
    """The corrected first observation: refresh the robot and object buffers from the simulator and rebuild the observation."""
    uenv = env.unwrapped
    for robot in uenv.robots.values():
        robot.update_value_buffers()
    for obj in uenv.objects.values():
        obj.update_value_buffers()
    process = getattr(env, '_process_obs', None) or getattr(env, '_process')
    return process(uenv.get_observations())


def eval_one_episode(env, agent, obj_state_tensor, print_rew=False, record_video=False,
                     show_reference=False, reset_validator=None):
    env.reset()
    if reset_validator is not None:
        reset_validator(env.unwrapped)
    obs = refresh_first_observation(env)   # never act on the stale observation env.reset() returns
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    joint_target_left = left_hand.residual_qpos
    joint_target_right = right_hand.residual_qpos
    
    eval_data = defaultdict(list)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            # get actions from the agent
            actions = agent.get_action(obs, is_deterministic=True) 
            demo_state = obj_state_tensor[env_step]
                    
            if show_reference: # visualize the demo traj and set zero action
                obj.set_object_state(
                    root_pos=demo_state[:3][None],
                    root_quat=demo_state[3:7][None],
                    joint_qpos=demo_state[7:8][None],
                    env_idxs=torch.tensor([1], dtype=torch.int32, device=device),
                )
                actions[-1, :] = -1.0 
                for robot, joints in zip([left_hand, right_hand], [joint_target_left, joint_target_right]):
                    robot.set_joint_position(
                        joint_targets=joints[env_step][None],
                        env_idxs=[1],
                    ) 
            obs, rew, dones, infos = env.step(actions) 
            if getattr(uenv, 'ablation_protocol', 'legacy') in ('v2', 'dexmachina'):
                # BaseEnv automatically resets terminated lanes before step()
                # returns. Use the physical endpoint snapshot for every lane.
                if 'eval_object_state_before_reset' not in infos:
                    raise RuntimeError('Ablation evaluation requires object state captured before automatic reset')
                obj_state = infos['eval_object_state_before_reset']
            else:
                obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
                obj_state = torch.cat([obj_pos, obj_quat, obj_arti], dim=-1)
            eval_data["obj_state"].append(obj_state.cpu().numpy())
            eval_data["demo_state"].append(demo_state.cpu().numpy())

            rew_dict = uenv.rew_dict
            for key in ['pos_dist', 'rot_dist', 'arti_dist']:
                eval_data[key].append(rew_dict[key].cpu().numpy())
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0 
        
        if print_rew:
            print(f"Step {env_step}: Reward: {rew.cpu().numpy()}")
            # perform operations for terminated episodes 
            rew_dict = uenv.rew_dict
            for k, v in rew_dict.items():
                if 'con' in k:
                    print(f"Step {env_step}: {k}: {v.cpu().numpy()}")
    eval_data = {k: np.stack(v) for k, v in eval_data.items()}
    frames = uenv.get_recorded_frames()
    return frames, eval_data


def main():

    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--eval_episodes', '-ne', type=int, default=1)
    parser.add_argument('--print_rew', '-pr', action='store_true')
    parser.add_argument('--show_reference', '-ref', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--all_policy_envs', action='store_true',
                        help='Control every evaluation environment; save separately from legacy evaluations')
    parser.add_argument('--eval-seed', type=int, default=None,
                        help='Seed Genesis, Python, NumPy and Torch before scene and model creation')
    parser.add_argument('--eval-reset-manifest', type=str, default=None,
                        help='Save/verify immutable common initial-state arrays before policy actions')
    parser.add_argument('--zero-pose-last-lane', action='store_true',
                        help="DexMachina's released checkpoints, as in the paper: the last lane starts from a zero hand pose, "
                             "as in DexMachina's own evaluator (the policy still controls every lane)")
    parser.add_argument('--accept-selection', action='append', default=[],
                        help='also accept this checkpoint selection rule in final_checkpoint.json (default: only '
                             'native_best_training_reward), e.g. best_zero_help_training_return')
    # Deprecated no-op, hidden: kept so older commands parse; the first observation is always re-read from the simulator.
    parser.add_argument('--first-obs', choices=('corrected', 'auto'), default='corrected', help=argparse.SUPPRESS)
    parser.add_argument('--output_render', '-or', action='store_true') # if not ture, don't show the retargeted reference
    parser.add_argument('--render_dir', '-out', type=str, default="rendered") # if not provided, save in the same folder as the checkpoint
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4") # if not provided, save in the same folder as the checkpoint
    
    args = parser.parse_args()
    if args.eval_seed is not None and args.eval_seed < 0:
        parser.error('--eval-seed must be nonnegative')
    if args.eval_reset_manifest and (args.eval_seed is None or not args.all_policy_envs or args.eval_episodes != 1):
        parser.error('--eval-reset-manifest requires --eval-seed, --all_policy_envs and exactly one episode')
    if args.eval_seed is not None:
        seed_evaluation(args.eval_seed)
    try:
        validate_evaluation_options(args.num_envs, args.eval_episodes,
                                    all_policy_envs=args.all_policy_envs,
                                    show_reference=args.show_reference)
    except ValueError as exc:
        parser.error(str(exc))

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}" # inpsire_ep1000 etc
    run_name = f"{args.checkpoint.split('/')[-3]}" # inpsire_ep1000 etc
    ckpt_data_folder = os.path.join(ckpt_path, evaluation_folder_name(
        ckpt_name, all_policy_envs=args.all_policy_envs,
    ))
    prepare_evaluation_directory(ckpt_data_folder, all_policy_envs=args.all_policy_envs)

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    # load to pkl
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    # Legacy evaluation leaves its last environment uncontrolled. The explicit
    # opt-in protocol counts only actual policy trials and keeps old files intact.
    env_kwargs['env_cfg']['eval_reference_env'] = not args.all_policy_envs
    if args.zero_pose_last_lane:
        env_kwargs['env_cfg']['zero_pose_last_lane'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        # env_kwargs['camera_res'] = (2048, 2048)
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        ) 
    for name, cfg in env_kwargs['object_cfgs'].items():
        print("Setting eval time obj gains to 0.0")
        cfg['actuated'] = False

    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    first_obs, first_obs_rule = first_observation_mode(env_kwargs.get('curriculum_cfg'), args.first_obs)
    print(f"First observation: {first_obs} ({first_obs_rule})")
    print("Removing any curriculum config during eval")
    env_kwargs.pop("curriculum_cfg", None)
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning', **(
        {'seed': args.eval_seed} if args.eval_seed is not None else {}))
    env = BaseEnv(
         **env_kwargs
    )
    study_protocol = env.ablation_protocol
    controlled_ablation_evaluation = study_protocol in ('v2', 'dexmachina')
    if controlled_ablation_evaluation:
        if args.eval_seed is None or not args.eval_reset_manifest or not args.all_policy_envs:
            raise ValueError('Ablation evaluation requires the seeded common reset manifest and all policy environments')
        # Robot.is_eval reserves its last initial pose for visualization. BaseEnv
        # must suppress that legacy reservation when constructing v2 robots.
        if any(robot.is_eval for robot in env.robots.values()):
            raise RuntimeError('Ablation robots must not reserve a reference environment during reset')
    demo_data = env_kwargs['demo_data']
    obj_state_tensor = gather_object_state_tensor(demo_data)

    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    if env.coordinate_actor_cfg.get('enabled', False):
        saved_agent_cfg = os.path.join(ckpt_path, "params", "agent.yaml")
        if not os.path.exists(saved_agent_cfg):
            raise FileNotFoundError(
                "Factorized Coordinate actor evaluation requires its saved "
                f"agent config: {saved_agent_cfg}"
            )
        agent_cfg_fname = saved_agent_cfg
        print(
            f"Loading saved {NETWORK_NAME} architecture from "
            f"{agent_cfg_fname}"
        )
    
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    saved_training_config = dict(agent_cfg['params']['config'])
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    marl_baseline = saved_training_config.get('marl_baseline')
    if marl_baseline is not None:
        # MAPPO/IPPO: the player runs the shared hand actor on each agent's local observation; no critic state needed.
        from dexmachina.rl.marl_baselines import MultiAgentHandsEnv
        env = MultiAgentHandsEnv(env, rl_device, clip_obs, clip_actions, mode=marl_baseline['mode'], with_states=False)
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    uenv = env.unwrapped 
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs
    if args.eval_seed is not None:
        agent_cfg['params']['seed'] = args.eval_seed

    # create runner from rl-games
    runner = Runner()
    if controlled_ablation_evaluation:
        from dexmachina.rl.ablation_ppo import register_ablation_ppo
        register_ablation_ppo(runner)
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    resume_path = os.path.abspath(args.checkpoint) 
    restore_trusted_local_checkpoint(agent, resume_path)
    agent.reset()

    protocol = build_evaluation_metadata(
        resume_path, num_envs=uenv.num_envs, eval_episodes=args.eval_episodes,
        all_policy_envs=args.all_policy_envs, show_reference=args.show_reference,
    )
    if args.eval_seed is not None:
        protocol['evaluation_seed'] = args.eval_seed
    protocol['first_observation'] = first_obs
    if args.zero_pose_last_lane:
        protocol['zero_pose_last_lane'] = True
    protocol['first_observation_rule'] = first_obs_rule
    if controlled_ablation_evaluation:
        protocol['study_protocol'] = study_protocol
        protocol['object_state_sampling'] = 'post_physics_pre_reset_v2'
    if study_protocol == 'dexmachina':
        from pathlib import Path
        from dexmachina.rl.final_checkpoint import validate_training_checkpoint
        primary, selection = validate_training_checkpoint(
            Path(ckpt_path), saved_training_config, study_protocol,
            accept_selections=('native_best_training_reward', *args.accept_selection),
        )
        if primary.resolve() != Path(resume_path).resolve():
            raise ValueError('DexMachina primary evaluation requires the native best training reward checkpoint')
        protocol.update(
            checkpoint_selection=selection['selection'], checkpoint_epoch=selection['epoch'],
            completed_training_updates=selection['completed_updates'],
            training_epoch_cap=agent_cfg['params']['config']['max_epochs'],
        )
    if protocol['policy_controlled_env_ids'] != uenv._step_env_idxs:
        raise RuntimeError('Evaluation metadata disagrees with actual policy-controlled environments')
    protocol_path = os.path.join(ckpt_data_folder, 'eval_protocol.json')
    with open(protocol_path, 'w') as stream:
        json.dump(protocol, stream, indent=2)

    for eps in range(args.eval_episodes):
        if args.eval_seed is not None:
            # Player construction may consume RNG; reset sampling is independent
            # of model layout and initialization.
            seed_evaluation(args.eval_seed + eps)
        def reset_validator(uenv):
            protocol['common_reset'] = verify_common_reset_state(
                args.eval_reset_manifest, evaluation_reset_arrays(uenv),
                seed=args.eval_seed, num_envs=uenv.num_envs,
            )
            with open(protocol_path, 'w') as stream:
                json.dump(protocol, stream, indent=2)
        frames, eval_data = eval_one_episode(
            env, agent, obj_state_tensor, args.print_rew, args.record_video, args.show_reference,
            reset_validator=reset_validator if args.eval_reset_manifest else None,
            )
        ckpt_eval_fname = os.path.join(ckpt_data_folder, f"eval_ep{eps}.npy")
        np.save(ckpt_eval_fname, eval_data)
        protocol['completed_episodes'] = eps + 1
        print(f"Saved eval data to {ckpt_eval_fname}")
        # try loading the data
        # eval_data = np.load(ckpt_eval_fname, allow_pickle=True).item() 
        if args.record_video: 
            # save video with moviepy
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip(frames, fps=int(1/uenv.dt/2))
            clip.write_videofile(video_fname)
            print(f"Saved video to {video_fname}")
    
    protocol['status'] = 'complete'
    with open(protocol_path, 'w') as stream:
        json.dump(protocol, stream, indent=2)
    print("Done evaluating")


if __name__ == '__main__':
    main()
