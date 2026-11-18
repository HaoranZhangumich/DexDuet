import os 
import math
import yaml
import torch  
import wandb 
import shutil
import pickle
import argparse
import numpy as np
import genesis as gs
from datetime import datetime 

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner 

from dexmachina.asset_utils import get_rl_config_path
from dexmachina.envs.base_env import BaseEnv 
from dexmachina.envs.constructors import get_common_argparser, get_all_env_cfg, parse_clip_string
from dexmachina.envs.learner_config import check_learner_args
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv
from dexmachina.rl.coordinate_network_builder import configure_coordinate_network


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save either a dictionary or class object.
        sort_keys: Whether to sort the keys in the output file. Defaults to False.
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True) 
    # make all the numpy arrays into lists, recursively
    def to_list(d):
        for k, v in d.items():
            if isinstance(v, dict):
                to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
    
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)


def configure_training_protocol(agent_cfg, args, learner_cfg):
    """PPO settings of --ablation_protocol v2 (fixed update budget) and dexmachina (DexMachina's native schedule)."""
    protocol = getattr(args, 'ablation_protocol', 'legacy')
    if protocol not in ('v2', 'dexmachina'):
        return
    from dexmachina.rl.ablation_ppo import configure_ablation_ppo
    config = agent_cfg['params']['config']
    config.update(ablation_protocol=protocol, training_seed=args.seed)
    if protocol == 'v2':
        agent_cfg['params']['network']['coordinate']['initialization_seed'] = args.seed
        config.update(fixed_epoch_budget=True, lr_schedule='identity',
                      score_to_win=float('inf'), initialization_seed=args.seed,
                      rollout_seed=args.seed + 10000)
    else:
        # Native initialization and scheduler; do not inject the v2 RNG streams.
        agent_cfg['params']['network']['coordinate'].pop('initialization_seed', None)
        config.update(fixed_epoch_budget=False, lr_schedule='adaptive',
                      schedule_type='standard', kl_threshold=0.008)
    configure_ablation_ppo(
        agent_cfg, ratio_mode='per_hand' if learner_cfg['per_hand_ppo'] else 'joint',
        value_mode=learner_cfg['value_mode'],
    )


def build_parser():
    parser = get_common_argparser() 
    # now add RL training args 
    parser.add_argument("--exp_name", "-exp", type=str, default="inspire", help="Experiment name.") 
    parser.add_argument("--horizon", '-ho', type=int, default=16, help="Number of steps per environment.")
    parser.add_argument("--checkpoint", '-ck', type=str, default=None, help="Checkpoint file to load.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0003, help="Learning rate for the agent.") 
    parser.add_argument("--wandb_project", "-wp", type=str, default="dexmachina", help="WandB project name.")
    parser.add_argument("--save_freq", "-sf", type=int, default=1000)
    parser.add_argument("--study_profile", choices=("auto", "pilot", "validation"), default="auto")
    parser.add_argument("--caps_temporal_coefficient", type=float, default=0.0)
    parser.add_argument("--caps_temporal_gate", choices=("none", "task_reward", "ref_motion", "task_reward_ref_motion"), default="none",
                        help="Scale each CAPS pair by the task reward (0-1) of its later state, by how slowly the demo object "
                             "moves into that state (ref_motion), or by both.")
    parser.add_argument("--caps_ref_lin_scale", type=float, default=0.00357,
                        help="Demo object translation per frame (m) at which the reference-motion gate is 0.5; default: median over the seven clips.")
    parser.add_argument("--caps_ref_rot_scale", type=float, default=0.035,
                        help="Demo object rotation per frame (rad) at which the reference-motion gate is 0.5; default: median over the seven clips.")
    parser.add_argument(
        "--coord_ablation_id",
        default=None,
        choices=["per-hand", "MAPPO", "IPPO"],
        help="Optional label of the run variant, stored in W&B and in env.yaml ablation_study.variant; DexDuet's runs use "
             "per-hand.",
    )
    parser.add_argument("--marl_baseline", choices=("mappo", "ippo"), default=None,
                        help="Train standard MAPPO/IPPO (rl_games multi-agent PPO, one agent per hand) instead of "
                             "the per-hand policy.")
    return parser


def main():
    args = build_parser().parse_args()
    check_learner_args(args)
    if args.marl_baseline is not None and (
        args.coord_ablation_id != args.marl_baseline.upper() or args.ablation_protocol != 'dexmachina'
        or not args.coordinate_actor or args.per_hand_ppo or args.value_mode != 'scalar'
        or args.caps_temporal_coefficient != 0
        or args.checkpoint is not None
    ):
        raise ValueError('MARL baselines need the matching ablation id, the native protocol, the per-hand observation '
                         'layout (--coordinate_actor), a scalar team reward and none of the per-hand learner options')
    if not math.isfinite(args.caps_temporal_coefficient) or args.caps_temporal_coefficient < 0:
        raise ValueError('CAPS temporal coefficient must be finite and nonnegative')
    if args.caps_temporal_coefficient > 0 and (
        args.ablation_protocol != 'dexmachina'
        or not args.coordinate_actor or not args.per_hand_ppo
        or args.value_mode != 'per_hand'
    ):
        raise ValueError('--caps_temporal_coefficient > 0 requires the per-hand policy settings (--ablation_protocol '
                         'dexmachina --coordinate_actor --per_hand_ppo --value_mode per_hand)')
    protocol = getattr(args, 'ablation_protocol', 'legacy')
    protocol_v2 = protocol == 'v2'
    module_protocol = protocol in ('v2', 'dexmachina')
    if protocol_v2:
        from dexmachina.rl.ablation_initialization import seed_all
        seed_all(args.seed)
    if module_protocol and args.checkpoint is not None:
        raise ValueError('runs with --ablation_protocol v2|dexmachina start from scratch; resuming is not implemented')

    
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end 
    
    hand_prefix = str(args.hand).split("_")[0]
    exp_name = hand_prefix + "-" + args.exp_name
    exp_name += f"_{obj_name}{start}-{end}-{subject_name}-u{use_clip}_B{args.num_envs}"
    exp_name += "_"+args.action_mode
    exp_name += f"_thres{args.early_reset_threshold}"
    exp_name += f"_ho{args.horizon}"
    exp_name += f"_imi{args.imi_rew_weight}"
    if args.marl_baseline is not None:
        exp_name += f"_{args.marl_baseline}"
    elif args.coordinate_actor:
        exp_name += "_factnomsg" if args.coord_no_partner_message else "_factmsg"
        if getattr(args, 'coord_local_critic', False):
            exp_name += "_lcrit"
    if args.per_hand_ppo:
        exp_name += "_handppo"
    if args.contact_rew_weight > 0:
        exp_name += f"_con{args.contact_rew_weight}"
    if args.rand_init_ratio > 0:
        exp_name += f"_rand{args.rand_init_ratio}"
    if args.bc_rew_weight > 0:
        exp_name += f"_bc{args.bc_rew_weight}"
    if args.caps_temporal_coefficient > 0:
        exp_name += f"_capsT{args.caps_temporal_coefficient:g}"
        if args.caps_temporal_gate != "none":
            exp_name += {"task_reward": "_gateTR", "ref_motion": "_gateRM", "task_reward_ref_motion": "_gateTRRM"}[args.caps_temporal_gate]
    if args.curr_schedule == "rsi":
        exp_name += "_rsikf" if args.rsi_keyframes else "_rsi"
        if args.rsi_keyframes and args.rsi_kf_confirm > 0:
            exp_name += f"c{args.rsi_kf_confirm}"
        if args.rsi_keyframes and args.rsi_kf_to_end:
            exp_name += "_kfend"
        if args.rsi_frame0_share > 0:
            exp_name += f"_f0s{args.rsi_frame0_share:g}"
    if args.cg_gate > 0:
        exp_name += f"_cg{args.cg_gate:g}"
    if args.hand_graph_weight > 0:
        exp_name += f"_hg{args.hand_graph_weight:g}"
    if args.ho_contact_weight > 0:
        exp_name += f"_hoc{args.ho_contact_weight:g}"
        if args.ho_contact_cap != 0.05:
            exp_name += f"c{args.ho_contact_cap * 100:g}"


    num_envs = args.num_envs   
    env_kwargs = get_all_env_cfg(args, device='cuda:0')
    env_kwargs['env_cfg']['use_rl_games'] = True
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning',
            **({'seed': args.seed} if protocol_v2 else {}))
    env = BaseEnv(
         **env_kwargs
    )  
    agent_cfg_fname = get_rl_config_path("rl_games_ppo_cfg")
    with open(agent_cfg_fname, encoding="utf-8") as f:
        agent_cfg = yaml.full_load(f)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["name"] = args.hand
    
    log_root_path = os.path.join("logs", "rl_games", args.hand)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = exp_name
    agent_cfg["params"]["config"]["max_epochs"] = int(args.max_epochs)
    agent_cfg["params"]["config"]["save_frequency"] = int(args.save_freq)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args.checkpoint is not None: 
        assert os.path.exists(args.checkpoint), f"Checkpoint file not found: {args.checkpoint}"
        # agent_cfg["params"]["load_checkpoint"] = True
        # agent_cfg["params"]["load_path"] = args.checkpoint     
    
    if args.marl_baseline is not None:
        from dexmachina.rl.marl_baselines import MultiAgentHandsEnv
        env = MultiAgentHandsEnv(env, rl_device, clip_obs, clip_actions, mode=args.marl_baseline)
    else:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, use_sil=False)
    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )

    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = int(args.num_envs * 8)
    agent_cfg["params"]["config"]["mini_epochs"] = max(1, int(args.num_envs / 4096 * 5)) # 5 epochs per 4096 samples
    agent_cfg["params"]["config"]["num_steps_per_env"] = args.horizon
    agent_cfg["params"]["config"]["learning_rate"] = args.learning_rate
    if args.marl_baseline is not None:
        # Native DexMachina PPO settings (as configure_training_protocol applies them), then stock rl_games MAPPO/IPPO.
        from dexmachina.rl.marl_baselines import configure_marl_agent
        agent_cfg['params']['config'].update(
            ablation_protocol=protocol, training_seed=args.seed, fixed_epoch_budget=False,
            lr_schedule='adaptive', schedule_type='standard', kl_threshold=0.008)
        configure_marl_agent(agent_cfg, args.marl_baseline)
    elif args.coordinate_actor:
        configure_coordinate_network(
            agent_cfg,
            env.unwrapped,
            env.unwrapped.coordinate_actor_cfg,
        )
    if module_protocol and args.marl_baseline is None:
        configure_training_protocol(agent_cfg, args, env.unwrapped.learner_cfg)

    if args.caps_temporal_coefficient > 0:
        wrist_indices = []
        for side, robot in env.unwrapped.robots.items():
            indices = env.unwrapped.action_idxs_to_robot[side]
            wrist_indices.extend(int(indices[int(robot.action_from_idxs[dof])])
                                 for dof in robot.wrist_dof_idxs)
        agent_cfg['params']['config']['caps_temporal'] = {
            'coefficient': args.caps_temporal_coefficient,
            'wrist_action_indices': wrist_indices,
            **({'gate': args.caps_temporal_gate} if args.caps_temporal_gate != 'none' else {}),
            **({'ref_lin_scale': args.caps_ref_lin_scale, 'ref_rot_scale': args.caps_ref_rot_scale}
               if args.caps_temporal_gate in ('ref_motion', 'task_reward_ref_motion') else {}),
        }
    elif args.caps_temporal_gate != 'none':
        raise ValueError('--caps_temporal_gate needs --caps_temporal_coefficient > 0')
    
    
    env_save_kwargs = env_kwargs.copy()
    # pop the demo data and retargeted data
    env_save_kwargs.pop('demo_data')
    env_save_kwargs.pop('retarget_data')
    if module_protocol:
        env_save_kwargs['ablation_study'] = dict(
            protocol=protocol, variant=args.coord_ablation_id, training_seed=args.seed,
            training_seed_count=1,
            profile=(args.study_profile if args.study_profile != 'auto' else
                     ('pilot' if args.max_epochs == 13000 and num_envs == 12000 else 'validation')),
        )
    

    # convert agent_cfg to dict:
    wandb_cfg = agent_cfg.copy()
    wandb_cfg['env_kwargs'] = env_save_kwargs
    # also save args
    wandb_cfg['clip'] = f"{obj_name}{start}-{end}-{subject_name}-u{use_clip}"
    wandb_cfg['hand'] = args.hand
    if module_protocol:
        wandb_cfg['study'] = dict(
            protocol=protocol, training_seed=args.seed, training_seed_count=1,
            **({'exact_update_budget': int(args.max_epochs)} if protocol_v2 else
               {'max_epoch_cap': int(args.max_epochs),
                'stopping': 'native_dexmachina', 'checkpoint_selection': 'native_best_training_reward'}),
            primary_metric='ADD-AUC',
            statistical_scope=f'single training seed on {args.clip}',
        )
    if args.coord_ablation_id is not None:
        wandb_cfg['ablation_id'] = args.coord_ablation_id
        wandb_cfg['coordinate_partner_message'] = bool(
            args.coordinate_actor and not args.coord_no_partner_message and args.marl_baseline is None
        )
        wandb_cfg['coordinate_local_critic'] = bool(getattr(args, 'coord_local_critic', False))
    if args.marl_baseline is not None:
        wandb_cfg['marl_baseline'] = dict(agent_cfg['params']['config']['marl_baseline'])
    
    run = wandb.init(
        project=args.wandb_project, 
        config=wandb_cfg,
        monitor_gym=True,
        save_code=True,
        name=exp_name,
    )
    if module_protocol:
        run.tags = tuple(run.tags or ()) + (f'ablation-{protocol}', 'single-seed', args.coord_ablation_id or 'unlabelled') + (
            ('marl-baseline', 'rl_games-multi-agent') if args.marl_baseline is not None else ())
        run.summary.update({'study/protocol': protocol, 'study/training_seed': args.seed,
                            'study/training_seed_count': 1,
                            'study/variant': args.coord_ablation_id,
                            'study/include_in_final_results': False,
                            'study/status': 'training'})

    # get wandb run name and id
    run_name = run.name
    run_id = run.id
    env_save_kwargs['wandb'] = dict(
        run_name=run_name,
        run_id=run_id,
        project=getattr(run, 'project', args.wandb_project),
        entity=getattr(run, 'entity', None),
        url=getattr(run, 'url', None),
    )
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "env.yaml"), env_save_kwargs)
    dump_yaml(os.path.join(log_root_path, exp_name, "params", "agent.yaml"), agent_cfg) 
    # also dump as pkl file 
    pickle.dump(env_kwargs, open(os.path.join(log_root_path, exp_name, "params", "env.pkl"), "wb")) 

    # create runner from rl-games
    observer = IsaacAlgoObserver()
    if args.marl_baseline is not None:
        from dexmachina.rl.marl_baselines import MarlAlgoObserver
        observer = MarlAlgoObserver()
    elif module_protocol:
        from dexmachina.rl.training_observer import TrainingObserver
        observer = TrainingObserver()
    runner = Runner(observer)
    if module_protocol and args.marl_baseline is None:
        from dexmachina.rl.ablation_ppo import register_ablation_ppo
        register_ablation_ppo(runner)
    runner.load(agent_cfg)

    # set seed of the env
    # env.seed(agent_cfg["params"]["seed"]) 
    # reset the agent and env
    runner.reset()
    # train the agent
    runner_args = {"train": True, "play": False, "sigma": None}
    if args.checkpoint is not None:
        runner_args["checkpoint"] = os.path.abspath(args.checkpoint) 
    runner.run(runner_args)

    if module_protocol:
        from dexmachina.rl.final_checkpoint import record_training_end
        selection = record_training_end(
            observer.algo, os.path.join(log_root_path, exp_name), args.hand, protocol,
            env.unwrapped.environment_transitions, max_epochs=args.max_epochs, num_envs=num_envs,
            horizon_length=agent_cfg['params']['config']['horizon_length'], marl_baseline=args.marl_baseline)
        run.summary['study/selected_checkpoint'] = selection['checkpoint']
        run.summary['study/final_epoch'] = int(observer.algo.epoch_num)
        run.summary['study/environment_transitions'] = env.unwrapped.environment_transitions
        if protocol == 'dexmachina':
            run.summary.update({'study/checkpoint_selection': selection['selection'],
                                'study/selected_checkpoint_epoch': selection['epoch'],
                                'study/selected_checkpoint_sha256': selection['checkpoint_sha256'],
                                'study/completed_updates': selection['completed_updates'],
                                'study/stop_reason': selection['stop_reason']})
        run.summary['study/status'] = 'trained_pending_evaluation'
        run.finish()

    # close the simulator
    exit()


if __name__ == "__main__":
    # run the main function
    main() 
    exit()
