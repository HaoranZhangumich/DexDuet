import os  
import argparse
import numpy as np
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.randomizations import get_randomization_cfg
from dexmachina.envs.curriculum import get_curriculum_cfg, get_v2_curriculum_cfg
from dexmachina.envs.maniptrans_curr import get_maniptrans_cfg
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg
from dexmachina.envs.rewards import RewardModule, get_reward_cfg
from dexmachina.envs.learner_config import add_learner_arguments, make_learner_config
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data 
from dexmachina.asset_utils import get_asset_path

def parse_clip_string(clip):
    vals = clip.split('-')
    if len(vals) == 3:
        print("Default to using subject s01 and clip 01")
        vals += ['s01', 'u01']
    assert len(vals) == 5, "Clip should be in format: obj_name-start-end-subject-clip"
    obj_name = vals[0]
    start = int(vals[1])
    end = int(vals[2])
    subject = vals[3]
    use_clip = vals[4].replace('u', '') # just 01/02
    return obj_name, start, end, subject, use_clip

def get_all_env_cfg(args, device, load_retarget_data=True):
    num_envs = args.num_envs
    ablation_protocol = getattr(args, 'ablation_protocol', 'legacy')
    v2_curr_cfg = None
    if ablation_protocol == 'dexmachina' and (
        not args.actuate_object or args.no_object or args.maniptrans
    ):
        raise ValueError('dexmachina ablations require the actuated object and native VOC curriculum')
    if ablation_protocol == 'v2':
        if not args.actuate_object or args.no_object or args.maniptrans:
            raise ValueError('v2 ablations require the actuated object and common linear assistance')
        v2_curr_cfg = get_v2_curriculum_cfg(int(args.max_epochs), int(args.num_zero_epoch),
                                             seed=getattr(args, 'seed', 42))
    if getattr(args, 'coord_no_partner_message', False) and not getattr(
        args, 'coordinate_actor', False
    ):
        raise ValueError(
            '--coord_no_partner_message requires --coordinate_actor'
        )
    if getattr(args, 'coord_local_critic', False) and not (
        getattr(args, 'coordinate_actor', False)
        and getattr(args, 'coord_no_partner_message', False)
    ):
        raise ValueError(
            '--coord_local_critic requires --coordinate_actor and --coord_no_partner_message'
        )
    if getattr(args, 'coordinate_actor', False):
        if args.no_object:
            raise ValueError('--coordinate_actor requires an object state')
        positive_dims = {
            'coord_encoder_dim': args.coord_encoder_dim,
            'coord_message_hidden_dim': args.coord_message_hidden_dim,
            'coord_message_dim': args.coord_message_dim,
        }
        positive_dims.update({
            f'coord_actor_units[{index}]': value
            for index, value in enumerate(args.coord_actor_units)
        })
        positive_dims.update({
            f'coord_critic_units[{index}]': value
            for index, value in enumerate(args.coord_critic_units)
        })
        for name, value in positive_dims.items():
            if int(value) < 1:
                raise ValueError(f'{name} must be positive, got {value}')
    # check if args has this attribute:
    last_n = args.last_n_frame if hasattr(args, 'last_n_frame') else -1
    reward_cfg = get_reward_cfg(last_n_frame=last_n)
    reward_cfg['imi_rew_weight'] = args.imi_rew_weight
    reward_cfg['imi_wrist_weight'] = args.imi_wrist_weight
    reward_cfg['contact_rew_weight'] = args.contact_rew_weight
    reward_cfg['contact_rew_function'] = args.contact_rew_function
    reward_cfg['contact_beta'] = args.contact_beta
    reward_cfg['wrist_frame_contact'] = not args.skip_wrist_frame_contact
    reward_cfg['mask_zero_contact'] = not args.nomask_zero_contact
    reward_cfg['use_retarget_contact'] = args.use_retarget_contact 
    reward_cfg['retarget_objframe'] = not args.retarget_worldfr
    reward_cfg['action_penalty'] = args.action_penalty
    reward_cfg['ref_motion_info'] = getattr(args, 'caps_temporal_gate', 'none') in ('ref_motion', 'task_reward_ref_motion')
    reward_cfg['cg_gate'] = args.cg_gate
    reward_cfg['cg_extra'] = args.cg_extra
    reward_cfg['hand_graph_weight'] = args.hand_graph_weight
    reward_cfg['hand_graph_beta_dir'] = args.hand_graph_beta_dir
    reward_cfg['hand_graph_beta_gap'] = args.hand_graph_beta_gap
    reward_cfg['hand_graph_thumb'] = args.hand_graph_thumb
    # a number, also when taken from the hand's retargeting scale: env.pkl must stay readable by older checkouts
    from dexmachina.envs.finger_spec import resolve_hand_graph_scale
    reward_cfg['hand_graph_scale'] = resolve_hand_graph_scale(args.hand, args.hand_graph_scale, needed=args.hand_graph_weight > 0)
    if args.hand_graph_weight > 0:
        print(f"Hand-graph fingertip-gap scale for {args.hand}: {reward_cfg['hand_graph_scale']}")
    reward_cfg['ho_contact_weight'] = args.ho_contact_weight
    for key in ('cap', 'tol', 'scale', 'touch', 'near'):
        reward_cfg[f'ho_contact_{key}'] = getattr(args, f'ho_contact_{key}')
    reward_cfg['hand_name'] = args.hand
    reward_cfg['ho_contact_hand'] = args.hand   # older name of hand_name, kept for older readers

    if args.objdex_baseline:
        print("Setting action mode to hybrid for objdex baseline and task rew beta to lower")
        args.action_mode = 'hybrid' 
        args.task_rew_betas = [30, 2, 10]
    
    task_betas = args.task_rew_betas
    assert len(task_betas) == 3, "Task reward betas should be of length 3"
    reward_cfg['obj_pos_beta'] = task_betas[0]
    reward_cfg['obj_rot_beta'] = task_betas[1]
    reward_cfg['obj_arti_beta'] = task_betas[2]
    reward_cfg['bc_rew_weight'] = args.bc_rew_weight
    reward_cfg['bc_beta'] = args.bc_beta
    reward_cfg['objdex_baseline'] = args.objdex_baseline    
    reward_cfg['multiply_all_rew'] = args.multiply_all_rew
    
    assert args.clip is not None, "Please provide a clip name"
    obj_name, start, end, subject, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end
    reward_cfg['ho_contact_object'] = obj_name
    
    retarget_data = dict()
    if args.use_teleop and load_retarget_data:
        raise NotImplementedError("Teleoperation data loading is not implemented yet.") 
    else:
        if load_retarget_data: 
            print(f"Loading retarget data for {args.arctic_object}")
            _, retarget_data = load_genesis_retarget_data(
                obj_name=args.arctic_object,
                hand_name=args.hand,
                frame_start=start, 
                frame_end=end,
                save_name=args.retarget_name,
                use_clip=use_clip,
                subject_name=subject,
            ) 
        demo_data = get_demo_data(
            obj_name=args.arctic_object,
            hand_name=args.hand,
            frame_start=start, 
            frame_end=end,
            use_clip=use_clip,
            subject_name=subject,
            load_retarget_contact=args.use_retarget_contact,
            load_hand_joints=args.hand_graph_weight > 0,
        ) 
        if args.ho_contact_weight > 0:
            from dexmachina.envs.ho_contact import load_human_targets
            demo_data.update(load_human_targets(str(get_asset_path('')), subject, obj_name, use_clip, start, end))
    
    ep_len = int(int(end) - int(start))
    if args.interp > 1:
        raise NotImplementedError("Interpolation is not implemented yet for retarget data.") 

    env_cfg = get_env_cfg(use_visualizer=args.vis, show_viewer=args.vis, show_fps=args.show_fps)
    rand_cfg = get_randomization_cfg(
        randomize=args.use_rand,
        on_friction=args.rand_friction,
        on_com=args.rand_com,
        on_mass=args.rand_mass,
        external_force=args.external_force,
        force_scale=args.force_scale,
        torque_scale=args.torque_scale
    )
    env_cfg['num_envs'] = num_envs
    env_cfg['early_reset_threshold'] = args.early_reset_threshold
    env_cfg['refresh_after_reset'] = not getattr(args, 'legacy_reset_obs', False)
    env_cfg['early_reset_aux_thres'] = dict(con=args.aux_reset_thres[0], imi=args.aux_reset_thres[1], bc=args.aux_reset_thres[2])
    env_cfg['episode_length'] = ep_len
    env_cfg['observe_tip_dist'] = args.observe_tip_dist
    env_cfg['observe_contact_force'] = True #ve_contact_force
    print(f"Setting observe_contact_force to True")
    env_cfg['use_contact_reward'] = args.contact_rew_weight > 0 or args.ho_contact_weight > 0   # the latter reads the contact buffers
    env_cfg['use_rl_games'] = args.use_rl_games
    env_cfg['rand_init_ratio'] = args.rand_init_ratio  
    env_cfg['chunk_ep_length'] = args.chunk_ep_length

    if args.record_interval > 0 or args.record_video:
        env_cfg["record_video"] = True
        # disable viewer 
        env_cfg['scene_kwargs']["use_visualizer"] = True
        env_cfg['scene_kwargs']["show_viewer"] = False
        env_cfg['scene_kwargs']["raytrace"] = args.raytrace
        if args.raytrace:
            print(f"Using white plane for raytracing")
            env_cfg['plane_urdf_path'] = 'assets/plane/plane_custom.urdf'
        env_cfg["max_video_frames"] = int(2 * ep_len)
    robot_cfgs = {
        'left': get_default_robot_cfg(name=args.hand, side='left'),
        'right': get_default_robot_cfg(name=args.hand, side='right')
    }
    for side in ['left', 'right']: 
        robot_cfgs[side]['action_mode'] = args.action_mode
        robot_cfgs[side]['hybrid_scales'] = tuple(args.hybrid_scales)
        robot_cfgs[side]['res_cap'] = args.res_cap
        robot_cfgs[side]['show_keypoints'] = args.show_kpts
        if args.hide_hand:
            robot_cfgs[side]['visualization'] = False

    obj_name = args.arctic_object
    object_cfgs = {
        obj_name: get_arctic_object_cfg(name=obj_name, convexify=args.convexify_object, texture_mesh=args.texture_object)
    } 
    if args.actuate_object:
        object_cfgs[obj_name]['actuated'] = True
        object_cfgs[obj_name]['kp'] = v2_curr_cfg['kp_init'] if v2_curr_cfg else args.kp_init
        object_cfgs[obj_name]['kv'] = v2_curr_cfg['kv_init'] if v2_curr_cfg else args.kv_init
        object_cfgs[obj_name]['force_range'] = v2_curr_cfg['force_range_init'] if v2_curr_cfg else args.force_range_init
        print('Setting batch_dofs_info=True for actuated object')
        env_cfg['scene_kwargs']['batch_dofs_info'] = True
    if args.color_object is not None:
        if args.color_object == 'gray':
            object_cfgs[obj_name]['color'] = (0.5, 0.5, 0.5, 1)
    if args.no_object:
        # print('WARNING: No object in the environment. ONLY using imitation reward.')
        reward_cfg['task_rew_weight'] = 0.0 
        object_cfgs = dict()

    assert len(args.upper_ratios) == len(args.lower_ratios) == 3, "Upper and lower ratios should have length 3"
    ups = list(args.upper_ratios)
    lows = list(args.lower_ratios)
    upper_ratios = dict(kp=ups[0], kv=ups[1], fr=ups[2])
    lower_ratios = dict(kp=lows[0], kv=lows[1], fr=lows[2])
    curr_rew_thres = list(args.curr_rew_thres)
    assert len(curr_rew_thres) == 4, "Curriculum reward thresholds should have length 4"
    zero_epoch = max(int(args.max_epochs - args.num_zero_epoch), 0)

    if args.maniptrans:
        assert not args.actuate_object, "Maniptrans curriculum only works with non-actuated objects"
        print("Using maniptrans curriculum")
        maniptrans_kwargs = dict(
            wait_epochs=args.wait_epochs,
            eps_finger_range=args.eps_finger_range,
            eps_object_Prange=args.eps_object_Prange,
            eps_object_Rrange=args.eps_object_Rrange,
            friction_range=args.friction_range,
            mode=args.maniptrans_mode,
            interval=args.interval,
            schedule=args.maniptrans_schedule,
            zero_epoch=zero_epoch,
        )
        curr_cfg = get_maniptrans_cfg(maniptrans_kwargs)
    else:
        curr_kwargs = dict(
            kp_init=args.kp_init,
            kv_init=args.kv_init,
            force_range_init=args.force_range_init, 
            rew_thresholds=dict(
                task=curr_rew_thres[0],
                con=curr_rew_thres[1],
                imi=curr_rew_thres[2],
                bc=curr_rew_thres[3],
            ),
            wait_epochs=args.wait_epochs,
            decay_rew=args.decay_rew,
            schedule=args.curr_schedule,
            fixed_mode=args.fixed_mode,
            interval=args.interval,
            first_ratio=args.first_ratio,
            first_stop_iter=args.first_stop_iter,
            second_stop_iter=args.second_stop_iter,
            
            deque_freq=args.deque_freq,
            grad_threshold=args.grad_threshold,
            gain_mode=args.gain_mode,

            uniform_mode=args.uniform_mode,
            upper_ratios=upper_ratios,
            lower_ratios=lower_ratios,
            deque_len=args.deque_len,
            decay_solimp=args.decay_solimp,
            solip_multiplier=args.solip_multiplier,
            resample_every_epoch=args.resample_every_epoch,
            skip_grad=args.skip_grad,
            zero_epoch=zero_epoch, # set this to last 
            dialback_ep_len=args.dialback_ep_len,
            dialback_min_epochs=args.dialback_min_epochs,
            dialback_ratios=dict(kp=args.dialback_ratios[0], kv=args.dialback_ratios[1], fr=args.dialback_ratios[2]),
        )
        if args.curr_schedule == 'rsi':
            if args.rsi_kf_to_end and not args.rsi_keyframes:
                raise ValueError('--rsi_kf_to_end needs --rsi_keyframes')
            curr_kwargs.update(
                rsi_bin_frames=args.rsi_bin_frames, rsi_bins=args.rsi_bins, rsi_uniform_mix=args.rsi_uniform_mix,
                rsi_credit_gamma=args.rsi_credit_gamma, rsi_fail_alpha=args.rsi_fail_alpha,
                rsi_success_up=args.rsi_success_up, rsi_success_down=args.rsi_success_down,
                rsi_min_count=args.rsi_min_count, rsi_min_gap=args.rsi_min_gap,
                rsi_deadline=args.rsi_deadline, rsi_zero_ramp=args.rsi_zero_ramp,
                rsi_min_horizon=args.rsi_min_horizon,
                rsi_keyframes=args.rsi_keyframes, rsi_kf_capacity=args.rsi_kf_capacity, rsi_kf_prob=args.rsi_kf_prob,
                rsi_kf_min_fill=args.rsi_kf_min_fill, rsi_kf_min_task=args.rsi_kf_min_task, rsi_kf_confirm=args.rsi_kf_confirm,
                rsi_kf_to_end=args.rsi_kf_to_end,
                rsi_grace_bank=args.rsi_grace_bank, rsi_frame0_share=args.rsi_frame0_share,
                rsi_step=upper_ratios['kp'], rsi_band=1.0 / lower_ratios['kp'],
            )
        curr_cfg = get_curriculum_cfg(curr_kwargs)
        if ablation_protocol == 'dexmachina':
            # Keep the native CLI-configured adaptive VOC; only its hard-stop
            # application is corrected under this separately named protocol.
            curr_cfg['ablation_protocol'] = 'dexmachina'
    if v2_curr_cfg is not None:
        curr_cfg = v2_curr_cfg
    
    env_kwargs = {
        'env_cfg': env_cfg,
        'robot_cfgs': robot_cfgs,
        'object_cfgs': object_cfgs,
        'reward_cfg': reward_cfg,
        'demo_data': demo_data,
        'retarget_data': retarget_data,
        'device': device,
        'visualize_contact': args.vis_contact,
        'rand_cfg': rand_cfg,
        'curriculum_cfg': curr_cfg,
        'group_collisions': args.group_collisions,
    }
    if getattr(args, 'coordinate_actor', False):
        env_kwargs['coordinate_actor_cfg'] = {
            'enabled': True,
            'message_enabled': not args.coord_no_partner_message,
            'critic_mode': 'local' if getattr(args, 'coord_local_critic', False) else 'central',
            'encoder_dim': args.coord_encoder_dim,
            'message_hidden_dim': args.coord_message_hidden_dim,
            'message_dim': args.coord_message_dim,
            'actor_units': list(args.coord_actor_units),
            'critic_units': list(args.coord_critic_units),
            'activation': args.coord_actor_activation,
        }
    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)  
    if args.n_envs_per_row is not None:
        env_kwargs['env_cfg']['n_envs_per_row'] = int(args.n_envs_per_row)
    learner_cfg = make_learner_config(args)
    if learner_cfg is not None:
        env_kwargs['learner_cfg'] = learner_cfg
    return env_kwargs

def get_common_argparser():
    """ add the commonly used command line arguments for env_cfg so that they can be used in multiple scripts """
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', '--num_envs', type=int, default=1)
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--overlay', '-ov', action='store_true')
    parser.add_argument('--n_envs_per_row', '-nrow', type=int, default=None)
    parser.add_argument('--texture_object', '-to', action='store_true', help='Show texture for the object and hide the actual URDF') 

    parser.add_argument('--arctic_object', '-ao', type=str, default='box')
    parser.add_argument('--hand', type=str, default='inspire_hand')
    parser.add_argument('--frame_start', '-fs', type=int, default=40)
    parser.add_argument('--frame_end', '-fe', type=int, default=200)
    parser.add_argument('--clip', '-cl', type=str, default="box-40-200-s01-u01")
    parser.add_argument('--show_markers', action='store_true', help='Whether to show contact markers')
    parser.add_argument('--interp', type=int, default=1, help='Interpolation multiplier for the demo data')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--action_mode', '-am', default='residual', choices=['residual', 'absolute', 'relative', 'hybrid','kinematic'])
    parser.add_argument('--show_kpts', action='store_true', help='Whether to show keypoints')
    parser.add_argument('--res_cap', action='store_true')
    parser.add_argument('--hybrid_scales', type=float, nargs='+', default=[0.04, 0.5])
    parser.add_argument('--hide_hand', action='store_true')

    parser.add_argument('--last_n_frame', type=int, default=-1)
    parser.add_argument('--early_reset_threshold', '-ert', type=float, default=0.5)
    parser.add_argument('--legacy_reset_obs', action='store_true',
                        help="DexMachina's original reset: the episode's first observation keeps stale object part poses and "
                             "wrist position (reproduces runs trained with the stale reset observation; docs/METHOD.md, section 5)")
    parser.add_argument('--aux_reset_thres', type=float, nargs='+', default=[0, 0, 0])
    parser.add_argument('--record_video', '-rv', action='store_true')
    parser.add_argument('--render_camera', '-rcam', type=str, default='front')
    parser.add_argument('--raytrace', '-ray', action='store_true')
    parser.add_argument('--record_interval', '-ri', type=int, default=-1)
    parser.add_argument('--use_teleop', action='store_true') 
    parser.add_argument('--retarget_name', '-rn', type=str, default='genesis')
    parser.add_argument('--teleop_fname', type=str, default='data/scripted/tmp_box/tmp_box.npz')
    parser.add_argument('--observe_tip_dist', '-obt', action='store_true')
    parser.add_argument('--observe_contact_force', '-obf', action='store_true')
    parser.add_argument('--task_rew_betas', '-trb', type=float, nargs='+', default=[10, 1, 5]) 

    parser.add_argument('--no_object', '-no_obj', action='store_true')
    parser.add_argument('--show_fps', '-fps', action='store_true')
    parser.add_argument('--color_object', '-co', default=None, type=str, help='Color of the object to be rendered')
    parser.add_argument('--action_penalty', '-ap', type=float, default=0.0)
    parser.add_argument('--imi_rew_weight', '-imi', type=float, default=0.0)
    parser.add_argument('--imi_wrist_weight', '-imw', type=float, default=0.5) 
    parser.add_argument('--contact_rew_weight', '-con', type=float, default=0.0)
    parser.add_argument('--bc_rew_weight', '-bc', type=float, default=0.0)
    parser.add_argument('--bc_beta', '-beta', type=float, default=500.0)
    parser.add_argument('--contact_beta', '-cbeta', type=float, default=10.0)
    parser.add_argument('--skip_wrist_frame_contact', '-swfc', action='store_true')
    parser.add_argument('--nomask_zero_contact',action='store_true')
    parser.add_argument('--contact_rew_function', '-crf', type=str, default='exp', choices=['exp', 'sigmoid'])
    parser.add_argument('--use_retarget_contact', '-urc', action='store_true')
    parser.add_argument('--retarget_worldfr', '-rwf', action='store_true')
    parser.add_argument('--cg_gate', type=float, default=0.0, help='contact-graph gate on the object reward, 0 = off, 1 = multiplicative')
    parser.add_argument('--cg_extra', type=float, default=0.25, help='weight of contacts the demo does not have')
    parser.add_argument('--hand_graph_weight', type=float, default=0.0,
                        help='reward w*exp(-beta_dir*dir_err - beta_gap*gap_err): palm and finger directions (object frame) and fingertip gaps vs the human hand; 0 = off')
    parser.add_argument('--hand_graph_beta_dir', type=float, default=5.0)
    parser.add_argument('--hand_graph_beta_gap', type=float, default=20.0)
    parser.add_argument('--hand_graph_thumb', type=float, default=0.5, help='thumb edge weight relative to the other direction edges')
    parser.add_argument('--hand_graph_scale', type=float, default=None,
                        help="human-to-robot length scale for the fingertip gaps (default: the hand's retargeting scaling_factor, "
                             "Inspire 1.2)")
    parser.add_argument('--ho_contact_weight', type=float, default=0.0,
                        help='hand-object contact reward, "where and how far" (0 = off): palm and outer finger segments match the '
                             'human hand\'s distance to each object part and nearest surface spot, every frame; added on top, '
                             'the DexMachina contact reward is unchanged (dexmachina/envs/ho_contact.py)')
    parser.add_argument('--ho_contact_cap', type=float, default=0.05, help='distances are capped here (m); farther counts as far')
    parser.add_argument('--ho_contact_tol', type=float, default=0.01, help='errors below this (m) count as perfect')
    parser.add_argument('--ho_contact_scale', type=float, default=0.02, help='score = exp(-(error - tol) / scale), m')
    parser.add_argument('--ho_contact_touch', type=float, default=0.01, help='closer than this (m) counts as touching (weight 1)')
    parser.add_argument('--ho_contact_near', type=float, default=0.5, help='weight of anchors that are near (< cap) but not touching')

    parser.add_argument('--kp', '-kp', type=float, default=200.0)
    parser.add_argument('--kv', '-kv', type=float, default=20.0)
    parser.add_argument('--force_range', '-fr', type=float, default=100.0)
    parser.add_argument('--obj_stiffness', '-ostiff', type=float, default=0.0)
    parser.add_argument('--obj_damping', '-odamp', type=float, default=0.0)

    parser.add_argument('--vis_contact', '-vc', action='store_true')
    parser.add_argument('--use_rl_games', '-rlg', action='store_true')
    parser.add_argument('--is_eval', '-eval', action='store_true')
    parser.add_argument('--rand_init_ratio', '-randr', type=float, default=0.0)   
    parser.add_argument('--chunk_ep_length', '-chunk', type=int, default=-1)
    
    parser.add_argument('--use_rand', '-rand', action='store_true')
    parser.add_argument('--rand_friction', '-rf', action='store_true')
    parser.add_argument('--rand_com', '-rc', action='store_true')
    parser.add_argument('--rand_mass', '-rm', action='store_true')
    parser.add_argument('--external_force', '-ef', action='store_true')
    parser.add_argument('--force_scale', '-fscale', type=float, default=5.0)
    parser.add_argument('--torque_scale', '-tscale', type=float, default=0.0)
    parser.add_argument('--convexify_object', '-cvf', action='store_true')
    parser.add_argument('--actuate_object', '-act', action='store_true')
    parser.add_argument('--objdex_baseline', '-objdex', action='store_true')
    parser.add_argument('--multiply_all_rew', '-mar', action='store_true')
    parser.add_argument('--group_collisions', '-gcol', action='store_true')

    parser.add_argument('--interval', type=int, default=500) 
    parser.add_argument('--first_ratio', type=float, default=0.3)
    parser.add_argument('--first_stop_iter', type=int, default=7000)
    parser.add_argument('--second_stop_iter', type=int, default=16000)
    parser.add_argument('--kp_init', type=float, default=100.0)
    parser.add_argument('--kv_init', type=float, default=10.0)
    parser.add_argument('--force_range_init', type=float, default=50.0)
    parser.add_argument("--max_epochs", "-me", type=int, default=5000, help="Maximum number of epochs to train the agent.")     
    parser.add_argument('--num_zero_epoch', '-nze', type=int, default=1000)
    parser.add_argument('--curr_schedule', type=str, default='uniform', choices=['fixed', 'exp', 'uniform', 'rsi'])
    # RSI + BeyondMimic-style adaptive start sampling (used only with --curr_schedule rsi);
    # step and band come from upper/lower gain ratios like the native uniform schedule.
    parser.add_argument('--rsi_bin_frames', type=int, default=0,
                        help='RSI start-bin width in frames, e.g. 15; the bin count then follows the clip length (0 = legacy fixed --rsi_bins count)')
    parser.add_argument('--rsi_bins', type=int, default=20, help='legacy fixed RSI bin count, used when --rsi_bin_frames is 0')
    parser.add_argument('--rsi_uniform_mix', type=float, default=0.3)
    parser.add_argument('--rsi_credit_gamma', type=float, default=0.75)
    parser.add_argument('--rsi_fail_alpha', type=float, default=0.02)
    parser.add_argument('--rsi_success_up', type=float, default=0.8)
    parser.add_argument('--rsi_success_down', type=float, default=0.4)
    parser.add_argument('--rsi_min_count', type=int, default=64)
    parser.add_argument('--rsi_min_gap', type=int, default=40)
    parser.add_argument('--rsi_deadline', type=int, default=9000)
    parser.add_argument('--rsi_zero_ramp', type=int, default=1000)
    parser.add_argument('--rsi_min_horizon', type=int, default=10)
    parser.add_argument('--rsi_keyframes', action='store_true', help='RSI starts from reached simulator states')
    parser.add_argument('--rsi_kf_capacity', type=int, default=256)
    parser.add_argument('--rsi_kf_prob', type=float, default=0.002)
    parser.add_argument('--rsi_kf_min_fill', type=int, default=32)
    parser.add_argument('--rsi_kf_min_task', type=float, default=0.6)
    parser.add_argument('--rsi_kf_confirm', type=int, default=0, help='store a keyframe only after its lane goes on well for this many frames')
    parser.add_argument('--rsi_kf_to_end', action='store_true',
                        help='store a keyframe only once its lane finishes the clip, and serve it only to lanes with at least the help it was reached with (overrides --rsi_kf_confirm)')
    parser.add_argument('--rsi_grace_bank', type=float, default=0.0)
    parser.add_argument('--rsi_frame0_share', type=float, default=0.0,
                        help='share of RSI starts always at frame 0, the evaluation start (0 = off)')
    parser.add_argument('--fixed_mode', type=str, default='uniform', choices=['exp', 'lin', 'uniform'])
    parser.add_argument('--decay_rew', '-drw', action='store_true')
    parser.add_argument('--wait_epochs', type=int, default=500)
    parser.add_argument('--decrease_rew', '-dr', action='store_true')
    parser.add_argument('--curr_rew_thres', type=float, nargs='+', default=[0.5, 0, 0, 0])
    parser.add_argument('--deque_freq', type=int, default=1)
    parser.add_argument('--grad_threshold', type=float, default=0.001)
    parser.add_argument('--gain_mode', '-gm', type=str, default='all', choices=['all', 'kpkv', 'fr'])
    
    # uniform-sampling schedule 
    parser.add_argument('--uniform_mode', '-um', type=str, default='slow', choices=['fast', 'slow'])
    parser.add_argument('--upper_ratios', type=float, nargs='+', default=[0.8, 0.95, 0.95])
    parser.add_argument('--lower_ratios', type=float, nargs='+', default=[0.7, 0.9, 0.9])
    parser.add_argument('--deque_len', type=int, default=30)
    parser.add_argument('--decay_solimp', '-ds', action='store_true')
    parser.add_argument('--solip_multiplier', '-solip', type=float, default=0.95)
    parser.add_argument('--resample_every_epoch', '-resample', type=int, default=-1)
    parser.add_argument('--skip_grad', action='store_true')

    parser.add_argument('--dialback_ep_len', type=int, default=30)
    parser.add_argument('--dialback_min_epochs', type=int, default=500)
    parser.add_argument('--dialback_ratios', type=float, nargs='+', default=[0.98, 1.0, 1.0])

    # DexDuet's per-hand policy: opt-in shared per-hand actor and Deep-Sets
    # partner encoder.
    parser.add_argument(
        '--coordinate_actor', '--coord_actor',
        action='store_true',
        help=(
            'Use a shared factorized per-hand actor with an optional learned '
            'partner message. The released centralized actor remains default.'
        ),
    )
    parser.add_argument(
        '--coord_no_partner_message',
        action='store_true',
        help='Ablation: factorized actor with the partner message fixed to zero.',
    )
    parser.add_argument(
        '--coord_local_critic',
        action='store_true',
        help="Ablation (IPPO): per-hand critic on that hand's local inputs only.",
    )
    parser.add_argument('--coord_encoder_dim', type=int, default=128)
    parser.add_argument('--coord_message_hidden_dim', type=int, default=64)
    parser.add_argument('--coord_message_dim', type=int, default=64)
    parser.add_argument(
        '--coord_actor_units', type=int, nargs='+', default=[256, 128]
    )
    parser.add_argument(
        '--coord_critic_units', type=int, nargs='+',
        default=[512, 512, 256, 128],
    )
    parser.add_argument(
        '--coord_actor_activation', type=str, default='elu',
        choices=['relu', 'tanh', 'elu', 'selu', 'swish', 'gelu'],
    )
    # additional arguments for maniptrans
    parser.add_argument('--maniptrans', '-mpt', action='store_true')
    parser.add_argument('--friction_range', type=float, nargs='+', default=[4, 1])
    parser.add_argument('--eps_finger_range', '-efr', type=float, nargs='+', default=[0.1, 0.05])
    parser.add_argument('--eps_object_Prange', '-eopr', type=float, nargs='+', default=[0.1, 0.03])
    parser.add_argument('--eps_object_Rrange', '-eorr', type=float, nargs='+', default=[1.57, 0.52])
    parser.add_argument('--maniptrans_mode', '-mpmode', type=str, default='fixed', choices=['fixed', 'auto'])
    parser.add_argument('--maniptrans_schedule', '-mpsch', type=str, default='exp', choices=['exp', 'linear'])
    return add_learner_arguments(parser)
