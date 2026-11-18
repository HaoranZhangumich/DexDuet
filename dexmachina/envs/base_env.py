import os  
import torch
import numpy as np
import genesis as gs 
from dexmachina.envs.robot import BaseRobot
from dexmachina.envs.object import ArticulatedObject
from dexmachina.envs.rewards import RewardModule
from dexmachina.envs.math_utils import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
)
from dexmachina.envs.contacts import get_filtered_contacts
from dexmachina.envs.randomizations import RandomizationModule
from dexmachina.envs.curriculum import Curriculum 
from dexmachina.envs.maniptrans_curr import ManipTransCurriculum 
from dexmachina.envs.learner_config import learner_cfg_from_legacy
from dexmachina.eval.protocol import policy_controlled_env_indices
from typing import Dict, List, Tuple, Union
from collections import deque
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

TABLE_HEIGHT = 0.6
OBJ_DEFAULT_POS = (0.0597, -0.2476,  1.0354)
OBJ_DEFAULT_ROT = (-0.6413,  0.2875,  0.6467, -0.2964)
CARDBOARD_POS = (0, -0.08, 0.90) 
CAMERA_RES=(160, 160)
ENV_SPACING=(1.0, 1.0)


def get_scene_cfg(
    dt=1/60, 
    zero_gravity=False, 
    show_viewer=False, 
    show_fps=False, 
    batch_dofs_info=False, 
    use_visualizer=False,
    n_rendered_envs=None,
    raytrace=False,
    visualize_contact=False,
    enable_joint_limit=True,
):
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0, 0, -9.81) if not zero_gravity else (0, 0, 0),
            #  gravity=(0, 0, 0),
        ), 
        vis_options=gs.options.VisOptions(
            n_rendered_envs=n_rendered_envs,
            show_world_frame=False,
            visualize_contact=visualize_contact,
            segmentation_level="entity", # "link" or "entity" or 'geom'
            # NOTE set to false to speed up rendering!!
            ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=enable_joint_limit,
            max_collision_pairs=100, # default was 100
            batch_dofs_info=batch_dofs_info,
        ),
        viewer_options=gs.options.ViewerOptions( 
            camera_pos=(0.5, 1.5, 1.8), # looking from behind
            camera_lookat=(0.0, -0.15, 1.0),
            camera_fov=30,
        ),
        use_visualizer=use_visualizer,
        show_viewer=show_viewer,
        show_FPS=show_fps, 
    )
    
    if raytrace:
        scene_cfg['renderer'] = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=10.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 1.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
    return scene_cfg

def get_env_cfg(
    dt=1/60, 
    use_visualizer=False, 
    show_viewer=False, 
    show_fps=False, 
    zero_gravity=False, 
):  
    scene_kwargs = dict(
        dt=dt, 
        zero_gravity=zero_gravity, 
        use_visualizer=use_visualizer,
        show_viewer=show_viewer, 
        show_fps=show_fps,
        batch_dofs_info=False, 
        # NOTE: this will be slower but allows different gains per env,turn this on for actuated object
        n_rendered_envs=None,
        raytrace=False,
        visualize_contact=False,
        enable_joint_limit=True, # enable joint limits for robots
        )
    camera_kwargs = dict(
        front=dict(
            res=(160, 160),
            # pos=(0.5, -1.5, 1.2),
            # lookat=(0.0, -0.15, 1.0),
            pos=( 0, -1.6,  2.2),
            lookat=(0.0, -0.1, 1.2),
            fov=30,
        ),
        back=dict(
            res=(160, 160),
            pos=(0.4, 1.5, 1.8),
            lookat=(0.0, -0.15, 1.0),
            fov=25,
        ),
    )
    env_cfg = {
        "num_envs": 1, 
        "episode_length": 10,
        "action_clip": 1.0,
        "action_scale": 1.0,
        "obs_clip": 5.0,
        "scene_kwargs": scene_kwargs,   
        "dt": dt,
        "early_reset_threshold": 0.0,
        "early_reset_interval": 5,
        "refresh_after_reset": True,
        "early_reset_aux_thres": dict(con=0, imi=0, bc=0),
        "record_video": False,
        "render_segmentation": False,
        "texture_cardbox": False,
        "max_video_frames": 0,
        "observe_tip_dist": False,
        "observe_contact_force": False,
        "use_contact_reward": False,
        'use_rl_games': True,
        "is_eval": False, 
        "rand_init_ratio": 0.0, # randomize initial states  
        "env_spacing": ENV_SPACING,
        "n_envs_per_row": None, # this will default to grid layout 
        "chunk_ep_length": -1,#chunk the episode length
        "plane_urdf_path": 'urdf/plane/plane.urdf',
        'camera_kwargs': camera_kwargs,
        'render_camera': 'front',  
    } 
    return env_cfg 

class BaseEnv:
    """
    Support multiple different embodiments, and either bimanual or single hand
    """
    def __init__(
        self, 
        env_cfg,
        robot_cfgs, # dict of robots 
        object_cfgs, # dict of objects
        reward_cfg, 
        demo_data,
        retarget_data=dict(), # dict of joint_pos after retargeting
        rand_cfg=dict(),
        curriculum_cfg=dict(), 
        coordinate_actor_cfg=None,
        learner_cfg=None,
        device=torch.device('cuda'),
        visualize_contact=False, 
        contact_marker_cfgs=dict(),
        group_collisions=False,
        render_figure=False,
        hide_cardbox=False, 
        postpone_build=False,
        wrench_cfg=None,  # key of run folders written before the rename to learner_cfg
    ):
        if wrench_cfg is not None:
            if learner_cfg is not None:
                raise ValueError('environment configuration has both wrench_cfg and learner_cfg')
            learner_cfg = learner_cfg_from_legacy(wrench_cfg)
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.demo_data = demo_data
        self.curr_cfg = curriculum_cfg
        self.coordinate_actor_cfg = coordinate_actor_cfg or {}
        self.learner_cfg = learner_cfg or None
        cfg = self.learner_cfg or {}
        self.per_hand_ppo = bool(cfg.get('per_hand_ppo', False))
        self.per_hand_value = bool(cfg.get('per_hand_value', self.per_hand_ppo))
        self.ablation_protocol = cfg.get('ablation_protocol', 'legacy')
        self.environment_transitions = 0
        self.epoch_num = 0
        self.group_collisions = group_collisions

        self.num_envs = env_cfg['num_envs']
        self.max_video_frames = env_cfg['max_video_frames']
        self.record_video = env_cfg['record_video']
        self.render_segmentation = env_cfg.get('render_segmentation', False)
        self.max_episode_length = int(env_cfg['episode_length'])
        self.chunk_ep_length = env_cfg['chunk_ep_length']
        if self.chunk_ep_length > 0:
            print("Chunking episode length to ", self.chunk_ep_length)
            self.max_episode_length = self.chunk_ep_length

        self.action_clip = env_cfg['action_clip']
        self.action_scale = env_cfg['action_scale']
        self.obs_clip = env_cfg['obs_clip']
        self.dt = env_cfg['dt'] 
        self.early_reset_threshold = env_cfg['early_reset_threshold']
        self.early_reset_interval = int(env_cfg['early_reset_interval']) 
        self.early_reset_aux_thres = env_cfg.get('early_reset_aux_thres', dict())
        # if true, return obs dict insteaf of obs
        self.use_rl_games = env_cfg['use_rl_games']
        # finger correspondence of each hand side (retargeting config + URDF), for the hand-graph and hand-object contact rewards
        self.finger_specs = dict()
        if float(reward_cfg.get('hand_graph_weight', 0.0)) > 0.0 or float(reward_cfg.get('ho_contact_weight', 0.0)) > 0.0:
            from dexmachina.envs.finger_spec import load_finger_spec
            hand = reward_cfg.get('hand_name') or reward_cfg.get('ho_contact_hand') or 'inspire_hand'
            self.finger_specs = {side: load_finger_spec(hand, side, robot_cfg=cfg) for side, cfg in robot_cfgs.items()}
        self.reward_module = RewardModule(
            reward_cfg, demo_data, retarget_data, device, finger_specs=self.finger_specs or None
            )

        self.reward_keys = self.reward_module.get_reward_keys()
        if self.num_envs > 3 and self.record_video:
            print("Warning: setting render env to 1 when there's more than 3 envs")
            env_cfg['scene_kwargs']['n_rendered_envs'] = 1
            self.max_video_frames = int(self.max_episode_length * 2)

        self.scene_cfg = get_scene_cfg(**env_cfg['scene_kwargs'])
        
        if self.group_collisions:
            self.scene_cfg['rigid_options'].enable_self_collision = True
            if hasattr(self.scene_cfg['rigid_options'], 'self_collision_group_filter'):
                print('Setting the SAME collision grouping to both hands')
                self.scene_cfg['rigid_options'].self_collision_group_filter = True
                collision_groups = robot_cfgs['left'].get('collision_groups', dict())
                self.scene_cfg['rigid_options'].link_group_mapping = collision_groups
            else:
                print('WARNING: this Genesis build has no self_collision_group_filter/link_group_mapping support; '
                      'falling back to full self-collision checking (enable_self_collision=True) without grouping.')
        if render_figure:
            print("Disabling gravity for figure rendering")
            self.scene_cfg['sim_options'].gravity = (0, 0, 0)
        self.demo_length = self.reward_module.get_demo_length()
        if self.chunk_ep_length <= 0:
            assert self.max_episode_length >= self.demo_length, f"Episode length {self.max_episode_length} != demo length {self.reward_module.demo_length}"
        
        self.is_finite_horizon = True # need this for rlgames
        
        self.table_height = TABLE_HEIGHT
        self.device = device 
        self.scene = gs.Scene(**self.scene_cfg)
        
        self.rigid_solver = None 
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver 

        self.robot_cfgs = robot_cfgs
        self.robots = dict()
        self.retarget_data = retarget_data
        robot_reference_eval = env_cfg['is_eval']
        # BaseRobot otherwise zeroes the final lane at construction and reset, even when evaluation explicitly uses
        # every policy lane. Applied under every protocol (ablation runs), so the unified-policy row (legacy
        # protocol) starts its evaluation from the same states as the per-hand rows; training is unaffected (is_eval).
        # env_cfg['zero_pose_last_lane'] (eval_rl_games.py --zero-pose-last-lane) keeps DexMachina's original behaviour,
        # with which the paper evaluates DexMachina's released checkpoints: the final lane starts from a zero hand pose.
        robot_reference_eval = robot_reference_eval and (env_cfg.get('eval_reference_env', True)
                                                         or env_cfg.get('zero_pose_last_lane', False))
        for k, cfg in robot_cfgs.items():
            self.robots[k] = BaseRobot(
                robot_cfg=cfg, 
                scene=self.scene,
                num_envs=self.num_envs,
                device=device,
                retarget_data=retarget_data.get(k, dict()),
                visualize_contact=visualize_contact,
                is_eval=robot_reference_eval,
                disable_collision=cfg.get('disable_collision', False),
                ) 
            # BaseRobot.set_kpt_links skips unknown names with a warning, which would shift every keypoint index after them
            if self.robots[k].missing_kpt_links:
                raise ValueError(f"{k} hand: keypoint links {self.robots[k].missing_kpt_links} are not links of "
                                 f"{cfg['urdf_path']} (retargeted kpt_names or the hand cfg's kpt_link_names)")
            # hand-graph nodes that are not keypoints: tracked in the robot's extra buffer; a wrist uses wrist_pose
            extra = self.reward_module.hand_graph_extra.get(k, [])
            others = [n for n in extra if n != self.robots[k].wrist_link_name]
            if others:
                self.robots[k].track_extra_links(others)
        self.robot_names = list(self.robots.keys())
        
        self.object_cfgs = object_cfgs
        # use retarget data to set base_init_pos, base_init_quat in obj_cfg
        self.objects = dict()
        for k, cfg in object_cfgs.items():
            # cfg['base_init_pos'] =  demo_data['obj_pos'][0]
            # cfg['base_init_quat'] = demo_data['obj_quat'][0]
            self.objects[k] = ArticulatedObject(
                cfg, 
                device=device,
                scene=self.scene,
                num_envs=self.num_envs,
                demo_data=demo_data,
                visualize_contact=visualize_contact, 
                disable_collision=cfg.get('disable_collision', False), # or render_figure,
                ) 
        
        self.object_names = list(self.objects.keys())
        self.object = None 
        if len(self.object_names) > 0: 
            self.object = self.objects[self.object_names[0]] # only support one object for now
        self.n_objects = len(self.object_names) # might be 0!!
       
        self.use_curriculum = False 
        self.curriculum = None
        if self.n_objects == 1 and self.object.actuated:
            self.use_curriculum = True 
            self.curriculum = Curriculum(
                self.curr_cfg, 
                task_object=self.object,
                reward_keys=self.reward_keys,
                num_envs=self.num_envs,
                achieved_length=0,
                max_episode_length=self.max_episode_length,
            )        
        elif self.curr_cfg.get('type', None) == 'maniptrans':
            print("Using ManipTrans curriculum for ablation")
            self.use_curriculum = True
            self.curriculum = ManipTransCurriculum(
                self.curr_cfg, 
                task_object=self.object,
                reward_keys=self.reward_keys,
                num_envs=self.num_envs,
                achieved_length=0,
                max_episode_length=self.max_episode_length,
                sim=self.scene.sim,
                rigid_solver=self.rigid_solver,
            )
        cardbox_size = (0.2,0.2,0.1)
        if self.n_objects == 1 and 'notebook' in self.object_names[0]:
            print("Adding a SMALLER cardboard box for notebook") 
            cardbox_size = (0.15, 0.15, 0.1) # wider: cardbox_size = (0.25, 0.2, 0.1)
        
        cardbox_surface = gs.surfaces.Rough(roughness=0.1, color=(167/255, 134/255, 103/255, 1.0))
        texture = env_cfg.get('texture_cardbox', False)
        if texture:
            if not isinstance(texture, str):
                raise ValueError('texture_cardbox needs the path of a texture image')
            cardbox_surface = gs.surfaces.Default( 
                diffuse_texture=gs.textures.ImageTexture(
                    image_path=texture,
                ),
            ) 
        self.cardboard_box = self.scene.add_entity(
            gs.morphs.Box(
                pos=CARDBOARD_POS,
                size=cardbox_size,
                fixed=True,
                visualization=(not hide_cardbox),
            ),
            surface=cardbox_surface,
        ) 
        
        self.contact_markers = dict()
        for name, marker_cfg in contact_marker_cfgs.items():
            num_vis_contacts = marker_cfg.get('num_vis_contacts', 0)
            color = marker_cfg.get('color', (1.0, 0.0, 0.0, 0.8))
            radius = marker_cfg.get('radius', 0.007)
            markers = [] 
            for _ in range(num_vis_contacts):
                marker = self.scene.add_entity(
                    gs.morphs.Sphere(
                        radius=radius, fixed=False, collision=False, # no collision
                        ),
                    surface=gs.surfaces.Smooth(color=color),
                )
                markers.append(marker)
            self.contact_markers[name] = markers
        
        plane_urdf_path = env_cfg.get('plane_urdf_path', 'urdf/plane/plane.urdf')
        self.ground = self.scene.add_entity(
            gs.morphs.URDF(file=plane_urdf_path, fixed=True)
        )

        self._recording = False
        self._recorded_frames = []
        
        self._floating_camera = None
        if self.record_video:
            assert gs.platform != 'macos', "Cannot render on macos"           
            self.render_camera = env_cfg.get('render_camera', 'front')
            self._add_camera(camera_kwargs=env_cfg['camera_kwargs'])

        self.env_cfg = env_cfg
        self.rand_cfg = rand_cfg
        if not postpone_build:
            self.build_scene()
            self.post_scene_build_setup()
        else:
            print("Scene created but not built yet") 
            
    def build_scene(self):
        env_cfg = self.env_cfg
        self.scene.build(
            n_envs=self.num_envs, 
            env_spacing=env_cfg.get('env_spacing', ENV_SPACING),
            n_envs_per_row=env_cfg.get('n_envs_per_row', None),
            )

    def post_scene_build_setup(self):
        """ call this separately to customize the scene after env.init()"""
        env_cfg = self.env_cfg  
        rand_cfg = self.rand_cfg
        self.setup_actions(self.robots) 
        self.observe_tip_dist = env_cfg['observe_tip_dist']
        if self.observe_tip_dist:
            # load vertices
            assert self.n_objects == 1, "Only support one object for now"
            obj = self.objects[self.object_names[0]]
            # Keep the sampled vertices on the GPU. compute_closest_vertice_dist_single() then
            # repeats them to (N, 300, 3) on the device instead of building that tensor on the CPU and copying it to
            # the GPU 4 times per step; .repeat() (not .expand()) keeps the same contiguous input, so the same values.
            self.obj_verts = {part: obj.sample_mesh_vertices(300, part).to(self.device) for part in ['top', 'bottom']}
        
        self.observe_contact_force = env_cfg.get('observe_contact_force', False)
        if self.n_objects == 0:
            self.observe_contact_force = False
            print("Disabling contact force observation because no object")
        self.use_contact_reward = env_cfg.get('use_contact_reward', False) 
        if self.observe_contact_force or self.use_contact_reward:
            self.num_obj_links = len(self.object.coll_idxs_global)
            self.num_robot_links = sum([len(robot.coll_idxs_global) for robot in self.robots.values()])

            self.filter_links_a = torch.tensor(self.object.coll_idxs_global, device=self.device)
            self.filter_links_b = torch.tensor(
                self.robots['left'].coll_idxs_global + self.robots['right'].coll_idxs_global, 
                device=self.device
                )
            self.num_left_contact_links = len(self.robots['left'].coll_idxs_global)

            print("num_obj_links", self.num_obj_links)
        self.ho_contact_points = None
        if self.reward_module.ho_contact_weight > 0.0:
            if not self.use_contact_reward:
                raise ValueError("The hand-object contact reward needs the contact buffers (use_contact_reward)")
            self.setup_ho_contact()
        

        self.obs_dim, self.obs_idxs = self.compute_obs_dim() 
        self.num_obs = self.obs_dim
        self.num_privileged_obs = None
        self.num_actions = self.action_dim # need for rsl
        self.rand_init_ratio = env_cfg.get('rand_init_ratio', 0.0) 
        self.initialize_value_buffers()
        
        # # NOTE! seems like scene must be built first
        for name, robot in self.robots.items():
            robot.post_scene_build_setup()
        
        for name, obj in self.objects.items():
            obj.post_scene_build_setup()
        
        if self.use_curriculum:
            self.curriculum.post_scene_build_setup()

        self.keyframes = None
        if self.use_curriculum and getattr(self.curriculum, 'schedule', None) == 'rsi' and self.curriculum.rsi_keyframes:
            from dexmachina.envs.keyframes import KeyframeBuffer
            c = self.curriculum
            self.keyframes = KeyframeBuffer(
                self, c.rsi_bin_lo.tolist(), c.rsi_bin_hi.tolist(), capacity=c.rsi_kf_capacity,
                capture_prob=c.rsi_kf_prob, min_fill=c.rsi_kf_min_fill, min_task_rew=c.rsi_kf_min_task,
                min_horizon=c.rsi_min_horizon, confirm=c.rsi_kf_confirm, to_end=bool(c.rsi_kf_to_end), seed=c.seed)
            c.keyframes = self.keyframes

        self.scene.step()
        self.extras = dict(log=dict())
        self.rew_dict = dict()
        self.obs_dict = dict()
        self.is_eval = env_cfg.get('is_eval', False) # if eval, have the option of skipping some env idxs for vis
        self._step_env_idxs = policy_controlled_env_indices(
            self.num_envs, is_eval=self.is_eval,
            eval_reference_env=env_cfg.get('eval_reference_env', True),
        )
        
        self.randomization = RandomizationModule(rand_cfg, self.rigid_solver, self.object, self.num_envs)
        if self.record_video: #  and self.num_envs > 2: 
            # NOTE: set the camera to only record the first env, must do this after the scene.build call
            offset = self.scene.rigid_solver.envs_offset.to_numpy()[0] # (3,)
            lookat_pos = offset + np.array([0, -0.1, 1.0])
            cam_pos = lookat_pos + np.array([0.0, -1.5, 1.2])
            self._set_camera(pos=cam_pos, lookat=lookat_pos, fov=30, name='front')


    def setup_actions(self, robots: Dict[str, BaseRobot]):
        action_dim = 0 
        idxs_to_robot = dict()
        for k, robot in robots.items():
            dim = robot.get_action_dim()
            idxs_to_robot[k] = [i for i in range(action_dim, action_dim+dim)]
            action_dim += dim
        self.action_dim = action_dim
        self.action_idxs_to_robot = idxs_to_robot
        # TODO: sim action latency as in Go2?
        # self.action_latency = 0 
    
    def update_contact_markers(self, contact_dict: Dict[str, torch.Tensor]):
        if len(self.contact_markers) == 0:
            return
        for name, contacts in contact_dict.items():
            if name not in self.contact_markers:
                continue
            markers = self.contact_markers[name]
            contacts = contacts.reshape((contacts.shape[0], -1, 3)) # squeeze middle dims
            # contacts shaped (num_envs, num_contacts, 6)
            num_vis = min(len(markers), contacts.shape[1])
            for i in range(num_vis):
                markers[i].set_pos(contacts[:, i, :3])
        
    def compute_obs_dim(self):
        
        obs_dim = 0
        obs_dim_info = dict()
        obs_idxs = dict()
        for k, robot in self.robots.items():
            dim, dim_info = robot.compute_obs_dim()
            obs_dim_info[k] = dim_info
            obs_idxs[k] = (obs_dim, obs_dim + dim)
            obs_dim += dim
        for k, obj in self.objects.items():
            dim, dim_info = obj.compute_obs_dim()
            obs_dim_info[k] = dim_info
            obs_idxs[k] = (obs_dim, obs_dim + dim)
            obs_dim += dim

        # get_observations appends phase immediately after object state.  Keep
        # this index map aligned with the existing tensor order (tip-distance
        # and contact-force observations follow it).
        obs_idxs['episode_length'] = (obs_dim, obs_dim + 1)
        obs_dim += 1

        if self.observe_tip_dist:
            for side in ('left', 'right'):
                dim = self.robots[side].n_kpts * 2
                obs_idxs[f'{side}_tip_dist'] = (obs_dim, obs_dim + dim)
                obs_dim += dim
        
        if self.observe_contact_force:
            dim = self.num_obj_links * self.num_robot_links
            obs_idxs['contact_force'] = (obs_dim, obs_dim + dim)
            obs_dim += dim

        if self.coordinate_actor_cfg.get('enabled', False):
            partner_state_dim = sum(
                7 + 2 * int(self.robots[name].ndof)
                for name in self.robot_names
            )
            obs_idxs['coordinate_partner_states'] = (
                obs_dim, obs_dim + partner_state_dim
            )
            obs_dim += partner_state_dim
            obs_idxs['coordinate_previous_actions'] = (
                obs_dim, obs_dim + self.action_dim
            )
            obs_dim += self.action_dim
            obs_idxs['coordinate_active_mask'] = (obs_dim, obs_dim + 2)
            obs_dim += 2
            obs_idxs['coordinate_presence_mask'] = (obs_dim, obs_dim + 2)
            obs_dim += 2
            obs_idxs['coordinate_message_gate'] = (obs_dim, obs_dim + 1)
            obs_dim += 1

        # return 20, obs_idxs
        return obs_dim, obs_idxs

    def get_coordinate_actor_layout(self):
        """Describe actor-visible slices for the factorized RL-Games model.

        Raw partner-hand and contact-force observations remain in the flat
        tensor for the centralized critic.  The factorized actor accesses a
        partner only through the explicitly listed wrist/q/qdot/action/active
        descriptor consumed by its Deep-Sets message encoder.
        """

        if not self.coordinate_actor_cfg.get('enabled', False):
            raise RuntimeError('Coordinate actor layout requested while disabled')
        if self.robot_names != ['left', 'right']:
            raise ValueError(
                "Coordinate actor currently requires ['left', 'right'], got "
                f"{self.robot_names}"
            )

        hand_obs_slices = [list(self.obs_idxs[name]) for name in self.robot_names]
        hand_feature_indices = []
        partner_start, _ = self.obs_idxs['coordinate_partner_states']
        for name in self.robot_names:
            partner_dim = 7 + 2 * int(self.robots[name].ndof)
            hand_feature_indices.append(
                list(range(partner_start, partner_start + partner_dim))
            )
            partner_start += partner_dim

        shared_feature_indices = []
        for name in self.object_names:
            start, stop = self.obs_idxs[name]
            shared_feature_indices.extend(range(start, stop))
        phase_start, phase_stop = self.obs_idxs['episode_length']
        shared_feature_indices.extend(range(phase_start, phase_stop))

        previous_start, _ = self.obs_idxs['coordinate_previous_actions']
        previous_action_slices = []
        for name in self.robot_names:
            action_indices = self.action_idxs_to_robot[name]
            previous_action_slices.append([
                previous_start + action_indices[0],
                previous_start + action_indices[-1] + 1,
            ])

        return {
            'hand_obs_slices': hand_obs_slices,
            'hand_feature_indices': hand_feature_indices,
            'shared_feature_indices': shared_feature_indices,
            'previous_action_slices': previous_action_slices,
            'active_mask_slice': list(
                self.obs_idxs['coordinate_active_mask']
            ),
            'presence_mask_slice': list(
                self.obs_idxs['coordinate_presence_mask']
            ),
            'message_gate_slice': list(
                self.obs_idxs['coordinate_message_gate']
            ),
        }
    
    def initialize_value_buffers(self):
        # assume robots and objects have been initialized with buffers
        self.obs_buf = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_task_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_con_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_imi_rew = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_bc_rew = torch.zeros(self.num_envs, device=self.device)
        self.obs_sum = torch.zeros(self.num_envs, self.obs_dim, device=self.device)

        # match isaaclab
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.nan_envs = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Task reward credited to the early-reset check only: mid-clip RSI starts get the bank an arriving lane
        # would have accumulated, so a brief low-reward stretch (the lid swings) does not end them at once.
        self.early_reset_bank = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        # NOTE this must be int dtype for indexing in match_demo_state
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.episode_start_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # this should be demo timestep

        if self.chunk_ep_length > 0:
            # each env idx has a different init timestep
            for i in range(self.num_envs):
                self.episode_start_buf[i] = i % self.chunk_ep_length
                self.episode_length_buf[i] = i % self.chunk_ep_length

        self.max_achieved_length = 0
        self.actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)

        # approximate dist from hand kpt to object surface
        num_obj_parts = 2 
        if self.observe_tip_dist:
            self.kpt_dists_left = torch.zeros((self.num_envs, self.robots['left'].n_kpts, num_obj_parts), device=self.device)
            self.kpt_dists_right = torch.zeros((self.num_envs, self.robots['right'].n_kpts, num_obj_parts), device=self.device)

        if self.observe_contact_force:
            self.contact_forces = torch.zeros((self.num_envs, self.num_obj_links,  self.num_robot_links, 3), device=self.device) 

        if self.use_contact_reward:
            self.contact_link_pos = torch.zeros((self.num_envs, self.num_obj_links, self.num_robot_links, 3), device=self.device)
            self.contact_link_valid = torch.zeros((self.num_envs, self.num_obj_links, self.num_robot_links), device=self.device, dtype=torch.bool)
        self.extras = dict() 
    # def progress_episode_length(self):
    #     self.episode_length_buf += 1
    #     for k, robot in self.robots.items():
    #         robot.episode_length_buf += 1
    #     for k, obj in self.objects.items():
    #         obj.episode_length_buf += 1
    def set_retarget_states(self, step, env_idxs=None):
        if env_idxs is None:
            env_idxs = list(range(self.num_envs))
        """directly set the state of the object and robots from retargeting data, use for visualization"""
        for k, robot in self.robots.items():
            if robot.residual_qpos is not None:
                robot.set_joint_position(
                    joint_targets=robot.residual_qpos[step][None].repeat(len(env_idxs), 1),
                    env_idxs=env_idxs,
                )

        if self.n_objects == 1:
            if self.object.demo_dofs is not None:
                targets = self.object.demo_dofs[step][None].repeat(len(env_idxs), 1) 
                self.object.entity.set_dofs_position(targets) 
        self.scene.step()
        self.episode_length_buf += 1 
        self._compute_intermediate_values()
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones() 
        self.reset_buf[:] = self.reset_terminated | self.reset_time_outs
        self._get_rewards()
        if self.record_video:
            self._render_headless()
        return  
    
    def pre_scene_step(self, actions: torch.Tensor):
        """ call this for only stepping the robot/objects"""
        self.last_actions[:] = self.actions
        self.actions[:] = self._prepare_actions(actions)
        
        for k, robot in self.robots.items():
            idxs = self.action_idxs_to_robot[k]
            robot.step(self.actions[:, idxs], self._step_env_idxs)
        for k, obj in self.objects.items():
            obj.step() 
            
    def step(self, actions: torch.Tensor):
        """
        actions: torch.Tensor of shape (num_envs, action_dim)
        """
        assert actions.shape[1] == self.action_dim
        assert actions.shape[0] == self.num_envs
        self.last_actions[:] = self.actions
        self.actions[:] = self._prepare_actions(actions)
        
        for k, robot in self.robots.items():
            idxs = self.action_idxs_to_robot[k]
            robot.step(self.actions[:, idxs], self._step_env_idxs)
        for k, obj in self.objects.items():
            obj.step()
            
        self.randomization.on_step(self.episode_length_buf)
        self.scene.step()  
        if self.learner_cfg:
            self.environment_transitions += self.num_envs
        self.episode_length_buf += 1
        # self.progress_episode_length() 
        self._compute_intermediate_values()
        
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones() 
        self.reset_buf[:] = self.reset_terminated | self.reset_time_outs
 
        rew_dict = self._get_rewards()
        if self.keyframes is not None:
            self.keyframes.capture(rew_dict['task_rew'])
        if "log" not in self.extras:
            self.extras["log"] = dict() 

        # maniptrans curriculum checks additional early resets
        if isinstance(self.curriculum, ManipTransCurriculum) and self.n_objects == 1 and "keypoint_dist" in rew_dict:
            # get the object position and rotation error from rew_dict   
            curriculm_reset = self.curriculum.determine_early_term(
                obj_pos_err=rew_dict['pos_dist'],
                obj_rot_err=rew_dict['rot_dist'],
                finger_pos_err=rew_dict['keypoint_dist'],
            ) 
            self.reset_buf[:] = self.reset_buf[:] | curriculm_reset

        if self.is_eval and self.ablation_protocol in ('v2', 'dexmachina') and self.n_objects == 1:
            # Capture the physical post-step state before automatic episode
            # resets replace terminal poses. Preserve the evaluator's existing
            # position, wxyz quaternion, and articulation coordinate convention.
            eval_object = self.objects[self.object_names[0]]
            self.extras['eval_object_state_before_reset'] = torch.cat(
                (eval_object.root_pos, eval_object.root_quat, eval_object.dof_pos), dim=-1,
            ).detach().clone()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # only log cum. episode reward if that env_idx is DONE 
            rew_dict['episode_rew'] = self.cumulative_task_rew[reset_env_ids] 
            self.reset_idx(reset_env_ids)  # rew and obs will be resetted   
        
        self.extras["log"].update(rew_dict) 
        if self.use_curriculum:
            rew_grads = self.curriculum.get_reward_grads()
            self.extras["log"].update(
                {f"grad/{k}": v for k, v in rew_grads.items()}
            )
            self.extras["log"].update(
                self.curriculum.get_current_gains()
                )
        # log contact forces
        if self.observe_contact_force:
            self.extras["log"]["contact_force"] = torch.norm(self.contact_forces, dim=-1).max().item()
        
        # get control_force 
        for side in ['left', 'right']:
            robot = self.robots[side]
            control_force = robot.get_control_force()
            self.extras["log"][f"{side}_control_force"] = control_force.mean().item()

        if self.record_video:
            self._render_headless()
        
        # update obs after potential reset_idx: 
        if self.use_rl_games:
            observations = self.get_observations()
            self.obs_buf[:] = observations['policy']
            return observations, self.rew_buf, self.reset_terminated, self.reset_time_outs, self.extras 
        else:
            self.obs_buf[:] = self.get_observations() # if resetted, the obs should be the initial ones
        self.obs_sum[:] += torch.abs(self.obs_buf)  
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _prepare_actions(self, actions: torch.Tensor):
        return (
            torch.clamp(actions, -self.action_clip, self.action_clip)
            * self.action_scale
        )
    
    def setup_ho_contact(self):
        """Hand-object contact reward: surface points of each hand's region links (link frames) and each region's contact links;
        the regions (palm and the hand's fingers) come from the finger spec."""
        from dexmachina.envs.ho_contact import PARTS, robot_anchor_points
        if tuple(self.object.link_names) != PARTS:
            raise ValueError(f"Hand-object contact expects object links {PARTS}, got {self.object.link_names}")
        hand = self.reward_module.hand_name
        self.ho_contact_points = dict()
        for side, robot in self.robots.items():
            spec = robot_anchor_points(str(robot.cfg['urdf_path']), hand, side,
                                       regions=self.reward_module.finger_specs[side].contact_regions())
            names = [link.name for link in robot.entity.links]
            coll = list(robot.coll_link_names)
            self.ho_contact_points[side] = dict(
                link_idx=torch.tensor([names.index(n) for n in spec['links']], dtype=torch.long, device=self.device),
                ranges=spec['link_ranges'], local=torch.tensor(spec['points'], dtype=torch.float32, device=self.device))
            self.reward_module.setup_ho_contact(side, spec['slices'], [[coll.index(n) for n in links] for links in spec['anchor_links']])

    def get_ho_contact_inputs(self):
        """World positions of each hand's anchor points (N, P, 3) and the object part poses."""
        from dexmachina.envs.math_utils import matrix_from_quat
        out = dict()
        for side, robot in self.robots.items():
            spec = self.ho_contact_points[side]
            pos = robot.entity.get_links_pos()[:, spec['link_idx']]
            quat = robot.entity.get_links_quat()[:, spec['link_idx']]
            rot = matrix_from_quat(quat.reshape(-1, 4)).view(quat.shape[0], quat.shape[1], 3, 3)
            out[f'points_{side}'] = torch.cat([torch.einsum('nij,pj->npi', rot[:, k], spec['local'][s:e]) + pos[:, k, None]
                                               for k, (s, e) in enumerate(spec['ranges'])], dim=1)
        out['part_pos'] = self.object.part_pos
        out['part_quat'] = self.object.part_quat
        return out

    def get_hand_graph_extra(self, side):
        """Positions (N, E, 3) of the hand-graph nodes of one hand that are not keypoints, in RewardModule.hand_graph_extra
        order (a wrist outside the keypoints first, from wrist_pose; then the robot's extra buffer)."""
        names = self.reward_module.hand_graph_extra.get(side, [])
        if not names:
            return None
        robot = self.robots[side]
        parts = []
        if names[0] == robot.wrist_link_name:
            parts.append(robot.wrist_pose[:, None, :3])
        if robot.extra_link_names:
            parts.append(robot.extra_kpt_pos)
        return torch.cat(parts, dim=1)

    def _get_rewards(self):
        if self.n_objects == 1:
            obj = self.objects[self.object_names[0]]
            obj_pos, obj_quat, obj_arti = obj.root_pos, obj.root_quat, obj.dof_pos
        else:
            obj_pos, obj_quat, obj_arti = None, None, None
        bc_dist = torch.cat([robot.get_bc_dist() for robot in self.robots.values()], dim=-1)
        reward_kwargs = dict(
            actions=self.actions,
            bc_dist=bc_dist,
            obj_pos=obj_pos,
            obj_quat=obj_quat,
            obj_arti=obj_arti,
            kpts_left=self.robots['left'].kpt_pos,
            kpts_right=self.robots['right'].kpt_pos,
            episode_length_buf=self.episode_length_buf,
            contact_link_pos_left=None,
            contact_link_valid_left=None,
            contact_link_pos_right=None,
            contact_link_valid_right=None,
            wrist_pose_left=self.robots['left'].wrist_pose,
            wrist_pose_right=self.robots['right'].wrist_pose,
            contact_forces=None,
            )
        if self.use_contact_reward:
            reward_kwargs.update(
                contact_link_pos_left=self.contact_link_pos[:, :, :self.num_left_contact_links], # N, 2, 13, 3
                contact_link_valid_left=self.contact_link_valid[:, :, :self.num_left_contact_links],
                contact_link_pos_right=self.contact_link_pos[:, :, self.num_left_contact_links:],
                contact_link_valid_right=self.contact_link_valid[:, :, self.num_left_contact_links:],
            )
        if self.observe_contact_force:
            reward_kwargs.update(
                contact_forces=self.contact_forces
            )
        if self.ho_contact_points is not None:
            reward_kwargs.update(ho_contact_inputs=self.get_ho_contact_inputs())
        if self.reward_module.hand_graph_weight > 0.0 and any(self.reward_module.hand_graph_extra.values()):
            reward_kwargs.update(hand_graph_extra_left=self.get_hand_graph_extra('left'),
                                 hand_graph_extra_right=self.get_hand_graph_extra('right'))
        rewards, rew_dict = self.reward_module.compute_reward(
            **reward_kwargs
        )

        if not self.use_rl_games:
            # scale the reward by 0.1 manually to match the scale in rl_games
            rewards *= 0.1

        self.rew_dict = rew_dict 
        self.rew_buf[:] = rewards
        # there's potentially nan values 
        self.rew_buf[self.nan_envs] = -1.0
        task_rewards = rew_dict['task_rew'] # use task_rew for reset
        task_rewards[self.nan_envs] = -1.0
        rew_dict['task_rew'] = task_rewards
        self.cumulative_task_rew[:] += task_rewards if self.n_objects == 1 else rewards 
        
        if 'con_rew' in rew_dict:
            con_rew = rew_dict['con_rew']
            con_rew[self.nan_envs] = -1.0
            self.cumulative_con_rew[:] += con_rew
        if 'imi_rew' in rew_dict:
            imi_rew = rew_dict['imi_rew']
            imi_rew[self.nan_envs] = -1.0
            self.cumulative_imi_rew[:] += imi_rew
        if 'bc_rew' in rew_dict:
            bc_rew = rew_dict['bc_rew']
            bc_rew[self.nan_envs] = -1.0
            self.cumulative_bc_rew[:] += bc_rew
        # log-only terms computed from NaN keypoints would turn their logged means into NaN
        for key, val in rew_dict.items():
            if key.startswith(('hand_graph_', 'ho_')) and torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == self.num_envs:
                val[self.nan_envs] = 0.0
        return rew_dict

    def _get_dones(self):
        stepped_length = self.episode_length_buf - self.episode_start_buf
        if self.chunk_ep_length > 0:
            timeout = stepped_length >= self.chunk_ep_length
        else:
            timeout = self.episode_length_buf >= self.max_episode_length  # returns true for end of episode
 
        timeout = timeout | self.nan_envs
        if self.is_eval:
            return timeout, timeout # always let it run to last frame for eval
        
        object_fell_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        first_obj = None
        if self.n_objects == 1:
            first_obj = self.objects[self.object_names[0]]
            object_fell_off = first_obj.root_pos[:, 2] < self.table_height
        
        need_reset = timeout | object_fell_off | self.nan_envs
        
        if self.early_reset_threshold > 0.0:
            # early reset curriculum  
            for interval in range(0, self.max_episode_length, self.early_reset_interval):
                need_reset = torch.where(
                    (stepped_length > interval),
                    self.cumulative_task_rew + self.early_reset_bank < self.early_reset_threshold * interval,
                    need_reset
                ) 
        
        for key, cum_rew in zip(
            ['con', 'imi', 'bc'],
            [self.cumulative_con_rew, self.cumulative_imi_rew, self.cumulative_bc_rew]
        ):  
            thres = self.early_reset_aux_thres.get(key, 0.0)
            if thres > 0.0:
                for interval in range(0, self.max_episode_length, 20):   
                    need_reset = torch.where(
                        (stepped_length > interval), cum_rew < thres * interval, need_reset
                    )
        return need_reset, timeout
    
    def prepare_sliced_contact(self, source='policy', part='top', side='left'):
        """ this does not have to always return the same shape """
        if source == 'policy' and self.use_contact_reward:
            _pos = self.contact_link_pos 
            # set the invalid positions all to 0
            _pos[~self.contact_link_valid] = 0
            part_idx = 0 if part == 'top' else 1
            _pos = _pos[:, part_idx, :]
            if side == 'left':
                _pos = _pos[:, :self.num_left_contact_links]    
            else:
                _pos = _pos[:, self.num_left_contact_links:]

        else:        
            _pos = self.reward_module.match_demo_state(f'contact_links_{side}', self.episode_length_buf) # (N, 2*nlinks, 4) -> last dim is part id
            part_id = 2 if part == 'bottom' else 1
            if len(_pos.shape) == 4:
               # for retargeted contact, the shape is (N, 2, nlinks, 4) 
               _pos = _pos.reshape(self.num_envs, -1, 4)
            # only take positions that has the correct part id
            demo_part_valid = (_pos[:, :, -1] == part_id)# skip valid mask& demo_valid_contact
            # set invalid positions to 0
            _pos[~demo_part_valid] = 0
            _pos = _pos[:, :, :3]
        return _pos 
        
    def _compute_intermediate_values(self):
        for name, robot in self.robots.items():
            robot.update_value_buffers()
            self.nan_envs[:] = self.nan_envs | robot.get_nan_envs()
        for name, obj in self.objects.items():
            obj.update_value_buffers() 
            self.nan_envs[:] = self.nan_envs | obj.get_nan_envs()

        if self.observe_contact_force or self.use_contact_reward:
            entity_a = self.object.entity
            contact_info = get_filtered_contacts(
                    entity_a=entity_a, 
                    entity_b=None,
                    filter_links_a=self.filter_links_a,
                    filter_links_b=self.filter_links_b,
                    return_link_force=self.observe_contact_force,
                    return_link_pos=self.use_contact_reward,
                    device=self.device ,
                )
                
            if self.observe_contact_force:
                contact_force = contact_info["contact_force_link_a"] # shape (n_env, n_obj_links, n_robot_links, 3)
                self.contact_forces[:] = contact_force
            if self.use_contact_reward and self.num_obj_links > 0:
                self.contact_link_pos[:] = contact_info['contact_pos_link_a']
                self.contact_link_valid[:] = contact_info['contact_pos_link_a_valid']
                # vis left hand contact for now:
            
        contact_dict = dict()
        for key in self.contact_markers.keys():
            source, part, side = key.split('_')
            contact_dict[key] = self.prepare_sliced_contact(source, part, side) 
        self.update_contact_markers(contact_dict)
            
    def get_observations(self):
        value_list = []
        all_obs_dict = dict()
        for name, robot in self.robots.items():
            obs_dict = robot.get_observations()
            value_list.extend(list(obs_dict.values()))
            all_obs_dict[name] = obs_dict
        for name, obj in self.objects.items():
            obs_dict = obj.get_observations()
            value_list.extend(list(obs_dict.values()))
            all_obs_dict[name] = obs_dict
        left = self.robots['left']
        right = self.robots['right'] 
        # contact_info = left.entity.get_contacts(obj.entity)
        # force, mask = contact_info['force_a'], contact_info['valid_mask']
        # print(force[mask].shape)
        if self.chunk_ep_length > 0:
            normalize_ep_len = 2.0 * self.episode_length_buf[:, None].float() / self.demo_length - 1.0
        else:
            normalize_ep_len = 2.0 * self.episode_length_buf[:, None].float() / self.max_episode_length - 1.0
        value_list.append(normalize_ep_len)

        # value_list = []
        # value_list.append(normalize_ep_len.repeat(1, 20))

        if self.observe_tip_dist:
            assert self.n_objects == 1, "Only support one object for now"
            obj = self.objects[self.object_names[0]]
            # compute the kpt distances to the object surface
            for side, dists_tensor in zip(['left', 'right'], [self.kpt_dists_left, self.kpt_dists_right]):
                robot = self.robots[side]
                for i, part in enumerate(['top', 'bottom']):
                    part_pose = obj.get_part_pose(part)
                    dists_tensor[:, :, i] = self.compute_closest_vertice_dist_single(
                        self.obj_verts[part], robot.kpt_pos, part_pose
                    )
                    name = f"{side}_kpt_dist_{part}"
                    # print(name, np.round(dists_tensor[:, :, i].cpu().numpy(), 2))
            value_list.extend([
                self.kpt_dists_left.flatten(start_dim=1),
                self.kpt_dists_right.flatten(start_dim=1),
                ])

        if self.observe_contact_force:
            force_norm = torch.norm(self.contact_forces, dim=-1) * 0.01 # scale down! max contact force can go to 1000+
            value_list.append(force_norm.flatten(start_dim=1))

        if self.coordinate_actor_cfg.get('enabled', False):
            # Both hands are always active and the message gate is open; the channels keep the actor's input layout.
            active_mask = torch.ones(
                (self.num_envs, 2), dtype=torch.float32, device=self.device
            )
            message_gate = torch.ones(
                (self.num_envs, 1), dtype=torch.float32, device=self.device
            )
            presence_mask = torch.ones_like(active_mask)
            object_rotation_inv = matrix_from_quat(
                self.object.root_quat
            ).transpose(1, 2)
            partner_states = []
            for name in self.robot_names:
                robot = self.robots[name]
                wrist_position_object = torch.einsum(
                    'nij,nj->ni',
                    object_rotation_inv,
                    robot.wrist_pose[:, :3] - self.object.root_pos,
                )
                wrist_rotation_object = quat_mul(
                    quat_conjugate(self.object.root_quat),
                    robot.wrist_pose[:, 3:7],
                )
                robot_obs = all_obs_dict[name]
                partner_states.append(torch.cat((
                    wrist_position_object,
                    wrist_rotation_object,
                    robot_obs['dof_pos'],
                    robot_obs['dof_vel'],
                ), dim=-1))
            value_list.extend((
                torch.cat(partner_states, dim=-1),
                self.last_actions,
                active_mask,
                presence_mask,
                message_gate,
            ))

        obs = torch.cat(value_list, dim=-1)
        self.obs_dict = all_obs_dict
        # if sum(torch.isnan(obs).flatten()) > 0:
        #     print("NAN OBSERVATIONS")
        #     breakpoint()
        nan_mask = torch.isnan(obs) 
        obs[nan_mask] = -self.obs_clip
        # if there's nan, need immediately reset 
        all_obs_dict['nan_mask'] = nan_mask
        obs = torch.clamp(obs, -self.obs_clip, self.obs_clip)

        if self.use_rl_games:
            return dict(policy=obs, itemized=all_obs_dict, critic=obs) # critic for sil
        return obs 
    
    def get_privileged_observations(self): 
        return None

    def normalize_episode_rew(self, rewards: torch.Tensor):
        # rewards should be shape (num_envs,)
        avg_rew = rewards / self.max_episode_length
        return avg_rew.mean().item()

    def reset_idx(self, env_idxs=[]):
        if len(env_idxs) == 0:
            return  
        self.randomization.on_reset_idx(env_idxs)
        progressed = self.episode_length_buf[env_idxs] - self.episode_start_buf[env_idxs]
        progressed_avg = torch.mean(progressed.float()).item()
        self.max_achieved_length = int(self.max_achieved_length * 0.5 + progressed_avg * 0.5)

        rsi_curriculum = self.use_curriculum and getattr(self.curriculum, "schedule", None) == "rsi"
        if rsi_curriculum:
            # Outcomes of the finishing episodes, read before the reset flags are overwritten.
            rsi_ended = self.reset_terminated[env_idxs] | self.reset_time_outs[env_idxs]
            rsi_failed = (self.reset_terminated[env_idxs] & ~self.reset_time_outs[env_idxs]) | self.nan_envs[env_idxs]

        if self.use_curriculum:
            episode_rewards = dict()
            if 'task' in self.reward_keys:
                episode_rewards['task'] = self.normalize_episode_rew(self.cumulative_task_rew[env_idxs])
            if 'con' in self.reward_keys:
                episode_rewards['con'] = self.normalize_episode_rew(self.cumulative_con_rew[env_idxs])
            if 'imi' in self.reward_keys:
                episode_rewards['imi'] = self.normalize_episode_rew(self.cumulative_imi_rew[env_idxs])
            if 'bc' in self.reward_keys:
                episode_rewards['bc'] = self.normalize_episode_rew(self.cumulative_bc_rew[env_idxs])
            self.curriculum.update_progress(episode_rewards, self.max_achieved_length)

        if self.chunk_ep_length > 0:
            self.episode_start_buf[env_idxs] = (env_idxs % self.chunk_ep_length).to(torch.int32).to(self.device)
            self.episode_length_buf[env_idxs] = self.episode_start_buf[env_idxs]
        else:
            self.episode_length_buf[env_idxs] = 0
            self.episode_start_buf[env_idxs] = 0 
        
        if self.rand_init_ratio > 0.0:
            # randomly sample non-zero initial t 
            # num_rand = int(self.rand_init_ratio * len(env_idxs)) + 1
            # treat this as probability 
            torand = torch.rand(len(env_idxs)) <= self.rand_init_ratio
            # randomly sample from any t within max_episode_length
            end_t = min(self.max_achieved_length + 1, self.max_episode_length - 1)
            rand_t = torch.randint(0, end_t, (len(env_idxs),), dtype=torch.int32, device=self.device)
            ep_starts = torch.zeros(len(env_idxs), dtype=torch.int32, device=self.device)
            ep_starts[torand] = rand_t[torand] 
            self.episode_length_buf[env_idxs] = ep_starts
            self.episode_start_buf[env_idxs] = ep_starts
            
        if rsi_curriculum:
            # Adaptive reference-state starts: sample each lane's next start frame and set
            # its bin's assistance before the robot and object states are written below.
            starts = self.curriculum.rsi_on_reset(env_idxs, rsi_ended, rsi_failed)
            kf_plan = None
            if self.keyframes is not None:
                kf_bins = self.curriculum.rsi_env_bin[torch.as_tensor(env_idxs).long().cpu()]
                kf_have, kf_frames, kf_bins, kf_slot = self.keyframes.sample(kf_bins)
                starts = torch.where(kf_have.cpu(), kf_frames.cpu(), starts)
                kf_plan = (kf_have, kf_bins, kf_slot)
                self.keyframes.mark_lineage(env_idxs, starts, kf_have)
            starts = starts.to(device=self.episode_length_buf.device, dtype=self.episode_length_buf.dtype)
            self.episode_length_buf[env_idxs] = starts
            self.episode_start_buf[env_idxs] = starts
            self.early_reset_bank[env_idxs] = torch.where(
                starts > 0, torch.full_like(starts, 0, dtype=torch.float32) + float(self.curriculum.rsi_grace_bank),
                torch.zeros_like(starts, dtype=torch.float32))

        for k, robot in self.robots.items():
            # need to rand init too
            robot.reset_idx(env_idxs, self.episode_start_buf[env_idxs])
        
        for k, obj in self.objects.items():
            obj.reset_idx(env_idxs, self.episode_start_buf[env_idxs])

        if rsi_curriculum and kf_plan is not None and bool(kf_plan[0].any()):
            # Keyframe lanes: replace the reference pose written above with a reached simulator state.
            kf_have, kf_bins, kf_slot = kf_plan
            kf_idx = torch.as_tensor(env_idxs, device=self.device).long()[kf_have]
            self.keyframes.restore(kf_idx, kf_bins[kf_have], kf_slot[kf_have])

        if self.env_cfg.get("refresh_after_reset", False):
            # DexMachina's robot/object reset_idx leave stale cached state: object.reset_idx zeroes the part poses and
            # robot.reset_idx reads the wrist from the keypoint array with a link index. Re-read the simulator so the first
            # observation of every episode matches its (correct) physical start state, in training as in evaluation.
            for robot in self.robots.values():
                robot.update_value_buffers()
            for obj in self.objects.values():
                obj.update_value_buffers()

        self.last_actions[env_idxs] = 0.0
        
        self.reset_buf[env_idxs] = True
        self.reset_terminated[env_idxs] = True
        self.reset_time_outs[env_idxs] = True

        # self.rew_buf[env_idxs] = 0.0
        self.cumulative_task_rew[env_idxs] = 0.0
        if not rsi_curriculum:
            self.early_reset_bank[env_idxs] = 0.0
        self.cumulative_con_rew[env_idxs] = 0.0
        self.cumulative_imi_rew[env_idxs] = 0.0
        self.cumulative_bc_rew[env_idxs] = 0.0
        # self.obs_buf[env_idxs] = 0.0
        self.obs_sum[env_idxs] = 0.0   

        # self._compute_intermediate_values()
        if self.observe_tip_dist:
            self.kpt_dists_left[env_idxs] = 0.0
            self.kpt_dists_right[env_idxs] = 0.0
        
        if self.observe_contact_force:
            self.contact_forces[env_idxs] = 0.0
        
        if self.use_contact_reward:
            self.contact_link_pos[env_idxs] = 0.0
            self.contact_link_valid[env_idxs] = False  
         
    def reset(self): 
        # reset all envs
        env_idxs = torch.arange(self.num_envs)
        self.reset_idx(env_idxs)  
        if self.use_rl_games:
            return self.get_observations(), dict()
        else:
            self.obs_buf[:] = self.get_observations()
        self.scene.step()
        return self.obs_buf, None #self.extras

    def transform_vertice_frame(self, vertices, pose):
        """ transform the vertices to the object frame """
        # vertices: (N, K, 3), pose: (N, 7)
        quat = pose[:, 3:7]
        matrices = matrix_from_quat(quat)
        offsets = pose[:, :3].unsqueeze(1)
        transformed = torch.einsum("nij,nkj->nki", matrices, vertices) + offsets
        return transformed
        
    def compute_closest_vertice_dist_single(self, verts, keypoint_pos, pose):
        # verts: (N, K, 3), keypoint_pos: (N, 3), pose: (N, 7)
        verts = verts.unsqueeze(0).repeat((keypoint_pos.shape[0], 1, 1)).to(pose.device)
        verts = self.transform_vertice_frame(verts, pose)
        dists = torch.cdist(keypoint_pos, verts, p=2)
        min_dists = torch.min(dists, dim=-1).values
        return min_dists
        
    def _add_camera(self, camera_kwargs):
        ''' Set camera position and direction, NOTE this must be done BEFORE scene.build()''' 
        cameras = dict()
        for name, kwargs in camera_kwargs.items():
            cam_pos = kwargs.get('pos', (0.0, -1.5, 1.2))
            lookat = kwargs.get('lookat', CARDBOARD_POS)
            res = kwargs.get('res', CAMERA_RES)
            if self.num_envs < 3:
                res = (500, 500)
            fov = kwargs.get('fov', 30)
            if 'renderer' in self.scene_cfg: # ray tracing cam
                res = kwargs.get('raytrace_res', (1024, 1024))
                fov = kwargs.get('raytrace_fov', 18) 
            GUI = kwargs.get('GUI', False) 
            cameras[name] = self.scene.add_camera(
                pos=cam_pos,
                lookat=lookat,
                res=res,
                fov=fov,
                GUI=GUI,
            ) 
        self._floating_camera = cameras.get(self.render_camera, None)
        self.cameras = cameras 
        self._recording = False
        self._recorded_frames = [] 
    
    def _set_camera(self, pos=None, lookat=None, fov=None, name='front'):
        if self.cameras.get(name, None) is None:
            print("Camera not initialized")
            return
        camera = self.cameras[name]
        if pos is not None or lookat is not None:
            camera.set_pose(pos=pos, lookat=lookat)
        if fov is not None and hasattr(camera, "set_params"):
            camera.set_params(fov=fov)
        return

    def start_recording(self):
        self._recorded_frames = []
        if self.record_video:
            self._recording = True

    def _render_headless(self): 
        if self._recording and len(self._recorded_frames) < self.max_video_frames and self.record_video:
            # obj_pos = self.objects[self.object_names[0]].root_pos.cpu().numpy()[-1] 
            # import time
            # start = time.time()
            frame, depth_arr, seg_arr, normal_arr = self._floating_camera.render(segmentation=self.render_segmentation)
            if self.render_segmentation: 
                frame = np.concatenate(
                    [frame, seg_arr[:,:,None]], axis=-1
                )
            # end = time.time()
            # print(end-start)
            self._recorded_frames.append(frame)

    def get_recorded_frames(self, wait_for_max=True): 
        """ Stops recording if frames were yielded """
        if len(self._recorded_frames) == 0: 
            return None
        if wait_for_max and len(self._recorded_frames) < self.max_video_frames:
            return None
        frames = self._recorded_frames
        self._recorded_frames = []
        self._recording = False
        return frames
    
    def export_video(self, path, wait_for_max=True):
        if len(self._recorded_frames) == 0:
            return 
        frames = self.get_recorded_frames(wait_for_max=wait_for_max)
        if frames is not None:
            rgb_frames = [frame[:,:,:3] for frame in frames]
            if frames[0].shape[-1] == 4: # fill background with white
                mask_frames = [frame[:,:,3:] for frame in frames]
                # use mask frames to fill in white background 
                rgb_frames = np.array(rgb_frames)
                mask_frames = np.array(mask_frames)
                mask_frames = np.repeat(mask_frames, 3, axis=-1) 
                # max_id = mask_frames.max()
                ground_id = self.ground.idx
                rgb_frames[mask_frames == ground_id] = 255
                rgb_frames = np.clip(rgb_frames, 0, 255).astype(np.uint8)                 
                rgb_frames = [rgb_frames[i] for i in range(len(rgb_frames))]
            path = path + ".mp4" if not path.endswith(".mp4") else path
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip(rgb_frames, fps=int(1/self.dt/2))
            clip.write_videofile(path) 
        return frames
    
    def randomize(self, env_idxs=None):
        if not self.rand_cfg.get('randomize', False):
            return
        if self.rand_cfg.get('friction', False):
            friction_range = self.rand_cfg['friction']
            self._randomize_link_friction(friction_range=friction_range, env_idxs=env_idxs)
        
        if self.rand_cfg.get('com', False):
            com_range = self.rand_cfg['com']
            self._randomize_com_displacement(com_range=com_range, env_idxs=env_idxs)

        if self.rand_cfg.get('mass', False):
            mass_range = self.rand_cfg['mass']
            self._randomize_mass(mass_range=mass_range, env_idxs=env_idxs)

    def set_curriculum(self, epoch_num):
        """Native protocols use the upcoming epoch; v2 uses completed updates."""
        self.epoch_num = epoch_num 
        verbose = epoch_num % 250 == 0
        reset_reward_tracker = False
        if self.use_curriculum:
            zero_gains, gains_decayed, reason = self.curriculum.set_curriculum(epoch_num)
            if verbose: 
                print(reason) 
            rew_weights_decayed = False
            if zero_gains and self.ablation_protocol != 'v2':
                rew_weights_decayed = self.curriculum.decay_reward_weights(self.reward_module)
            if self.ablation_protocol != 'v2':
                reset_reward_tracker = (rew_weights_decayed or gains_decayed)
        return reset_reward_tracker

