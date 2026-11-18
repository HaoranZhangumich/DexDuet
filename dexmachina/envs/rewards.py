import os
import torch
import numpy as np
from os.path import join
from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.reward_utils import position_distance, rotation_distance, chamfer_distance, transform_contact
from dexmachina.envs.math_utils import matrix_from_quat

# Hand graph (--hand_graph_weight): nodes, finger edges, fingertip gaps and the palm frame come per hand and side from the
# retargeting's finger correspondence (dexmachina/envs/finger_spec.py). Per present finger: an edge from its base to its tip
# (MANO bases 13/1/4/10/7, tips 16-20); the palm is described by its normal, (index base - wrist) x (outermost finger base -
# wrist), and its forward direction, wrist to middle base (to the midpoint of the index and middle bases when no finger lies
# beyond the middle): together they fix the hand's rotation, including the roll about the fingers' own axis, which the finger
# edges alone barely see. Fingers the robot lacks are left out of the edges, the gaps and their normalisations.
# On hands with fewer than four fingers (finger_spec.palm_normal_mode 'calibrated_axis', e.g. Dex3) the palm normal is the
# human (index base - wrist) x (little base - wrist) against R_wrist(t) a on the robot, where a is a fixed axis of the wrist
# link calibrated once per clip over the retargeted reference (calibrate_palm_axis) and stored in reward_cfg
# ['hand_graph_palm_axis'], so a run's env.pkl carries it to evaluation.
# The five-finger Inspire tables below are what the Inspire spec resolves to (tests/test_finger_spec.py checks it).
MANO_TIP_JOINTS = (16, 17, 18, 19, 20)
FINGER_PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4))
HAND_GRAPH_NODES = {'wrist': 0, 'TH_L1': 13, 'FF_L1': 1, 'MF_L1': 4, 'RF_L1': 10, 'LF_L1': 7,
                    'TH_Tip': 16, 'FF_Tip': 17, 'MF_Tip': 18, 'RF_Tip': 19, 'LF_Tip': 20}
HAND_GRAPH_FINGERS = (('TH_L1', 'TH_Tip'), ('FF_L1', 'FF_Tip'), ('MF_L1', 'MF_Tip'), ('RF_L1', 'RF_Tip'), ('LF_L1', 'LF_Tip'))
# a calibrated palm axis below this mean resultant length means the human normal does not keep a direction in the wrist
# frame (e.g. a mirrored or broken reference); the axis would then be meaningless
MIN_PALM_AXIS_RESULTANT = 0.5


def calibrate_palm_axis(wrist_quat, human_normal):
    """The palm-normal axis of a 'calibrated_axis' hand (dexmachina/envs/finger_spec.py): the fixed unit axis a of the wrist
    link (link frame) whose world direction R_wrist(t) a best matches the human palm normal n(t) over a reference clip,
    a = normalize(sum_t R_wrist(t)^T n(t)), which maximises sum_t cos(R_wrist(t) a, n(t)).
    wrist_quat (T, 4): wrist-link orientations, (w, x, y, z) as robot.wrist_pose[:, 3:] and the retargeted wrist_pose[:, 3:];
    human_normal (T, 3): unit human palm normals in the same world frame.
    Float64 on the CPU with plain torch ops (no TorchScript, no randomness): the result does not depend on the device or on
    what ran before. Returns (a (3,) float64, the mean resultant length |mean_t R^T n| in [0, 1], 1 when the human normal is
    fixed in the wrist frame)."""
    q = torch.as_tensor(wrist_quat).detach().to(device='cpu', dtype=torch.float64)
    n = torch.as_tensor(human_normal).detach().to(device='cpu', dtype=torch.float64)
    if q.dim() != 2 or q.shape[1] != 4 or n.shape != (q.shape[0], 3) or q.shape[0] == 0:
        raise ValueError(f"Palm-axis calibration needs (T, 4) wrist quaternions and (T, 3) normals, got {tuple(q.shape)} "
                         f"and {tuple(n.shape)}")
    if not (torch.isfinite(q).all() and torch.isfinite(n).all()):
        raise ValueError("Palm-axis calibration: non-finite wrist quaternions or human palm normals")
    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, u = q[:, :1], -q[:, 1:]                             # the conjugate quaternion rotates by R^T
    t = 2.0 * torch.cross(u, n, dim=-1)
    local = n + w * t + torch.cross(u, t, dim=-1)          # R(t)^T n(t), the human normal in the wrist-link frame
    mean = local.mean(dim=0)
    length = torch.norm(mean)
    if not length > 0:
        raise ValueError("Palm-axis calibration: the human palm normals cancel out in the wrist frame")
    return mean / length, float(length)


def get_reward_cfg(last_n_frame=-1):
    reward_cfg = {
        "obj_pos_beta": 20.0,
        "obj_rot_beta": 5.0,
        "obj_arti_beta": 20.0,
        "obj_pos_weight": 2.0,
        "obj_rot_weight": 3.0,
        "obj_arti_weight": 5.0,  
        
        "last_n_frame": last_n_frame,
        "multiply_task_rew": True,
        "multiply_all_rew": False, 
        "task_rew_weight": 1.0,

        "exp_kpt_first": True,
        "imi_rew_weight":  0.0, 
        "imi_wrist_weight": 0.0, # do a weighted avg between fingertip and wrist poses
        "imi_wrist_rot_beta": 3.0,
        "imi_wrist_pos_beta": 10.0,
        "imi_fingertip_beta": 20.0,

        "bc_rew_weight": 0.0,
        "bc_beta": 500.0,

        "contact_rew_weight": 0.0,
        "contact_rew_function": "exp", # exp or sigmoid
        "wrist_frame_contact": True,
        "contact_beta": 30.0,
        "contact_a": 100.0,
        "contact_b": 5.0,
        "multiply_frame_contact": True,
        "mask_zero_contact": True, # if both policy and demo has no contact, reward is 0 (1 if False)
        "contact_phase_penalty": 0,

        "mask_well_track": False, 
        "scale_well_track": 1.0,
        "force_penalty": 0.1,  # ~60 contact pairs in each env
        "action_penalty": 0.0,
        "objdex_baseline": False,
        "use_retarget_contact": False,
        "ref_motion_info": False,  # report the demo object's per-frame motion (for the reference-aware smoothness gate)
        # grasp-fidelity gate on the object reward (PhysHOI/SkillMimic style): rew = task_rew * gate + hand terms
        "cg_gate": 0.0,            # kappa for the contact graph: gate = (1 - kappa) + kappa * r_CG
        "cg_extra": 0.25,          # weight of contacts the policy makes that the demo does not
        "hand_graph_weight": 0.0,
        "hand_graph_beta_dir": 5.0,   # exp(-beta_dir * dir_err); dir_err = mean of the palm and finger (1 - cos) errors, object frame
        "hand_graph_beta_gap": 20.0,  # exp(-beta_gap * mean |robot gap - scaled human gap|), metres
        "hand_graph_thumb": 0.5,      # weight of the thumb edge relative to the other finger edges
        "hand_graph_scale": None,     # human-to-robot length scale for the fingertip gaps; None: the hand's retargeting scale
        "retarget_objframe": True, # if True, the contact is in the object frame, otherwise in the wrist frame

    } 
    return reward_cfg

class RewardModule:
    def __init__(self, reward_cfg, demo_data, retarget_data, device, finger_specs=None):
        self.cfg = reward_cfg 
        self.last_n_frame = reward_cfg.get("last_n_frame", -1)
        self.exp_kpt_first = reward_cfg.get("exp_kpt_first", False)
        self.imi_rew_weight = reward_cfg["imi_rew_weight"]        
        self.imi_wrist_weight = reward_cfg.get("imi_wrist_weight", 0.0)
        self.task_rew_weight = reward_cfg["task_rew_weight"] 
        self.bc_rew_weight = reward_cfg.get("bc_rew_weight", 0.0)
        self.bc_wrist_weight = reward_cfg.get("bc_wrist_weight", 0.0)  
        self.demo_data = demo_data  
        self.multiply_task_rew = reward_cfg["multiply_task_rew"]
        self.multiply_all_rew = reward_cfg.get("multiply_all_rew", False)
        self.mask_well_track = reward_cfg.get("mask_well_track", False)
        self.scale_well_track = reward_cfg.get("scale_well_track", 1.0)
        self.obj_pos_beta = reward_cfg["obj_pos_beta"]
        self.obj_rot_beta = reward_cfg["obj_rot_beta"]
        self.obj_arti_beta = reward_cfg["obj_arti_beta"]
        if reward_cfg.get("objdex_baseline", False):
            print("Using the lambdas from ObjDex baseline")
            self.obj_pos_beta = 1.0
            self.obj_rot_beta = 20.0
            self.obj_arti_beta = 5.0
            self.imi_rew_weight = 0.0
            self.bc_rew_weight = 0.0
            self.task_rew_weight = 1.0
            self.contact_rew_weight = 0.0
        
        self.use_imi_rew = self.imi_rew_weight > 0.0
        self.use_bc_rew = self.bc_rew_weight > 0.0
        self.obj_pos_weight = reward_cfg["obj_pos_weight"] # not use because we multiply the rewards
        self.obj_rot_weight = reward_cfg["obj_rot_weight"]
        self.obj_arti_weight = reward_cfg["obj_arti_weight"]  

        self.wrist_frame_contact = reward_cfg.get("wrist_frame_contact", False)
        self.use_retarget_contact = reward_cfg.get("use_retarget_contact", False) # if True, the loaded contact is from retargeted hands
        self.retarget_objframe = reward_cfg.get("retarget_objframe", True) # if True, the contact is in the object frame, otherwise in the wrist frame
        self.contact_rew_weight = reward_cfg.get("contact_rew_weight", 0.0)
        self.contact_beta = reward_cfg.get("contact_beta", 10.0)
        self.contact_a = reward_cfg.get("contact_a", 100.0)
        self.contact_b = reward_cfg.get("contact_b", 5.0)
        self.sigmoid_offset = torch.tensor(-1 / (1.0 + np.exp(-self.contact_b)), device=device) + 1.0
        self.contact_phase_penalty = reward_cfg.get("contact_phase_penalty", 0.0)
        self.multiply_frame_contact = reward_cfg.get("multiply_frame_contact", True) 
        self.mask_zero_contact = reward_cfg.get("mask_zero_contact", True)
        self.contact_rew_function = reward_cfg.get("contact_rew_function", "exp")
        self.ref_motion_info = bool(reward_cfg.get("ref_motion_info", False))
        self.cg_gate = float(reward_cfg.get("cg_gate", 0.0))
        self.cg_extra = float(reward_cfg.get("cg_extra", 0.25))
        self.use_grasp_gate = self.cg_gate > 0.0
        if self.use_grasp_gate and not (self.contact_rew_weight > 0.0 and self.use_retarget_contact):
            raise ValueError("The grasp gate needs the retargeted demo contacts (contact_rew_weight > 0 and use_retarget_contact)")
        self.hand_graph_weight = float(reward_cfg.get("hand_graph_weight", 0.0))
        self.hand_graph_beta_dir = float(reward_cfg.get("hand_graph_beta_dir", 5.0))
        self.hand_graph_beta_gap = float(reward_cfg.get("hand_graph_beta_gap", 20.0))
        self.hand_graph_thumb = float(reward_cfg.get("hand_graph_thumb", 0.5))
        # hand-object contact, "where and how far" (dexmachina/envs/ho_contact.py): added on top, DexMachina's contact term unchanged
        self.ho_contact_weight = float(reward_cfg.get("ho_contact_weight", 0.0))
        # finger correspondence per side (dexmachina/envs/finger_spec.py): from the env (its robot cfgs), else from the hand name
        self.hand_name = reward_cfg.get("hand_name") or reward_cfg.get("ho_contact_hand") or "inspire_hand"
        self.finger_specs = dict(finger_specs or {})
        if (self.hand_graph_weight > 0.0 or self.ho_contact_weight > 0.0) and not self.finger_specs:
            from dexmachina.envs.finger_spec import load_finger_spec
            self.finger_specs = {side: load_finger_spec(self.hand_name, side) for side in ('left', 'right')}
        scale = reward_cfg.get("hand_graph_scale", None)
        if scale is None:
            scale = self.finger_specs['left'].scale if self.finger_specs else 1.2
        self.hand_graph_scale = float(scale)
        if (self.hand_graph_weight < 0 or self.hand_graph_beta_dir <= 0 or self.hand_graph_beta_gap <= 0
                or self.hand_graph_thumb < 0 or self.hand_graph_scale <= 0):
            raise ValueError("Hand-graph reward needs weight >= 0, betas > 0, thumb >= 0 and scale > 0")
        self.hand_graph_index = dict()
        self.hand_graph_extra = dict()        # per side: hand-graph nodes that are not robot keypoints (the env passes them)
        self.hand_graph_palm_axis = dict()    # per side: calibrated wrist-link palm axis (3,) of a 'calibrated_axis' hand, else None
        if self.ho_contact_weight < 0:
            raise ValueError("Hand-object contact reward needs weight >= 0")
        self.ho_contact, self.ho_contact_spec = None, dict()
        if self.ho_contact_weight > 0.0:
            from dexmachina.envs.ho_contact import PARTS, HandObjectContact, grid_path
            obj_name = reward_cfg.get("ho_contact_object")
            if obj_name is None:
                raise ValueError("The hand-object contact reward needs ho_contact_object (the ARCTIC object name)")
            assets = str(reward_cfg.get("ho_contact_assets") or get_asset_path(""))
            self.ho_contact = HandObjectContact(
                {k: reward_cfg.get(f"ho_contact_{k}") for k in ("cap", "tol", "scale", "touch", "near")},
                [grid_path(assets, obj_name, part) for part in PARTS], device)
        self.load_demo(demo_data, retarget_data, device)

    def load_demo(self, demo_data, retarget_data, device):
        self.demo_tensors = dict()
        demo_keys = ["obj_pos", "obj_quat", "obj_arti"]
        if self.contact_rew_weight > 0.0:
            demo_keys += ["contact_links_left", "contact_links_right"]
        
        for key in demo_keys:
            assert key in demo_data, f"{key} not in demo_data" 
            self.demo_tensors[key] = torch.tensor(
                demo_data[key], dtype=torch.float32, device=device
            )

        if self.ref_motion_info:
            # how far the demo object moves from the previous frame to this one: metres and radians per frame
            pos = self.demo_tensors["obj_pos"]; quat = self.demo_tensors["obj_quat"]
            quat = quat / torch.norm(quat, dim=-1, keepdim=True)
            lin = torch.zeros(pos.shape[0], dtype=torch.float32, device=device); rot = torch.zeros_like(lin)
            lin[1:] = torch.norm(pos[1:] - pos[:-1], dim=-1)
            rot[1:] = 2.0 * torch.acos(torch.clamp((quat[1:] * quat[:-1]).sum(dim=-1).abs(), max=1.0))
            lin[0] = lin[1]; rot[0] = rot[1]
            self.demo_tensors["obj_step_lin"] = lin; self.demo_tensors["obj_step_rot"] = rot

        if self.use_imi_rew:
            for side in ['left', 'right']:
                key = f"kpts_{side}"
                self.demo_tensors[key] = torch.tensor(
                    retarget_data[side]['kpts_data']['kpt_pos'], 
                    dtype=torch.float32, device=device
                )
        if self.hand_graph_weight > 0.0:
            from dexmachina.envs.finger_spec import hand_graph_nodes
            quat = self.demo_tensors["obj_quat"]
            rot = matrix_from_quat(quat / torch.norm(quat, dim=-1, keepdim=True))                       # (T, 3, 3) demo object
            unit = lambda v: v / torch.norm(v, dim=-1, keepdim=True)
            self.hand_graph_finger_weights, self.hand_graph_pairs, self.hand_graph_forward_mid = dict(), dict(), dict()
            for side in ['left', 'right']:
                spec = self.finger_specs[side]
                key = f"hand_joints_{side}"
                if key not in demo_data:
                    raise ValueError(f"The hand-graph reward needs the human hand joints ({key}) in the demo data")
                joints = torch.tensor(np.asarray(demo_data[key]), dtype=torch.float32, device=device)   # (T, 21, 3) MANO
                # palm frame: normal (index base - wrist) x (outermost base - wrist), or x (little base - wrist) on a
                # 'calibrated_axis' hand; forward wrist -> middle base, or -> the midpoint of the index and middle bases when no
                # finger lies beyond the middle (same nodes on the robot)
                wrist, index, middle, outer = (joints[:, i] for i in spec.human_palm_nodes)
                forward = middle - wrist if spec.forward_to_middle else 0.5 * (index + middle) - wrist
                palm = torch.stack([unit(torch.cross(index - wrist, outer - wrist, dim=-1)), unit(forward)], dim=1)
                self.hand_graph_palm_axis[side] = None
                if spec.palm_normal_mode == 'calibrated_axis':
                    self.hand_graph_palm_axis[side] = self.load_palm_axis(side, palm[:, 0], retarget_data, device)
                # human directions in the demo object's frame (R^T v), so they follow the simulated object at run time
                self.demo_tensors[f"hand_graph_palm_{side}"] = torch.einsum('tij,tei->tej', rot, palm)          # (T, 2, 3)
                fingers = unit(joints[:, list(spec.human_tips)] - joints[:, list(spec.human_bases)])
                self.demo_tensors[f"hand_graph_fingers_{side}"] = torch.einsum('tij,tei->tej', rot, fingers)
                tips = joints[:, list(spec.human_tips)]
                a, b = zip(*spec.gap_pairs)
                self.demo_tensors[f"hand_graph_gaps_{side}"] = self.hand_graph_scale * torch.norm(tips[:, list(a)] - tips[:, list(b)], dim=-1)
                # robot nodes by link name in the retargeted keypoints; nodes that are not keypoints are extra nodes
                names = [str(n) for n in retarget_data[side]['kpts_data']['kpt_names']]
                index_lists, extra = hand_graph_nodes(spec, names)
                as_index = lambda ids: torch.tensor(ids, dtype=torch.long, device=device)
                self.hand_graph_index[side] = {k: as_index(v) for k, v in index_lists.items()}
                self.hand_graph_extra[side] = extra
                self.hand_graph_pairs[side] = (list(a), list(b))
                self.hand_graph_forward_mid[side] = not spec.forward_to_middle
                w = torch.tensor(spec.finger_weights(self.hand_graph_thumb), dtype=torch.float32, device=device)
                self.hand_graph_finger_weights[side] = w / w.sum()
            self.hand_graph_finger_weight = self.hand_graph_finger_weights['left']
            unused = [s for s in (self.cfg.get("hand_graph_palm_axis") or {}) if self.hand_graph_palm_axis.get(s) is None]
            if unused:
                raise ValueError(f"reward_cfg['hand_graph_palm_axis'] has an axis for {unused}, but the palm normal of "
                                 f"{self.hand_name} there comes from its fingers (finger_spec.palm_normal_mode)")
        if self.ho_contact_weight > 0.0:
            from dexmachina.envs.ho_contact import ANCHORS
            self.ho_contact_regions = dict()
            for side in ['left', 'right']:
                # the human target files hold all six regions (ANCHORS order); keep the robot's regions, picked by name
                regions = tuple(self.finger_specs[side].regions)
                pick = None if regions == ANCHORS else [ANCHORS.index(r) for r in regions]
                self.ho_contact_regions[side] = regions
                for key in (f"ho_dist_{side}", f"ho_spot_{side}"):
                    if key not in demo_data:
                        raise ValueError(f"The hand-object contact reward needs the human targets ({key}) in the demo data")
                    value = np.asarray(demo_data[key])
                    if pick is not None:
                        value = value[:, pick]
                    self.demo_tensors[key] = torch.tensor(value, dtype=torch.float32, device=device)
        if self.contact_rew_weight > 0.0 or (self.use_imi_rew and self.imi_wrist_weight > 0.0):
            # load wrist pose for contact reward
            for side in ['left', 'right']:
                key = f"wrist_pose_{side}"
                if isinstance(retarget_data[side]['wrist_pose'], torch.Tensor):
                    self.demo_tensors[key] = retarget_data[side]['wrist_pose'].clone().to(device)
                else:
                    self.demo_tensors[key] = torch.tensor(
                        retarget_data[side]['wrist_pose'], dtype=torch.float32, device=device
                    )
        
        # check all the data have the same first dim size 
        assert all(
            [self.demo_tensors[key].shape[0] == self.demo_tensors["obj_pos"].shape[0] for key in self.demo_tensors.keys()]
        ), f"First dim size mismatch: {[self.demo_tensors[key].shape[0] for key in self.demo_tensors.keys()]}"
        self.demo_length = self.demo_tensors["obj_pos"].shape[0] 
        print(f"Loaded demo data with length {self.demo_length}")
    
    def load_palm_axis(self, side, human_normal, retarget_data, device):
        """The palm axis (3,) float32 of one side of a 'calibrated_axis' hand (finger_spec.palm_normal_mode): its robot palm
        normal is R_wrist(t) @ axis. reward_cfg['hand_graph_palm_axis'][side] (three floats; at evaluation it comes from the
        run's params/env.pkl) is used exactly as stored. Otherwise the axis is calibrated on this clip's retargeted reference,
        retarget_data[side]['wrist_pose'] (T, 7: position, wxyz quaternion) against the human palm normal human_normal (T, 3)
        of the same frames (calibrate_palm_axis), rounded to float32 and written into reward_cfg, which the trainer saves in
        params/env.pkl and env.yaml after building the env."""
        stored = (self.cfg.get("hand_graph_palm_axis") or {}).get(side)
        try:
            pose = retarget_data[side]['wrist_pose']
        except (KeyError, TypeError):
            pose = None

        def fit():
            p = torch.as_tensor(pose).detach().to(device='cpu', dtype=torch.float64)
            if p.dim() != 2 or p.shape[1] != 7 or p.shape[0] != human_normal.shape[0]:
                raise ValueError(f"{side} hand: the retargeted wrist_pose has shape {tuple(p.shape)}; the palm-axis calibration "
                                 f"needs (T, 7) over the {human_normal.shape[0]} demo frames")
            axis64, length = calibrate_palm_axis(p[:, 3:7], human_normal)
            if length < MIN_PALM_AXIS_RESULTANT:
                raise ValueError(f"{side} hand: the human palm normal has no stable direction in the {self.hand_name} wrist frame "
                                 f"over the retargeted reference (mean resultant length {length:.3f} < {MIN_PALM_AXIS_RESULTANT})")
            return axis64.to(torch.float32), length

        if stored is not None:
            values = [float(x) for x in stored]
            axis = torch.tensor(values, dtype=torch.float32)
            if len(values) != 3 or not torch.isfinite(axis).all() or abs(float(torch.norm(axis.double())) - 1.0) > 1e-4:
                raise ValueError(f"reward_cfg['hand_graph_palm_axis']['{side}'] must be a unit 3-vector, got {stored!r}")
            source = "stored in reward_cfg"
            if pose is not None:                       # for the log only: how far this clip's own calibration would be
                try:
                    fitted, _ = fit()
                    a64, f64 = axis.double(), fitted.double()
                    diff = torch.rad2deg(torch.atan2(torch.norm(torch.cross(a64, f64, dim=-1)), torch.dot(a64, f64))).item()
                    source += f"; this clip's own calibration is {diff:.4f} deg away"
                except ValueError as e:
                    source += f"; this clip does not calibrate: {e}"
        elif pose is None:
            raise ValueError(f"{side} hand: the {self.hand_name} palm normal is a calibrated wrist-link axis; it needs the "
                             f"retargeted wrist_pose or reward_cfg['hand_graph_palm_axis']")
        else:
            axis, length = fit()
            axes = dict(self.cfg.get("hand_graph_palm_axis") or {})
            axes[side] = [float(x) for x in axis.tolist()]
            self.cfg["hand_graph_palm_axis"] = axes
            source = f"calibrated on the retargeted reference, mean resultant length {length:.4f}"
        print(f"Hand-graph palm normal ({self.hand_name} {side}): wrist-link axis {[round(x, 6) for x in axis.tolist()]} ({source})")
        return axis.to(device)

    def get_demo_length(self):  
        return self.demo_length
    
    def match_demo_state(self, demo_key, episode_length_buf):
        """ returns shape (num_envs, num_features) """
        assert demo_key in self.demo_tensors, f"Key {demo_key} not found in demo tensors"
        demo_t = torch.where(episode_length_buf >= self.demo_length, self.demo_length-1, episode_length_buf)
        # print(demo_t, episode_length_buf)
        return self.demo_tensors[demo_key][demo_t]
    
    def compute_task_reward(self, obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, episode_length_buf):
        if obj_pos is None or obj_quat is None or obj_arti is None or self.task_rew_weight == 0.0:
            # dummy reward
            task_rew = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device)
            return task_rew, dict(task_rew=task_rew)
        
        demo_arti = self.match_demo_state("obj_arti", episode_length_buf)
        pos_dist = position_distance(obj_pos, demo_pos)
        rot_dist = rotation_distance(obj_quat, demo_quat)
        # arti_dist = torch.mean((obj_arti - demo_arti)**2, dim=-1)

        if len(obj_arti.shape) > 1:
            obj_arti = obj_arti.flatten(start_dim=0)
            
        arti_dist = (obj_arti - demo_arti)**2 / 2.0
        
        # these should all be shape (num_envs, )!!
        obj_pos_rew = torch.exp(-self.obj_pos_beta * pos_dist)  
        obj_rot_rew = torch.exp(-self.obj_rot_beta * rot_dist)
        obj_arti_rew = torch.exp(-self.obj_arti_beta * arti_dist)
        assert obj_pos_rew.shape == obj_rot_rew.shape == obj_arti_rew.shape, "Shape mismatch"
        # if joint is closed, don't compute arti reward 
        if self.multiply_task_rew:
            task_rew = self.task_rew_weight * obj_pos_rew * obj_rot_rew * obj_arti_rew # scale is stil [0,1]
            # print(f"========= weighted task rew: {task_rew} | rew weight: {self.task_rew_weight * 10 } obj pos rew: {obj_pos_rew} | obj rot rew: {obj_rot_rew} | obj arti rew: {obj_arti_rew} =========")
        else:
            # obj_arti_rew = torch.where(demo_arti <= 0.0001, torch.zeros_like(obj_arti_rew), obj_arti_rew)
            task_rew = self.task_rew_weight * (
                self.obj_pos_weight * obj_pos_rew + 
                self.obj_rot_weight * obj_rot_rew + 
                self.obj_arti_weight * obj_arti_rew
                )
            
        # give a bonus if obj state is well tracked and articulation joint is open 
        well_track = (pos_dist < 0.005) & (rot_dist < 0.1) & (arti_dist < 0.1)  & (demo_arti > 0.1)
        if self.scale_well_track > 1.0:
            task_rew = torch.where(well_track, task_rew * self.scale_well_track, task_rew)

        rew_dict = dict(
            pos_dist=pos_dist,
            rot_dist=rot_dist,
            arti_dist=arti_dist,
            obj_pos_rew=obj_pos_rew,
            obj_rot_rew=obj_rot_rew,
            obj_arti_rew=obj_arti_rew,
            task_rew=task_rew.clone(), # NOTE: otherwise it's the same tensor as the total reward
            well_track=well_track,
        )  

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame, 
                torch.zeros(task_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(task_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            task_rew[tomask] = 0.0
        return task_rew, rew_dict

    def compute_keypoint_dist(self, keypoint_pos, episode_length_buf, left_hand=True):
        demo_key = "kpts_left" if left_hand else "kpts_right"
        demo_kpts = self.match_demo_state(demo_key, episode_length_buf) 
        return position_distance(keypoint_pos, demo_kpts)
    
    def compute_wrist_reward(self, wrist_pose, episode_length_buf, side='left'):
        demo_wrist = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        wrist_rot_dist = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
        wrist_pos_dist = position_distance(wrist_pose[:, :3], demo_wrist[:, :3])
        rot_beta = self.cfg["imi_wrist_rot_beta"]
        pos_beta = self.cfg["imi_wrist_pos_beta"]
        wrist_rew = (torch.exp(-rot_beta * wrist_rot_dist) + torch.exp(-pos_beta * wrist_pos_dist)) / 2.0 
        return wrist_rew, wrist_pos_dist, wrist_rot_dist

    def compute_imitation_reward(
        self,
        wrist_pose_left,
        wrist_pose_right,
        kpts_left: torch.Tensor,
        kpts_right: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ): 
        fingertip_dist_left = self.compute_keypoint_dist(kpts_left, episode_length_buf, left_hand=True)
        fingertip_dist_right = self.compute_keypoint_dist(kpts_right, episode_length_buf, left_hand=False)
        fingertip_dist = torch.mean( (fingertip_dist_left + fingertip_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
        beta = self.cfg["imi_fingertip_beta"]
        if self.exp_kpt_first:
            fingertip_rew_left = torch.exp(- beta * fingertip_dist_left)
            fingertip_rew_right = torch.exp(- beta * fingertip_dist_right)
            fingertip_rew = torch.mean( (fingertip_rew_left + fingertip_rew_right) / 2.0 , dim=-1) # (B, num_links) -> (B,) 
        else:  
            fingertip_rew = torch.exp(-self.cfg["imi_fingertip_beta"] * fingertip_dist) 
        # 
        if self.imi_wrist_weight > 0.0:
            # do a rotation + position distance for wrist pose
            wrist_rew_left, pos_dist_left, rot_dist_left = self.compute_wrist_reward(wrist_pose_left, episode_length_buf, side='left')
            wrist_rew_right, pos_dist_right, rot_dist_right = self.compute_wrist_reward(wrist_pose_right, episode_length_buf, side='right')
            wrist_rew = (wrist_rew_left + wrist_rew_right) / 2.0
            imi_rew = self.imi_wrist_weight * wrist_rew + (1.0 - self.imi_wrist_weight) * fingertip_rew

            wrist_dist = torch.mean( (pos_dist_left + pos_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
            keypoint_dist = self.imi_wrist_weight * wrist_dist + (1.0 - self.imi_wrist_weight) * fingertip_dist
        else:
            imi_rew = fingertip_rew
            keypoint_dist = fingertip_dist
            
        imi_rew *= self.imi_rew_weight

        rew_dict = dict(
            kpts_dist_left=fingertip_dist_left,
            kpts_dist_right=fingertip_dist_right,  
            imi_rew=imi_rew, 
            keypoint_dist=keypoint_dist,
        )
        if self.imi_wrist_weight > 0.0:
            rew_dict["wrist_pdist_left"] = pos_dist_left
            rew_dict["wrist_pdist_right"] = pos_dist_right
            rew_dict["wrist_rdist_left"] = rot_dist_left
            rew_dict["wrist_rdist_right"] = rot_dist_right
            rew_dict["wrist_rew_left"] = wrist_rew_left  
            rew_dict["wrist_rew_right"] = wrist_rew_right 
            rew_dict["fingertip_rew"] = fingertip_rew
            if not self.exp_kpt_first:
                rew_dict["fingertip_dist"] = fingertip_dist
                

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame, 
                torch.zeros(imi_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(imi_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            imi_rew[tomask] = 0.0
        return imi_rew, rew_dict 

    def contact_dist_to_rew(self, dist, function='exp'):
        if function == 'exp':
            return torch.exp(-self.contact_beta * dist)
        elif function == 'sigmoid':
            axb = self.contact_a * dist + self.contact_b
            rew = 1.0 / (1.0 + torch.exp(-axb)) + self.sigmoid_offset.to(dist.device) 
            return rew
        else:
            raise ValueError(f"Unknown contact reward function: {function}")

    def compute_hand_contact_reward(
        self,
        contact_link_pos, # shape (N, num_obj_links * num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links * num_hand_links, 1)
        wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7)
        episode_length_buf,
        demo_obj_pose, # shape (N, 7)
        side='left',
    ):
        demo_wrist_pose = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf) 
        # N, num_links * 2, 4 (last dim is contact pair ID) -> NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top' 
        demo_positions = demo_contacts[:, :, :3]
        demo_valid_contact = demo_contacts[:, :, -1] > 0.0 # (part id is <= 0 if no contact)
        chamfer_dists = dict()
        contact_rewards = dict()

        positions = contact_link_pos[:, :, :3]
        valid_mask = contact_link_pos[:, :, -1] > 0.0 
        for part_id in [1, 2]:
            # consider contact invalid if part_id is different
            demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
            part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                 
            for frame in ['obj', 'wrist']:
                if not self.wrist_frame_contact and frame == 'wrist':
                    continue
                demo_pose = demo_obj_pose if frame == 'obj' else demo_wrist_pose
                demo_in_frame = transform_contact(
                    demo_positions, demo_pose
                )
                pose = obj_pose if frame == 'obj' else wrist_pose
                in_frame = transform_contact(
                    positions, pose
                )
                dist = chamfer_distance(
                    in_frame, demo_in_frame, part_valid, demo_part_valid
                )
                chamfer_dists[f"CD_{side}_p{part_id}_frame_{frame}"] = dist
                rew = self.contact_dist_to_rew(dist, self.contact_rew_function)
                both_zero = (part_valid.sum(dim=-1) == 0) & (demo_part_valid.sum(dim=-1) == 0)
                if self.mask_zero_contact:
                    rew = torch.where(both_zero, torch.zeros_like(rew), rew)
                else:
                    rew = torch.where(both_zero, torch.ones_like(rew), rew)
                contact_rewards[f"conrew_{side}_p{part_id}_frame_{frame}"] = rew
            
        # per-part contact should be added, because sometimes full contact coverage is not feasible 
        if self.multiply_frame_contact:
            # multiply the rew in obj & wrist frame such that one does not dominate the other
            contact_rews = []
            for part_id in [1, 2]:
                obj_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_obj"]
                if not self.wrist_frame_contact:
                    wrist_rew = 1.0
                else:
                    wrist_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_wrist"] 
                multi_rew = obj_rew * wrist_rew
                contact_rewards[f"mul_conrew_{side}_p{part_id}"] = multi_rew
                contact_rews.append(multi_rew)
            contact_rew = torch.stack(contact_rews, dim=-1).mean(dim=-1)

        else: # sum & average all contact rewards
            contact_rew = sum(contact_rewards.values()) / len(contact_rewards)

        contact_dict = {**chamfer_dists, **contact_rewards, f"contact_rew_{side}": contact_rew}
        # add penalty for scenarios where demo has 0 contact but env has contact
        if self.contact_phase_penalty != 0.0: 
            mismatch = torch.zeros_like(contact_rew).bool()
            num_mismatch = torch.zeros_like(contact_rew)
            for part_id in [1, 2]:
                demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
                part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                part_mismatch = (demo_part_valid.sum(dim=-1) == 0) & (part_valid.sum(dim=-1) > 0)
                mismatch = mismatch | part_mismatch
                num_mismatch += part_mismatch.float()
            contact_rew[mismatch] += self.contact_phase_penalty
            contact_dict['num_mismatch'] = num_mismatch

        return contact_rew, contact_dict
    
    def compute_matched_contact_per_hand(
        self,
        contact_link_pos, # shape (N, num_obj_links, num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links, num_hand_links, 1)
        episode_length_buf,
        # wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7) 
        demo_obj_pose,
        side='left',
        max_distance=1.0,
    ):
        """ 
        Instead of computing point cloud reward, here we assume demo contacts and policy contacts
        are from the same set of dex hand links and hence each contact point has a matched target position in the demo 
        """ 
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf) 
        # NOTE the retargeted contacts are of shape (N, num_obj_parts=2, num_links, 4), first row is 'top' and second row is 'bottom'!
        # need to flip the order since policy contact has object link bottom first 
        demo_contacts = demo_contacts.clone()[:, [1, 0], :, :] # (N, num_obj_links, num_hand_links, 4)
        assert demo_contacts.shape[1] == contact_link_pos.shape[1], f"Shape mismatch: {demo_contacts.shape} vs {contact_link_pos.shape}"
        assert demo_contacts.shape[2] == contact_link_pos.shape[2], f"Shape mismatch: {contact_link_pos.shape} vs {demo_contacts.shape}"
        demo_valids = demo_contacts[:, :, :, -1] > 0.0 # (part id is <= 0 if no contact)
         
        # need to reshape this to (N, num_obj_links * num_hand_links, 3) to do the transformation first
        if self.retarget_objframe:
            bsize, nparts, nlinks = demo_contacts.shape[:3]
            demo_pos_global = demo_contacts[:,:,:,:3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links * num_hand_links, 3)
            demo_pos = transform_contact(demo_pos_global, demo_obj_pose)        
            demo_pos = demo_pos.reshape(bsize, nparts, nlinks, 3)  

            policy_pos_global = contact_link_pos[:, :, :, :3].reshape(
                bsize, nparts * nlinks, 3
            ) # (N, num_obj_links, num_hand_links, 3)
            policy_pos = transform_contact(policy_pos_global, obj_pose)
            # then reshape it back!
            policy_pos = policy_pos.reshape(bsize, nparts, nlinks, 3)
        else:
            demo_pos = demo_contacts[:, :, :, :3]
            policy_pos = contact_link_pos[:, :, :, :3]
        
        
        both_invalid_mask = torch.logical_not(demo_valids) & torch.logical_not(contact_link_valid)
        # only one valid 
        one_valid_mask = torch.logical_xor(demo_valids, contact_link_valid)
        # compute distance between demo and policy contact points
        dist = position_distance(demo_pos, policy_pos)
        if self.mask_zero_contact:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.ones_like(dist) * max_distance, dist)
        else:
            # if both invalid, set distance to 0
            dist = torch.where(both_invalid_mask, torch.zeros_like(dist), dist)
        # if one valid, set distance to a large value
        dist = torch.where(one_valid_mask, torch.ones_like(dist) * max_distance, dist) # (B, 2, nlinks)
        
        # different ways to compute the distance: 
        # if take mean, encourages all contacts to be close to the targets 
        # if take min, encourages only one contact to be close to the target
        # part_dist = torch.mean(dist, dim=-1) # (N, num_obj_links=2)  
        return dist 
    
    def hand_graph_palm_frame(self, side, kpts, wrist_pose=None):
        """Robot palm frame of one hand, (N, 2, 3) unit vectors in the world: the palm normal and the forward direction.
        kpts (N, K, 3) holds the hand-graph nodes (the keypoints, then the extra nodes if any). The normal is (index base -
        wrist) x (outermost base - wrist) ('fingers'), or R_wrist @ hand_graph_palm_axis[side] on a 'calibrated_axis' hand,
        with R_wrist from wrist_pose (N, 7: position, then the wrist link's wxyz quaternion, as robot.wrist_pose)."""
        unit = lambda v: v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
        wrist, index, middle, outer = (kpts[:, i] for i in self.hand_graph_index[side]['palm'])
        forward = 0.5 * (index + middle) - wrist if self.hand_graph_forward_mid[side] else middle - wrist
        axis = self.hand_graph_palm_axis.get(side)
        if axis is None:
            normal = unit(torch.cross(index - wrist, outer - wrist, dim=-1))
        else:
            if wrist_pose is None:
                raise ValueError(f"The {side} hand graph of {self.hand_name} needs the wrist pose (its palm normal is a "
                                 f"calibrated wrist-link axis)")
            quat = wrist_pose[:, 3:7]
            rot = matrix_from_quat(quat / torch.norm(quat, dim=-1, keepdim=True))
            normal = unit(torch.einsum('nij,j->ni', rot, axis.to(rot.dtype)))
        return torch.stack([normal, unit(forward)], dim=1)

    def compute_hand_graph_reward(self, kpts_left, kpts_right, obj_quat, episode_length_buf, extra_left=None, extra_right=None,
                                  wrist_pose_left=None, wrist_pose_right=None):
        """Hand graph against the human (MANO) hand at the same frame, in the object frame, over the fingers the robot has.
        dir_err = mean of (palm: 1 - cos of the palm normal and of the palm's forward direction) and (fingers: weighted mean
        1 - cos of the base-to-tip edges); gap_err = mean |robot fingertip gap - scaled human gap| (metres). Per hand
        exp(-beta_dir * dir_err - beta_gap * gap_err), averaged over hands and weighted. Translation-free: where the hand is
        comes from the other terms. extra_left/right (N, E, 3): positions of the hand-graph nodes that are not keypoints
        (hand_graph_extra), needed only when there are any. wrist_pose_left/right (N, 7), robot.wrist_pose: needed only on a
        'calibrated_axis' hand (hand_graph_palm_frame)."""
        if obj_quat is None:
            raise ValueError("The hand-graph reward needs the object orientation")
        rot = matrix_from_quat(obj_quat / torch.norm(obj_quat, dim=-1, keepdim=True))          # (N, 3, 3) simulated object
        unit = lambda v: v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
        to_world = lambda key: torch.einsum('nij,nej->nei', rot, self.match_demo_state(key, episode_length_buf))
        out, rews = dict(), []
        for side, kpts, extra, wrist_pose in (('left', kpts_left, extra_left, wrist_pose_left),
                                              ('right', kpts_right, extra_right, wrist_pose_right)):
            idx = self.hand_graph_index[side]
            if self.hand_graph_extra.get(side):
                if extra is None or extra.shape[1] != len(self.hand_graph_extra[side]):
                    raise ValueError(f"The {side} hand graph needs the positions of {self.hand_graph_extra[side]} (not keypoints)")
                kpts = torch.cat([kpts, extra], dim=1)
            a, b = self.hand_graph_pairs[side]
            palm = self.hand_graph_palm_frame(side, kpts, wrist_pose)
            palm_cos = (palm * to_world(f"hand_graph_palm_{side}")).sum(dim=-1).clamp(-1.0, 1.0)                # (N, 2)
            fingers = unit(kpts[:, idx['dst']] - kpts[:, idx['src']])
            finger_cos = (fingers * to_world(f"hand_graph_fingers_{side}")).sum(dim=-1).clamp(-1.0, 1.0)       # (N, fingers)
            palm_err = (1.0 - palm_cos).mean(dim=-1)
            finger_err = ((1.0 - finger_cos) * self.hand_graph_finger_weights[side]).sum(dim=-1)
            dir_err = 0.5 * (palm_err + finger_err)
            tips = kpts[:, idx['tips']]
            gaps = torch.norm(tips[:, a] - tips[:, b], dim=-1)
            gap_err = (gaps - self.match_demo_state(f"hand_graph_gaps_{side}", episode_length_buf)).abs().mean(dim=-1)
            rews.append(torch.exp(-self.hand_graph_beta_dir * dir_err - self.hand_graph_beta_gap * gap_err))
            out[f"hand_graph_dir_err_{side}"] = dir_err
            out[f"hand_graph_finger_err_{side}"] = finger_err
            out[f"hand_graph_gap_err_{side}"] = gap_err
            out[f"hand_graph_palm_deg_{side}"] = torch.rad2deg(torch.acos(palm_cos[:, 0]))
        hand_rew = self.hand_graph_weight * torch.stack(rews, dim=-1).mean(dim=-1)
        out["hand_graph_rew"] = hand_rew
        return hand_rew, out

    def compute_matched_contact_reward(
        self,
        contacts_link_left, # shape (N, num_obj_links, num_hand_links) 
        contacts_link_valid_left, # shape (N, num_obj_links, num_hand_links)
        contacts_link_right,
        contacts_link_valid_right,
        obj_pose,
        demo_obj_pose,
        episode_length_buf,
    ):
        rews = dict()
        contact_rew = 0
        for side, contacts, valids in zip(
            ['left', 'right'],
            [contacts_link_left, contacts_link_right],
            [contacts_link_valid_left, contacts_link_valid_right],
        ):
            part_dist = self.compute_matched_contact_per_hand(
                contacts, valids, episode_length_buf, 
                obj_pose, demo_obj_pose, side=side,
            )
            for i, part in enumerate(['bottom', 'top']): 
                con_dist = part_dist[:, i]
                if self.exp_kpt_first:
                    con_rew = self.contact_dist_to_rew(con_dist, self.contact_rew_function).mean(dim=-1) 
                else:
                    con_rew = self.contact_dist_to_rew(con_dist.mean(dim=-1), self.contact_rew_function) 
                rews[f"conrew_{side}_{part}"] = con_rew
                rews[f"matched_condist_{side}_{part}"] = con_dist
                contact_rew += con_rew
        contact_rew /= 4.0
        contact_rew *= self.contact_rew_weight
        rews['con_rew'] = contact_rew
        return contact_rew, rews


    def compute_contact_reward(
        self,
        obj_pose,
        wrist_pose_left,
        wrist_pose_right,
        contacts_link_left,
        contacts_link_valid_left, # shape (N, num_obj_links, num_hand_links)
        contacts_link_right,
        contacts_link_valid_right,
        episode_length_buf,
    ):
        demo_obj_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_obj_quat = self.match_demo_state("obj_quat", episode_length_buf)
        demo_obj_pose = torch.cat([demo_obj_pos, demo_obj_quat], dim=1)
        contact_rew_left, contact_dict_left = self.compute_hand_contact_reward(
            contacts_link_left, contacts_link_valid_left, 
            wrist_pose_left, obj_pose, episode_length_buf, demo_obj_pose, side='left'
        )
        contact_rew_right, contact_dict_right = self.compute_hand_contact_reward(
            contacts_link_right, contacts_link_valid_right, 
            wrist_pose_right, obj_pose, episode_length_buf, demo_obj_pose, side='right'
        )
        contact_rew = (contact_rew_left + contact_rew_right) / 2.0 
     
        # add bonus to reward more contact points
        # num_points = torch.sum(contacts_link_valid_left, dim=-1) + torch.sum(contacts_link_valid_right, dim=-1)
        # more_points_rew = torch.mean(num_points.float() / 2.0, dim=-1) # average over num_links, then over batch # max is 1.0
        # contact_rew += more_points_rew * 0.01 
        
        contact_rew *= self.contact_rew_weight
        contact_dict = {**contact_dict_left, **contact_dict_right} # merge the two dicts, should have no key overlap
        contact_dict['con_rew'] = contact_rew
        return contact_rew, contact_dict

    def reshape_contact_with_label(self, contact_link_pos, contact_link_valid):
        """ 
        Reshape the environment contact to better match with demos:
        In: 
            contact_link_pos_left: (N, 2, num_hand_links, 3), 2 is each object part
            contact_link_valid_left: (N, 2, num_hand_links)
        Out: 
            reshaped_contact_link_pos_left: (N, 2*num_hand_links, 4): 4 is (x, y, z, part_id), part_id is 1 or 2 or 0 if no valid
            reshaped contact_link_valid: (N, 2, num_hand_links) -> (N, 2*num_hand_links)
        """
        # first, add a part_id dim so (N, 2, num_links, 4) and last dim is part_id label
        labeled = torch.cat(
            [contact_link_pos, torch.zeros_like(contact_link_pos)[:, :, :, 0:1]], dim=-1
        ) # (N, 2, num_links, 4) 
        # fill in the labels 1 or 2 if valid contact
        valid_reshaped = contact_link_valid[..., None] # (N, 2, num_links, 1)
        valid = torch.cat(
            [valid_reshaped, torch.zeros_like(valid_reshaped)], dim=-1
        ) # -> (N, 2, num_links, 2)
        for part_idx in [0, 1]: # bottom, top (sice obj urdf is flipped for parsing error)
            label_id = part_idx + 1 
            # Need flipping, because in demos, id=2 is bottom
            label_id = 1 if label_id == 2 else 2 
            labeled[:, part_idx, :, -1] = torch.where(
                contact_link_valid[:, part_idx, :],  label_id, 0
            ) 
            valid[:, part_idx, :, 1] = torch.where(
                contact_link_valid[:, part_idx, :], label_id, 0
            )

        # reshape to (N, num_links*2, 4)
        contact_link_reshaped = labeled.view(
            labeled.shape[0], -1, labeled.shape[-1]
        )
        contact_valid_reshaped = valid.view(
            valid.shape[0], -1, valid.shape[-1]
        )
        
        return contact_link_reshaped, contact_valid_reshaped

    def compute_grasp_gate(
        self,
        contact_link_valid_left,   # (N, num_obj_parts, num_hand_links) bool, sim part order (bottom, top)
        contact_link_valid_right,
        episode_length_buf,
    ):
        """Grasp-fidelity gate in [0, 1]: contact graph (which links touch which part) against the retargeted demo."""
        out = dict()
        gate = torch.ones(episode_length_buf.shape[0], dtype=torch.float32, device=episode_length_buf.device)
        demo_valid = dict()
        for side in ['left', 'right']:
            # demo contacts are stored (T, parts, links, 4) with parts ordered (top, bottom); flip to the sim order
            d = (self.match_demo_state(f"contact_links_{side}", episode_length_buf)[..., 3] > 0.0)[:, [1, 0]]   # (N, 2, links)
            demo_valid[side] = d
        if self.cg_gate > 0.0:
            miss = 0.0; extra = 0.0; need = 0.0
            for side, valid in (('left', contact_link_valid_left), ('right', contact_link_valid_right)):
                sim = valid.bool(); d = demo_valid[side]
                assert sim.shape == d.shape, f"contact mask shape {tuple(sim.shape)} != demo {tuple(d.shape)}"
                miss = miss + (d & ~sim).sum(dim=(1, 2)).float()
                extra = extra + (~d & sim).sum(dim=(1, 2)).float()
                need = need + d.sum(dim=(1, 2)).float()
            cg = torch.clamp(1.0 - (miss + self.cg_extra * extra) / torch.clamp(need, min=1.0), 0.0, 1.0)
            out.update(cg_rew=cg, cg_missing=miss, cg_extra=extra, cg_need=need)
            gate = gate * ((1.0 - self.cg_gate) + self.cg_gate * cg)
        out["grasp_gate"] = gate
        return gate, out

    def setup_ho_contact(self, side, slices, anchor_link_idx):
        """Called by the env once its robots exist: the point slices of each region (the hand's regions: palm, then its
        fingers) in the per-hand point tensor, and each region's contact-link indices (proximal to distal) into that hand's
        contact buffers."""
        regions = getattr(self, 'ho_contact_regions', {}).get(side)
        if regions is not None and len(slices) != len(regions):
            raise ValueError(f"{side} hand: {len(slices)} contact regions from the env, the reward expects {list(regions)}")
        self.ho_contact_spec[side] = dict(slices=[tuple(s) for s in slices], links=[list(map(int, idx)) for idx in anchor_link_idx])

    def compute_ho_contact_reward(self, inputs, contact_pos_left, contact_valid_left, contact_pos_right, contact_valid_right,
                                  episode_length_buf):
        """Hand-object contact, "where and how far" (dexmachina/envs/ho_contact.py); one score per hand, averaged, weighted."""
        from dexmachina.envs.ho_contact import anchor_contacts
        out, hands = dict(), []
        check_sum = check_n = 0.0
        for side, pos, valid in (('left', contact_pos_left, contact_valid_left), ('right', contact_pos_right, contact_valid_right)):
            if side not in self.ho_contact_spec:
                raise ValueError("setup_ho_contact was not called for the " + side + " hand")
            spec = self.ho_contact_spec[side]
            touching, touch_pos = anchor_contacts(valid.bool(), pos, spec['links'])
            d_r, s_r, grid_d = self.ho_contact.robot_distances(
                inputs[f'points_{side}'], spec['slices'], inputs['part_pos'], inputs['part_quat'], touching, touch_pos)
            d_h = self.match_demo_state(f"ho_dist_{side}", episode_length_buf)
            s_h = self.match_demo_state(f"ho_spot_{side}", episode_length_buf)
            hand, det = self.ho_contact.score(d_r, s_r, d_h, s_h)
            hands.append(hand)
            w = det['weight']
            out[f"ho_hand_{side}"] = hand
            out[f"ho_err_{side}"] = (w * det['err']).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1e-6)
            # robot at the surface where the human is not (>= 2 x touch away), and the reverse
            out[f"ho_unwanted_{side}"] = ((d_r < self.ho_contact.touch) & (d_h >= 2 * self.ho_contact.touch)).float().sum(dim=(1, 2))
            out[f"ho_missed_{side}"] = ((d_h < self.ho_contact.touch) & (d_r >= 2 * self.ho_contact.touch)).float().sum(dim=(1, 2))
            # consistency: grid distance of anchors the simulator reports touching (should be ~0)
            t = touching.transpose(1, 2)                                                    # (N, A, parts)
            check_sum = check_sum + (grid_d * t).sum()
            check_n = check_n + t.sum()
        rew = self.ho_contact_weight * (hands[0] + hands[1]) / 2.0
        out["ho_contact_rew"] = rew
        out["ho_contact_check"] = (check_sum / torch.clamp(check_n, min=1.0)).expand_as(rew).clone()
        return rew, out

    def compute_reward(
        self, 
        actions,
        bc_dist, # (num_envs, left+right hand num of joints) 
        obj_pos,
        obj_quat,
        obj_arti,
        kpts_left,
        kpts_right,
        contact_link_pos_left, 
        contact_link_valid_left,  # shape (N, num_obj_links, num_hand_links)
        contact_link_pos_right,
        contact_link_valid_right,
        wrist_pose_left,
        wrist_pose_right,
        contact_forces, # shape (N, num_obj_links, num_hand_links, 3)
        episode_length_buf,
        ho_contact_inputs=None, # hand-object contact only: anchor points per hand (world) and object part poses, from the env
        hand_graph_extra_left=None, # hand graph only: positions of its nodes that are not keypoints (N, E, 3), if any
        hand_graph_extra_right=None,
    ):
        demo_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_quat = self.match_demo_state("obj_quat", episode_length_buf) 
        rew, rew_dict = self.compute_task_reward(
            obj_pos, obj_quat, obj_arti, 
            demo_pos, demo_quat,
            episode_length_buf
        )
        if self.ref_motion_info:
            rew_dict["ref_obj_step_lin"] = self.match_demo_state("obj_step_lin", episode_length_buf)
            rew_dict["ref_obj_step_rot"] = self.match_demo_state("obj_step_rot", episode_length_buf)
        if self.use_grasp_gate:
            # gate only the summed object reward; rew_dict['task_rew'] stays ungated for resets, curriculum and keyframes
            gate, gate_dict = self.compute_grasp_gate(contact_link_valid_left, contact_link_valid_right, episode_length_buf)
            rew = rew * gate
            rew_dict.update(gate_dict)
        if self.bc_rew_weight > 0.0:
            bc_rew = self.bc_rew_weight * torch.exp(-self.cfg["bc_beta"] * bc_dist)
            bc_rew = torch.mean(bc_rew, dim=-1)
            rew_dict["bc_dist"] = bc_dist.mean(dim=-1)
            rew_dict["bc_rew"] = bc_rew
            # rew += bc_rew

        if self.use_imi_rew: 
            imi_rew, imi_rew_dict = self.compute_imitation_reward(
                wrist_pose_left, wrist_pose_right,
                kpts_left, kpts_right, episode_length_buf
            )
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the imitation reward
                imi_rew = torch.where(rew_dict["well_track"], torch.zeros_like(imi_rew), imi_rew) 
            # rew += imi_rew
            rew_dict.update(imi_rew_dict)
 
        if self.contact_rew_weight > 0.0: 
            obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
            demo_pose = torch.cat([demo_pos, demo_quat], dim=1)
            if self.use_retarget_contact:
                contact_rew, contact_dict = self.compute_matched_contact_reward(
                    contact_link_pos_left, contact_link_valid_left,
                    contact_link_pos_right, contact_link_valid_right,
                    obj_pose, demo_pose, episode_length_buf,
                )
            else:
                left_reshaped, left_valid = self.reshape_contact_with_label(
                    contact_link_pos_left, contact_link_valid_left
                    )
                right_reshaped, right_valid = self.reshape_contact_with_label(
                    contact_link_pos_right, contact_link_valid_right
                    )
                contact_rew, contact_dict = self.compute_contact_reward(
                    obj_pose, 
                    wrist_pose_left, 
                    wrist_pose_right,
                    left_reshaped, left_valid, 
                    right_reshaped, right_valid,
                    episode_length_buf
                ) 
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the contact reward
                contact_rew = torch.where(rew_dict["well_track"], torch.zeros_like(contact_rew), contact_rew)   
  
            # rew += contact_rew
            rew_dict.update(contact_dict)
        
        if self.multiply_all_rew:
            aux_rew = 1.0
            if self.use_imi_rew:
                aux_rew *= imi_rew
            if self.contact_rew_weight > 0.0:
                aux_rew *= contact_rew
            if self.bc_rew_weight > 0.0:
                aux_rew *= bc_rew
            rew += aux_rew * 0.5
        else:
            if self.use_imi_rew:
                rew += imi_rew
            if self.contact_rew_weight > 0.0:
                rew += contact_rew
            if self.bc_rew_weight > 0.0:
                rew += bc_rew
        if self.hand_graph_weight > 0.0:
            hand_rew, hand_dict = self.compute_hand_graph_reward(kpts_left, kpts_right, obj_quat, episode_length_buf,
                                                                 hand_graph_extra_left, hand_graph_extra_right,
                                                                 wrist_pose_left, wrist_pose_right)
            rew += hand_rew
            rew_dict.update(hand_dict)
        if self.ho_contact_weight > 0.0:
            if ho_contact_inputs is None or contact_link_valid_left is None or contact_link_valid_right is None:
                raise ValueError("The hand-object contact reward needs the env's anchor points, part poses and contact buffers")
            ho_rew, ho_dict = self.compute_ho_contact_reward(
                ho_contact_inputs, contact_link_pos_left, contact_link_valid_left,
                contact_link_pos_right, contact_link_valid_right, episode_length_buf)
            rew += ho_rew
            rew_dict.update(ho_dict)


        if self.cfg["force_penalty"] > 0.0 and contact_forces is not None: # shape (B, 2, num_links*2, 3)
            force_norm = torch.norm(contact_forces, dim=-1).flatten(start_dim=1) # shape (B, 2*num_links*2) -> this goes to up to 5k
            high_force = torch.where(force_norm > 500.0, force_norm - 500.0, torch.zeros_like(force_norm))
            high_force = torch.mean(high_force, dim=-1) # shape (B, )
            force_penalty = self.cfg["force_penalty"] * high_force 
            rew -= force_penalty
            rew_dict["force_penalty"] = force_penalty
        
        # rew *= 0.02 
        # penalize action to stay close to 0 
        if self.cfg["action_penalty"] > 0.0:
            action_penalty = torch.mean(actions**2, dim=-1) * self.cfg["action_penalty"]
            rew -= action_penalty
            rew_dict["action_penalty"] = action_penalty  
        return rew, rew_dict
 
    def get_reward_keys(self):
        keys = []
        if self.task_rew_weight > 0:
            keys.append('task')
        if self.imi_rew_weight > 0:
            keys.append('imi')
        if self.contact_rew_weight > 0:
            keys.append('con')
        if self.bc_rew_weight > 0:
            keys.append('bc')
        return keys 