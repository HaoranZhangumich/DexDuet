import os
import math
import torch
import numpy as np
import genesis as gs
from collections import deque

def get_curriculum_cfg(kwargs=dict()):
    curr_cfg = {
        "kp_init": 1000.0,
        "kv_init": 10.0,
        "force_range_init": 50.0, 
        "wait_epochs": 2000,
        "decay_rew": False,
        "schedule": "exp", # or exp or uniform 
        "deque_len": 100, # length of reward deques 
        "interval": 500, # epoch count
        "first_ratio": 0.3, # determines how much to reduce the gain at the end of firt stage
        "first_stop_iter": 7000, # epoch count
        "second_stop_iter": 16000, # epoch count
        "fixed_mode": "exp", # or exp 

        # if schedule=exp: auto linear 
        "deque_freq": 1, # epoch count
        "grad_threshold": 0.0001,
        "gain_mode": "all", # which terms to reduce gain

        # if schedule=uniform: auto uniform distribution of gains
        "uniform_mode": "fast", # or slow
        # "upper_ratio": 0.9, # upper = curr_upper * (upper_ratio)
        # "lower_ratio": 0.8, # if fast: lower=curr_lower * lower_ratio, if slow: lower=curr_upper * lower_ratio
        "upper_ratios": dict(kp=0.5, kv=0.9, fr=0.95),
        "lower_ratios": dict(kp=0.5, kv=0.8, fr=0.9),
        "seed": 42,
        "decay_solimp": False,
        "solip_multiplier": 0.98,
        "d0_lower": 0.9,
        "dmid_lower": 0.95,
        "tconst_lower": 0.1,
        "tconst_upper": 0.15, # 0.2 is too soft
        "rew_thresholds": dict(task=0.5, con=0.05, imi=0.05, bc=0.05),
        "resample_every_epoch": -1, # if true and using uniform, re-sample gains every epoch
        "skip_grad": False, # if true, skip gradient check
        "zero_epoch": 30000, # set all zeros once beyond this hardstop
        "dialback_ep_len": 50,
        "dialback_min_epochs": 500,
        "dialback_ratios": dict(kp=0.98, kv=0.98, fr=0.98), # don't fully reset to prev gains, instead decay by this ratio 
    }
    # update with kwargs
    curr_cfg.update(kwargs)
    return curr_cfg

def get_v2_curriculum_cfg(max_epochs, num_zero_epoch, seed=42):
    """Common score-independent assistance for the v2 ablations.

    Counts refer to completed PPO updates. With 13,000 updates and a 1,000
    update zero tail, rollout 12,001 is the first fully unassisted rollout.
    """
    if (type(max_epochs) is not int or type(num_zero_epoch) is not int
            or max_epochs < 1 or not 0 <= num_zero_epoch <= max_epochs):
        raise ValueError("v2 assistance requires max_epochs > 0 and 0 <= num_zero_epoch <= max_epochs")
    return get_curriculum_cfg(dict(
        ablation_protocol="v2", schedule="linear_updates", fixed_mode="lin", gain_mode="all",
        kp_init=80.0, kv_init=5.0, force_range_init=50.0,
        zero_epoch=max_epochs - num_zero_epoch, max_epochs=max_epochs, num_zero_epoch=num_zero_epoch,
        wait_epochs=0, interval=1, decay_rew=False, decay_solimp=False, resample_every_epoch=-1,
        seed=seed,
    ))


SCHEDULE_OPTIONS=["fixed", "exp", "uniform", "linear_updates", "rsi"]

# Reference-state initialization (DeepMimic RSI) with BeyondMimic-style adaptive start
# sampling (arXiv 2508.08241) over the VOC assistance. The demo clip is split into start
# bins; every episode begins at a sampled reference frame with the virtual object
# controller holding the object there, each bin's assistance decays on that bin's own
# success rate instead of the global episode-length gate, and start bins are drawn with
# probability raised by recent failures in and after the bin (backward credit kernel),
# mixed with a uniform floor so easy bins are never forgotten.
RSI_DEFAULTS = dict(
    rsi_bin_frames=0,       # start-bin width in frames (15 = 0.25 s at 60 Hz, bin count follows the clip length); 0 = legacy
    rsi_bins=20,            # legacy fixed bin count, used only when rsi_bin_frames is 0 (the default)
    rsi_uniform_mix=0.3,    # uniform share of start sampling (BeyondMimic lambda)
    rsi_credit_gamma=0.75,  # bin s draws credit gamma^tau from the failure rate of bin s+tau (per bin = per rsi_bin_frames)
    rsi_fail_alpha=0.02,    # per-episode EMA step for each bin's sampled failure rate
    rsi_success_up=0.8,     # window success rate that lowers a bin's assistance one step
    rsi_success_down=0.4,   # below this the bin's assistance moves back up one step
    rsi_step=0.9,           # one native decay step (upper ratio 0.9)
    rsi_band=2.0,           # per-env gains sample in [top/band, top] (native lower ratio 0.5)
    rsi_min_count=64,       # finished episodes a bin needs per gain decision
    rsi_min_gap=40,         # updates between gain moves of the same bin
    rsi_zero_below=0.05,    # success-driven decay below this snaps to zero (native cutoff)
    rsi_deadline=9000,      # updates by which the floor reaches zero_below in every bin
    rsi_zero_ramp=1000,     # afterwards the floor ramps linearly to exactly zero
    rsi_min_horizon=10,     # episodes never start closer than this to the demo end
    rsi_keyframes=False,    # start bins > 0 from reached simulator states (envs/keyframes.py) once filled
    rsi_kf_capacity=256,    # stored states per bin
    rsi_kf_prob=0.002,      # per healthy lane and step capture probability
    rsi_kf_min_fill=32,     # stored states a bin needs before its starts use them
    rsi_kf_min_task=0.6,    # capture only lanes tracking this well (step and episode-average task reward)
    rsi_kf_confirm=0,       # frames a lane must go on well before its picked state is stored (0: store at once)
    rsi_kf_to_end=False,    # store a picked state only once its lane finishes the clip; serve it only with >= the help it had
    rsi_grace_bank=0.0,     # task reward credited to the early-reset check of mid-clip starts
    rsi_frame0_share=0.0,   # share of starts always put at frame 0 (the evaluation's start), whatever the bin sampling says
)
RSI_INT_KEYS = ("rsi_bin_frames", "rsi_bins", "rsi_min_count", "rsi_min_gap", "rsi_deadline", "rsi_zero_ramp", "rsi_min_horizon",
                "rsi_kf_capacity", "rsi_kf_min_fill", "rsi_kf_confirm")

class Curriculum:
    def __init__(self, curr_cfg, task_object, reward_keys, num_envs, achieved_length, max_episode_length):
        self.curr_cfg = curr_cfg
        self.ablation_protocol = curr_cfg.get('ablation_protocol', 'legacy')
        self.num_envs = num_envs
        self.object = task_object
        self.init_gains = {
            "kp": curr_cfg['kp_init'],
            "kv": curr_cfg['kv_init'],
            "fr": curr_cfg['force_range_init'],
        }
        self.curr_gains = self.init_gains.copy()
        self.curr_gains_lower = self.init_gains.copy() # use for uniform mode
        self.gain_history = [] # keep a list of gains for prev epochs 
        self.gain_mode = curr_cfg['gain_mode']
        self.decay_rew = curr_cfg['decay_rew']
        # first figure out which terms to decay:
        decay_terms = []
        if self.gain_mode == "all":
            decay_terms = ["kp", "kv", "fr"]
        elif "kp" in self.gain_mode:
            decay_terms.append("kp")
        elif "kv" in self.gain_mode:
            decay_terms.append("kv")
        elif "fr" in self.gain_mode:
            decay_terms.append("fr")
        else:
            raise ValueError("Invalid gain mode")
        self.decay_terms = decay_terms
        self.wait_epochs = curr_cfg['wait_epochs']

        # next get the params for different schedules
        self.schedule = curr_cfg['schedule']
        assert self.schedule in SCHEDULE_OPTIONS, "Invalid schedule"

        self.fixed_mode = curr_cfg['fixed_mode']
        assert self.fixed_mode in ['lin', 'exp', 'uniform'], "Invalid fixed mode"
        
        self.uniform_mode = curr_cfg['uniform_mode']
        self.obj_ndof = 7 # base + articulation
        assert self.uniform_mode in ['fast', 'slow'], "Invalid uniform mode"
        self.deque_appends = 0
        self.deque_append_freq = curr_cfg['deque_freq'] 

        self.decay_solimp = curr_cfg['decay_solimp']
        self.solip_multiplier = curr_cfg['solip_multiplier']
        self.d0_lower = curr_cfg['d0_lower']
        self.dmid_lower = curr_cfg['dmid_lower']
        self.tconst_lower = curr_cfg['tconst_lower']
        self.tconst_upper = curr_cfg['tconst_upper']

        self.upper_ratios = curr_cfg.get('upper_ratios', dict(kp=0.9, kv=0.9, fr=0.9))
        self.lower_ratios = curr_cfg.get('lower_ratios', dict(kp=0.8, kv=0.8, fr=0.8))

        self.rew_deques = dict()
        self.rew_grads = dict() 
        self.deque_len = curr_cfg['deque_len'] 
        for key in reward_keys:
            assert key in ["task", "imi", "bc", "con"], f"Invalid reward key: {key}"
            self.rew_deques[key] = deque(maxlen=self.deque_len)
            self.rew_grads[key] = 1.0 
        self.ep_lens = deque(maxlen=self.deque_len) 

        self.reward_keys = reward_keys
        self.grad_threshold = curr_cfg['grad_threshold']  
        self.rew_thresholds = curr_cfg['rew_thresholds']
        self.max_episode_length = max_episode_length
        # self.achieved_length = achieved_length # need to update this from env!
        self.seed = curr_cfg['seed']
        self.num_epoch_since_last_decay = 0
        self.num_epoch_since_zero = 0
        self.resample_every_epoch = curr_cfg['resample_every_epoch'] 
        if self.schedule != "linear_updates":
            # Legacy behavior; v2 uses the explicitly seeded training process.
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.skip_grad = curr_cfg['skip_grad']
        self.zero_epoch = curr_cfg.get('zero_epoch', 50000)
        if self.schedule == "linear_updates":
            if type(self.zero_epoch) is not int or self.zero_epoch < 0:
                raise ValueError("Linear assistance requires a nonnegative completed-update denominator")
            if self.zero_epoch == 0:
                self.curr_gains = {key: 0.0 for key in self.init_gains}
                self.curr_gains_lower = self.curr_gains.copy()
        if self.schedule == "rsi":
            self._rsi_init(curr_cfg)

    def post_scene_build_setup(self):
        if self.schedule == "rsi":
            none = torch.zeros(self.num_envs, dtype=torch.bool)
            self.rsi_on_reset(torch.arange(self.num_envs), none, none)
            self.rsi_env_active[:] = False  # the first episodes start at the next reset
        if self.schedule == "linear_updates":
            self.reset_object_gains()
        if self.decay_solimp:
            new_params = torch.tensor([self.tconst_upper, 1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0]) 
            self.object.entity._solver.set_geom_sol_params(new_params)
        return 
        
    def set_fixed_decay(self, epoch_num):
        """
        2 stage fast and slow linear decay, or 1 stage exponential decay, only depending on epoch_num
        """ 
        interval = self.curr_cfg['interval']
        if epoch_num % interval != 0 or epoch_num == 0:
            return False, "wrong interval"
        if self.fixed_mode == "exp" or self.fixed_mode == "uniform":
            for k in self.decay_terms:
                fixed_ratio = self.upper_ratios.get(k, 0.9)
                # if self.curr_gains[k] < 1 and k == 'kp':
                #     fixed_ratio = 0.96
                self.curr_gains[k] *= fixed_ratio
            # if uniform, decay lower bound as well
            if self.fixed_mode == "uniform":
                for k in self.decay_terms:
                    low_ratio = self.lower_ratios[k]
                    if self.uniform_mode == "fast":
                        self.curr_gains_lower[k] *= low_ratio
                    elif self.uniform_mode == "slow":
                        self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio
        elif self.fixed_mode == "lin":
            # two stage linear decay
            first_stop = self.curr_cfg['first_stop_iter']
            second_stop = self.curr_cfg['second_stop_iter']
            if epoch_num < first_stop:
                mid_gains = {k: v * self.curr_cfg['first_ratio'] for k, v in self.init_gains.items()}
                frac = epoch_num / first_stop
                new_gains = {
                    "kp": self.init_gains['kp'] * (1 - frac) + mid_gains['kp'] * frac,
                    "kv": self.init_gains['kv'] * (1 - frac) + mid_gains['kv'] * frac,
                    "fr": self.init_gains['fr'] * (1 - frac) + mid_gains['fr'] * frac,
                }
            elif epoch_num < second_stop:
                frac = (epoch_num - first_stop) / (second_stop - first_stop)
                new_gains = {
                    "kp": mid_gains['kp'] * (1 - frac) + self.init_gains['kp'] * frac,
                    "kv": mid_gains['kv'] * (1 - frac) + self.init_gains['kv'] * frac,
                    "fr": mid_gains['fr'] * (1 - frac) + self.init_gains['fr'] * frac,
                }
            else:
                new_gains = {k: 0.0 for k in self.init_gains}
            
            for k in self.decay_terms:
                self.curr_gains[k] = new_gains[k]
        else:
            raise ValueError("Invalid fixed mode")
        return True, ""

    def determine_decay(self, epoch_num):
        reduce_gains = True 
        reason = f"Epoch {epoch_num}: "
        if epoch_num < self.wait_epochs:
            reason += f"waiting for {self.wait_epochs} epochs"
            reduce_gains = False
        if self.schedule == "fixed": # always True
            return reduce_gains, reason
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                reduce_gains = False 
                reason += f"{key} deque too short: {len(self.rew_deques[key]) } "
                break 
            grad = self.rew_grads[key] 
            rew_mean = np.mean(self.rew_deques[key])
            rew_thres = self.rew_thresholds.get(key, 0.01)
            if rew_mean < rew_thres:
                reason += f"{key} reward too low: {rew_mean} "
                reduce_gains = False
            elif not self.skip_grad and abs(grad) > self.grad_threshold:
                reason += f"{key} grad too low/high: {grad} "
                reduce_gains = False 
            if not reduce_gains:
                break # no need to check other terms 
         
        if self.num_epoch_since_last_decay < 40:
            reason += f"too soon until last decay: {self.num_epoch_since_last_decay} epochs "
            reduce_gains = False
        
        if len(self.ep_lens) < self.deque_len:
            reason += f"ep lens deque too short: {len(self.ep_lens)} "
            reduce_gains = False
        else:
            achieved_len = np.mean(self.ep_lens)
            if achieved_len < self.max_episode_length - 2:
                reason += f"max achieved length: {achieved_len} too low"
                reduce_gains = False
        return reduce_gains, reason 

    def determine_dialback(self, epoch_num):
        """ if the acheived ep length has been too low for too long, dial back the gains """
        max_past = self.curr_cfg["dialback_min_epochs"]
        min_ep_len = self.curr_cfg["dialback_ep_len"]
        ratios = self.curr_cfg["dialback_ratios"]
        dialed = False
        # print(f"Epoch {epoch_num}: Dialback check: {len(self.ep_lens)} past ep lens")
        if len(self.ep_lens) < self.deque_len:
            return dialed
        achieved_len = np.mean(self.ep_lens) 
        # print(f"Epoch {epoch_num}: Achieved length: {achieved_len} {self.num_epoch_since_last_decay}")
        if achieved_len < min_ep_len:
            if self.num_epoch_since_last_decay > max_past:
                # dial back the gains
                upper_gains = self.gain_history[-1]['gains']
                lower_gains = self.gain_history[-1]['lower']
                for k in self.decay_terms:
                    if k in upper_gains:
                        ratio = ratios.get(k, 1.0)
                        upper_gains[k] *= ratio
                        lower_gains[k] *= ratio
                self.curr_gains = upper_gains
                self.curr_gains_lower = lower_gains
                dialed = True
                # TODO: also decay the solimp params
        return dialed

    def set_auto_exp_decay(self, epoch_num): 
        new_gains = dict()
        for k, v in self.curr_gains.items():
            ratio = self.upper_ratios[k]
            new_gains[k] = v * ratio
        if "kp" in new_gains and new_gains['kp'] < 0.05 or "fr" in new_gains and new_gains['fr'] < 0.01:
            new_gains = {k: 0.0 for k in new_gains}
        for k in self.decay_terms:
            self.curr_gains[k] = new_gains[k]
        return True, ""

    def set_auto_uniform_decay(self, epoch_num):
        """ update an upper and lower bound for gains """  
        for k in self.decay_terms: 
            upp_ratio = self.upper_ratios[k]
            self.curr_gains[k] = self.curr_gains[k] * upp_ratio
        if self.uniform_mode == "fast":
            # use self.curr_gains as upper bound 
            for k in self.decay_terms: 
                low_ratio = self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains_lower[k] * low_ratio
        elif self.uniform_mode == "slow":
            # use self.curr_gains as upper bound 
            for k in self.decay_terms:
                low_ratio = self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio
        else:
            raise ValueError("Invalid uniform mode") 
            
        if ('kp' in self.decay_terms and self.curr_gains['kp'] < 0.05) or ('fr' in self.decay_terms and self.curr_gains['fr'] < 0.01):
            for k in self.decay_terms:
                self.curr_gains[k] = 0.0
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0 
        # if the lower bound kp is within 0.1 thres, reduce it to 0 
        if ('kp' in self.decay_terms and self.curr_gains_lower.get('kp', 1) < 0.1):
            for k in self.decay_terms:
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0
        return True, ""
    
    # ------------------------------------------------------------------ RSI
    def _rsi_init(self, cfg):
        if self.gain_mode != "all":
            raise ValueError("RSI assistance scales kp and kv together; use gain_mode=all")
        for key, default in RSI_DEFAULTS.items():
            value = cfg.get(key, default)
            setattr(self, key, int(value) if key in RSI_INT_KEYS else float(value))
        startable = self.max_episode_length - self.rsi_min_horizon   # starts never closer than this to the end
        if self.rsi_bin_frames > 0:
            # Fixed-duration bins (BeyondMimic-style): the count follows the clip. The last bin may be shorter.
            self.rsi_bins = math.ceil(startable / self.rsi_bin_frames)
        elif self.rsi_bin_frames < 0:
            raise ValueError("rsi_bin_frames must be >= 0")
        if not (self.rsi_bins > 1 and 0 <= self.rsi_uniform_mix <= 1 and 0 <= self.rsi_credit_gamma < 1
                and 0 < self.rsi_fail_alpha <= 1 and 0 <= self.rsi_success_down < self.rsi_success_up <= 1
                and 0 < self.rsi_step < 1 and self.rsi_band >= 1 and self.rsi_min_count > 0
                and self.rsi_min_gap >= 0 and self.rsi_zero_below > 0 and self.rsi_deadline > 0
                and self.rsi_zero_ramp > 0 and 0 < self.rsi_min_horizon < self.max_episode_length
                and 0 <= self.rsi_frame0_share < 1):
            raise ValueError("Invalid RSI assistance configuration")
        kp0 = float(self.init_gains["kp"])
        if kp0 <= 0:
            raise ValueError("RSI assistance needs a positive initial kp")
        self.rsi_kv_per_kp = float(self.init_gains["kv"]) / kp0
        self.rsi_fr = float(self.init_gains["fr"])
        if self.rsi_bin_frames > 0:
            lo = torch.arange(self.rsi_bins, dtype=torch.long) * self.rsi_bin_frames
            self.rsi_bin_hi = (lo + self.rsi_bin_frames).clamp(max=startable)
            self.rsi_bin_lo = lo
        else:   # legacy: rsi_bins equal slices of the whole clip
            edges = torch.linspace(0, self.max_episode_length, self.rsi_bins + 1)
            self.rsi_bin_hi = edges[1:].floor().long().clamp(max=startable)
            self.rsi_bin_lo = torch.minimum(edges[:-1].floor().long(), self.rsi_bin_hi - 1).clamp(min=0)
        if not bool((self.rsi_bin_hi > self.rsi_bin_lo).all()):
            raise ValueError("Every RSI start bin needs at least one start frame")
        widths = (self.rsi_bin_hi - self.rsi_bin_lo).tolist()
        print(f"RSI start bins: {self.rsi_bins} over frames 0-{startable - 1} of a {self.max_episode_length}-frame clip "
              f"({'fixed ' + str(self.rsi_bin_frames) + '-frame width' if self.rsi_bin_frames > 0 else 'legacy count'}; "
              f"widths {min(widths)}-{max(widths)})")
        self.rsi_top = torch.full((self.rsi_bins,), kp0, dtype=torch.float64)
        self.rsi_rate = torch.zeros(self.rsi_bins, dtype=torch.float64)
        self.rsi_n = torch.zeros(self.rsi_bins, dtype=torch.float64)
        self.rsi_f = torch.zeros(self.rsi_bins, dtype=torch.float64)
        self.rsi_keyframes = bool(self.rsi_keyframes)
        self.keyframes = None
        self.rsi_last_move = torch.zeros(self.rsi_bins, dtype=torch.long)
        self.rsi_env_bin = torch.zeros(self.num_envs, dtype=torch.long)
        self.rsi_env_help = torch.zeros(self.num_envs, dtype=torch.float64)   # bin gain top each lane's episode started with
        self.rsi_env_active = torch.zeros(self.num_envs, dtype=torch.bool)
        self.rsi_counts = dict(down=0, up=0, forced=0, decisions=0)
        self.rsi_rng = torch.Generator().manual_seed(int(self.seed))
        self.curr_gains = dict(kp=kp0, kv=kp0 * self.rsi_kv_per_kp, fr=self.rsi_fr)
        self.curr_gains_lower = dict(self.curr_gains)

    def _rsi_floor(self, t):
        """Assistance ceiling every bin must sit at or below at completed update t."""
        kp0 = float(self.init_gains["kp"])
        if t < self.rsi_deadline:
            return kp0 * (self.rsi_zero_below / kp0) ** (t / self.rsi_deadline)
        return self.rsi_zero_below * max(0.0, 1.0 - (t - self.rsi_deadline) / self.rsi_zero_ramp)

    def _rsi_start_probs(self):
        credit = torch.zeros(self.rsi_bins, dtype=torch.float64)
        acc = 0.0
        for b in range(self.rsi_bins - 1, -1, -1):
            acc = float(self.rsi_rate[b]) + self.rsi_credit_gamma * acc
            credit[b] = acc
        total = float(credit.sum())
        if total > 0:
            adaptive = credit / total
        else:
            adaptive = torch.full((self.rsi_bins,), 1.0 / self.rsi_bins, dtype=torch.float64)
        return self.rsi_uniform_mix / self.rsi_bins + (1.0 - self.rsi_uniform_mix) * adaptive

    def _rsi_sample_starts(self, n):
        bins = torch.multinomial(self._rsi_start_probs(), n, replacement=True, generator=self.rsi_rng)
        lo, hi = self.rsi_bin_lo[bins], self.rsi_bin_hi[bins]
        u = torch.rand(n, generator=self.rsi_rng, dtype=torch.float64)
        starts = lo + (u * (hi - lo).double()).floor().long()
        if self.rsi_keyframes:
            starts[bins == 0] = 0   # bin 0 starts from the evaluation's initial state: the root of every keyframe
        if self.rsi_frame0_share > 0:
            # Anchor: a fixed share of starts at the evaluation's own initial state. The bins judge success by survival, and
            # a policy that grabs late but recovers survives, so the opening can look solved while it is not (Mixer-170).
            anchor = torch.rand(n, generator=self.rsi_rng, dtype=torch.float64) < self.rsi_frame0_share
            bins[anchor] = 0
            starts[anchor] = 0
        return bins, starts.clamp(min=0, max=self.max_episode_length - self.rsi_min_horizon)

    def _rsi_apply(self, idx, bins):
        n, ndof = idx.numel(), self.obj_ndof
        top = self.rsi_top[bins]
        u = 1.0 / self.rsi_band + torch.rand((n, ndof), generator=self.rsi_rng, dtype=torch.float64) * (1.0 - 1.0 / self.rsi_band)
        device = self.object.device
        kp = (top[:, None] * u).to(device=device, dtype=torch.float32)
        kv = kp * self.rsi_kv_per_kp
        fr = (top[:, None] > 0).expand(n, ndof).to(device=device, dtype=torch.float32) * self.rsi_fr
        self.object.set_joint_gains(kp=kp, kv=kv, force_range=fr, env_idxs=idx.to(device))

    def rsi_on_reset(self, env_idxs, ended, failed):
        """Score the finished episodes by start bin, then draw new starts and per-bin gains."""
        idx = torch.as_tensor(env_idxs).long().cpu()
        if idx.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        ended = torch.as_tensor(ended).bool().cpu()
        failed = torch.as_tensor(failed).bool().cpu()
        scored = self.rsi_env_active[idx] & ended
        if scored.any():
            bins_e = self.rsi_env_bin[idx[scored]]
            n_b = torch.bincount(bins_e, minlength=self.rsi_bins).double()
            f_b = torch.bincount(bins_e, weights=failed[scored].double(), minlength=self.rsi_bins)
            self.rsi_n += n_b
            self.rsi_f += f_b
            has = n_b > 0
            w = 1.0 - (1.0 - self.rsi_fail_alpha) ** n_b[has]
            self.rsi_rate[has] = (1.0 - w) * self.rsi_rate[has] + w * (f_b[has] / n_b[has])
        bins, starts = self._rsi_sample_starts(idx.numel())
        self.rsi_env_bin[idx] = bins
        self.rsi_env_help[idx] = self.rsi_top[bins]
        self.rsi_env_active[idx] = True
        self._rsi_apply(idx, bins)
        return starts

    def _rsi_set_curriculum(self, t):
        t = int(t)
        kp0 = float(self.init_gains["kp"])
        before = self.rsi_top.clone()
        floor = self._rsi_floor(t)
        reason = (f"Epoch {t}: RSI tops [{float(self.rsi_top.min()):.4g}, {float(self.rsi_top.max()):.4g}]"
                  f" zero {int((self.rsi_top == 0).sum())}/{self.rsi_bins} floor {floor:.4g}")
        if t >= self.wait_epochs:
            for b in range(self.rsi_bins):
                if self.rsi_n[b] < self.rsi_min_count or t - int(self.rsi_last_move[b]) < self.rsi_min_gap:
                    continue
                success = 1.0 - float(self.rsi_f[b] / self.rsi_n[b])
                self.rsi_n[b] = 0.0
                self.rsi_f[b] = 0.0
                self.rsi_counts["decisions"] += 1
                top = float(self.rsi_top[b])
                if success >= self.rsi_success_up and top > 0:
                    new = top * self.rsi_step
                    self.rsi_top[b] = 0.0 if new < self.rsi_zero_below else new
                    self.rsi_counts["down"] += 1
                    self.rsi_last_move[b] = t
                elif success < self.rsi_success_down and top < kp0:
                    new = self.rsi_zero_below if top == 0 else min(top / self.rsi_step, kp0)
                    if new <= floor:
                        self.rsi_top[b] = new
                        self.rsi_counts["up"] += 1
                        self.rsi_last_move[b] = t
        # Deadline floor: only acts on bins a full step behind it, so it
        # never pre-empts success-driven moves that keep pace with the deadline.
        behind = self.rsi_top > (floor / self.rsi_step) if floor > 0 else self.rsi_top > 0
        if bool(behind.any()):
            self.rsi_top[behind] = max(floor, 0.0)
            self.rsi_counts["forced"] += 1
            self.rsi_last_move[behind] = t
            reason += f" | floor forced {int(behind.sum())} bins"
        zero_gains = bool((self.rsi_top == 0).all())
        if zero_gains:
            self.num_epoch_since_zero += 1
        moved = bool((self.rsi_top != before).any())
        top_max, top_min = float(self.rsi_top.max()), float(self.rsi_top.min())
        self.curr_gains = dict(kp=top_max, kv=top_max * self.rsi_kv_per_kp, fr=self.rsi_fr)
        self.curr_gains_lower = dict(kp=top_min, kv=top_min * self.rsi_kv_per_kp, fr=self.rsi_fr)
        return zero_gains, moved, reason

    def update_progress(self, rewards, achieved_length):
        self.deque_appends += 1
        if self.deque_appends % self.deque_append_freq == 0:
            for k, val in rewards.items():
                assert k in self.rew_deques, f"Invalid key: {k}"
                self.rew_deques[k].append(val)
            self.deque_appends = 0 
        # self.achieved_length = achieved_length
        self.ep_lens.append(achieved_length) 

    def get_current_gains(self):
        if self.schedule == "rsi":
            top_max, top_min = float(self.rsi_top.max()), float(self.rsi_top.min())
            return dict(
                kp=top_max, kv=top_max * self.rsi_kv_per_kp, fr=self.rsi_fr,
                kp_lower=top_min, kv_lower=top_min * self.rsi_kv_per_kp, fr_lower=self.rsi_fr,
                rsi_mean_top=float(self.rsi_top.mean()), rsi_bins_zero=float((self.rsi_top == 0).sum()),
                rsi_mean_fail=float(self.rsi_rate.mean()), rsi_max_fail=float(self.rsi_rate.max()),
                rsi_worst_bin=float(int(self.rsi_rate.argmax())), rsi_pending=float(self.rsi_n.sum()),
                rsi_moves_down=float(self.rsi_counts["down"]), rsi_moves_up=float(self.rsi_counts["up"]),
                rsi_forced=float(self.rsi_counts["forced"]),
                **({f"rsi_{k}": v for k, v in self.keyframes.stats().items()} if self.keyframes is not None else {}),
            )
        ret_gains = self.curr_gains.copy()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            ret_gains.update(
                {f"{k}_lower": v for k, v in self.curr_gains_lower.items()}
            )
        if self.decay_solimp:
            ret_gains.update(
                # dict(d0_lower=self.d0_lower, dmid_lower=self.dmid_lower)
                dict(tconst_lower=self.tconst_lower, tconst_upper=self.tconst_upper)
            )
        return ret_gains

    def reset_object_gains(self):
        rand_gains = dict()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            if any([self.curr_gains[key] == 0 for key in self.decay_terms]):
                rand_gains = {k: 0.0 for k in self.decay_terms}
            else:
                # sample for every joint dim and env dim! shape should be (num_envs, num_joints)
                for key in self.decay_terms:
                    lower = self.curr_gains_lower[key]
                    upper = self.curr_gains[key] 
                    # uniform from lower to upper, sample shape (num_envs, num_joints)
                    rand_gains[key] = torch.rand((self.num_envs, self.obj_ndof), dtype=torch.float32) * (upper - lower) + lower   
        else:
            # same params for all 
            for key in self.decay_terms:
                rand_gains[key] = self.curr_gains[key] 
        # this should handle missing shapes:
        self.object.set_joint_gains(
            kp=rand_gains.get("kp", None),
            kv=rand_gains.get("kv", None),
            force_range=rand_gains.get("fr", None),
            env_idxs=None,
            **({"zero_velocity": False} if self.schedule == "linear_updates" else {}),
        )
    
    def reset_solimp(self): 
        self.tconst_upper = max(self.tconst_upper * self.solip_multiplier, 0.02)
        self.tconst_lower = max(self.tconst_lower * self.solip_multiplier, 0.02)
        if self.tconst_upper == 0.02:
            tconst = 0.02
        else:
            tconst = np.random.uniform(self.tconst_lower, self.tconst_upper)
        new_params = torch.tensor([tconst,1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0]) 
        self.object.entity._solver.set_geom_sol_params(new_params)
        return 

    def get_reward_grads(self):
        return self.rew_grads

    def update_reward_grads(self):
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                continue 
            grad = np.gradient(self.rew_deques[key]).mean()
            self.rew_grads[key] = grad

    def decay_reward_weights(self, reward_module):
        decayed = False
        if not self.decay_rew:
            return decayed
        info = ""
        for attr in ['imi_rew_weight', 'bc_rew_weight', 'contact_rew_weight']:
            if hasattr(reward_module, attr):
                val = getattr(reward_module, attr)
                if val > 0.01:
                    setattr(reward_module, attr, val * 0.95)
                    info += f"{attr}: to {val * 0.95} "
                    decayed = True
        # also decay the rew thresholds
        for key in self.rew_thresholds.keys():
            self.rew_thresholds[key] = self.rew_thresholds[key] * 0.95
        if len(info) > 0:
            print(info)
        return decayed

    def set_curriculum(self, epoch_num):
        if self.schedule == "linear_updates":
            return self.set_linear_update_assistance(epoch_num)
        if self.schedule == "rsi":
            return self._rsi_set_curriculum(int(epoch_num))
        # if gains are already zero, no need to reset
        self.update_reward_grads()
        self.num_epoch_since_last_decay += 1
        learning_stablized, reason = self.determine_decay(epoch_num) 
        zero_gains = True 
        for key in self.decay_terms:
            if self.curr_gains[key] > 0.0:
                zero_gains = False
                break 
        # add to the count of epochs since last zero gains
        if zero_gains:
            self.num_epoch_since_zero += 1 
        gains_decayed = False 
        if epoch_num > self.zero_epoch:
            if not zero_gains:
                # set all gains to zero
                for key in self.decay_terms:
                    self.curr_gains[key] = 0.0
                    self.curr_gains_lower[key] = 0.0
                reason += f"Epoch {epoch_num}: Gains set to zero"  
                gains_decayed = True 
                if self.ablation_protocol == 'dexmachina':
                    # The native hard-stop used to update only Python bounds
                    # and return before the simulator received the zero gains.
                    # Preserve that historical behavior for legacy runs.
                    self.reset_object_gains()
                    zero_gains = True
            return zero_gains, gains_decayed, reason 
        
        if zero_gains or not learning_stablized:
            return zero_gains, gains_decayed, reason 
        
        # decide if need to dial back the gains
        dialed_back = self.determine_dialback(epoch_num) 
        if dialed_back:
            print(f"Epoch {epoch_num}: Dialed back gains")
            gains_decayed = True # if the gains are dialed back, don't append to the history
        else:
            self.gain_history.append(
                dict(gains=self.curr_gains.copy(), lower=self.curr_gains_lower.copy())
            )

            if self.schedule == "fixed":
                gains_decayed, reason = self.set_fixed_decay(epoch_num)
            elif self.schedule == "exp":
                gains_decayed, reason = self.set_auto_exp_decay(epoch_num)
            elif self.schedule == "uniform":
                gains_decayed, reason = self.set_auto_uniform_decay(epoch_num)
            else:
                raise ValueError("Invalid schedule")

        if gains_decayed:
            new_gains = self.get_current_gains()
            print(f"Epoch {epoch_num}: New Gains: {new_gains}")
            self.reset_object_gains() 
            # also clear the deques!!
            for key in self.rew_deques.keys():
                self.rew_deques[key].clear()
                self.rew_grads[key] = 1.0
            self.ep_lens.clear()
            if self.decay_solimp:
                self.reset_solimp()
            self.deque_appends = 0 
            self.num_epoch_since_last_decay = 0
        elif self.resample_every_epoch > 0 and epoch_num % self.resample_every_epoch == 0:
            self.reset_object_gains()
        return zero_gains, gains_decayed, reason

    def set_linear_update_assistance(self, completed_updates):
        """Apply exact gains from completed updates without score gates or dialback."""
        if type(completed_updates) is not int or completed_updates < 0:
            raise ValueError("Assistance progress must be a nonnegative completed-update count")
        alpha = max(0.0, 1.0 - completed_updates / self.zero_epoch) if self.zero_epoch else 0.0
        gains = {key: initial * alpha for key, initial in self.init_gains.items()}
        changed = gains != self.curr_gains
        self.curr_gains = gains
        self.curr_gains_lower = gains.copy()
        self.num_epoch_since_zero = max(0, completed_updates - self.zero_epoch)
        if changed:
            self.reset_object_gains()
        return alpha == 0.0, changed, f"Completed updates {completed_updates}: fixed assistance alpha={alpha:.8f}"
    
