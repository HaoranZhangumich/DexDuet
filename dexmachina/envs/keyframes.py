"""Reached-state keyframes for RSI starts.

Per start bin, a ring buffer of full simulator states that training lanes actually reached: every robot's and the
object's qpos and dof velocities, the robots' PD targets and the lane's last action. RSI starts in a bin with enough
stored states resume from one of them instead of the retargeted reference pose, so the policy starts in its own
posture, contacts and velocities. Bin 0 keeps the evaluation's frame-0 initial state.
Only lanes with a genuine history are captured: those that started at frame 0 or from a stored state. Lanes started
from the retargeted pose spend their first steps recovering from it, so their states are never stored.

Confirmed capture (confirm > 0): the object reward lags a lost grasp by ten frames or more, so a state can look healthy
when its episode is already lost, and such states can fill a bin. A picked
state is therefore held back and stored only once its lane has gone on for `confirm` more frames (or reached the end of
the clip) with the task reward over that stretch still at min_task_rew; if the lane fails first, the state is dropped.

Confirmed to the end (to_end=True): a picked state is stored only when its lane reaches the end of the clip, with the task
reward from the picked frame to the end still at min_task_rew, so a short confirmation window cannot let through states that
fail later (for example, states just before a two-hand turn that pass 30 frames and then fail the turn). Each state
also records the assistance (bin gain top) its lane had, and is served only while its bin's assistance is at least that: a
state reached with help is not handed to a lane that has less help than it needed.
"""
import torch


class KeyframeBuffer:
    def __init__(self, env, bin_lo, bin_hi, capacity=256, capture_prob=0.002, min_fill=32, min_task_rew=0.6,
                 min_stepped=15, min_horizon=10, confirm=0, to_end=False, seed=0):
        self.env, self.device = env, env.device
        self.bin_lo = torch.as_tensor(bin_lo, dtype=torch.long, device=self.device)
        self.bin_hi = torch.as_tensor(bin_hi, dtype=torch.long, device=self.device)
        self.nb, self.cap = len(bin_lo), int(capacity)
        self.capture_prob, self.min_fill, self.min_task_rew = float(capture_prob), int(min_fill), float(min_task_rew)
        self.min_stepped, self.min_horizon = int(min_stepped), int(min_horizon)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.entities = [(f'robot_{k}', r.entity, r) for k, r in env.robots.items()]
        self.entities += [(f'object_{k}', o.entity, None) for k, o in env.objects.items()]
        n = env.num_envs
        self.buf = {}
        for name, ent, robot in self.entities:
            nq, nd = ent.get_qpos().reshape(n, -1).shape[1], ent.get_dofs_velocity().reshape(n, -1).shape[1]
            self.buf[f'{name}/qpos'] = torch.zeros(self.nb, self.cap, nq, device=self.device)
            self.buf[f'{name}/vel'] = torch.zeros(self.nb, self.cap, nd, device=self.device)
            if robot is not None:
                self.buf[f'{name}/targets'] = torch.zeros(self.nb, self.cap, robot.curr_targets.shape[1], device=self.device)
        self.buf['actions'] = torch.zeros(self.nb, self.cap, env.action_dim, device=self.device)
        self.to_end = bool(to_end)
        if self.to_end:
            self.buf['help'] = torch.zeros(self.nb, self.cap, 1, device=self.device)
            confirm = 10 ** 9            # hold every picked state until its lane ends; only a finished clip commits it
        self.frame = torch.zeros(self.nb, self.cap, dtype=torch.long, device=self.device)
        self.count = torch.zeros(self.nb, dtype=torch.long, device=self.device)
        self.ptr = torch.zeros(self.nb, dtype=torch.long, device=self.device)
        self.captured_total, self.served, self.requested = 0, 0, 0
        self.clean = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.confirm, self.dropped_total = int(confirm), 0
        if self.confirm > 0:   # one held-back state per lane
            self.pend = {k: torch.zeros(n, v.shape[2], device=self.device, dtype=v.dtype) for k, v in self.buf.items()}
            self.pend_frame = torch.zeros(n, dtype=torch.long, device=self.device)
            self.pend_cum = torch.zeros(n, device=self.device)
            self.pend_on = torch.zeros(n, dtype=torch.bool, device=self.device)

    def bin_of(self, frame):
        return (torch.bucketize(frame, self.bin_hi, right=True)).clamp(max=self.nb - 1)

    @torch.no_grad()
    def capture(self, step_task_rew):
        """Store a random share of healthy, physically evolved lanes; call after rewards, before resets."""
        env = self.env
        frame = env.episode_length_buf.long()
        stepped = frame - env.episode_start_buf.long()
        healthy = self.clean & (~env.reset_buf) & (~env.nan_envs) & (stepped >= self.min_stepped)
        healthy &= frame <= env.max_episode_length - self.min_horizon
        healthy &= step_task_rew >= 0.3
        healthy &= env.cumulative_task_rew / stepped.clamp(min=1) >= self.min_task_rew
        pick = healthy & (torch.rand(env.num_envs, generator=self.gen, device=self.device) < self.capture_prob)
        if self.confirm > 0:
            self._resolve_pending(step_task_rew, frame)
            pick &= ~self.pend_on
        idx = pick.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        bins = self.bin_of(frame[idx])
        keep = bins > 0
        idx, bins = idx[keep], bins[keep]
        if idx.numel() == 0:
            return
        states = {}
        for name, ent, robot in self.entities:
            states[f'{name}/qpos'] = ent.get_qpos(envs_idx=idx).reshape(idx.numel(), -1)
            states[f'{name}/vel'] = ent.get_dofs_velocity(envs_idx=idx).reshape(idx.numel(), -1)
            if robot is not None:
                states[f'{name}/targets'] = robot.curr_targets[idx]
        states['actions'] = env.actions[idx]
        if self.to_end:
            states['help'] = env.curriculum.rsi_env_help[idx.cpu()].to(device=self.device, dtype=torch.float32)[:, None]
        if self.confirm > 0:
            for key, val in states.items():
                self.pend[key][idx] = val.to(self.pend[key].dtype)
            self.pend_frame[idx] = frame[idx]
            self.pend_cum[idx] = env.cumulative_task_rew[idx]
            self.pend_on[idx] = True
            return
        self._store(states, bins, frame[idx])

    @torch.no_grad()
    def _resolve_pending(self, step_task_rew, frame):
        """Store the held-back states whose lanes went on well, drop those whose lanes failed or drifted."""
        env, on = self.env, self.pend_on
        if not bool(on.any()):
            return
        elapsed = frame - self.pend_frame
        window = (env.cumulative_task_rew - self.pend_cum) / elapsed.clamp(min=1)
        good = (window >= self.min_task_rew) & (~env.nan_envs)
        finished = env.reset_buf & env.reset_time_outs                      # reached the end of the clip
        survived = (~env.reset_buf) & (elapsed >= self.confirm) & (step_task_rew >= 0.3)
        if self.to_end:
            survived = torch.zeros_like(survived)                          # only the end of the clip confirms
        commit = on & good & (finished | survived)
        drop = on & (~commit) & (env.reset_buf | (elapsed >= self.confirm))
        idx = commit.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            self._store({key: val[idx] for key, val in self.pend.items()}, self.bin_of(self.pend_frame[idx]), self.pend_frame[idx])
        self.dropped_total += int(drop.sum())
        self.pend_on[commit | drop] = False

    def _store(self, states, bins, fr):
        for b in bins.unique().tolist():
            m = (bins == b).nonzero(as_tuple=False).squeeze(-1)
            k = m.numel()
            slots = (self.ptr[b] + torch.arange(k, device=self.device)) % self.cap
            for key, val in states.items():
                self.buf[key][b, slots] = val[m].to(self.buf[key].dtype)
            self.frame[b, slots] = fr[m]
            self.ptr[b] = (self.ptr[b] + k) % self.cap
            self.count[b] = torch.clamp(self.count[b] + k, max=self.cap)
        self.captured_total += int(bins.numel())

    @torch.no_grad()
    def sample(self, bins):
        """For lanes starting in `bins` (bin 0 excluded): which can use a keyframe, its frame and slot."""
        bins = bins.to(self.device).long()
        if self.to_end:
            # only states reached with no more assistance than the bin gives now
            top = self.env.curriculum.rsi_top.to(device=self.device, dtype=torch.float32)
            filled = torch.arange(self.cap, device=self.device)[None, :] < self.count[:, None]
            valid = filled & (self.buf['help'][:, :, 0] <= top[:, None] + 1e-6)                 # (nb, cap)
            have = (bins > 0) & (valid.sum(dim=1)[bins] >= self.min_fill)
            w = valid[bins].float()
            w[w.sum(dim=1) == 0] = 1.0                                                          # unused rows (have is False)
            slot = torch.multinomial(w, 1, replacement=True, generator=self.gen).squeeze(-1) if bins.numel() else bins
        else:
            have = (bins > 0) & (self.count[bins] >= self.min_fill)
            slot = (torch.rand(bins.numel(), generator=self.gen, device=self.device) * self.count[bins].clamp(min=1)).long()
        frames = self.frame[bins, slot]
        self.requested += int((bins > 0).sum()); self.served += int(have.sum())
        return have, frames, bins, slot

    @torch.no_grad()
    def restore(self, env_idxs, bins, slot):
        """Overwrite the (already reset) lanes with stored states and refresh cached buffers."""
        env = self.env
        idx = torch.as_tensor(env_idxs, device=self.device).long()
        if idx.numel() == 0:
            return
        for name, ent, robot in self.entities:
            ent.set_qpos(self.buf[f'{name}/qpos'][bins, slot], envs_idx=idx, zero_velocity=False)
            ent.set_dofs_velocity(self.buf[f'{name}/vel'][bins, slot], envs_idx=idx)
            if robot is not None:
                t = self.buf[f'{name}/targets'][bins, slot]
                robot.curr_targets[idx] = t
                robot.prev_targets[idx] = t
        env.actions[idx] = self.buf['actions'][bins, slot]
        env.last_actions[idx] = self.buf['actions'][bins, slot]
        for _, _, robot in self.entities:
            if robot is not None:
                robot.update_value_buffers()
        for obj in env.objects.values():
            obj.update_value_buffers()

    def mark_lineage(self, env_idxs, starts, have):
        idx = torch.as_tensor(env_idxs, device=self.device).long()
        self.clean[idx] = (starts.to(self.device) == 0) | have.to(self.device)

    def stats(self):
        out = dict(kf_bins_ready=float((self.count >= self.min_fill).sum()), kf_captured=float(self.captured_total),
                   kf_clean_frac=float(self.clean.float().mean()),
                   kf_served_frac=float(self.served / max(1, self.requested)))
        if self.confirm > 0:
            out['kf_dropped_frac'] = float(self.dropped_total / max(1, self.dropped_total + self.captured_total))
        return out
