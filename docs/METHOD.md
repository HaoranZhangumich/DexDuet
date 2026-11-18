# DexDuet: method and implementation notes

This document maps DexDuet's parts to the code and flags, and describes the evaluation protocol and the issues in Genesis
and in DexMachina's released code that the paper's numbers depend on (sections 3, 4, 5 and 7). The paper's method section and supplementary have the full description.
Reproduction commands are in [../REPRODUCE.md](../REPRODUCE.md); the trainer options of the paper's runs are listed in
[OPTIONS.md](OPTIONS.md); setting up a machine is [SETUP_NEW_MACHINE.md](SETUP_NEW_MACHINE.md).

## 1. What DexDuet changes

DexDuet trains a bimanual policy from one ARCTIC human demonstration in Genesis, on top of DexMachina. It changes the four
parts below and keeps everything else from DexMachina: the retargeting, the per-task reward weights (including DexMachina's
own contact reward), the PPO settings, the stopping rule and the checkpoint selection (best training reward).

| Part | What it is | Options |
|---|---|---|
| Per-hand policy | The bimanual policy is factorized into two per-hand policies that share one actor network. Each hand's actor sees its own observation and the other hand's state and previous action (the partner input, encoded by a shared partner encoder). PPO clips each hand's probability ratio separately; the advantages come from two value heads of a centralized critic. | `--coordinate_actor --per_hand_ppo --value_mode per_hand` (with `--ablation_protocol dexmachina`, which per-hand PPO needs) |
| Assisted reference-state initialization (A-RSI) | Episodes start throughout the demonstration. Each start segment has its own level of object assistance (DexMachina's virtual object controller), which is withdrawn as episodes from that segment succeed, under a deadline that reaches zero assistance before the end of training. Mid-demonstration episodes start from simulator states that earlier assisted episodes reached (keyframes); a keyframe is kept only once its episode reaches the end of the demonstration. Start segments are drawn more often where episodes fail, and a quarter of the starts are at frame 0, where evaluation starts. | `--curr_schedule rsi --rsi_keyframes --rsi_grace_bank 8 --rsi_kf_confirm 30 --rsi_kf_to_end --rsi_frame0_share 0.25`; segments of 15 frames (`--rsi_bin_frames 15`) on Box-200, Ketchup-100, Mixer-170, Mixer-300 and Waffleiron-300, 20 equal segments on Ketchup-300 and Notebook-300 |
| Hand graph reward (weight 0.5) | Each hand's palm normal, palm direction and finger directions, and its fingertip gaps, compared with the human (MANO) hand in the object frame. | `--hand_graph_weight 0.5` |
| Hand-object contact reward (weight 1.0) | For each region of each hand (palm, thumb, each other finger) and each object part: the distance to the part and the closest point on it, robot against human, scored with a tolerance. Pairs in which the robot or the human touches the part weigh most. | `--ho_contact_weight 1.0` (assets from `dexmachina/hand_proc/ho_contact_assets.py`) |

Evaluation adds **Contact-AUC** (`dexmachina/eval/slip_record.py` records a rollout, `dexmachina/eval/slip_auc.py` scores
it). For every window in which a human hand holds an object part without sliding, it takes the contact drift of the same
robot hand on that part (how far its touching keypoints move in the part's frame over 6 steps) and scores the share of
thresholds from 0.5 to 5 cm that the drift stays below. A window in which the robot hand does not touch the part, or in which
the object is lost, scores 0. Contact-AUC is kinematic: it checks whether the robot keeps the human's contacts, and it is
not a measure of how much disturbance a grasp resists.

`scripts/train_coordinates_hand.py --task <task>` builds the exact training command of the paper's runs, trains, then
evaluates the run and scores ADD-AUC and Contact-AUC.

### Names in the code

Flags, file names and run-name fragments map to the paper's terms as follows:

| In the code, flags and run names | In the paper |
|---|---|
| `coordinates-hand` (launcher file name, default run names, W&B project) | DexDuet |
| `--coordinate_actor`, `--coord_ablation_id per-hand`, `factmsg`, `coordinate_actor_critic`, "partner message" | per-hand policy, partner input |
| `--per_hand_ppo`, `--value_mode per_hand`, `handppo` | per-hand PPO and the two value heads |
| `--curr_schedule rsi`, `rsikfc30_kfend_f0s0.25`, help, bins | A-RSI, assistance, segments |
| `--curr_schedule uniform`, VOC, `--ablation_protocol dexmachina` | DexMachina's single-level curriculum (virtual object controller) |
| `hg0.5`, `--hand_graph_*` | hand graph reward |
| `hoc1`, `--ho_contact_*` | hand-object contact reward |
| `slip_record.py`, `slip_auc.py`, `slip_*` keys, `slip/` folders | Contact-AUC |
| `baselines/track2_metrics.py` | SP-SR, MP-SR and MPPE with the V2D challenge kit's definitions |
| rows a-g, `abl-<row>-<label>` run names (`noir`: no interaction rewards) | the rows of the ablation table |

Run folders and commands written before the per-hand options were renamed use older names (`wrench_cfg` in env.yaml,
`nn/wrench_final.pth` for the terminal snapshot, `_cwsteam_handppo_credit0` in run names, and `--wrench_per_hand_ppo`,
`--wrench_value_mode` and zero-weight `--wrench_*` options in commands); the code still loads these folders and accepts
these option names.

## 2. Other hands

`--hand` selects the robot hand (`inspire_hand` by default; `allegro_hand`, `xhand`, `schunk_hand`, and in code also
`dex3_hand`, `ability_hand`, `dexrobot_hand`). Both interaction rewards follow the finger correspondence of the hand's
retargeting (`dexmachina/envs/finger_spec.py`, from `assets/<hand>/retarget_config.yaml` and the URDF):

- Each robot finger is matched to the human finger it is retargeted from. Human fingers without a robot finger (Allegro's
  little finger; Dex3's ring and little fingers) are left out of the finger edges, the fingertip gaps, the contact regions
  and their normalisations.
- The fingertip-gap scale is the hand's retargeting scale (Inspire 1.2, XHand 1.4, Schunk 1.3, Allegro 1.7, Dex3 1.5).
- The contact tolerances are the same for every hand.
- On the long demonstrations (Ketchup-300, Notebook-300, Waffleiron-300) the other hands use DexMachina's non-Inspire
  weights from its released configs (imitation 0.2, contact 2.0, behavior cloning 0.2). The paper's Schunk runs use
  behavior cloning 0.3 there (`--bc-weight 0.3`, which `scripts/reproduce_paper.sh main` sets for Schunk).

Palm normal (`finger_spec.palm_normal_mode`, threshold `PALM_NORMAL_MIN_FINGERS = 4`). A hand with four or more fingers
uses the outermost present finger bases on both hands (Inspire index x little, Allegro index x ring). A hand with fewer
fingers (Dex3: thumb, index, middle) has no knuckles that span the palm: the human index and middle knuckles are only
2.2-2.5 cm apart, and the retargeting, which matches three fingertips, rolls the robot palm 30-50 degrees relative to the
human's. There, the human normal is the true palm normal (index base x little base), and the robot normal is a fixed axis a
of the palm link (`wrist_link_name`), turned with it: R_wrist(t) a. The axis is calibrated once per clip on the run's own
retargeted reference, a = normalize(sum_t R_ref(t)^T n_human(t)), the axis with the largest summed cosine to the human normal
(`rewards.calibrate_palm_axis`; float64, no randomness). It is stored in `reward_cfg['hand_graph_palm_axis']`, so the run's
`params/env.pkl` carries it and evaluation uses the stored axis. Median palm-normal angle between the retargeted reference
and the human hand on Box-200 (`scripts/reference_check.py`, left / right): Dex3 7.9 / 15.2 degrees (65 / 46 with the finger
rule), Allegro 8.3 / 18.2, Inspire 12.5 / 21.2.

The Dex3 environment builds and steps with both rewards (`scripts/hand_env_check.py`), but no Dex3 result comes from this
repository; the G1 + Dex3 study runs on the V2D challenge's own runtime, in a separate repository (README).

The other-hand code leaves the Inspire path unchanged: its rewards, contact anchor points, launcher commands and the random
state after the CPU part of the environment construction are identical to those of the Inspire-only reward code that the
paper's Inspire runs used, from before the other-hand code was added (`tests/test_inspire_golden.py`, against a golden file
that holds ARCTIC-derived data and is not included; `tests/test_launcher_hand.py`; on the GPU,
`scripts/shadow_reward_replay.py` replays 300 steps and compares 51 reward outputs bit for bit).

## 3. Evaluation protocol

Every method is evaluated the same way (`dexmachina/rl/eval_rl_games.py`):

- 100 parallel rollouts from the first frame, all controlled by the policy (`--all_policy_envs`), deterministic actions, no
  assistance, evaluation seed 1042, Genesis with the collision-pair fix (section 4);
- the checkpoint with the best training reward (`nn/<hand>.pth`);
- the first observation re-read from the simulator after the reset (below);
- a frozen start state per hand and demonstration (`--eval-reset-manifest`): the first evaluation writes
  `evaluation_reset.json` and `.npz`, and every later evaluation must reproduce them byte for byte before its first action.
  DexMachina's released checkpoints carry their own retargeted reference, so they get their own start states.

**First observation after a reset.** DexMachina's reset leaves stale values in the first observation: `object.reset_idx`
zeroes the cached part poses, and `robot.reset_idx` reads the wrist position from the keypoint array with a link index (about
11 cm off). The physical start state is correct; only the observation is wrong until the simulator is read again. The
evaluator always re-reads the simulator before the first action. The original stale mode was removed; `--first-obs` is still
accepted so older commands parse, and always means corrected.

**Last evaluation environment.** DexMachina's released evaluator leaves the last of its 100 environments uncontrolled and
starts it from a zero hand pose. Here the policy controls every environment. As in the paper, DexMachina's checkpoints keep
the zero-pose start of the last lane (`--zero-pose-last-lane` of `eval_rl_games.py`, `slip_record.py` and the hold test,
which `scripts/reproduce_paper.sh` sets for them), and MPPE is reported over lanes 0-98. Every other run, including ablation
row (e), starts all 100 lanes from the reference start. For Contact-AUC, when DexMachina's zero-pose lane resets early,
`slip_record.py --allow-last-lane-reset` keeps recording with that lane's states marked as lost.

Metrics: ADD-AUC (`dexmachina/eval/compute_add.py`, DexMachina's code unchanged), SP-SR, MP-SR and MPPE
(`baselines/track2_metrics.py`, the V2D challenge kit's definitions; `baselines/test_track2_metrics.py` checks it against the
kit), Contact-AUC (`slip_record.py` + `slip_auc.py`).

## 4. Genesis collision-pair overflow fix

Genesis' broad phase writes each environment's candidate collision pairs into a buffer with min(possible pairs after the
collision-group filter, `max_collision_pairs`) slots per environment, without a bounds check. DexMachina sets
`max_collision_pairs=100` (`dexmachina/envs/base_env.py`), and the runtime pair check does not apply the group filter, so an
environment can produce more pairs than it has slots (during evaluation, the number of pairs peaks at 111 to 157 on six of
the seven demonstrations). The extra writes land in other
environments' slots: one environment's contacts corrupt another's, and results depend on what the other environments do.
Waffleiron-300 is the most affected task.

The fix, `environment/patches/genesis/0001-*.patch` (Genesis commit `fca4168` on top of MandiZhao/Genesis `e9df2d8`):

- the pair buffer gets the exact number of pairs the runtime rules can emit, independent of `max_collision_pairs`;
- the contact capacity and the constraint buffers derived from it are unchanged, so runs without overflow stay bit-identical;
- every pair and contact write is bounds-checked, and dropped items are counted per environment;
- `Collider.check_overflow()` raises with the environment ids and counts if anything was dropped. It runs every 50 detection
  calls and at every reset.

`scripts/setup_dexmachina_env.sh` applies it with the committer identity that reproduces the pinned hash. To check a
checkout, from the repository root:

```bash
git -C Genesis rev-parse HEAD                  # must print fca4168f562b0762130932cda68ba09d011336ae
python scripts/preflight.py --stage sources    # also checks that the imported Genesis has Collider.check_overflow
```

The launcher and `scripts/reproduce_paper.sh` refuse to run on a Genesis without `Collider.check_overflow`.

## 5. Reset observation fix in training

The stale first observation of section 3 also affects training: DexMachina's reset leaves the cached object part poses and
wrist position stale, so every training episode starts from an observation that does not match its (correct) physical
start. With the fix, `base_env.reset_idx` re-reads robots and
objects after every reset (`env_cfg['refresh_after_reset']`, on by default). `--legacy_reset_obs` restores the old reset.

## 6. Which code trained the paper's runs

The paper's runs were trained as follows:

- Training uses Genesis 0.2.1 (commit `e9df2d8`) or the same build with the patch of section 4 (`fca4168`). DexDuet's Inspire
  checkpoints of the main table are trained on `e9df2d8`, with the stale reset observation of section 5 at starts that do not
  restore a keyframe, except seven: the five Waffleiron-300 runs are trained on `fca4168` with the same reset, and the
  Mixer-170 and Mixer-300 runs with seed 66 on `fca4168` with the reset fix. Its checkpoints for the other three hands and
  the ablation rows (b)-(g) are trained on `fca4168` with the reset fix. A run's `config/env.yaml` shows the reset it used
  (`env_cfg.refresh_after_reset`; absent in the runs with the stale reset).
- Every evaluation on the DexMachina suite uses `fca4168`.

This repository trains on `fca4168` with the reset fix, so a run trained with it matches the conditions of the other hands
and of the ablation rows. `--legacy_reset_obs`, added to the training command that `--dry-run` prints, restores the old
reset; most of the paper's Inspire runs also used the unpatched Genesis, which this repository does not train on.

## 7. End of DexMachina's curriculum

DexMachina's curriculum withdraws the remaining assistance by a hard stop 1,000 updates before the update cap (500 on the
demonstrations with a 5,000-update cap). In the released code, this stop sets the assistance gains to zero in Python but
does not pass them to the simulator. In the released runs whose curriculum has stalled, as on Ketchup-300, the object
therefore stays assisted until the end of training, so the released checkpoints of these runs are never trained without
assistance, although they are evaluated without it.

The paper's DexMachina rows evaluate the authors' released checkpoints as they are. When this repository trains with
DexMachina's curriculum (`--curr_schedule uniform` with `--ablation_protocol dexmachina`, as ablation rows (f) and (g) do),
the hard stop also sends the zero gains to the simulator (`dexmachina/envs/curriculum.py`, `set_curriculum`), so the last
updates of those runs are unassisted.
