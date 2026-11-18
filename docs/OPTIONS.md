# Trainer options

The options of `dexmachina/rl/train_rl_games.py` that the paper's runs set. `scripts/train_coordinates_hand.py` sets all of
them for DexDuet, and `baselines/ablation_rows.py` changes them for each ablation row. [METHOD.md](METHOD.md) describes the
method and maps the names used in the code to the paper's. The trainer keeps further options from development; every run of
the paper leaves them at their defaults, which switch them off.

## Per-hand policy

| Option | Default | What it does |
|---|---|---|
| `--coordinate_actor` | off | Two per-hand policies that share one actor network; each sees its own observation and the partner input. |
| `--per_hand_ppo` | off | PPO clips the probability ratio of each hand separately (needs `--ablation_protocol dexmachina`). |
| `--value_mode per_hand` | per_hand with `--per_hand_ppo`, else scalar | Two value heads of the centralized critic, one per hand. |
| `--coord_ablation_id per-hand` | none | A label of the run variant, stored in W&B and in the run's `params/env.yaml`; it does not change training. |
| `--ablation_protocol dexmachina` | legacy | DexMachina's PPO settings and stopping rule, run through the learner that supports per-hand PPO. The best-training-reward checkpoint is recorded in `final_checkpoint.json`, which evaluation checks, and the curriculum's final zero gains reach the simulator ([METHOD.md](METHOD.md)). |

Without these options the trainer uses DexMachina's unified policy, as in ablation row (e).

## A-RSI

| Option | Default | What it does |
|---|---|---|
| `--curr_schedule rsi` | uniform | Episodes start throughout the demonstration, and each start segment has its own level of assistance, lowered as episodes from that segment succeed. `uniform` is DexMachina's single-level curriculum, as in ablation rows (f) and (g). |
| `--rsi_bin_frames 15` | 0 (20 equal segments) | Segments of 15 frames. The paper uses it on Box-200, Ketchup-100, Mixer-170, Mixer-300 and Waffleiron-300, and 20 equal segments on Ketchup-300 and Notebook-300. |
| `--rsi_keyframes` | off | Episodes in the middle of the demonstration start from simulator states that earlier episodes reached (keyframes). |
| `--rsi_kf_to_end` | off | A state is kept as a keyframe only once its episode reaches the end of the demonstration, and it is served only while its segment gives at least the assistance that episode had. |
| `--rsi_kf_confirm 30` | 0 | A state is kept only after its episode continues for 30 more frames. `--rsi_kf_to_end` takes precedence. |
| `--rsi_grace_bank 8` | 0 | Task reward credited to the early-reset check of episodes that start in the middle of the demonstration. |
| `--rsi_frame0_share 0.25` | 0 | A quarter of the starts are at the first frame, where evaluation starts. |

## Interaction rewards

| Option | Default | What it does |
|---|---|---|
| `--hand_graph_weight 0.5` | 0 | Hand graph reward: palm normal, palm direction, finger directions and fingertip gaps of each hand against the human hand, in the object frame. `--hand_graph_beta_dir` 5, `--hand_graph_beta_gap` 20, `--hand_graph_thumb` 0.5; the fingertip-gap scale is the hand's retargeting scale. |
| `--ho_contact_weight 1.0` | 0 | Hand-object contact reward: for the palm and each finger of each hand and each object part, the distance to the part and the closest point on it against the human's. `--ho_contact_cap` 0.05, `--ho_contact_tol` 0.01, `--ho_contact_scale` 0.02, `--ho_contact_touch` 0.01, `--ho_contact_near` 0.5 (metres, except the last, a weight). Needs the assets built by `dexmachina/hand_proc/ho_contact_assets.py`. |

Both rewards follow the finger correspondence of the hand's retargeting (`dexmachina/envs/finger_spec.py`), so a hand with
fewer fingers than the human's uses only the matched fingers.

## Evaluation (`dexmachina/rl/eval_rl_games.py`)

| Option | What it does |
|---|---|
| `--all_policy_envs` | The policy controls all 100 rollouts. |
| `--eval-seed 1042` | Evaluation seed. |
| `--eval-reset-manifest` | The frozen start state per hand and demonstration, checked byte for byte before the first action. |
| `--zero-pose-last-lane` | For DexMachina's released checkpoints: the last rollout starts from DexMachina's zero hand pose, as in their evaluator. |
| `--accept-selection` | Accepts a checkpoint that was not selected by the best training reward (two Inspire runs, REPRODUCE.md, section 2.2). |

Contact-AUC: `python -m dexmachina.eval.slip_record` records an evaluation rollout, and `python -m dexmachina.eval.slip_auc
<recording> --mano <MANO export>` scores it.
