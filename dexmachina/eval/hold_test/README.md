# Holding after the last frame

The hold test of the paper (main text, and the supplementary section "Holding After the Last Frame"). Each policy
replays its scored evaluation, and after the last frame it keeps acting for 180 steps (3 s) without a reset, with every
reference frozen at the last frame. A run holds the object if the object stays within 5 cm of its final pose for the
whole 3 s, and drops it if the object ends the 3 s more than 10 cm below its final pose or off the table. The paper
reports the share of rollouts that hold the object (`SR@5` below), the share that drop it (`Drop`) and the number of
runs that end more than 3 cm from the final pose (off track). Both DexDuet and DexMachina are trained only on episodes
that end at the last frame, so the hold lies outside the training distribution of both.

| File | What |
|---|---|
| `run.py` | the rollout: the scored evaluation replayed, then the hold; mechanics checks (`hold_eval.json`), `hold_rollout.npz`, `eval100/eval_ep0.npy` |
| `add.py` | the official part ADD (`dexmachina/eval/compute_add.py`) of every hold frame against the final reference (`hold_add.npz`) |
| `metrics.py` | per-run metrics and the replay checks (`hold_metrics.json`) |
| `analyze.py` | aggregation over runs, hands and training seeds, and the tables (`TABLES.md`, `per_run.csv`, `aggregates.csv`) |

The rollout mechanics, the metric functions and the aggregation are those that produced the paper's tables, with
command-line inputs.

## Protocol

**Rollout** (`run.py`). The scored 100-lane evaluation is replayed exactly as `dexmachina/rl/eval_rl_games.py` runs it: the
run's `params/env.pkl`, every lane policy-controlled, evaluation seed 1042, deterministic actions, corrected first
observation, no early reset, no randomisation, object not actuated, and the start state checked byte for byte against a copy
of the evaluation's start-state manifest. DexMachina's checkpoints keep their zero-pose last lane
(`--zero-pose-last-lane`), as in their evaluation. Before the last clip step, a patch pins the step counters of the
environment, robots and object to the last frame L-1, so every reference (wrist residual target, the object goal in the
observation, the phase input, the reward targets) stays at the last frame, and disables the time-out; lanes with a non-finite
state are reset by the evaluator as before and count as failures. The rollout then continues for H = 180 steps (one step is
1/60 s). Recorded per step and lane: the object state, the physical pre-reset object state, the hand-object contacts and the
reset flags.

**Replay checks** (`metrics.replay_checks`): the start state equals the manifest; clip rows 0..L-2 equal the scored rollout
bit for bit; the pre-reset state of the last clip step equals the scored one where the scored outputs hold it; during the
patched steps every counter is L-1 and every reference is the last frame; no reset of a finite lane; no collision-pair
overflow. A run that fails a check is flagged ("replay not identical"). Flagged runs stay in the tables, and
`aggregates.csv` also gives every value without them.

**Metrics** (`add.py`, `metrics.py`), against the frozen final reference g:

- **SR@5** (the paper's hold rate) and SR@3: share of lanes whose object ADD stays within 5 / 3 cm for the whole hold.
- **Drop** (the paper's drop rate): share of lanes whose object ends more than 10 cm below g, or below 0.60 m (off the
  table) while g is above it.
- **Hold-ADD-AUC**: ADD-AUC (500 mesh vertices per part, thresholds 1 to 9 cm, both parts) over all (hold step, lane)
  pairs, h = 1..180. A non-finite lane counts as ADD = infinity. The paper does not report it.
- Drift at 1, 2 and 3 s (position, rotation, hinge, ADD, displacement since the clip end) and the share of hold steps with
  hand-object contact (any link, finger links, per hand).
- **On track**: the same over the lanes whose object is within 3 cm (ADD) of g at the last frame. A run with no such lane
  has no on-track value. Lanes are identical within a run in this evaluation, apart from the zero-pose last lane of DexMachina's
  checkpoints, so this selects runs, and the paper's
  off-track count is the number of runs without such a lane.

Demonstrations are classed by their last frame: the object rests on the support box at the end of Box-200 and Notebook-300,
and is held in the air at the end of the other five.

**Analysis** (`analyze.py`): per task, mean ± sd over seeds; per hand and pooled, the mean over runs ± the sd over the
training-seed means, as in the paper's tables. It also writes the difference DexDuet - DexMachina with a 95% percentile
bootstrap (10,000 resamples, `numpy.random.default_rng(0)` per comparison, runs resampled within each hand, demonstration
and method), which the paper does not report.

## Running it

The test needs an evaluated run: its checkpoint, the start-state manifest of the evaluation and the evaluation's
`eval_ep0.npy` (`scripts/reproduce_paper.sh evaluate` produces all three; a released run is staged first). On one GPU
(about 1.2 GB of GPU memory; about 4 minutes per run on an RTX PRO 6000, including the ADD step):

```bash
bash scripts/reproduce_paper.sh hold-test --run <evaluated run folder> --task ketchup-300 --seed 15            # DexDuet
bash scripts/reproduce_paper.sh hold-test --run <evaluated run folder> --task ketchup-300 --seed 15 \
    --label DexMachina --group dexmachina                                                                    # DexMachina
bash scripts/reproduce_paper.sh hold-test --analyze       # tables over every result in $DEXDUET_OUT/hold_test
```

or directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m dexmachina.eval.hold_test.run --checkpoint <run>/nn/inspire_hand.pth \
    --manifest <evaluation_reset.json of its evaluation> --out hold_test/ketchup-300_seed15 --task ketchup-300 --seed 15
CUDA_VISIBLE_DEVICES=0 python -m dexmachina.eval.hold_test.add hold_test/ketchup-300_seed15
python -m dexmachina.eval.hold_test.metrics hold_test/ketchup-300_seed15
python -m dexmachina.eval.hold_test.analyze hold_test
```

The paper runs it on the checkpoints of its main table on the DexMachina suite: five DexDuet runs and five of DexMachina's
released checkpoints per hand and demonstration, 280 runs.

## Checks

- On three stored runs of the paper's test (Inspire DexDuet Box-200 seed 15 and Waffleiron-300 seed 24, released
  checkpoints staged with a rebuilt `env.pkl`, and DexMachina's Allegro Box-200 seed-15 checkpoint), this code reproduces
  `hold_rollout.npz` bit for bit. The hold ADD agrees to within 1.2e-7 m (float32 rounding of the GPU kinematics between
  processes), and every per-run metric agrees with the stored one to its six stored digits.
- `metrics.py` applied to the stored rollouts and hold ADD of the 280 runs reproduces all of their per-run metrics and replay
  flags, and `analyze.py` then reproduces every aggregate of the stored analysis exactly.

## Inputs

Everything the test needs can be rebuilt from the released checkpoints and your own ARCTIC data: the checkpoint, the run's
`env.pkl` (`scripts/stage_release_run.py`), the start-state manifest and the evaluation rollout (`reproduce_paper.sh
evaluate`). The paper's own evaluation recordings are not released (they contain the ARCTIC reference trajectory), so the
replay check compares against your evaluation, which reproduces the paper's when your generated assets match
`environment/generated-assets-sha256.txt`. DexMachina's rows need the authors' released checkpoints (REPRODUCE.md,
section 2.3). The MANO export is not needed.
