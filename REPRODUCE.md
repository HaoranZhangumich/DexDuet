# Reproducing the paper

This guide maps each result of the DexDuet paper whose code is in this repository to the commands that reproduce it. All of
them are on the **DexMachina suite**: seven ARCTIC demonstrations (Box-200, Ketchup-100, Ketchup-300, Mixer-170, Mixer-300,
Notebook-300, Waffleiron-300) in Genesis, with Inspire, Allegro, XHand and Schunk hands.

| Paper result | What you run | Section |
|---|---|---|
| Main table, DexMachina suite: DexDuet rows (four hands) | `reproduce_paper.sh from-release` (released checkpoints) or `main` (training from scratch) | 2, 3.1 |
| Main table, DexMachina suite: DexMachina rows | `reproduce_paper.sh dexmachina` on DexMachina's released checkpoints | 2.3 |
| Ablation table, rows (a)-(g) | `reproduce_paper.sh from-release --suite ablation` or `ablation` (training from scratch) | 2, 3.2 |
| ADD-AUC, SP-SR, MP-SR, MPPE and Contact-AUC of any run | `reproduce_paper.sh evaluate`, `contact-auc`, `kit-metrics` | 4 |
| Holding after the last frame (main text and supplementary) | `reproduce_paper.sh hold-test` | 5 |
| Tables from the per-run results | `reproduce_paper.sh table` | 2.4 |

Results whose code is elsewhere are listed in section 6.

Every command below runs from the repository root with the `dexmachina` environment active, on one GPU
(`CUDA_VISIBLE_DEVICES`). `bash scripts/reproduce_paper.sh --dry-run <command> ...` prints the exact commands without running
them. Outputs go to `$DEXDUET_OUT` (default `paper_repro/`):

| Folder | Contents |
|---|---|
| `summaries/<label>/<hand>/<task>/seed<S>.json` | one file per evaluated run: ADD-AUC, SP-SR, MP-SR, MPPE (all lanes and lanes 0-98), Contact-AUC, updates, checkpoint hash, start-state check |
| `tables/` | `TABLES.md`, `tables.csv`, `per_run.csv` |
| `resets/<group>/<hand>/<task>/` | the frozen start state of each hand and demonstration (`evaluation_reset.json` and `.npz`) |
| `staged/` | released checkpoints turned into evaluable run folders |
| `logs/<label>__<hand>__<task>__seed<S>/` | one log per step |
| `hold_test/` | hold test outputs |

## Protocol

The paper's protocol, which every command below follows:

- **Simulator.** Genesis `e9df2d8` with the collision-pair overflow patch, `fca4168` ([docs/METHOD.md](docs/METHOD.md),
  section 4). `scripts/setup_dexmachina_env.sh` installs it, and the scripts refuse a Genesis without it.
- **Evaluation.** 100 parallel rollouts from the first frame, all controlled by the policy, deterministic actions, no
  assistance, evaluation seed 1042, first observation read from the simulator after the reset, and a frozen start state per
  hand and demonstration that every evaluation must reproduce byte for byte. Checkpoint: the best training reward,
  `nn/<hand>.pth`.
- **Seeds.** Training seeds 15, 24, 42, 66 and 113 for the main table; 15, 42 and 113 for the ablations.
- **Metrics.** ADD-AUC (`dexmachina/eval/compute_add.py`, DexMachina's code); SP-SR, MP-SR and MPPE with the V2D challenge
  kit's definitions (`baselines/track2_metrics.py`; SP-SR averages the errors over the two parts, MP-SR tests every part);
  Contact-AUC (`dexmachina/eval/slip_record.py` + `slip_auc.py`).
  ADD-AUC, the success rates and Contact-AUC use all 100 lanes; MPPE uses lanes 0-98, because DexMachina's evaluator starts
  lane 99 from a zero hand pose, which can diverge (DexDuet's MPPE is the same either way).
- **Tables.** Each cell is the mean over runs ± the standard deviation over the training-seed means, where a seed mean is the
  mean over that seed's demonstrations. DexMachina's checkpoints are grouped by the training seed of their release config.

## 1. Setup

[docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md) covers a new machine step by step. In short:

```bash
bash scripts/reproduce.sh setup              # dexmachina and arctic_env environments; Genesis with the overflow patch
bash scripts/reproduce.sh download-arctic    # ARCTIC, MANO and SMPL-X, with your own accounts
bash scripts/reproduce.sh prepare            # process the six ARCTIC sequences; retarget and map contacts for the four hands
```

DexDuet also needs the MANO hand vertices of the demonstrations (Contact-AUC and the contact reward's human targets) and the
contact reward's object distance grids ([docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md), sections 4.2 and 4.3):

```bash
SEQS="s01/ketchup_use_02 s01/ketchup_use_01 s01/mixer_use_01 s01/waffleiron_use_01 s01/box_use_01 s02/notebook_use_02"
conda run -n arctic_env python dexmachina/hand_proc/export_mano_vertices.py --raw arctic/data/arctic_data/data/raw_seqs \
    --mano arctic/unpack/body_models/mano --out ~/data/dexduet/mano_export $SEQS
conda activate dexmachina
python dexmachina/hand_proc/ho_contact_assets.py grids --assets dexmachina/assets box mixer ketchup notebook waffleiron
python dexmachina/hand_proc/ho_contact_assets.py human --assets dexmachina/assets --mano ~/data/dexduet/mano_export $SEQS
export CH_MANO_EXPORT=~/data/dexduet/mano_export
python scripts/preflight.py --stage train --hands inspire_hand
```

These steps are also what evaluating a released checkpoint needs. `scripts/stage_release_run.py`, which every
release command runs first, rebuilds a run's environment from them: it reads the generated assets under `dexmachina/assets/`
(`arctic/processed`, `retargeted`, `retargeter_results`, `contact_retarget` and `ho_contact`). If any of them is missing, it
lists the missing files with the step that creates each one.

For Inspire, `(cd dexmachina/assets && sha256sum -c ../../environment/generated-assets-sha256.txt)` checks the generated
files against the ones DexDuet's main-table Inspire runs used (row (a) of the ablation table is part of these runs). When
all of them match, these checkpoints evaluate to the per-run values in their `RUN_INFO.json` (`table_metrics`) exactly on an RTX PRO 6000 (section 2.1; on other
GPUs see section 2.2). The runs of ablation rows
(b)-(g) were trained with other processings of the same six ARCTIC sequences, whose files differ from the listed ones.
Each run with the per-hand policy records the sha256 of its own processed demonstration in `config/env.yaml`
(`source_sha256`), and `scripts/stage_release_run.py` prints a notice when your file differs from it. Evaluated with the
listed files, the ablation runs can therefore differ from their reported values. Files you generate yourself can differ slightly too, and
then so can the start state and the scores.

## 2. From the released checkpoints (fast route)

### 2.1 Download

The checkpoints are on Hugging Face, in [oldTOM/dexduet-checkpoints](https://huggingface.co/oldTOM/dexduet-checkpoints).
Download, check and unpack them (`DEXDUET_CKPT_REPO` names another repository):

```bash
bash scripts/download_checkpoints.sh          # every package into ./checkpoints (or $DEXDUET_CHECKPOINTS)
```

| Package | Unpacks to | Contents |
|---|---|---|
| `dexduet_dexmachina_suite.zip` | `dexduet/<hand>/<task>/seed<S>/` | DexDuet's main-table runs: 4 hands x 7 demonstrations x 5 seeds |
| `dexduet_ablations.zip` | `ablations/row_<b..g>/inspire_hand/<task>/seed<S>/` | ablation rows (b)-(g): 6 rows x 7 demonstrations x 3 seeds; row (a) is the suite's Inspire runs |

Each run folder holds the evaluated checkpoint `<hand>.pth`, its training configuration (`config/agent.yaml`,
`config/env.yaml`) and `RUN_INFO.json`: the method or ablation row, hand, task, training seed and original run name, the
checkpoint record (file, sha256, size in bytes, epoch, selection rule and completed PPO updates) and the run's evaluated
values (`table_metrics`). Runs with the per-hand policy (every run except those of ablation row (e)) also hold
`config/initialization.json` (hashes of the initial network) and `final_checkpoint.json` (the training-end record that the
evaluator checks the checkpoint against). Every download is checked against `CHECKPOINTS.sha256`.

The demonstration and the retargeted robot reference of each run (the run's `env.pkl`) and the start states are not
included, because they are derived from the licensed ARCTIC data. `scripts/stage_release_run.py` rebuilds them from your own
ARCTIC data (section 1) and the shipped `config/env.yaml`, exactly as the training launcher builds them. We checked this
path on an RTX PRO 6000 with two released Inspire runs (Box-200 seed 15 and Waffleiron-300 seed 24): the rebuilt `env.pkl`
of the first equals the original one, the start states of both are byte-identical to the paper's, and both reproduce their
reported ADD-AUC, SP-SR, MP-SR, MPPE and Contact-AUC exactly.

### 2.2 Evaluate DexDuet

```bash
bash scripts/reproduce_paper.sh from-release --suite main                         # main table; downloads first if needed
bash scripts/reproduce_paper.sh from-release --suite main --hand inspire_hand --task box-200 --seed 15   # one run
bash scripts/reproduce_paper.sh from-release --suite ablation                     # rows (a)-(g), 147 runs
```

For each run, `from-release` stages the run folder, evaluates it (`dexmachina.rl.eval_rl_games`), computes ADD-AUC
(`dexmachina.eval.compute_add`), SP-SR, MP-SR and MPPE (`baselines.track2_metrics`) and Contact-AUC
(`dexmachina.eval.slip_record` + `slip_auc`), writes the run's summary and finally the tables. The tables also compare
every run with the values it was reported with (`RUN_INFO.json`). Runs with a summary are skipped, so you can stop and
restart it. On one RTX PRO 6000 a run takes about 4 to 5 minutes and less than 2 GB of GPU
memory, so the main-table runs take about 10 GPU-hours and the ablation runs about 9; several evaluations can share a GPU.

GPU evaluation is not bit-reproducible across machines: on another GPU model, driver or library build, a released
checkpoint can evaluate to values that differ from its reported ones. The tables' check against the released values
(section 2.4) lists every run whose reproduced values differ from its `table_metrics`, with both values.

Two Inspire checkpoints (Mixer-300 seed 66 and Waffleiron-300 seed 24) are periodic saves chosen by
the best training return without assistance, not the best-reward save; `final_checkpoint.json` records this (and the run's
`RUN_INFO.json` says so in `notes`), and the evaluation accepts it explicitly (`eval_rl_games.py --accept-selection
best_zero_help_training_return`).

### 2.3 Evaluate DexMachina's released checkpoints

The DexMachina rows evaluate the authors' own checkpoints, which this repository does not redistribute. Download their
main-results archive (`dexmachina_main_results_data.zip`) from the link in DexMachina's evaluation guide
([dexmachina/eval/eval.md](dexmachina/eval/eval.md), from the [DexMachina repository](https://github.com/MandiZhao/dexmachina);
`scripts/reproduce_reference_inspire.sh` downloads the same archive), unzip it, then:

```bash
bash scripts/reproduce_paper.sh dexmachina --archive <path>/dexmachina_main_results_data                     # 4 hands
bash scripts/reproduce_paper.sh dexmachina --archive <path>/dexmachina_main_results_data --hand inspire_hand
```

Each checkpoint keeps its own `env.pkl` (DexMachina's retargeting), so these runs have their own start states
(`resets/dexmachina/`). `scripts/relocate_run.py` points the copy's asset paths at this repository, and the repository's
default agent configuration (DexMachina's network) goes next to it. The training seed of each checkpoint comes from its
release config (`config.json`, `params.seed`). As in DexMachina's evaluator, the last of the 100 lanes starts from their zero
hand pose (`--zero-pose-last-lane`). When that lane resets early, Contact-AUC is recorded with `slip_record.py
--allow-last-lane-reset`, which marks the lane's later states as lost.

### 2.4 Tables

```bash
bash scripts/reproduce_paper.sh table          # writes $DEXDUET_OUT/tables/TABLES.md, tables.csv, per_run.csv
```

The paper's values, for comparison (mean ± sd over training-seed means; MPPE in cm, the rest in %):

| Robot | Method | ADD-AUC | SP-SR | MP-SR | MPPE | Contact-AUC |
|---|---|---|---|---|---|---|
| Inspire | DexMachina | 52.5 ± 10.2 | 53.7 ± 11.8 | 53.7 ± 11.8 | 40.7 ± 15.6 | 33.6 ± 7.8 |
| Inspire | DexDuet | 88.7 ± 2.3 | 94.3 ± 7.8 | 94.3 ± 7.8 | 1.50 ± 0.22 | 79.6 ± 2.6 |
| Allegro | DexMachina | 65.8 ± 14.4 | 67.9 ± 18.4 | 67.9 ± 18.4 | 23.4 ± 15.4 | 35.8 ± 8.3 |
| Allegro | DexDuet | 94.8 ± 0.9 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.02 ± 0.09 | 68.0 ± 1.9 |
| XHand | DexMachina | 60.7 ± 5.0 | 62.3 ± 7.7 | 62.2 ± 7.7 | 28.7 ± 6.1 | 38.2 ± 4.4 |
| XHand | DexDuet | 93.6 ± 2.1 | 100.0 ± 0.0 | 97.1 ± 6.4 | 1.09 ± 0.19 | 79.5 ± 3.5 |
| Schunk | DexMachina | 73.7 ± 8.7 | 79.2 ± 16.2 | 76.4 ± 16.1 | 12.1 ± 11.6 | 45.2 ± 6.5 |
| Schunk | DexDuet | 94.1 ± 0.3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.06 ± 0.02 | 80.4 ± 2.2 |

| Row | Per-hand | A-RSI | Hand graph | Hand-object contact | ADD-AUC | Contact-AUC | Updates |
|---|---|---|---|---|---|---|---|
| (a) | ✓ | ✓ | ✓ | ✓ | 89.7 ± 2.1 | 80.5 ± 3.2 | 5,524 |
| (b) | ✓ | ✓ | | ✓ | 87.5 ± 1.1 | 71.5 ± 1.6 | 6,721 |
| (c) | ✓ | ✓ | ✓ | | 86.4 ± 3.9 | 65.4 ± 4.5 | 5,836 |
| (d) | ✓ | ✓ | | | 76.7 ± 8.8 | 49.0 ± 5.1 | 6,745 |
| (e) | | ✓ | ✓ | ✓ | 74.1 ± 1.7 | 62.1 ± 2.6 | 6,255 |
| (f) | ✓ | | | | 79.4 ± 5.9 | 47.6 ± 4.5 | 6,845 |
| (g) | ✓ | | ✓ | ✓ | 88.4 ± 1.6 | 74.2 ± 3.2 | 6,608 |

An empty per-hand mark means DexMachina's unified policy; an empty A-RSI mark means DexMachina's single-level curriculum. Row
(a) is DexDuet's Inspire runs with seeds 15, 42 and 113.

## 3. Training from scratch

Training needs about 26 GB of GPU memory per run (12,000 environments). In the paper, the 140 main-table runs stopped
after 4,350 to 11,500 PPO updates (1.67 to 4.42 billion transitions), and the 126 runs of ablation rows (b)-(g) after
4,350 to 13,000 updates (each run's `RUN_INFO.json`, `checkpoint.completed_updates`). Genesis runs cannot resume: a run
folder left by an interrupted training must be moved away before the run is started again.

Check the setup first with a short run (256 environments, 20 updates, then the full evaluation; a few minutes):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh main --task Box-200 --seed 15 --smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh ablation --row b --task Box-200 --seed 15 --smoke
```

### 3.1 DexDuet (main table)

```bash
for hand in inspire_hand allegro_hand xhand schunk_hand; do
  for task in Box-200 Ketchup-100 Ketchup-300 Mixer-170 Mixer-300 Notebook-300 Waffleiron-300; do
    for seed in 15 24 42 66 113; do
      CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh main --hand $hand --task $task --seed $seed
done; done; done
bash scripts/reproduce_paper.sh table
```

`main` runs `scripts/train_coordinates_hand.py`, which trains with the paper's flags, evaluates the best-reward checkpoint
and scores ADD-AUC and Contact-AUC; `main` then adds SP-SR, MP-SR and MPPE and the summary. Run folders are
`logs/rl_games/<hand>/<prefix>-coordinates-hand_<task>_seed<S>_*`. For Schunk on Ketchup-300, Waffleiron-300 and Notebook-300
(group A), `main` sets behavior cloning to 0.3, as in the paper's Schunk runs. Run several jobs in parallel by giving each its own GPU, or two per
GPU.

This repository trains with the collision-pair fix and the corrected reset observation. The paper's other three hands and its
ablation rows were trained this way too; most of its Inspire checkpoints were trained before both fixes
([docs/METHOD.md](docs/METHOD.md), section 6). Training is not bit-reproducible across machines, so a run trained here can
differ from the released one, most where a policy fails.

### 3.2 Ablation rows

```bash
for row in b c d e f g; do
  for task in Box-200 Ketchup-100 Ketchup-300 Mixer-170 Mixer-300 Notebook-300 Waffleiron-300; do
    for seed in 15 42 113; do
      CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh ablation --row $row --task $task --seed $seed
done; done; done
```

Row (a) is DexDuet on Inspire with seeds 15, 42 and 113 (section 3.1). `baselines/ablation_rows.py` builds each row's
arguments from the launcher's full-method command and changes only what the row removes
(`python baselines/ablation_rows.py --diff` lists the changes); [baselines/README.md](baselines/README.md) describes the rows
and `baselines/run_all.sh` queues them over several GPUs. Runs stop by the same rule as DexDuet, except that without A-RSI
(rows f and g) DexMachina's curriculum can run to the update cap (13,000 on Ketchup-300).

## 4. Metrics of a single run

```bash
bash scripts/reproduce_paper.sh evaluate --run <run folder> --task box-200 --seed 15 [--label DexDuet]
bash scripts/reproduce_paper.sh contact-auc --run <run folder> --task box-200 --seed 15
bash scripts/reproduce_paper.sh kit-metrics --run <evaluated run folder>
```

`--run` is a run folder with `nn/<hand>.pth` and `params/`, or a released run folder (it is staged first). Use `--label
DexMachina --group dexmachina` for DexMachina's checkpoints, so they get their own start states. Results inside the run
folder:

| File | Contents |
|---|---|
| `<hand>_eval_all_policy/eval_ep0.npy`, `eval_protocol.json` | the evaluation rollout (100 lanes) and its protocol record |
| `<hand>_eval_all_policy/add_stats.json` | ADD-AUC (`auc.overall`) |
| `<hand>_eval_all_policy/track2_metrics.json` | SP-SR, MP-SR, MPPE, and per-rollout values |
| `slip/recording.npz`, `slip/record.json` | the re-recorded rollout with the hand keypoints |
| `slip/slip_auc.json` | Contact-AUC (`slip_auc`), coverage and lost share |

`baselines/track2_metrics.py` implements the V2D kit's definitions of SP-SR, MP-SR and MPPE for the two parts of an
articulated object. With the kit at hand, `KIT=<path to the kit> python baselines/test_track2_metrics.py` checks it against
the kit's own functions. The paper's numbers were computed with the kit's own code; on the released Inspire runs checked
above, `track2_metrics.py` gives the same values.

## 5. Holding after the last frame

The hold test of the main text and the supplementary lets each policy act for 3 s after the last frame with every reference
frozen there. A run holds the object if the object stays within 5 cm of its final pose for the whole 3 s, and drops it if
the object ends the 3 s more than 10 cm below its final pose or off the table.
[dexmachina/eval/hold_test/README.md](dexmachina/eval/hold_test/README.md) gives the protocol.

```bash
bash scripts/reproduce_paper.sh hold-test --run <evaluated run folder> --task ketchup-300 --seed 15          # DexDuet
bash scripts/reproduce_paper.sh hold-test --run <evaluated run folder> --task ketchup-300 --seed 15 \
    --label DexMachina --group dexmachina                                                                   # DexMachina
bash scripts/reproduce_paper.sh hold-test --analyze        # tables over $DEXDUET_OUT/hold_test
```

Run it on every evaluated run of the main table, 280 runs: DexMachina's released checkpoints and DexDuet's checkpoints of
the main table. A run takes about 4 minutes and
about 1.2 GB of GPU memory. Everything it needs is rebuilt from the released checkpoints and your ARCTIC data. The analysis
writes the hold rate as `SR@5`, the drop rate as `Drop` and the number of runs on track; a hand has 35 runs per method, so
its off-track count is 35 minus the runs on track. The paper's values (in % of rollouts, mean ± sd over the five training
seeds; off track: runs that end more than 3 cm from the final pose, of 35 per hand and 140 in all):

| Hand | Method | Held ↑ | Dropped ↓ | Off track ↓ |
|---|---|---|---|---|
| Inspire | DexMachina | 17.0 ± 11.8 | 71.7 ± 14.2 | 18 |
| Inspire | DexDuet | 37.1 ± 16.3 | 51.4 ± 7.8 | 5 |
| Allegro | DexMachina | 14.1 ± 10.0 | 80.0 ± 7.7 | 16 |
| Allegro | DexDuet | 22.9 ± 12.8 | 62.9 ± 16.3 | 5 |
| XHand | DexMachina | 5.7 ± 7.7 | 71.7 ± 17.3 | 18 |
| XHand | DexDuet | 25.7 ± 15.6 | 54.3 ± 18.6 | 1 |
| Schunk | DexMachina | 25.5 ± 11.8 | 68.9 ± 15.5 | 10 |
| Schunk | DexDuet | 42.9 ± 10.1 | 51.4 ± 16.3 | 0 |
| All | DexMachina | 15.6 ± 3.2 | 73.1 ± 6.9 | 62 |
| All | DexDuet | 32.1 ± 7.1 | 55.0 ± 6.0 | 11 |

## 6. Results whose code is not in this repository

- **V2D challenge (Unitree G1 with Dex3 hands)**: DexDuet, DexDuet without interaction rewards and CHORD (PPO) on the
  challenge's runtime are in [DexDuet-V2D](https://github.com/HaoranZhangumich/DexDuet-V2D).
- **SPIDER suite and ManipTrans suite**: DexDuet and the baselines of these two benchmarks are not in this repository.
- Not included: ConTrack on the DexMachina suite with XHand hands, the comparison of runs at equal tracking, the perceptual
  study and the real-robot experiments.
