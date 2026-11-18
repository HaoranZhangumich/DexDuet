# Ablation rows

The paper's ablation table (Inspire hands, the seven demonstrations of the DexMachina suite, training seeds 15, 42 and 113)
compares DexDuet with six variants. Each variant changes DexDuet in one place and keeps everything else, including the
reward weights, the stopping rule, the update caps and the evaluation.

| Row | Per-hand policy | A-RSI | Hand graph reward | Hand-object contact reward | What changes |
|---|---|---|---|---|---|
| (a) | ✓ | ✓ | ✓ | ✓ | DexDuet itself (`scripts/train_coordinates_hand.py`) |
| (b) | ✓ | ✓ | | ✓ | `--hand_graph_weight 0` |
| (c) | ✓ | ✓ | ✓ | | `--ho_contact_weight 0` |
| (d) | ✓ | ✓ | | | both weights 0 |
| (e) | | ✓ | ✓ | ✓ | DexMachina's unified policy: the per-hand actor and per-hand PPO flags removed |
| (f) | ✓ | | | | (d) with DexMachina's single-level curriculum: every `--rsi_*` flag removed, `--curr_schedule uniform` |
| (g) | ✓ | | ✓ | ✓ | DexDuet with DexMachina's single-level curriculum, as (f) is to (d) |

`ablation_rows.py` builds each row's training arguments from the launcher's own command for the full method
(`scripts/train_coordinates_hand.py`), so the rows cannot drift from it:

```bash
python baselines/ablation_rows.py --row b --task box-200 --seed 15           # one argument per line
python baselines/ablation_rows.py --diff                                     # each row's changes
python baselines/ablation_rows.py --row b --task box-200 --seed 15 --name    # run name
```

Run names are `abl-<row>-<label>_<task>_seed<seed>` (labels `nohg`, `nohoc`, `noir`, `unified`, `noir-noarsi`, `noarsi`;
`noir`: no interaction rewards).

## Running one row

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh ablation --row b --task Box-200 --seed 15 --smoke   # setup check, minutes
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh ablation --row b --task Box-200 --seed 15
```

`reproduce_paper.sh ablation` trains with the row's arguments (`dexmachina/rl/train_rl_games.py`), then evaluates the
best-reward checkpoint with the paper's protocol (100 rollouts, evaluation seed 1042, the frozen start state shared with
DexDuet's runs) and computes ADD-AUC, SP-SR, MP-SR, MPPE and Contact-AUC. The run's summary goes to
`$DEXDUET_OUT/summaries/row_<row>/inspire_hand/<task>/seed<seed>.json`, with the number of PPO updates until training
stopped.

## Running every row

`run_all.sh` queues all 126 runs (6 rows x 7 demonstrations x 3 seeds) over the given GPUs, longest demonstrations first:

```bash
DRY_RUN=1 GPUS="0 1 2 3" bash baselines/run_all.sh                                    # the plan
CH_MANO_EXPORT=<MANO export> GPUS="0 1 2 3" PER_GPU=2 bash baselines/run_all.sh
ROWS="a" GPUS="0" bash baselines/run_all.sh                                           # row (a): DexDuet, same seeds
bash scripts/reproduce_paper.sh table                                                 # the table
```

A run uses about 26 GB of GPU memory. Two runs per GPU share it well; a third adds little. In the paper, the 126 runs stop
after 4,350 to 13,000 updates; without A-RSI (rows f and g), DexMachina's curriculum can run to the update cap (13,000 on
Ketchup-300). Genesis runs cannot resume: a run folder left by an interrupted training must be moved away before its job is
started again.

The released checkpoints of all six rows can be evaluated without training (`reproduce_paper.sh from-release --suite
ablation`; [../REPRODUCE.md](../REPRODUCE.md), section 2). Each run folder's `RUN_INFO.json` gives the row, its components,
the checkpoint record and the values the run was reported with (`table_metrics`, with the number of updates). GPU
evaluation is not bit-reproducible across machines, so an evaluation on another machine can differ from these values.

## Other files

`track2_metrics.py` computes SP-SR, MP-SR and MPPE of an evaluation with the V2D challenge kit's definitions
(`python -m baselines.track2_metrics --input <run>/<hand>_eval_all_policy/eval_ep0.npy`). `test_track2_metrics.py` checks it
against the kit's own functions: `KIT=<path to the V2D submission kit> python baselines/test_track2_metrics.py`.
