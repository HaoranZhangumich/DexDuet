# Reproducing the DexMachina ARCTIC benchmark

This document trains and evaluates DexMachina itself, with its own (upstream) scripts. The DexDuet paper does not train
DexMachina: it evaluates the authors' released checkpoints (`scripts/reproduce_paper.sh dexmachina`,
[../REPRODUCE.md](../REPRODUCE.md), section 2.3).

This repository has an end-to-end, fail-fast pipeline for the seven-task
DexMachina benchmark. It follows the upstream instructions for the
[DexMachina environment](https://mandizhao.github.io/dexmachina-docs/0_install.html),
[ARCTIC processing](https://mandizhao.github.io/dexmachina-docs/2_process_demo.html),
[retargeting](https://mandizhao.github.io/dexmachina-docs/3_retargeting.html), and
[RL training/evaluation](https://mandizhao.github.io/dexmachina-docs/4_rl_training.html).

The pipeline uses two isolated Python 3.10 environments because ARCTIC requires
PyTorch 1.13/CUDA 11.6 while upstream DexMachina uses PyTorch 2.5.1. On RTX
50-series (Blackwell) GPUs, setup installs PyTorch 2.7.0 with CUDA 12.8 in
`dexmachina` instead, and ARCTIC processing falls back to CPU. Source revisions
are pinned in `environment/source-revisions.txt`; Genesis gets the broad-phase
overflow fix from `environment/patches/genesis/`.

Our evaluation protocol differs from the authors': the evaluator always uses
the corrected first observation, on the fixed Genesis (see
[METHOD.md](METHOD.md), sections 3 and 4), so re-evaluated
checkpoints need not match the published numbers.

## 1. Recreate or validate the software

From the repository root:

```bash
./scripts/reproduce.sh setup
. scripts/activate_env.sh dexmachina
python scripts/preflight.py --stage sources
```

The setup is idempotent: it creates missing environments/checkouts and refreshes
the three editable installs. `WANDB_MODE` defaults to `offline`; set it before
activation if online logging is wanted.

## 2. Download the licensed subset needed here

### Recompute the published Inspire result without licensed data

To first validate the authors' reporting pipeline from their public result
archive, run:

```bash
./scripts/reproduce.sh reference-inspire
```

This downloads the 3.86 GB public archive, integrity-checks it, selectively
extracts the 35 Inspire runs' configs, ADD arrays, and evaluation rollouts, and
runs `group_results.py` again. Its output is kept under
`$HOME/dexmachina_data/reference/recomputed/`, outside the repository. This
reproduces the published metrics, but is separate from training policies.

### Inputs for training from scratch

Register for [ARCTIC](https://arctic.is.tue.mpg.de/),
[SMPL-X](https://smpl-x.is.tue.mpg.de/), and
[MANO](https://mano.is.tue.mpg.de/). Enter credentials through
hidden prompts rather than putting a password in shell history or the repository:

```bash
bash scripts/download_arctic_data.sh --prompt --same-credentials
```

Omit `--same-credentials` if your three accounts use different credentials.
Existing six-variable environment credential injection is still supported by
`./scripts/reproduce.sh download-arctic`; never print the values or use shell
tracing. The downloader uses the pinned official URL lists, verified
HTTPS and archive validation, then unpacks the required raw sequences, metadata
and body models. It does not download the 649 GB image archive. The required raw
sequences are about 215 MB according to the
[ARCTIC data guide](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md).
The environment versions and SMPL-X edit follow the
[ARCTIC setup guide](https://github.com/zc-alexfan/arctic/blob/master/docs/setup.md).

Validate the licensed inputs:

```bash
. scripts/activate_env.sh arctic_env
python scripts/preflight.py --stage arctic --allow-cpu
```

## 3. Process ARCTIC, retarget, and map contacts

For the verified Inspire Hand reproduction (six unique raw clips backing seven
tasks):

```bash
DEXMACHINA_HANDS=inspire_hand ./scripts/reproduce.sh prepare
```

For all four paper hands:

```bash
./scripts/reproduce.sh prepare
```

The second command processes the same six ARCTIC clips, then performs 24
full-length retargets (six clips by four hands) and maps contacts. Retargeting
uses frames `0:600`; task-specific slicing happens later during training. This
avoids a subtle double-slicing error.

The stages can also be run separately:

```bash
./scripts/prepare_arctic_clip.sh --subject s01 --object box --use 01
DEXMACHINA_HANDS=inspire_hand ./examples/retarget_missing.sh
DEXMACHINA_HANDS=inspire_hand ./examples/contact_retarget_missing.sh
```

`process_arctic.py` does not need the undocumented `mano_urdf` module for
normal DexMachina use. Its `configs_left/configs_right` export, which retargeting
and RL do not use, is available with `--export_mano_configs` for users who have
that unpublished dependency.

## 4. Train

Check every trajectory and contact file, including the 600-frame invariant:

```bash
./scripts/reproduce.sh preflight-inspire
```

Run the verified Inspire Hand sweep (seven tasks by five seeds = 35 policies):

```bash
./scripts/reproduce.sh train-inspire
```

To do a cheaper seven-policy validation first, restrict the sweep to seed 42:

```bash
DEXMACHINA_SEEDS=42 ./scripts/reproduce.sh train-inspire
```

Batch size and local concurrency can also be overridden for a faster,
non-paper-exact validation. For example, a one-seed run with two dynamically
scheduled workers:

```bash
DEXMACHINA_SEEDS=42 \
DEXMACHINA_NUM_ENVS=4096 \
DEXMACHINA_MAX_PARALLEL=2 \
./scripts/run_inspire_train_eval.sh
```

This preserves the seven task/epoch/curriculum profiles but changes PPO batch
size (and therefore the automatically derived minibatch and mini-epoch
settings) from the released batch-12000 paper configuration.

The Inspire script's task-specific parameters and epoch counts were copied from
the reference run configurations. Checkpoints and saved environment configs are
written below `logs/rl_games/inspire_hand/`. Run directories include the seed,
and a repeated invocation skips only runs that have both a checkpoint and a
successful-completion marker, so the long sweep can be resumed safely.

`./scripts/reproduce.sh train-all` launches 140 policies (seven tasks, four
hands, five seeds) with `examples/reproduce_benchmark.sh`. Its non-Inspire settings
are inferred from the released grouping configuration, so use the Inspire sweep as the
strictly verified reproduction path unless you also compare the other-hand
arguments against the authors' released run configs.

## 5. Evaluate and compare ADD-AUC

Evaluate all Inspire checkpoints with 20 parallel evaluation environments,
compute ADD/AUC, and aggregate across seeds:

```bash
./scripts/reproduce.sh eval-inspire
```

For an unattended run that evaluates automatically after all 35 policies
finish, use:

```bash
./scripts/run_inspire_train_eval.sh
```

The number of fully completed policies is:

```bash
find logs/rl_games/inspire_hand -name .training_complete -type f | wc -l
```

The grouped result is written to
`dexmachina/eval/stats/inspire_hand_stats_AUC.json`. To evaluate one checkpoint:

```bash
. scripts/activate_env.sh dexmachina
CKPT=logs/rl_games/inspire_hand/<run>/nn/inspire_hand.pth
python scripts/preflight.py --stage eval --checkpoint "$CKPT"
python -m dexmachina.rl.eval_rl_games --checkpoint "$CKPT" --num_envs 20
python -m dexmachina.eval.compute_add \
  --input "${CKPT%/nn/*}/inspire_hand_eval/eval_ep0.npy"
```

The authors' reference grouped values are tracked in
`dexmachina/eval/stats/dexmachina_main_stats_AUC.json`; each cell uses five
seeds. The authors also link their checkpoints, configs, and precomputed ADD
data from `dexmachina/eval/eval.md`. Compare generated statistics only after
confirming `n_seeds: 5` for each task—partial sweeps intentionally produce
different uncertainty estimates.

## Operational notes

- Training and evaluation require an NVIDIA GPU. A batch size of 12,000 is the
  paper setting and can need substantial VRAM; reduce `-B` only for debugging,
  not for a claimed reproduction.
- ARCTIC preprocessing uses `cuda:0` when its pinned wheel can execute there,
  and otherwise falls back to CPU (as it does on RTX 50-series GPUs).
- Scripts stop at preflight instead of silently training on short, missing, or
  corrupt demonstrations.
- Generated/licensed data, logs, and the external source checkouts are ignored
  by Git.
