# Setting up DexDuet on a new machine

Instructions for building the training and evaluation environment on a Linux GPU machine and checking that it reproduces
the paper's numbers. Commands assume the repository root as the current directory unless stated. The method is described in
[METHOD.md](METHOD.md), the reproduction of each paper result in [../REPRODUCE.md](../REPRODUCE.md).

The commands and timings below were checked on Ubuntu 22.04 with NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB, driver
580.82.09). GPU evaluation is not bit-reproducible across machines: on another GPU model, driver or library build, a
released checkpoint can evaluate to values that differ slightly from its reported ones.

## 0. What "done" means

1. The CPU tests pass (section 6).
2. A smoke training run finishes and evaluates (section 6).
3. A released checkpoint evaluates to its reported numbers on this machine (section 6).

## 1. System

- Ubuntu 22.04 (other recent Linux distributions should work) and an NVIDIA driver >= 535. No system CUDA toolkit is needed:
  torch ships its own CUDA runtime. A training run needs about 26 GB of GPU memory (12,000 environments); an evaluation
  needs about 1.5 GB.
- Packages:
  ```bash
  sudo apt-get update && sudo apt-get install -y git curl wget unzip rsync build-essential cmake tmux ffmpeg \
      libgl1 libegl1 libglib2.0-0 libxrender1 libsm6 libxext6
  ```
  Rendering is headless EGL (no display or Xvfb needed).
- Conda: install Miniforge to `~/miniconda3` (the scripts look for conda there, or in `$CONDA_EXE`):
  ```bash
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p ~/miniconda3 && ~/miniconda3/bin/conda init bash
  ```

## 2. Code

```bash
git clone https://github.com/HaoranZhangumich/DexDuet.git
cd DexDuet
```

Pinned external sources (`environment/source-revisions.txt`), cloned by the setup scripts into the repository:

| Source | Where | Revision |
|---|---|---|
| Genesis (DexMachina author's fork) | `Genesis/` | MandiZhao/Genesis `e9df2d8f7d30595d65cabdc618c6b5ca8cc657e2` + the collision-pair overflow fix (`environment/patches/genesis/`) = `fca4168f562b0762130932cda68ba09d011336ae` |
| rl_games (fork) | `Genesis/rl_games/` | MandiZhao/rl_games `2c1a7718a3eed71e7c230b6bb8fb2897b0b1d0c4` |
| ARCTIC | `arctic/` | zc-alexfan/arctic `49f3eb4b7f01663172576357bb3d95695d7a7689` |

`scripts/setup_dexmachina_env.sh` checks out Genesis `e9df2d8`, applies the patch with `git am` and checks that the result is
`fca4168` (the same SHA on every machine); `python scripts/preflight.py --stage sources` checks it again. The forks are
personal repositories of DexMachina's author; if one disappears, copy `Genesis/` (with `Genesis/rl_games/`) from an existing
installation.

## 3. Python environments

```bash
bash scripts/reproduce.sh setup          # = scripts/setup_arctic_env.sh, then scripts/setup_dexmachina_env.sh
```

- `dexmachina` (Python 3.10, torch 2.5.1+cu124): training, evaluation, Contact-AUC. Editable installs of `Genesis/`,
  `Genesis/rl_games/` and this repository (`pip install --no-deps -e .`). On RTX 50-series and other Blackwell GPUs the setup
  script installs torch 2.7.0+cu128 instead.
- `arctic_env` (Python 3.10, torch 1.13 cu116, pytorch3d 0.7.3, smplx with the MANO patch, chumpy): ARCTIC processing and
  the MANO export only.
- Reference versions: `environment/pip-freeze-dexmachina.txt`, `environment/pip-freeze-arctic_env.txt`,
  `environment/conda-env-*.yml`. Keep wandb at 0.12.21 in `dexmachina`.
- Tests need pytest: `conda run -n dexmachina pip install pytest`.
- Run the code with the repository first on `PYTHONPATH` (`scripts/train_coordinates_hand.py` and
  `scripts/reproduce_paper.sh` do this). If another checkout is installed in the same environment, Python would otherwise
  import that one.

## 4. Data

### 4.1 Licensed ARCTIC, MANO and SMPL-X data

You need your own accounts on the [ARCTIC](https://arctic.is.tue.mpg.de/), [MANO](https://mano.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/) websites. Then:

```bash
bash scripts/download_arctic_data.sh --prompt --same-credentials   # or: bash scripts/reproduce.sh download-arctic
```

Layout used by the code: `arctic/data/arctic_data/data/{raw_seqs/<subject>/<seq>.*.npy, meta/...}` and
`arctic/unpack/body_models/{mano/MANO_LEFT.pkl, mano/MANO_RIGHT.pkl, smplx/...}`. The seven tasks use six sequences:

| Task | Clip | ARCTIC sequence |
|---|---|---|
| Ketchup-300 | `ketchup-40-340-s01-u02` | `s01/ketchup_use_02` |
| Ketchup-100 | `ketchup-30-130-s01-u01` | `s01/ketchup_use_01` |
| Mixer-300, Mixer-170 | `mixer-40-340-s01-u01`, `mixer-30-200-s01-u01` | `s01/mixer_use_01` |
| Waffleiron-300 | `waffleiron-40-340-s01-u01` | `s01/waffleiron_use_01` |
| Box-200 | `box-30-230-s01-u01` | `s01/box_use_01` |
| Notebook-300 | `notebook-40-340-s02-u02` | `s02/notebook_use_02` |

### 4.2 Generated assets (under `dexmachina/assets/`, git-ignored)

| Asset | Size (Inspire) | Built by |
|---|---|---|
| `arctic/processed/<s>/<seq>.npy` | 21 MB | `bash scripts/reproduce.sh prepare` (ARCTIC processing in `arctic_env`, then `process_arctic.py`; per sequence: `scripts/prepare_arctic_clip.sh --subject s01 --object ketchup --use 02`) |
| `retargeted/<hand>/<s>/<seq>_vector_para.pt`, `retargeter_results/...` | 10 MB | the same command (`examples/retarget_missing.sh`; full 0-600 frame range, about 5 min per sequence and hand) |
| `contact_retarget/<hand>/<s>/<seq>.npy` | 7 MB | the same command (`examples/contact_retarget_missing.sh`) |
| `ho_contact/grids/<obj>_{bottom,top}.npz` | 63 MB | `python dexmachina/hand_proc/ho_contact_assets.py grids --assets dexmachina/assets box mixer ketchup notebook waffleiron` |
| `ho_contact/human/<s>/<seq>.npz` | 1 MB | `python dexmachina/hand_proc/ho_contact_assets.py human --assets dexmachina/assets --mano <MANO export> s01/ketchup_use_02 s01/ketchup_use_01 s01/mixer_use_01 s01/waffleiron_use_01 s01/box_use_01 s02/notebook_use_02` |

`DEXMACHINA_HANDS=inspire_hand bash scripts/reproduce.sh prepare` restricts retargeting to one hand. For Inspire, compare
the result with the files DexDuet's main-table Inspire runs used (ablation rows (b)-(g) used other processings of the
same sequences; each run with the per-hand policy, rows (b), (c), (d), (f) and (g), records its own in `config/env.yaml`,
and row (e) records none; REPRODUCE.md, section 1):

```bash
(cd dexmachina/assets && sha256sum -c ../../environment/generated-assets-sha256.txt)   # 40 files, all must say OK
```

ARCTIC processing falls back to the CPU on some GPUs; that is expected. Regenerated files can differ from the listed ones in
the last bits, and then the evaluation start state and the scores can differ slightly too.

### 4.3 MANO export (Contact-AUC and the contact reward's human targets)

```bash
conda run -n arctic_env python dexmachina/hand_proc/export_mano_vertices.py --raw arctic/data/arctic_data/data/raw_seqs \
    --mano arctic/unpack/body_models/mano --out ~/data/dexduet/mano_export \
    s01/ketchup_use_02 s01/ketchup_use_01 s01/mixer_use_01 s01/waffleiron_use_01 s01/box_use_01 s02/notebook_use_02
export CH_MANO_EXPORT=~/data/dexduet/mano_export
```

The training launcher reads `--mano` or `$CH_MANO_EXPORT` (default `~/data/coordinates_hand/mano_export`).

## 5. Accounts

- W&B (optional): training logs offline by default (`WANDB_MODE=offline`). To log online, `wandb login` and set
  `WANDB_MODE=online`; the launcher's default project is `coordinates-hand` (`WANDB_PROJECT` overrides it).
- Hugging Face (optional): only to download the released checkpoints (`scripts/download_checkpoints.sh`).

## 6. Check the installation

```bash
conda activate dexmachina
# 1. tests (CPU, under a minute; tests that need generated assets or licensed data skip without them)
CUDA_VISIBLE_DEVICES= python -m pytest -q tests

# 2. the training command, then a smoke run: 256 environments, 20 updates, then the full evaluation (a few minutes)
python scripts/train_coordinates_hand.py --task Box-200 --dry-run
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh main --task Box-200 --seed 15 --smoke

# 3. a released checkpoint (REPRODUCE.md, section 2): evaluate it and compare with its reported numbers
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_paper.sh evaluate --run checkpoints/dexduet/inspire_hand/box-200/seed15
cat paper_repro/summaries/DexDuet/inspire_hand/box-200/seed15.json      # vs table_metrics in the run's RUN_INFO.json
```

Step 3 passes when the start-state check passes (the evaluator stops otherwise) and the scores match the run's
`table_metrics`. On an RTX PRO 6000, with generated assets that match `environment/generated-assets-sha256.txt`, the
released Inspire Box-200 seed-15 run reproduces them exactly (ADD-AUC 96.08, SP-SR 100, MP-SR 100, MPPE 0.83 cm, Contact-AUC
94.15).

## 7. Running experiments

- Two training runs per GPU share it well; a third adds almost no throughput. Avoid NVIDIA MPS for Genesis training unless
  every job on that GPU is yours: an abrupt exit of one MPS client can stop every job on the GPU, and Genesis runs cannot
  resume.
- One training run (REPRODUCE.md, section 3):
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/reproduce_paper.sh main --task Ketchup-300 --seed 42 > ketchup-300_seed42.out 2>&1 &
  ```
  Outputs: `logs/rl_games/<hand>/<prefix>-<run name>_*/`: `nn/<hand>.pth` (the best-reward checkpoint),
  `final_checkpoint.json`, `params/`, `<hand>_eval_all_policy/{eval_ep0.npy, eval_protocol.json, add_stats.json,
  track2_metrics.json}` and `slip/` (Contact-AUC); the summary goes to `paper_repro/summaries/`.
- Ablation rows: [../baselines/README.md](../baselines/README.md).

## 8. Known pitfalls

- Run the code with the repository first on `PYTHONPATH`; an editable install of another checkout would otherwise shadow it.
- A run folder copied from another machine needs `scripts/relocate_run.py` before evaluation (its `env.pkl` stores the old
  machine's paths); a released run folder needs `scripts/stage_release_run.py` (`reproduce_paper.sh` does both).
- Genesis must be the patched revision `fca4168` (section 2); `python scripts/preflight.py --stage sources` checks it.
- Contact-AUC needs the MANO export (section 4.3). Without it, training and ADD-AUC still finish but the Contact-AUC step
  fails; run `reproduce_paper.sh contact-auc` (or the launcher with `--skip-train`) once the export exists.
- When many jobs share a machine, cap each job's CPU thread pools (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`; `reproduce_paper.sh` does this). Otherwise each job starts one thread per core, which can exhaust a
  container's process limit.
- `.gitignore` covers the generated assets, `resets/`, `checkpoints/` and `paper_repro/`. Licensed data and files
  derived from it must not be committed or published.
- `tests/test_inspire_golden.py` checks the Inspire hand graph and hand-object contact rewards bit for bit against a golden
  file, `tests/golden/inspire_golden.npz`. The file holds data derived from ARCTIC, so it is not included (and is
  git-ignored); `tests/golden/make_inspire_golden.py` writes it from your own data and the code of your checkout. The test
  skips without the file or without the retargeted Inspire Box-200 data, processed ARCTIC and `assets/ho_contact`; its
  values can differ in the last digit on other hardware. In a checkout without generated assets,
  `python -m pytest -q tests` gives 177 passed and 14 skipped.
- The development tests in `dexmachina/tests` run on the CPU; the real-clip check in `test_ref_gate.py` skips without the
  generated Inspire assets.
