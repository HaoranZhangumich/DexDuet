"""Reproduce the DexDuet paper's results on the DexMachina suite (implementation of scripts/reproduce_paper.sh; REPRODUCE.md).

Every command calls the repository's own tools; nothing here re-implements training, evaluation or a metric:

  scripts/train_coordinates_hand.py   DexDuet training + evaluation        baselines/ablation_rows.py    ablation-row arguments
  dexmachina/rl/eval_rl_games.py      evaluation (100 lanes, seed 1042)    dexmachina/eval/compute_add   ADD-AUC
  dexmachina/eval/slip_record + slip_auc   Contact-AUC                     baselines/track2_metrics      SP-SR, MP-SR, MPPE
  scripts/stage_release_run.py        released checkpoint -> run folder    scripts/relocate_run.py       DexMachina's checkpoints
  dexmachina/eval/hold_test           hold after the last frame            scripts/paper_table.py        tables

Environment: DEXDUET_PYTHON (interpreter of the dexmachina environment; default: this one), DEXDUET_OUT (outputs; default
<repo>/paper_repro), DEXDUET_CHECKPOINTS (unpacked release; default <repo>/checkpoints), CH_MANO_EXPORT (MANO export, needed
for Contact-AUC), CUDA_VISIBLE_DEVICES (one GPU). Run `scripts/reproduce_paper.sh <command> --help` for the options.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('DEXDUET_OUT') or REPO / 'paper_repro').resolve()
CHECKPOINTS = Path(os.environ.get('DEXDUET_CHECKPOINTS') or REPO / 'checkpoints').resolve()
PY = os.environ.get('DEXDUET_PYTHON') or sys.executable
MANO = os.environ.get('CH_MANO_EXPORT')
HANDS = ('inspire_hand', 'allegro_hand', 'xhand', 'schunk_hand')
MAIN_SEEDS = (15, 24, 42, 66, 113)
ABLATION_SEEDS = (15, 42, 113)
ROWS = ('b', 'c', 'd', 'e', 'f', 'g')
SCHUNK_LONG_CLIP_BC = 0.3   # the paper's Schunk runs: bc 0.3 on Ketchup-300, Notebook-300, Waffleiron-300 (launcher default 0.2)
EVAL_SEED, LANES = 1042, 100
DRY_RUN = False


def launcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location('train_coordinates_hand', REPO / 'scripts' / 'train_coordinates_hand.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAUNCHER = launcher()
TASKS = {key.lower(): key for key in LAUNCHER.TASKS}          # 'box-200' -> 'Box-200'


def task_name(task):
    if task.lower() not in TASKS:
        sys.exit(f'unknown task {task}; one of {sorted(TASKS)}')
    return task.lower()


def clip_of(task):
    return LAUNCHER.TASKS[TASKS[task_name(task)]][0]


def child_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(key, '1')
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    env.setdefault('WANDB_MODE', 'offline')
    return env


def run(cmd, log=None, check=True):
    """Run one step from the repository root; its output goes to `log` (and a one-line status to the terminal)."""
    cmd = [str(c) for c in cmd]
    print('$', shlex.join(cmd) + (f'   > {log}' if log else ''), flush=True)
    if DRY_RUN:
        return 0
    if log is None:
        result = subprocess.run(cmd, cwd=REPO, env=child_env())
    else:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            result = subprocess.run(cmd, cwd=REPO, env=child_env(), stdout=f, stderr=subprocess.STDOUT)
    if check and result.returncode != 0:
        raise StepFailed(f'step failed (exit {result.returncode}): {shlex.join(cmd)}' + (f'; see {log}' if log else ''))
    return result.returncode


class StepFailed(RuntimeError):
    pass


def require_gpu_setup(need_mano=True):
    if DRY_RUN:
        return
    if need_mano and not (MANO and Path(MANO).is_dir()):
        sys.exit('set CH_MANO_EXPORT to the MANO export folder (docs/SETUP_NEW_MACHINE.md, section 4.3); Contact-AUC needs it')
    check = subprocess.run([PY, '-c', 'from genesis.engine.solvers.rigid.collider_decomp import Collider; import sys; '
                            'sys.exit(0 if hasattr(Collider, "check_overflow") else 1)'],
                           cwd=REPO, env=child_env(), capture_output=True)
    if check.returncode != 0:
        sys.exit('the imported Genesis lacks the collision-pair overflow fix (fca4168, docs/METHOD.md): set up the environment '
                 'with scripts/setup_dexmachina_env.sh, or put a patched Genesis checkout on PYTHONPATH')


def manifest_path(group, hand, task):
    """One frozen start state per (group, hand, task): every run of the group must start from it byte for byte."""
    return OUT / 'resets' / group / hand / task_name(task) / 'evaluation_reset.json'


def run_key(label, hand, task, seed):
    return f'{label}__{hand}__{task_name(task)}__seed{seed}'


def summary_path(label, hand, task, seed):
    return OUT / 'summaries' / label / hand / task_name(task) / f'seed{seed}.json'


def git_head():
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------- evaluate
def evaluate_run(run_dir, task, hand, label, seed, group, dexmachina=False, accept_selection=None, updates=None):
    """The paper's per-run evaluation: eval_rl_games -> compute_add -> track2_metrics -> slip_record -> slip_auc -> summary."""
    run_dir, task = Path(run_dir), task_name(task)
    checkpoint = run_dir / 'nn' / f'{hand}.pth'
    clip, key = clip_of(task), run_key(label, hand, task, seed)
    obj = clip.split('-')[0]
    logs = OUT / 'logs' / key
    manifest = manifest_path(group, hand, task)
    if not DRY_RUN and not checkpoint.is_file():
        raise StepFailed(f'no checkpoint {checkpoint}')
    evaluation = run_dir / f'{hand}_eval_all_policy'
    protocol = evaluation / 'eval_protocol.json'
    complete = protocol.exists() and json.loads(protocol.read_text()).get('status') == 'complete'
    if not complete:
        if evaluation.exists() and not DRY_RUN:
            shutil.rmtree(evaluation)          # created with exist_ok=False; a failed attempt would block every retry
        run([PY, '-m', 'dexmachina.rl.eval_rl_games', '--checkpoint', checkpoint, '--num_envs', LANES, '--all_policy_envs',
             '--eval-seed', EVAL_SEED, '--eval-reset-manifest', manifest, *(['--zero-pose-last-lane'] if dexmachina else []),
             *(['--accept-selection', accept_selection] if accept_selection else [])], logs / 'eval.log')
    if DRY_RUN or not (evaluation / 'add_stats.json').exists():
        run([PY, '-m', 'dexmachina.eval.compute_add', '--input', evaluation / 'eval_ep0.npy', '--obj_name', obj, '--overwrite'],
            logs / 'compute_add.log')
    if DRY_RUN or not (evaluation / 'track2_metrics.json').exists():
        run([PY, '-m', 'baselines.track2_metrics', '--input', evaluation / 'eval_ep0.npy', '--obj_name', obj],
            logs / 'track2_metrics.log')
    contact_auc(run_dir, task, hand, label, seed, group, dexmachina)
    if DRY_RUN:
        return None
    return write_summary(run_dir, task, hand, label, seed, group, updates)


def contact_auc(run_dir, task, hand, label, seed, group, dexmachina=False):
    """Contact-AUC: re-record the evaluation rollout from the same start state (slip_record), then score it (slip_auc)."""
    run_dir, task = Path(run_dir), task_name(task)
    checkpoint = run_dir / 'nn' / f'{hand}.pth'
    slip, logs = run_dir / 'slip', OUT / 'logs' / run_key(label, hand, task, seed)
    record = [PY, '-m', 'dexmachina.eval.slip_record', '--checkpoint', checkpoint, '--clip', clip_of(task),
              '--manifest', manifest_path(group, hand, task), '--task', TASKS[task], '--out', slip,
              *(['--zero-pose-last-lane'] if dexmachina else [])]
    if DRY_RUN or not (slip / 'record.json').exists():
        if slip.exists() and not DRY_RUN:
            shutil.rmtree(slip)
        code = run(record, logs / 'slip_record.log', check=not dexmachina)
        if code != 0:
            # DexMachina's checkpoints only, as in the paper: its evaluator starts the last lane from a zero hand pose, and
            # when that lane resets early, the recording keeps going with that lane's states set to NaN (scored as lost).
            if 'unexpected reset' not in Path(logs / 'slip_record.log').read_text():
                raise StepFailed(f'slip_record failed; see {logs / "slip_record.log"}')
            shutil.rmtree(slip, ignore_errors=True)
            run(record + ['--allow-last-lane-reset'], logs / 'slip_record_last_lane.log')
    score = [PY, '-m', 'dexmachina.eval.slip_auc', slip, '--mano', MANO or '<CH_MANO_EXPORT>']
    if DRY_RUN or not (slip / 'slip_auc.json').exists() or '"slip_auc"' not in (slip / 'slip_auc.json').read_text():
        code = run(score, logs / 'slip_auc.log', check=False)
        if code != 0 and not DRY_RUN:
            # slip_auc.py writes slip_auc.json before its summary print; a run with no touching window (coverage 0) only
            # fails that print. Keep the written result; anything else is an error.
            text = Path(logs / 'slip_auc.log').read_text()
            written = (slip / 'slip_auc.json').exists() and 'slip_auc' in json.loads((slip / 'slip_auc.json').read_text())
            if not (written and 'unsupported format string passed to NoneType' in text):
                raise StepFailed(f'slip_auc failed; see {logs / "slip_auc.log"}')


def completed_updates(run_dir, log=None):
    run_dir = Path(run_dir)
    for name in ('final_checkpoint.json', 'train_complete.json'):
        if (run_dir / name).exists():
            value = json.loads((run_dir / name).read_text()).get('completed_updates')
            if value is not None:
                return value
    info = run_dir / 'RUN_INFO.json'
    if info.exists():
        record = json.loads(info.read_text())
        value = record.get('completed_updates', (record.get('checkpoint') or {}).get('completed_updates'))
        if value is not None:
            return value
    if log and Path(log).exists():   # unified-policy row (e): no final_checkpoint.json; rl_games prints the epoch
        epochs = [int(m) for m in re.findall(r'epoch: (\d+)', Path(log).read_text(errors='replace'))]
        return max(epochs) if epochs else None
    return None


def release_table_metrics(run_dir):
    """A released run's reported values (RUN_INFO.json table_metrics, in % and cm), to compare the reproduction with;
    None for runs trained here."""
    info = Path(run_dir) / 'RUN_INFO.json'
    return json.loads(info.read_text()).get('table_metrics') if info.exists() else None


def write_summary(run_dir, task, hand, label, seed, group, updates=None):
    run_dir = Path(run_dir)
    evaluation = run_dir / f'{hand}_eval_all_policy'
    add = json.loads((evaluation / 'add_stats.json').read_text())
    kit = json.loads((evaluation / 'track2_metrics.json').read_text())
    contact = json.loads((run_dir / 'slip' / 'slip_auc.json').read_text())
    protocol = json.loads((evaluation / 'eval_protocol.json').read_text())
    mppe = [v for v in kit.get('per_rollout_mppe_cm', [])[:LANES - 1] if v is not None]
    summary = dict(
        label=label, hand=hand, task=task_name(task), seed=int(seed), run_dir=str(run_dir),
        checkpoint_sha256=sha256(run_dir / 'nn' / f'{hand}.pth'),
        add_auc=add['auc']['overall'], sp_sr=kit['sp_sr'], mp_sr=kit['mp_sr'], mppe_cm=kit['mppe_cm'],
        mppe_cm_lanes0_98=sum(mppe) / len(mppe) if mppe else None, nan_rollouts=kit.get('nan_rollouts', 0),
        contact_auc=contact.get('slip_auc'), contact_coverage=contact.get('coverage'),
        updates=updates if updates is not None else completed_updates(run_dir),
        checkpoint_selection=protocol.get('checkpoint_selection'), checkpoint_epoch=protocol.get('checkpoint_epoch'),
        start_state_manifest=str(manifest_path(group, hand, task)),
        start_state_check=(protocol.get('common_reset') or {}).get('verification'),
        evaluation=dict(lanes=LANES, eval_seed=EVAL_SEED, first_observation=protocol.get('first_observation')),
        release_table_metrics=release_table_metrics(run_dir), repo_commit=git_head())
    path = summary_path(label, hand, task, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1) + '\n')
    contact_text = 'n/a' if summary['contact_auc'] is None else f"{100 * summary['contact_auc']:.1f}"
    print(f"[{label} {hand} {task} seed {seed}] ADD-AUC {100 * summary['add_auc']:.1f}  SP-SR {100 * summary['sp_sr']:.1f}  "
          f"MP-SR {100 * summary['mp_sr']:.1f}  MPPE {summary['mppe_cm']:.2f} cm  Contact-AUC {contact_text}  -> {path}",
          flush=True)
    return summary


# ---------------------------------------------------------------------------------------------------------------- release
# The released packages (scripts/download_checkpoints.sh) and the folder each unpacks to under $DEXDUET_CHECKPOINTS.
PACKAGES = {'dexduet_dexmachina_suite.zip': 'dexduet',        # dexduet/<hand>/<task>/seed<S>/
            'dexduet_ablations.zip': 'ablations'}              # ablations/row_<b..g>/inspire_hand/<task>/seed<S>/
SUITE_PACKAGES = {'main': ['dexduet_dexmachina_suite.zip'],
                  'ablation': ['dexduet_dexmachina_suite.zip', 'dexduet_ablations.zip'],   # row (a) = the suite's Inspire runs
                  'all': list(PACKAGES)}


def release_runs(suite, hands, tasks, seeds, rows):
    """(label, hand, task, seed, folder) of the unpacked release, filtered."""
    out = []
    if suite in ('main', 'all'):
        for hand in hands:
            for task in tasks:
                for seed in seeds or MAIN_SEEDS:
                    out.append(('DexDuet', hand, task, seed, CHECKPOINTS / 'dexduet' / hand / task / f'seed{seed}'))
    if suite in ('ablation', 'all'):
        if suite == 'ablation':   # row (a) = DexDuet's Inspire runs with the ablation seeds
            for task in tasks:
                for seed in seeds or ABLATION_SEEDS:
                    out.append(('DexDuet', 'inspire_hand', task, seed, CHECKPOINTS / 'dexduet' / 'inspire_hand' / task / f'seed{seed}'))
        for row in rows:
            for task in tasks:
                for seed in seeds or ABLATION_SEEDS:
                    out.append((f'row_{row}', 'inspire_hand', task, seed,
                                CHECKPOINTS / 'ablations' / f'row_{row}' / 'inspire_hand' / task / f'seed{seed}'))
    return out


def stage_release(folder, label, hand, task, seed):
    """Released folder -> evaluable run folder (rebuilt env.pkl). Returns (run dir, accepted selection or None)."""
    base = OUT / 'staged' / label / hand / task_name(task) / f'seed{seed}'
    staged = [p.parent for p in base.glob('*/STAGED.json')] if base.exists() else []
    final = Path(folder) / 'final_checkpoint.json'
    selection = json.loads(final.read_text()).get('selection') if final.exists() else None
    accept = selection if selection not in (None, 'native_best_training_reward') else None
    if staged:
        return staged[0], accept
    run([PY, 'scripts/stage_release_run.py', folder, '--out', base, '--task', task], OUT / 'logs' /
        run_key(label, hand, task, seed) / 'stage.log')
    if DRY_RUN:
        return base / '<original run name>', accept
    staged = [p.parent for p in base.glob('*/STAGED.json')]
    if len(staged) != 1:
        raise StepFailed(f'staging {folder} produced no run folder under {base}')
    return staged[0], accept


def cmd_from_release(a):
    missing = [p for p in SUITE_PACKAGES[a.suite] if not (CHECKPOINTS / PACKAGES[p]).is_dir()]
    if missing and not a.skip_download:
        run(['bash', REPO / 'scripts' / 'download_checkpoints.sh', *missing])
    require_gpu_setup()
    done, failed, absent = [], [], []
    for label, hand, task, seed, folder in release_runs(a.suite, a.hand, a.task, a.seed, a.row):
        if summary_path(label, hand, task, seed).exists() and not DRY_RUN:
            done.append(run_key(label, hand, task, seed))
            continue
        if not folder.is_dir() and not DRY_RUN:
            absent.append(str(folder))
            continue
        try:
            run_dir, accept = stage_release(folder, label, hand, task, seed)
            evaluate_run(run_dir, task, hand, label, seed, 'dexduet', accept_selection=accept)
            done.append(run_key(label, hand, task, seed))
        except StepFailed as exc:
            failed.append(f'{run_key(label, hand, task, seed)}: {exc}')
            print('FAILED', failed[-1], flush=True)
    print(f'from-release: {len(done)} runs evaluated or already done, {len(failed)} failed, {len(absent)} not in the package')
    for item in failed + [f'not in the package: {p}' for p in absent]:
        print('  ', item)
    if not DRY_RUN:
        cmd_table(argparse.Namespace(results=None))
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------------------------ train
def cmd_main(a):
    require_gpu_setup()
    task, hand = task_name(a.task), a.hand
    group = 'smoke' if a.smoke else 'dexduet'
    label = 'DexDuet_smoke' if a.smoke else 'DexDuet'
    if summary_path(label, hand, task, a.seed).exists() and not DRY_RUN:
        print(f'{summary_path(label, hand, task, a.seed)} exists: nothing to do')
        return 0
    experiment = f'coordinates-hand_{task}_seed{a.seed}' + ('_smoke' if a.smoke else '')
    prefix = LAUNCHER.hand_prefix(hand)
    found = sorted((REPO / 'logs' / 'rl_games' / hand).glob(f'{prefix}-{experiment}_*'))
    trained = len(found) == 1 and (found[0] / 'final_checkpoint.json').exists()
    if found and not trained and not DRY_RUN:
        sys.exit(f'{found[0]} is left from an unfinished training (Genesis runs cannot resume); move it away and start again')
    manifest = manifest_path(group, hand, task)
    bc = ['--bc-weight', SCHUNK_LONG_CLIP_BC] if hand == 'schunk_hand' and TASKS[task] in LAUNCHER.LONG_CLIP_TASKS else []
    log = OUT / 'logs' / run_key(label, hand, task, a.seed) / 'train_and_evaluate.log'
    # The launcher trains, evaluates and scores ADD-AUC and Contact-AUC. Its last step, slip_auc, exits non-zero after
    # writing its result when a policy never touches the object (coverage 0); evaluate_run below runs every step that is
    # still missing (all of them for a run trained earlier) and accepts that case.
    if not trained:
        run([PY, 'scripts/train_coordinates_hand.py', '--task', TASKS[task], '--hand', hand, '--seed', a.seed,
             '--experiment', experiment, '--reset-manifest', manifest, '--mano', MANO or '<CH_MANO_EXPORT>', *bc,
             *(['--smoke'] if a.smoke else [])], log, check=False)
    if DRY_RUN:
        evaluate_run(f'<run folder of {experiment}>', task, hand, label, a.seed, group)
        return 0
    found = sorted((REPO / 'logs' / 'rl_games' / hand).glob(f'{prefix}-{experiment}_*'))
    if len(found) != 1 or not (found[0] / 'final_checkpoint.json').exists():
        raise StepFailed(f'training of {experiment} did not finish; see {log}')
    evaluate_run(found[0], task, hand, label, a.seed, group)
    return 0


def cmd_ablation(a):
    require_gpu_setup()
    row, task, hand = a.row, task_name(a.task), 'inspire_hand'
    label = f'row_{row}' + ('_smoke' if a.smoke else '')
    group = 'smoke' if a.smoke else 'dexduet'
    argv = subprocess.run([PY, 'baselines/ablation_rows.py', '--row', row, '--task', task, '--seed', str(a.seed)],
                          cwd=REPO, env=child_env(), capture_output=True, text=True, check=True).stdout.split('\n')
    argv = [x for x in argv if x != '']
    experiment = argv[argv.index('-exp') + 1] + ('_smoke' if a.smoke else '')
    argv[argv.index('-exp') + 1] = experiment
    if a.smoke:   # the launcher's setup check: same flags, 256 envs, 20 updates, assistance off at the end
        for flag, value in (('-B', '256'), ('--max_epochs', '20'), ('--num_zero_epoch', '0'), ('--save_freq', '10'),
                            ('--study_profile', 'validation')):
            if flag in argv:
                argv[argv.index(flag) + 1] = value
    runs = REPO / 'logs' / 'rl_games' / hand
    found = sorted(runs.glob(f'inspire-{experiment}_*'))
    key = run_key(label, hand, task, a.seed)
    train_log = OUT / 'logs' / key / 'train.log'
    # A finished training leaves final_checkpoint.json (per-hand rows) or, written here, train_complete.json (row e).
    finished = lambda d: (d / 'final_checkpoint.json').exists() or (d / 'train_complete.json').exists()
    if found and not finished(found[0]) and not DRY_RUN:
        sys.exit(f'{found[0]} is left from an unfinished training (Genesis runs cannot resume); move it away and start again')
    if not found or DRY_RUN:
        run([PY, 'dexmachina/rl/train_rl_games.py', *argv], train_log)
        found = sorted(runs.glob(f'inspire-{experiment}_*'))
        if len(found) == 1 and not DRY_RUN:
            (found[0] / 'train_complete.json').write_text(json.dumps(dict(
                completed_updates=completed_updates(found[0], train_log), train_log=str(train_log)), indent=1) + '\n')
    if DRY_RUN:
        evaluate_run(f'<run folder of {experiment}>', task, hand, label, a.seed, group)
        return 0
    if len(found) != 1:
        sys.exit(f'expected one run folder {runs}/inspire-{experiment}_*, found {len(found)}')
    updates = completed_updates(found[0], train_log)
    evaluate_run(found[0], task, hand, label, a.seed, group, updates=updates)
    return 0


# ---------------------------------------------------------------------------------------------------------------- DexMachina
def dexmachina_runs(archive, hands, tasks, seeds=None):
    """(hand, task, seed, run folder) of DexMachina's released checkpoints (run_paths.json of their main-results data)."""
    archive = Path(archive)
    paths = json.loads((archive / 'run_paths.json').read_text())
    clip_to_task = {clip_of(t).replace('-', '', 1): t for t in TASKS}      # 'box30-230-s01-u01' -> 'box-200'
    out = []
    for hand in hands:
        for clip, entries in paths.get(hand, {}).items():
            task = clip_to_task.get(clip)
            if task is None or task not in tasks:
                continue
            for entry in entries:
                root = archive / 'runs' / Path(entry).name
                seed = json.loads((root / 'config.json').read_text())['params']['seed']
                if seeds and int(seed) not in seeds:
                    continue
                folders = [d for d in (root / 'logs' / 'rl_games' / hand).iterdir() if (d / 'nn' / f'{hand}.pth').is_file()]
                if len(folders) != 1:
                    sys.exit(f'{root}: expected one run folder with nn/{hand}.pth under logs/rl_games/{hand}')
                out.append((hand, task, int(seed), folders[0]))
    return out


def cmd_dexmachina(a):
    require_gpu_setup()
    failed = []
    for hand, task, seed, folder in dexmachina_runs(a.archive, a.hand, a.task, a.seed):
        label = 'DexMachina'
        if summary_path(label, hand, task, seed).exists():
            continue
        staged = OUT / 'staged' / label / hand / task / f'seed{seed}' / folder.name
        try:
            if not (staged / 'params' / 'env.pkl.orig').exists():
                (staged / 'nn').mkdir(parents=True, exist_ok=True)
                (staged / 'params').mkdir(exist_ok=True)
                shutil.copy2(folder / 'nn' / f'{hand}.pth', staged / 'nn' / f'{hand}.pth')
                shutil.copy2(folder / 'params' / 'env.pkl', staged / 'params' / 'env.pkl')
                run([PY, 'scripts/relocate_run.py', staged], OUT / 'logs' / run_key(label, hand, task, seed) / 'relocate.log')
            if not (staged / 'params' / 'agent.yaml').exists():
                # the release has no agent config; its network is the repository's default one, which the evaluator
                # loads by itself and the Contact-AUC recorder reads from params/agent.yaml
                shutil.copy2(REPO / 'dexmachina' / 'rl' / 'configs' / 'rl_games_ppo_cfg.yaml', staged / 'params' / 'agent.yaml')
            evaluate_run(staged, task, hand, label, seed, 'dexmachina', dexmachina=True)
        except StepFailed as exc:
            failed.append(f'{run_key(label, hand, task, seed)}: {exc}')
            print('FAILED', failed[-1], flush=True)
    for item in failed:
        print('  ', item)
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------------------------- single steps
def resolve_run(a):
    """--run: a run folder (nn/<hand>.pth) or a released folder (config/env.yaml), which is staged first."""
    folder = Path(a.run).resolve()
    if (folder / 'config' / 'env.yaml').exists() and not (folder / 'params' / 'env.pkl').exists():
        info = json.loads((folder / 'RUN_INFO.json').read_text()) if (folder / 'RUN_INFO.json').exists() else {}
        hand = a.hand or info.get('hand_key')
        task = a.task or (info.get('task') or '').lower()
        seed = a.seed if a.seed is not None else info.get('training_seed')
        if not (hand and task and seed is not None):
            sys.exit('give --hand, --task and --seed for this released folder')
        run_dir, accept = stage_release(folder, a.label, hand, task, seed)
        return run_dir, hand, task_name(task), seed, accept
    hand = a.hand or next((h for h in HANDS + ('dex3_hand', 'ability_hand', 'dexrobot_hand') if (folder / 'nn' / f'{h}.pth').exists()), None)
    if hand is None or a.task is None or a.seed is None:
        sys.exit('give --task and --seed (and --hand unless nn/<hand>.pth identifies it)')
    return folder, hand, task_name(a.task), a.seed, None


def cmd_evaluate(a):
    require_gpu_setup()
    run_dir, hand, task, seed, accept = resolve_run(a)
    evaluate_run(run_dir, task, hand, a.label, seed, a.group, dexmachina=a.group == 'dexmachina',
                 accept_selection=accept)
    return 0


def cmd_contact_auc(a):
    require_gpu_setup()
    run_dir, hand, task, seed, _ = resolve_run(a)
    contact_auc(run_dir, task, hand, a.label, seed, a.group, dexmachina=a.group == 'dexmachina')
    if not DRY_RUN:
        result = json.loads((Path(run_dir) / 'slip' / 'slip_auc.json').read_text())
        print(f"Contact-AUC {result.get('slip_auc')} (coverage {result.get('coverage')}) -> {Path(run_dir) / 'slip' / 'slip_auc.json'}")
    return 0


def cmd_kit_metrics(a):
    folder = Path(a.run).resolve()
    evals = sorted(folder.glob('*_eval_all_policy/eval_ep0.npy'))
    if not evals and not DRY_RUN:
        sys.exit(f'no <hand>_eval_all_policy/eval_ep0.npy under {folder}: run `evaluate` first')
    for path in evals or [folder / '<hand>_eval_all_policy' / 'eval_ep0.npy']:
        run([PY, '-m', 'baselines.track2_metrics', '--input', path, *(['--obj_name', a.obj_name] if a.obj_name else [])])
    return 0


def cmd_hold_test(a):
    """Hold after the last frame: the scored evaluation replayed, then 180 more steps with every reference frozen."""
    if a.analyze:
        run([PY, '-m', 'dexmachina.eval.hold_test.analyze', OUT / 'hold_test', '--out', OUT / 'tables' / 'hold_test'])
        return 0
    require_gpu_setup(need_mano=False)
    run_dir, hand, task, seed, _ = resolve_run(a)
    dexmachina = a.group == 'dexmachina'
    evaluation = Path(run_dir) / f'{hand}_eval_all_policy'
    if not (evaluation / 'eval_ep0.npy').exists() and not DRY_RUN:
        sys.exit(f'{evaluation}/eval_ep0.npy is missing: run `evaluate` on this run first (the hold test replays it)')
    key = run_key(a.label, hand, task, seed)
    out, logs = OUT / 'hold_test' / key, OUT / 'logs' / key
    if out.exists() and not DRY_RUN:
        sys.exit(f'{out} exists (outputs are never overwritten)')
    slip = Path(run_dir) / 'slip' / 'recording.npz'
    table_add = (100 * json.loads((evaluation / 'add_stats.json').read_text())['auc']['overall']
                 if (evaluation / 'add_stats.json').exists() else None)
    run([PY, '-m', 'dexmachina.eval.hold_test.run', '--checkpoint', Path(run_dir) / 'nn' / f'{hand}.pth',
         '--manifest', manifest_path(a.group, hand, task), '--reference', evaluation / 'eval_ep0.npy', '--out', out,
         '--hold-steps', 180, '--method', 'DexMachina' if dexmachina else 'DexDuet', '--task', task, '--seed', seed,
         *(['--slip-recording', slip] if slip.exists() else []), *(['--table-add', table_add] if table_add is not None else []),
         *(['--zero-pose-last-lane'] if dexmachina else [])], logs / 'hold_run.log')
    run([PY, '-m', 'dexmachina.eval.hold_test.add', out], logs / 'hold_add.log')
    run([PY, '-m', 'dexmachina.eval.hold_test.metrics', out], logs / 'hold_metrics.log')
    if not DRY_RUN:
        m = json.loads((out / 'hold_metrics.json').read_text())
        print(f"Hold-ADD-AUC {m['u_hold_add_auc']:.1f}  SR@5 {m['u_hold_sr5']:.1f}  drop {m['u_drop_rate']:.1f}  on-track lanes "
              f"{m['n_ontrack']}  flagged {m['flagged']} -> {out / 'hold_metrics.json'}")
    return 0


def cmd_table(a):
    results = Path(a.results).resolve() if a.results else OUT / 'summaries'
    run([PY, 'scripts/paper_table.py', results, '--out', OUT / 'tables'])
    return 0


# ---------------------------------------------------------------------------------------------------------------- CLI
def main():
    global DRY_RUN
    ap = argparse.ArgumentParser(prog='scripts/reproduce_paper.sh', description=__doc__.split('\n')[0])
    ap.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    sub = ap.add_subparsers(dest='command', required=True)

    def filters(p, rows=False):
        p.add_argument('--hand', nargs='+', default=list(HANDS), choices=HANDS)
        p.add_argument('--task', nargs='+', default=sorted(TASKS), type=task_name)
        p.add_argument('--seed', nargs='+', type=int, default=None)
        if rows:
            p.add_argument('--row', nargs='+', default=list(ROWS), choices=ROWS)

    p = sub.add_parser('from-release', help='download the released checkpoints, evaluate them, write the tables')
    p.add_argument('--suite', choices=('main', 'ablation', 'all'), default='all')
    p.add_argument('--skip-download', action='store_true', help='use what is unpacked under DEXDUET_CHECKPOINTS')
    filters(p, rows=True)
    p.set_defaults(func=cmd_from_release)

    p = sub.add_parser('main', help='train DexDuet on one task, hand and seed, then evaluate it (main table)')
    p.add_argument('--task', required=True, type=task_name)
    p.add_argument('--hand', default='inspire_hand', choices=HANDS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='setup check: 256 environments, 20 updates')
    p.set_defaults(func=cmd_main)

    p = sub.add_parser('ablation', help='train one ablation row (b..g) on one task and seed (Inspire), then evaluate it')
    p.add_argument('--row', required=True, choices=ROWS)
    p.add_argument('--task', required=True, type=task_name)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true', help='setup check: 256 environments, 20 updates')
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser('dexmachina', help="evaluate DexMachina's released checkpoints (their main-results data, unzipped)")
    p.add_argument('--archive', required=True, help='the unzipped dexmachina_main_results_data folder (run_paths.json, runs/)')
    filters(p)
    p.set_defaults(func=cmd_dexmachina)

    for name, func, text in (('evaluate', cmd_evaluate, 'evaluate one run: ADD-AUC, SP-SR, MP-SR, MPPE, Contact-AUC'),
                             ('contact-auc', cmd_contact_auc, 'Contact-AUC of one evaluated run'),
                             ('hold-test', cmd_hold_test, 'hold after the last frame of one evaluated run (or --analyze)')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--run', required=name != 'hold-test',
                       help='run folder (nn/<hand>.pth) or released run folder')
        p.add_argument('--task', default=None)
        p.add_argument('--hand', default=None)
        p.add_argument('--seed', type=int, default=None)
        p.add_argument('--label', default='DexDuet', help='name of the row in the tables (DexDuet, DexMachina, row_b, ...)')
        p.add_argument('--group', default='dexduet', choices=('dexduet', 'dexmachina', 'smoke'),
                       help='start-state group: dexmachina for its released checkpoints, dexduet for everything trained here')
        if name == 'hold-test':
            p.add_argument('--analyze', action='store_true', help=f'run the analysis over all {name} results')
        p.set_defaults(func=func)

    p = sub.add_parser('kit-metrics', help='SP-SR, MP-SR and MPPE (V2D challenge definitions) of an evaluated run')
    p.add_argument('--run', required=True)
    p.add_argument('--obj-name', dest='obj_name', default=None)
    p.set_defaults(func=cmd_kit_metrics)

    p = sub.add_parser('table', help='aggregate the per-run summaries into the paper tables')
    p.add_argument('--results', default=None, help='summaries folder (default $DEXDUET_OUT/summaries)')
    p.set_defaults(func=cmd_table)

    a = ap.parse_args()
    DRY_RUN = a.dry_run
    if a.command == 'hold-test' and not a.analyze and not a.run:
        ap.error(f'{a.command} needs --run (or --analyze)')
    try:
        sys.exit(a.func(a))
    except StepFailed as exc:
        sys.exit(f'FAILED: {exc}')


if __name__ == '__main__':
    main()
