"""Turn a released DexDuet run folder into a run folder that dexmachina/rl/eval_rl_games.py can evaluate on this machine.

A released run ships its checkpoint and configuration but not params/env.pkl: that file holds the ARCTIC demonstration and
its retargeted robot reference, which are derived from licensed data and cannot be redistributed. This script rebuilds it
from the shipped config/env.yaml (everything except those two arrays) and your own processed ARCTIC data and retargeting
under dexmachina/assets/ (docs/SETUP_NEW_MACHINE.md, section 4), the same way scripts/train_coordinates_hand.py builds it
for training:

  demo_data      dexmachina.envs.demo_data.get_demo_data (+ ho_contact.load_human_targets when the contact reward is on)
  retarget_data  dexmachina.envs.demo_data.load_genesis_retarget_data (retargeting name 'para', as in every paper run)

Asset paths are repointed at this repository, and a configuration stored before two options were renamed is read into the
current layout (learner_cfg, dexmachina/envs/learner_config.py). Release layout (input):

  <hand>/<task>/seed<S>/<hand>.pth, RUN_INFO.json, config/{env.yaml, agent.yaml}; with the per-hand policy also
  final_checkpoint.json and config/initialization.json

Staged layout (output; the folder name is the original run name, which final_checkpoint.json records):

  <out>/<original run name>/nn/<hand>.pth, params/{env.pkl, env.yaml, agent.yaml}, final_checkpoint.json, RUN_INFO.json

  python scripts/stage_release_run.py checkpoints/dexduet/inspire_hand/box-200/seed15 --out paper_repro/staged/x
  python scripts/stage_release_run.py <release run> --out <dir> --compare-env-pkl <original params/env.pkl>   # check

Prerequisites (REPRODUCE.md, section 1): the generated assets under dexmachina/assets (arctic/processed, retargeted,
retargeter_results, contact_retarget, ho_contact). The script lists whatever is missing, with the step that creates it.

environment/generated-assets-sha256.txt lists the generated Inspire files of DexDuet's main-table runs; with files that match
it, the rebuilt env.pkl of such a run equals the one it was trained and evaluated with. The runs of ablation rows (b)-(g)
were trained with other processings of the same ARCTIC sequences. A run with the per-hand policy records the sha256 of its
own processed demonstration in config/env.yaml (source_sha256); when the file here differs, the report says so ("notice")
and the run is evaluated with the file here. With your own processing or retargeting the start state can differ slightly, and
so can the scores.
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path, PurePath, PurePosixPath

import yaml

REPO = Path(__file__).resolve().parents[1]
SAVE_ONLY_KEYS = ('ablation_study', 'wandb')   # written to env.yaml by train_rl_games.py, never part of env.pkl


class ConfigLoader(yaml.SafeLoader):
    """Safe YAML loading plus the two Python tags that train_rl_games.py's env.yaml uses (tuples and POSIX paths)."""


ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: tuple(loader.construct_sequence(node)))
ConfigLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath',
                             lambda loader, node: Path(*loader.construct_sequence(node)))


def load_env_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=ConfigLoader)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def launcher():
    spec = importlib.util.spec_from_file_location('train_coordinates_hand', REPO / 'scripts' / 'train_coordinates_hand.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def learner_config():
    """dexmachina/envs/learner_config.py, loaded by path (standard library only; no simulator import)."""
    spec = importlib.util.spec_from_file_location('learner_config', REPO / 'dexmachina' / 'envs' / 'learner_config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upgrade_legacy_env_kwargs(kw):
    """Read a configuration stored before the learner options were renamed into the current layout (in place)."""
    return learner_config().upgrade_legacy_env_kwargs(kw)


def task_key(task):
    keys = {key.lower(): key for key in launcher().TASKS}
    if task.lower() not in keys:
        raise SystemExit(f'unknown task {task}; one of {sorted(keys)}')
    return keys[task.lower()]


def repoint_assets(kw, repo=REPO):
    """Repoint every asset path at <repo>/dexmachina/assets (scripts/relocate_run.py's rule)."""
    spec = importlib.util.spec_from_file_location('relocate_run', REPO / 'scripts' / 'relocate_run.py')
    relocate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relocate)
    changed, missing = [], []
    relocate.repoint(kw, repo, changed, missing)
    if missing:
        raise SystemExit('missing in this repository (the generated assets: REPRODUCE.md, section 1):\n'
                         + '\n'.join(f'{p}: {new}' for p, _old, new in missing))
    return changed


SETUP_HINT = ('These files come from your own ARCTIC download and the data steps of REPRODUCE.md, section 1 '
              '(scripts/reproduce.sh setup, download-arctic and prepare, then the MANO export and dexmachina/hand_proc/'
              'ho_contact_assets.py grids and human; docs/SETUP_NEW_MACHINE.md, section 4).')


def processed_demonstration(clip, assets=REPO / 'dexmachina' / 'assets'):
    """The processed ARCTIC file of a clip such as box-30-230-s01-u01 (constructors.parse_clip_string)."""
    obj, _start, _end, subject, use = clip.split('-')
    return Path(assets) / 'arctic' / 'processed' / subject / f"{obj}_use_{use.replace('u', '')}.npy"


def recorded_demonstration_sha256(kw):
    """The sha256 of the processed demonstration that a released config/env.yaml records (source_sha256 next to a
    source_path under arctic/processed/), or None. Runs with the per-hand policy record it; row (e) and DexMachina's do not."""
    found, todo = set(), [kw]
    while todo:
        item = todo.pop()
        if isinstance(item, dict):
            if isinstance(item.get('source_sha256'), str) and '/arctic/processed/' in str(item.get('source_path', '')):
                found.add(item['source_sha256'])
            todo.extend(item.values())
        elif isinstance(item, (list, tuple)):
            todo.extend(item)
    return found.pop() if len(found) == 1 else None


def demonstration_check(recorded, clip, assets=REPO / 'dexmachina' / 'assets'):
    """The processed demonstration here against the one the run recorded: (check, notice or None)."""
    path = processed_demonstration(clip, assets)
    here = sha256(path)
    check = dict(file=str(path), recorded=recorded, here=here, matches=None if recorded is None else recorded == here)
    notice = None
    if check['matches'] is False:
        notice = (f"{path}: sha256 {here} differs from the processed demonstration this run recorded (config/env.yaml "
                  f"source_sha256 {recorded}); the run is evaluated with the file here, so its scores can differ from the "
                  "reported ones (REPRODUCE.md, section 1)")
    return check, notice


def missing_inputs(kw, clip, hand, retarget_name='para'):
    """The licensed-data files that rebuilding this run's env.pkl reads, with the step that creates each one."""
    obj, _start, _end, subject, use = clip.split('-')     # e.g. box-30-230-s01-u01 (constructors.parse_clip_string)
    use = use.replace('u', '')
    assets = REPO / 'dexmachina' / 'assets'
    reward = kw['reward_cfg']
    need = [(processed_demonstration(clip, assets), 'scripts/reproduce.sh prepare (ARCTIC processing)')]
    retarget = assets / 'retargeted' / hand / subject / f'{obj}_use_{use}_vector_{retarget_name}.npy'
    need.append((retarget if retarget.exists() else retarget.with_suffix('.pt'), 'scripts/reproduce.sh prepare (retargeting)'))
    if reward.get('use_retarget_contact'):
        need.append((assets / 'contact_retarget' / hand / subject / f'{obj}_use_{use}.npy',
                     'scripts/reproduce.sh prepare (contact mapping)'))
    if reward.get('ho_contact_weight', 0) > 0:
        need += [(assets / 'ho_contact' / 'grids' / f'{obj}_{part}.npz', 'dexmachina/hand_proc/ho_contact_assets.py grids')
                 for part in ('bottom', 'top')]
        need.append((assets / 'ho_contact' / 'human' / subject / f'{obj}_use_{use}.npz',
                      'dexmachina/hand_proc/ho_contact_assets.py human (needs the MANO export)'))
    return [(path, how) for path, how in need if not path.exists()]


def rebuild_data(kw, clip, hand, retarget_name='para'):
    """demo_data and retarget_data as dexmachina.envs.constructors.get_all_env_cfg builds them for this run."""
    from dexmachina.asset_utils import get_asset_path
    from dexmachina.envs.constructors import parse_clip_string
    from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data
    obj_name, start, end, subject, use_clip = parse_clip_string(clip)
    reward = kw['reward_cfg']
    _, retarget_data = load_genesis_retarget_data(obj_name=obj_name, hand_name=hand, frame_start=start, frame_end=end,
                                                  save_name=retarget_name, use_clip=use_clip, subject_name=subject)
    demo_data = get_demo_data(obj_name=obj_name, hand_name=hand, frame_start=start, frame_end=end, use_clip=use_clip,
                              subject_name=subject, load_retarget_contact=bool(reward.get('use_retarget_contact', False)),
                              load_hand_joints=reward.get('hand_graph_weight', 0) > 0)
    if reward.get('ho_contact_weight', 0) > 0:
        from dexmachina.envs.ho_contact import load_human_targets
        demo_data.update(load_human_targets(str(get_asset_path('')), subject, obj_name, use_clip, start, end))
    if int(end) - int(start) != kw['env_cfg']['episode_length']:
        raise SystemExit(f"clip {clip} has {int(end) - int(start)} frames, the run's episode_length is "
                         f"{kw['env_cfg']['episode_length']}")
    return demo_data, retarget_data


def compare(a, b, path='', out=None, tol=0.0):
    """Paths where two env kwargs differ (arrays compared exactly unless tol > 0)."""
    import numpy as np
    import torch
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append(f'{path}/{key}: only in {"rebuilt" if key in a else "original"}')
            else:
                compare(a[key], b[key], f'{path}/{key}', out, tol)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and not (
            isinstance(a, PurePath) or isinstance(b, PurePath)):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f'{path}[{i}]', out, tol)
    elif torch.is_tensor(a) or torch.is_tensor(b) or isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        y = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
        if x.shape != y.shape or x.dtype != y.dtype:
            out.append(f'{path}: shape/dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}')
        elif not (np.array_equal(x, y, equal_nan=x.dtype.kind == 'f') if tol == 0 else np.allclose(x, y, atol=tol, equal_nan=True)):
            out.append(f'{path}: values differ (max |diff| {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.3g})')
    elif isinstance(a, PurePath) or isinstance(b, PurePath):
        if str(a) != str(b):
            out.append(f'{path}: {a} vs {b}')
    elif a != b and not (a is None and b is None):
        out.append(f'{path}: {a!r} vs {b!r}')
    return out


def read_release_run(src):
    src = Path(src)
    info = json.loads((src / 'RUN_INFO.json').read_text()) if (src / 'RUN_INFO.json').exists() else {}
    final = json.loads((src / 'final_checkpoint.json').read_text()) if (src / 'final_checkpoint.json').exists() else {}
    checkpoints = sorted(src.glob('*.pth'))
    if len(checkpoints) != 1:
        raise SystemExit(f'{src}: expected one <hand>.pth, found {[c.name for c in checkpoints]}')
    checkpoint = checkpoints[0]
    hand = info.get('hand_key') or checkpoint.stem
    # the evaluator matches final_checkpoint.json's <run>/nn/<file> against the staged path, so its run name wins
    run_name = Path(final['checkpoint']).parts[-3] if final.get('checkpoint') else info.get('original_run_name')
    if not run_name:
        raise SystemExit(f'{src}: no RUN_INFO.json original_run_name and no final_checkpoint.json checkpoint path')
    return dict(src=src, info=info, final=final, checkpoint=checkpoint, hand=hand, run_name=run_name,
                task=info.get('task'), seed=info.get('training_seed'))


def stage(src, out, task=None, *, retarget_name='para', compare_env_pkl=None):
    run = read_release_run(src)
    task = task_key(task or run['task'] or '')
    clip = launcher().TASKS[task][0]
    expected = (run['info'].get('checkpoint') or {}).get('sha256') or run['final'].get('checkpoint_sha256')
    actual = sha256(run['checkpoint'])
    if expected and expected != actual:
        raise SystemExit(f"{run['checkpoint']}: sha256 {actual} differs from the release record {expected}")
    selection = run['final'].get('selection') or (run['info'].get('checkpoint') or {}).get('selection')
    kw = load_env_yaml(run['src'] / 'config' / 'env.yaml')
    recorded_demonstration = recorded_demonstration_sha256(kw)
    for key in SAVE_ONLY_KEYS:
        kw.pop(key, None)
    upgrade_legacy_env_kwargs(kw)
    missing = missing_inputs(kw, clip, run['hand'], retarget_name)
    if missing:
        raise SystemExit(f"cannot rebuild the environment of {run['src']}: missing\n"
                         + '\n'.join(f'  {path}   <- {how}' for path, how in missing) + '\n' + SETUP_HINT)
    changed = repoint_assets(kw)
    kw['demo_data'], kw['retarget_data'] = rebuild_data(kw, clip, run['hand'], retarget_name)
    report = dict(source=str(run['src']), task=task, clip=clip, hand=run['hand'], seed=run['seed'], checkpoint_sha256=actual,
                  selection=selection, repointed_paths=len(changed))
    report['demonstration'], notice = demonstration_check(recorded_demonstration, clip)
    if notice:
        report['notice'] = notice
        print(f'notice: {notice}', file=sys.stderr)
    if compare_env_pkl:
        original = upgrade_legacy_env_kwargs(pickle.load(open(compare_env_pkl, 'rb')))
        repoint_assets(original)
        report['compare_env_pkl'] = dict(original=str(compare_env_pkl), differences=compare(kw, original))
    staged = Path(out) / run['run_name']
    if staged.exists():
        raise SystemExit(f'{staged} exists; stage into a new folder')
    (staged / 'nn').mkdir(parents=True)
    (staged / 'params').mkdir()
    shutil.copy2(run['checkpoint'], staged / 'nn' / f"{run['hand']}.pth")
    with open(staged / 'params' / 'env.pkl', 'wb') as f:
        pickle.dump(kw, f)
    shutil.copy2(run['src'] / 'config' / 'env.yaml', staged / 'params' / 'env.yaml')
    shutil.copy2(run['src'] / 'config' / 'agent.yaml', staged / 'params' / 'agent.yaml')
    for name in ('final_checkpoint.json', 'RUN_INFO.json'):
        if (run['src'] / name).exists():
            shutil.copy2(run['src'] / name, staged / name)
    report['staged'] = str(staged)
    report['checkpoint'] = str(staged / 'nn' / f"{run['hand']}.pth")
    (staged / 'STAGED.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('release_run', help='<hand>/<task>/seed<S> folder of the release')
    ap.add_argument('--out', required=True, help='folder to stage into (the run folder is created inside it)')
    ap.add_argument('--task', default=None, help="task, e.g. Box-200 (default: RUN_INFO.json)")
    ap.add_argument('--reference', default=None, help=argparse.SUPPRESS)   # older option, no longer needed; ignored
    ap.add_argument('--retarget-name', default='para', help="retargeting save name (every paper run: 'para')")
    ap.add_argument('--compare-env-pkl', default=None, help='original params/env.pkl to compare the rebuilt one with')
    args = ap.parse_args()
    report = stage(args.release_run, args.out, args.task, retarget_name=args.retarget_name,
                   compare_env_pkl=args.compare_env_pkl)
    print(json.dumps(report, indent=1))
    if args.compare_env_pkl and report['compare_env_pkl']['differences']:
        sys.exit(f"rebuilt env.pkl differs from {args.compare_env_pkl} in {len(report['compare_env_pkl']['differences'])} places")


if __name__ == '__main__':
    main()
