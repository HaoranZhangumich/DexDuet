"""Make a training run folder copied from another machine evaluable here (evaluation only; the weights are untouched).

A run's params/env.pkl stores asset paths from the machine that trained it: the object URDF and meshes and the robot URDFs.
This script repoints every asset path nested in env.pkl at this repository's dexmachina/assets, both absolute ones
(.../dexmachina/assets/...) and the paths relative to dexmachina/ that DexMachina's released checkpoints use (assets/...),
and reads a configuration stored before two learner options were renamed into the current layout (learner_cfg,
dexmachina/envs/learner_config.py). The original is kept as params/env.pkl.orig. The checkpoint itself is matched by
<run>/nn/<file> and sha256, so final_checkpoint.json needs no change.

  python scripts/relocate_run.py <run dir>
"""
import argparse
import importlib.util
import pickle
import shutil
import sys
from pathlib import Path, PurePath

REPO = Path(__file__).resolve().parents[1]


def repoint(value, repo, changed, missing, path=''):
    """Repoint every asset path nested in `value` at <repo>/dexmachina/assets: absolute paths that contain
    /dexmachina/assets/ (object meshes, robot URDFs, which are Path objects) and paths relative to dexmachina/ ('assets/...',
    as in DexMachina's released checkpoints). Records (path, old, new) in `changed`, or in `missing` if the new file is absent."""
    if isinstance(value, dict):
        for key in list(value):
            value[key] = repoint(value[key], repo, changed, missing, f'{path}/{key}')
        return value
    if isinstance(value, list):
        return [repoint(v, repo, changed, missing, f'{path}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(repoint(v, repo, changed, missing, f'{path}[{i}]') for i, v in enumerate(value))
    if isinstance(value, (str, PurePath)):
        text = str(value)
        if '/dexmachina/assets/' in text:
            relative = text.split('/dexmachina/assets/', 1)[1]
        elif text.startswith('assets/'):
            relative = text[len('assets/'):]
        else:
            return value
        new = repo / 'dexmachina/assets' / relative
        if str(new) == text:
            return value
        (changed if new.exists() else missing).append((path, text, str(new)))
        return type(value)(new) if isinstance(value, PurePath) else str(new)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--reference', default=None, help=argparse.SUPPRESS)   # older option, no longer needed; ignored
    ap.add_argument('--repo', default=str(REPO), help='repository whose dexmachina/assets the paths point to (default: this one)')
    a = ap.parse_args()
    run, repo = Path(a.run_dir), Path(a.repo).resolve()
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('learner_config', repo / 'dexmachina' / 'envs' / 'learner_config.py')
    learner_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(learner_config)

    pkl, orig = run / 'params/env.pkl', run / 'params/env.pkl.orig'
    if not orig.exists():
        shutil.copy2(pkl, orig)
    kw = learner_config.upgrade_legacy_env_kwargs(pickle.load(open(orig, 'rb')))
    changed, missing = [], []
    kw = repoint(kw, repo, changed, missing)
    if missing:
        sys.exit('missing in this repository (the generated assets: REPRODUCE.md, section 1):\n'
                 + '\n'.join(f'{p}: {new}' for p, _old, new in missing))
    pickle.dump(kw, open(pkl, 'wb'))
    for key, old, new in changed:
        print(f'{key}: {old} -> {new}')
    print(f'wrote {pkl} ({len(changed)} paths); original kept as {orig}')


if __name__ == '__main__':
    main()
