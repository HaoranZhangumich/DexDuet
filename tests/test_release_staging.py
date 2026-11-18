"""scripts/stage_release_run.py: released configs load with only the two Python YAML tags the trainer writes, a stored learner
configuration is read into the current layout, asset paths are repointed at this repository, and the comparison reports
every difference.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_release_staging.py
"""
import hashlib
import importlib.util
from pathlib import Path, PosixPath

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def stage():
    spec = importlib.util.spec_from_file_location('stage_release_run', ROOT / 'scripts' / 'stage_release_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_loader_tags(stage, tmp_path):
    path = tmp_path / 'env.yaml'
    path.write_text('a: !!python/tuple\n- 1\n- 2\nurdf: !!python/object/apply:pathlib.PosixPath\n- /\n- x\n- dexmachina\n'
                    '- assets\n- inspire_hand\n- left.urdf\nshared: &id1 !!python/tuple [3, 4]\nagain: *id1\n')
    kw = stage.load_env_yaml(path)
    assert kw['a'] == (1, 2) and kw['again'] == (3, 4)
    assert kw['urdf'] == PosixPath('/x/dexmachina/assets/inspire_hand/left.urdf')
    path.write_text('bad: !!python/object/apply:os.system ["true"]\n')
    with pytest.raises(yaml.constructor.ConstructorError):
        stage.load_env_yaml(path)


def test_repoint_assets(stage):
    kw = {'object_cfgs': {'box': {'urdf_path': '/old/machine/dexmachina/assets/arctic/box/box.urdf', 'name': 'box'}},
          'robot_cfgs': {'left': {'urdf_path': PosixPath('/other/dexmachina/assets/inspire_hand/left_xyz_copy.urdf')}}}
    changed = stage.repoint_assets(kw)
    assert len(changed) == 2
    assert kw['object_cfgs']['box']['urdf_path'] == str(ROOT / 'dexmachina/assets/arctic/box/box.urdf')
    assert kw['robot_cfgs']['left']['urdf_path'] == ROOT / 'dexmachina/assets/inspire_hand/left_xyz_copy.urdf'
    assert isinstance(kw['robot_cfgs']['left']['urdf_path'], PosixPath)


def test_compare(stage):
    a = {'x': np.arange(3.0), 'y': (1, 2), 'z': {'p': PosixPath('/a')}}
    assert stage.compare(a, {'x': np.arange(3.0), 'y': [1, 2], 'z': {'p': PosixPath('/a')}}) == []
    diff = stage.compare(a, {'x': np.arange(3.0) + 1e-9, 'y': (1, 3), 'w': 0, 'z': {'p': PosixPath('/b')}})
    assert any(d.startswith('/x') for d in diff) and any(d.startswith('/y') for d in diff)
    assert any('/w' in d for d in diff) and any(d.startswith('/z/p') for d in diff)


def test_tasks_and_clips(stage):
    assert stage.task_key('box-200') == 'Box-200'
    assert stage.launcher().TASKS['Notebook-300'][0] == 'notebook-40-340-s02-u02'


def test_missing_inputs_name_the_setup_step(stage):
    kw = {'reward_cfg': {'use_retarget_contact': True, 'ho_contact_weight': 1.0},
          'learner_cfg': {'ablation_protocol': 'dexmachina', 'per_hand_ppo': True, 'value_mode': 'per_hand',
                          'per_hand_value': True}}
    missing = stage.missing_inputs(kw, 'box-30-230-s01-u01', 'no_such_hand')
    paths = [str(p) for p, _how in missing]
    assert all(not p.exists() for p, _how in missing)
    assert any(p.endswith('retargeted/no_such_hand/s01/box_use_01_vector_para.pt') for p in paths)
    assert any(p.endswith('contact_retarget/no_such_hand/s01/box_use_01.npy') for p in paths)
    hows = ' '.join(how for _p, how in missing)
    assert 'scripts/reproduce.sh prepare' in hows
    assert not any('/arctic/data/' in p for p in paths)       # the per-hand policy needs no ARCTIC checkout
    none = stage.missing_inputs({'reward_cfg': {}}, 'box-30-230-s01-u01', 'no_such_hand')
    assert 1 <= len(none) <= 2 and all('prepare' in how for _p, how in none)   # processed demo (if absent), retargeting


def test_demonstration_check(stage, tmp_path):
    clip = 'box-30-230-s01-u01'
    demo = stage.processed_demonstration(clip, tmp_path)
    assert demo == tmp_path / 'arctic' / 'processed' / 's01' / 'box_use_01.npy'
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b'processed demonstration')
    here = hashlib.sha256(b'processed demonstration').hexdigest()
    kw = {'reward_cfg': {}, 'wrench_cfg': released_wrench_cfg(reference_metadata={
        'source_path': '<REPO>/dexmachina/assets/arctic/processed/s01/box_use_01.npy', 'source_sha256': here})}
    assert stage.recorded_demonstration_sha256(kw) == here
    assert stage.recorded_demonstration_sha256({'reward_cfg': {}}) is None          # row (e), DexMachina's runs
    check, notice = stage.demonstration_check(here, clip, tmp_path)
    assert check == dict(file=str(demo), recorded=here, here=here, matches=True) and notice is None
    check, notice = stage.demonstration_check('0' * 64, clip, tmp_path)
    assert check['matches'] is False and 'source_sha256 ' + '0' * 64 in notice and 'REPRODUCE.md, section 1' in notice
    check, notice = stage.demonstration_check(None, clip, tmp_path)
    assert check['matches'] is None and notice is None


def released_wrench_cfg(**changes):
    """The learner block of a released per-hand run as shipped: the learner keys and the recorded demonstration."""
    block = dict(reward_weight=0.0, per_hand_ppo=True, credit_weight=0.0, ablation_protocol='dexmachina',
                 value_mode='per_hand', per_hand_value=True,
                 reference_metadata={'source_path': '<REPO>/dexmachina/assets/arctic/processed/s01/box_use_01.npy',
                                     'source_sha256': '0' * 64})
    block.update(changes)
    return block


def training_wrench_cfg(**changes):
    """The same block as training wrote it before the options were renamed (15 keys; placeholder paths)."""
    block = dict(enabled=True, mode='team', reference='/old/machine/refs/box_30_230_s01_u01.npz',
                 clip='box-30-230-s01-u01', reward_weight=0.0, per_hand_ppo=True, credit_weight=0.0,
                 ablation_protocol='dexmachina', value_mode='per_hand', per_hand_value=True, credit_mode='individual',
                 reference_sha256='0' * 64,
                 reference_metadata={'urdf_path': '/old/machine/dexmachina/assets/arctic/box/box.urdf',
                                     'geometry_sha256': {'/old/machine/dexmachina/assets/arctic/box/box.urdf': '1' * 64},
                                     'source_path': '/old/machine/dexmachina/assets/arctic/processed/s01/box_use_01.npy',
                                     'source_sha256': '0' * 64},
                 runtime_normal_policy='genesis_live_direction_canonicalization_v1', runtime_normal_tolerance=0.001)
    block.update(changes)
    return block


@pytest.mark.parametrize('block', [released_wrench_cfg, training_wrench_cfg])
def test_stored_learner_config_is_upgraded(stage, block):
    old = {'object_cfgs': {'box': {'urdf_path': '/old/machine/dexmachina/assets/arctic/box/box.urdf'}},
           'demo_data': {'obj_pos': np.zeros((3, 3))}, 'wrench_cfg': block()}
    new = {'object_cfgs': {'box': {'urdf_path': '/old/machine/dexmachina/assets/arctic/box/box.urdf'}},
           'demo_data': {'obj_pos': np.zeros((3, 3))},
           'learner_cfg': {'ablation_protocol': 'dexmachina', 'per_hand_ppo': True, 'value_mode': 'per_hand',
                           'per_hand_value': True}}
    assert stage.recorded_demonstration_sha256(old) == '0' * 64
    upgraded = stage.upgrade_legacy_env_kwargs(old)
    assert upgraded is old and 'wrench_cfg' not in old
    assert list(old['learner_cfg']) == ['ablation_protocol', 'per_hand_ppo', 'value_mode', 'per_hand_value']
    assert stage.compare(old, new) == []
    assert stage.compare(stage.upgrade_legacy_env_kwargs(old), new) == []          # idempotent
    changed = stage.repoint_assets(old)
    assert [path for path, _old, _new in changed] == ['/object_cfgs/box/urdf_path']   # no reference metadata left
    # row (e) and DexMachina's own runs: no learner block before or after
    assert stage.upgrade_legacy_env_kwargs({'reward_cfg': {}}) == {'reward_cfg': {}}
    for change in (dict(reward_weight=0.5), dict(credit_weight=0.3)):
        with pytest.raises(ValueError, match='wrench-guidance rewards or credit'):
            stage.upgrade_legacy_env_kwargs({'wrench_cfg': released_wrench_cfg(**change)})
    with pytest.raises(ValueError, match='both'):
        stage.upgrade_legacy_env_kwargs({'wrench_cfg': released_wrench_cfg(), 'learner_cfg': new['learner_cfg']})
