"""scripts/train_coordinates_hand.py and baselines/ablation_rows.py: the training commands of the paper's runs (every task, rows
(a)-(g), four hands) match tests/golden/training_commands.json token for token; other hands get their hand, their retargeting
scale for the fingertip gaps, DexMachina's long-clip weights, and hand-specific run / checkpoint / eval paths.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_launcher_hand.py
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'golden' / 'training_commands.json'
SCRIPT = 'dexmachina/rl/train_rl_games.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def new():
    return load(ROOT / 'scripts' / 'train_coordinates_hand.py', 'launcher_new')


@pytest.fixture(scope='module')
def rows():
    return load(ROOT / 'baselines' / 'ablation_rows.py', 'ablation_rows_test')


def test_commands_match_the_golden_file(new, rows):
    golden = json.loads(GOLDEN.read_text())
    assert len(golden) == 68
    for key, expected in golden.items():
        row, hand, task, mode = key.split('/')
        if row == 'a':
            experiment = f'coordinates-hand_{task.lower()}_seed15' + ('_smoke' if mode == 'smoke' else '')
            command = new.training_command(task, 15, experiment, 'coordinates-hand', 'python', smoke=mode == 'smoke',
                                           hand=hand)
            assert command[1] == str(ROOT / SCRIPT)
            command = ['python', SCRIPT, *command[2:]]
        else:
            command = rows.row_args(row, task.lower(), 15)
        assert command == expected.split(' '), key


def test_old_positional_reference_argument_is_rejected(new, rows):
    with pytest.raises(TypeError):
        new.training_command('Box-200', 15, Path('/r/ref.npz'), 'exp', 'proj', 'python')
    with pytest.raises(TypeError):
        rows.row_args('b', 'box-200', 15, 'refs/x.npz')


def flag(command, name):
    return command[command.index(name) + 1]


def test_other_hands_get_their_hand_scale_and_weights(new):
    args = (42, 'exp', 'proj', 'python')
    for hand, scale in (('allegro_hand', '1.7'), ('dex3_hand', '1.5'), ('xhand', '1.4'), ('schunk_hand', '1.3')):
        long_clip = new.training_command('Ketchup-300', *args, hand=hand)
        assert flag(long_clip, '--hand') == hand and flag(long_clip, '--hand_graph_scale') == scale
        assert (flag(long_clip, '-imi'), flag(long_clip, '-con'), flag(long_clip, '-bc')) == ('0.2', '2.0', '0.2')   # authors' configs
        assert long_clip.count('--hand') == 1 and long_clip.count('--hand_graph_scale') == 1
        short = new.training_command('Box-200', *args, hand=hand)
        inspire = new.training_command('Box-200', *args)
        assert [x for x in short if x not in (hand, scale)] == [x for x in inspire if x not in ('inspire_hand', '1.2')]
    assert flag(new.training_command('Ketchup-300', *args), '-imi') == '0.3'          # Inspire keeps its own recipe
    assert flag(new.training_command('Box-200', *args, hand='allegro_hand', hand_graph_scale=1.25), '--hand_graph_scale') == '1.25'


def test_run_paths_follow_the_hand(new, tmp_path, monkeypatch):
    monkeypatch.setattr(new, 'REPO', tmp_path)
    run = tmp_path / 'logs/rl_games/allegro_hand' / 'allegro-myexp_box30-230-s01-u01_B12000_hybrid'
    run.mkdir(parents=True)
    (tmp_path / 'logs/rl_games/inspire_hand' / 'inspire-myexp_box30-230').mkdir(parents=True)
    assert new.run_dir('myexp', 'allegro_hand') == run
    assert new.run_dir('myexp').name == 'inspire-myexp_box30-230'
    assert new.hand_prefix('xhand') == 'xhand' and new.hand_prefix('dex3_hand') == 'dex3'
