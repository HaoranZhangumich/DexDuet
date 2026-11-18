"""dexmachina/rl/final_checkpoint.py: what training writes when it returns (terminal snapshot, final_checkpoint.json with the
native best-training-reward checkpoint) and what evaluation accepts, including run folders copied to another path and
folders written before the terminal snapshot was renamed.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_final_checkpoint.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.rl import final_checkpoint as fc  # noqa: E402

KEYS = ['checkpoint', 'selection', 'checkpoint_sha256', 'epoch', 'completed_updates', 'environment_transitions',
        'stop_reason', 'terminal_checkpoint']


class FakeAlgo:
    def __init__(self, epoch_num, stop_reason=None):
        self.epoch_num = epoch_num
        if stop_reason is not None:
            self.training_stop_reason = stop_reason
        self.saved = []

    def save(self, filename):
        self.saved.append(filename)
        torch.save({'epoch': self.epoch_num}, filename + '.pth')


def make_run(tmp_path, name='inspire-exp_box30-230-s01-u01_B64_hybrid', hand='inspire_hand', best_epoch=7):
    run = tmp_path / name
    (run / 'nn').mkdir(parents=True)
    torch.save({'epoch': best_epoch, 'model': {'w': torch.arange(3.0)}}, run / 'nn' / f'{hand}.pth')
    return run


def test_dexmachina_record(tmp_path):
    run = make_run(tmp_path)
    algo = FakeAlgo(12, stop_reason='native_zero_gain_patience')
    record = fc.record_training_end(algo, str(run), 'inspire_hand', 'dexmachina', 12 * 64 * 32, max_epochs=5000,
                                    num_envs=64, horizon_length=32)
    assert list(record) == KEYS
    checkpoint = run / 'nn' / 'inspire_hand.pth'
    assert record == dict(checkpoint=str(checkpoint), selection='native_best_training_reward',
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), epoch=7,
                          completed_updates=12, environment_transitions=12 * 64 * 32,
                          stop_reason='native_zero_gain_patience', terminal_checkpoint=str(run / 'nn' / 'terminal.pth'))
    assert (run / 'nn' / 'terminal.pth').is_file() and algo.saved == [str(run / 'nn' / 'terminal')]
    assert json.loads((run / 'final_checkpoint.json').read_text()) == record
    with pytest.raises(FileExistsError):      # never overwritten
        fc.record_training_end(algo, str(run), 'inspire_hand', 'dexmachina', 12 * 64 * 32, max_epochs=5000,
                               num_envs=64, horizon_length=32)
    expected = dict(name='inspire_hand', max_epochs=5000, num_actors=64, horizon_length=32)
    assert fc.validate_training_checkpoint(run, expected, 'dexmachina') == (checkpoint, record)


def test_default_stop_reason_and_selected_epoch_range(tmp_path):
    run = make_run(tmp_path)
    record = fc.record_training_end(FakeAlgo(9), run, 'inspire_hand', 'dexmachina', 9, max_epochs=10, num_envs=1,
                                    horizon_length=1)
    assert record['stop_reason'] == 'native_training_loop_returned'
    late = make_run(tmp_path, name='late', best_epoch=20)
    with pytest.raises(RuntimeError, match='outside the completed training range'):
        fc.record_training_end(FakeAlgo(9), late, 'inspire_hand', 'dexmachina', 9, max_epochs=10, num_envs=1,
                               horizon_length=1)
    with pytest.raises(RuntimeError, match='checkpoint missing'):
        fc.record_training_end(FakeAlgo(9), late, 'allegro_hand', 'dexmachina', 9, max_epochs=10, num_envs=1,
                               horizon_length=1)


def test_v2_endpoint_record(tmp_path):
    run = make_run(tmp_path)
    record = fc.record_training_end(FakeAlgo(10), str(run), 'inspire_hand', 'v2', 10 * 4 * 8, max_epochs=10, num_envs=4,
                                    horizon_length=8)
    assert record == dict(checkpoint=str(run / 'nn' / 'terminal.pth'), selection='training_endpoint', epoch=10,
                          environment_transitions=320)
    assert fc.validate_training_checkpoint(run, {}, 'v2') == (run / 'nn' / 'terminal.pth', None)
    with pytest.raises(RuntimeError, match='update endpoint'):
        fc.record_training_end(FakeAlgo(9), str(make_run(tmp_path, 'a')), 'inspire_hand', 'v2', 9 * 32, max_epochs=10,
                               num_envs=4, horizon_length=8)
    with pytest.raises(RuntimeError, match='fixed budget'):
        fc.record_training_end(FakeAlgo(10), str(make_run(tmp_path, 'b')), 'inspire_hand', 'v2', 319, max_epochs=10,
                               num_envs=4, horizon_length=8)
    with pytest.raises(ValueError):
        fc.record_training_end(FakeAlgo(10), str(make_run(tmp_path, 'c')), 'inspire_hand', 'legacy', 0, max_epochs=10,
                               num_envs=4, horizon_length=8)


def test_v2_folder_written_before_the_rename(tmp_path):
    run = make_run(tmp_path)
    torch.save({'epoch': 10}, run / 'nn' / 'wrench_final.pth')
    assert fc.validate_training_checkpoint(run, {}, 'v2') == (run / 'nn' / 'wrench_final.pth', None)
    torch.save({'epoch': 10}, run / 'nn' / 'terminal.pth')
    assert fc.validate_training_checkpoint(run, {}, 'v2') == (run / 'nn' / 'terminal.pth', None)
    with pytest.raises(RuntimeError, match='Missing checkpoint'):
        fc.validate_training_checkpoint(make_run(tmp_path, 'empty'), {}, 'v2')


def test_released_record_accepted_after_relocation(tmp_path):
    """A released folder keeps the record written on the training machine: the old run name and terminal file name."""
    name = 'inspire-coordinates-hand_box-200_seed15_box30-230-s01-u01_B12000_hybrid_imi0.2_factmsg_cwsteam_handppo_credit0'
    origin = make_run(tmp_path / 'machine_a', name=name, best_epoch=3065)
    checkpoint = origin / 'nn' / 'inspire_hand.pth'
    (origin / 'final_checkpoint.json').write_text(json.dumps(dict(
        checkpoint=str(checkpoint), selection='native_best_training_reward',
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), epoch=3065, completed_updates=5000,
        environment_transitions=5000 * 12000 * 16, stop_reason='max_epochs',
        terminal_checkpoint=str(origin / 'nn' / 'wrench_final.pth'))))
    staged = tmp_path / 'machine_b' / 'staged' / name
    shutil.copytree(origin, staged)
    shutil.rmtree(tmp_path / 'machine_a')
    expected = dict(name='inspire_hand', max_epochs=5000, num_actors=12000, horizon_length=16)
    primary, endpoint = fc.validate_training_checkpoint(staged, expected, 'dexmachina')
    assert primary == staged / 'nn' / 'inspire_hand.pth' and endpoint['epoch'] == 3065
    # the folder must keep the recorded run name ...
    renamed = tmp_path / 'machine_b' / 'other_name'
    shutil.copytree(staged, renamed)
    with pytest.raises(RuntimeError, match='native best training reward'):
        fc.validate_training_checkpoint(renamed, expected, 'dexmachina')
    # ... and the recorded checkpoint content
    torch.save({'epoch': 3065, 'model': {'w': torch.zeros(3)}}, staged / 'nn' / 'inspire_hand.pth')
    with pytest.raises(RuntimeError, match='native best training reward'):
        fc.validate_training_checkpoint(staged, expected, 'dexmachina')
