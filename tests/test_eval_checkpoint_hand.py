"""dexmachina.rl.final_checkpoint.validate_training_checkpoint (called by eval_rl_games.py for every DexMachina-protocol run)
looks for rl_games' best checkpoint under the agent's name, i.e. the hand: nn/allegro_hand.pth for an Allegro run. Before, it
always looked for nn/inspire_hand.pth, so evaluating any other hand failed with 'Missing checkpoint .../nn/inspire_hand.pth'.
Without a name in the expected settings it keeps DEFAULT_HAND, the Inspire hand.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_eval_checkpoint_hand.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dexmachina.rl.final_checkpoint import DEFAULT_HAND as HAND, validate_training_checkpoint  # noqa: E402


def make_run(tmp_path, name, epoch=19, completed=20, num_actors=256, horizon=16):
    run = tmp_path / f'run_{name}'
    (run / 'nn').mkdir(parents=True)
    ckpt = run / 'nn' / f'{name}.pth'
    torch.save({'epoch': epoch}, ckpt)
    (run / 'final_checkpoint.json').write_text(json.dumps(dict(
        checkpoint=str(ckpt), selection='native_best_training_reward', epoch=epoch, completed_updates=completed,
        environment_transitions=completed * num_actors * horizon,
        checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())))
    expected = dict(max_epochs=completed, num_actors=num_actors, horizon_length=horizon)
    return run, ckpt, expected


@pytest.mark.parametrize('hand', ['allegro_hand', 'dex3_hand', 'xhand'])
def test_other_hand_checkpoint_is_found_by_the_agent_name(tmp_path, hand):
    run, ckpt, expected = make_run(tmp_path, hand)
    checkpoint, endpoint = validate_training_checkpoint(run, dict(expected, name=hand), 'dexmachina')
    assert checkpoint == ckpt and endpoint['epoch'] == 19
    with pytest.raises(RuntimeError, match=f'Missing checkpoint .*/nn/{HAND}.pth'):
        validate_training_checkpoint(run, expected, 'dexmachina')        # no name: DEFAULT_HAND


def test_inspire_unchanged_with_and_without_name(tmp_path):
    assert HAND == 'inspire_hand'
    run, ckpt, expected = make_run(tmp_path, 'inspire_hand')
    assert validate_training_checkpoint(run, expected, 'dexmachina')[0] == ckpt
    assert validate_training_checkpoint(run, dict(expected, name='inspire_hand'), 'dexmachina')[0] == ckpt
