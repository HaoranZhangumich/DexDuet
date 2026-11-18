"""CPU tests of the learner options (dexmachina/envs/learner_config.py): defaults, the paper's commands in their current and
their older spelling, rejected settings, reading stored run folders, and the value size the wrapper reports.

learner_config.py is loaded by path, so most tests run without the simulator; the parser tests that need the trainer's
full argument parser skip without Genesis.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q dexmachina/tests/test_learner_config.py
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('learner_config_test', ROOT / 'dexmachina' / 'envs' / 'learner_config.py')
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

PAPER_LEARNER = dict(ablation_protocol='dexmachina', per_hand_ppo=True, value_mode='per_hand', per_hand_value=True)
NEW_TOKENS = '--coordinate_actor --coord_ablation_id per-hand --per_hand_ppo --ablation_protocol dexmachina --value_mode per_hand'
# the same options in their older spelling (scripts/train_coordinates_hand.py before the learner options were renamed)
OLD_TOKENS = ('--coordinate_actor --coord_ablation_id per-hand --wrench_guidance team --wrench_reward_weight 0.0 '
              '--wrench_credit_weight 0.0 --wrench_per_hand_ppo --ablation_protocol dexmachina --wrench_value_mode per_hand '
              '--wrench_credit_mode individual')
OLD_REFERENCE = '--wrench_reference REF'


def small_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinate_actor', action='store_true')
    parser.add_argument('--coord_ablation_id', default=None)
    parser.add_argument('--no_object', action='store_true')
    parser.add_argument('--maniptrans', action='store_true')
    return lc.add_learner_arguments(parser)


def parse(tokens):
    return small_parser().parse_args(tokens.split())


def current(namespace):
    return {key: value for key, value in vars(namespace).items() if not key.startswith('legacy_')}


def released_block(**changes):
    """A stored learner block as training wrote it before the options were renamed: all 15 keys (placeholder paths).
    Released run folders keep the six learner keys and reference_metadata.source_path/source_sha256."""
    block = dict(enabled=True, mode='team', reference='/old/machine/refs/box_30_230_s01_u01.npz',
                 clip='box-30-230-s01-u01', reward_weight=0.0, per_hand_ppo=True, credit_weight=0.0,
                 ablation_protocol='dexmachina', value_mode='per_hand', per_hand_value=True, credit_mode='individual',
                 reference_sha256='0' * 64, reference_metadata={'source_path': '/old/machine/arctic/box.npy'},
                 runtime_normal_policy='genesis_live_direction_canonicalization_v1', runtime_normal_tolerance=0.001)
    block.update(changes)
    return block


def test_defaults_keep_dexmachina_learner():
    args = parse('')
    assert args.ablation_protocol == 'legacy' and args.per_hand_ppo is False and args.value_mode is None
    assert lc.resolve_value_mode(args) == 'scalar'
    assert lc.make_learner_config(args) is None
    assert lc.make_learner_config(parse('--coordinate_actor')) is None


def test_paper_options_in_both_spellings():
    new, old = parse(NEW_TOKENS), parse(f'{OLD_TOKENS} {OLD_REFERENCE}')
    assert current(new) == current(old)
    assert lc.make_learner_config(new) == lc.make_learner_config(old) == PAPER_LEARNER
    assert list(lc.make_learner_config(new)) == list(PAPER_LEARNER)
    assert lc.resolve_value_mode(parse('--coordinate_actor --per_hand_ppo --ablation_protocol dexmachina')) == 'per_hand'
    # the native protocol alone (no per-hand PPO): joint ratio, scalar value
    assert lc.make_learner_config(parse('--coordinate_actor --ablation_protocol dexmachina')) == dict(
        ablation_protocol='dexmachina', per_hand_ppo=False, value_mode='scalar', per_hand_value=False)
    # the MARL baselines' stored spelling
    marl = parse('--coordinate_actor --wrench_guidance team --wrench_reward_weight 0 --wrench_credit_weight 0 '
                 '--ablation_protocol dexmachina --wrench_value_mode scalar')
    assert marl.value_mode == 'scalar' and lc.make_learner_config(marl)['per_hand_ppo'] is False
    assert lc.make_learner_config(parse('--coordinate_actor --wrench_guidance none --ablation_protocol dexmachina'))


@pytest.mark.parametrize('tokens, message', [
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_reward_weight 0.5', 'no longer accepted'),
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_reward_weight -1', 'no longer accepted'),
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_reward_weight nan', 'no longer accepted'),
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_credit_weight 0.3', 'no longer accepted'),
    ('--wrench_reward_weight 1', 'no longer accepted'),
    ('--wrench_guidance none --wrench_reward_weight 0.5', 'no longer accepted'),
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_guidance team --wrench_reward_weight 0 '
     '--wrench_credit_weight 0.3', 'other than 0'),
    # the old default reward weight was 1: guidance without an explicit 0 would have been on
    ('--coordinate_actor --ablation_protocol dexmachina --wrench_guidance team', 'older default weight was 1.0'),
    ('--coordinate_actor --wrench_guidance team', 'older default weight was 1.0'),
    # zero weights, but not the protocol of the paper's per-hand commands
    ('--coordinate_actor --wrench_guidance hand --wrench_reward_weight 0', 'ablation_protocol legacy is no longer'),
    ('--coordinate_actor --wrench_guidance team --wrench_reward_weight 0 --wrench_credit_weight 0',
     'ablation_protocol legacy is no longer'),
    (OLD_TOKENS.replace(' --ablation_protocol dexmachina', ''), 'ablation_protocol legacy is no longer'),
    ('--coordinate_actor --per_hand_ppo', 'need --ablation_protocol'),
    ('--coordinate_actor --wrench_per_hand_ppo', 'need --ablation_protocol'),
    ('--coordinate_actor --value_mode per_hand', 'need --ablation_protocol'),
    ('--per_hand_ppo --ablation_protocol dexmachina', 'needs the per-hand actor'),
    ('--coordinate_actor --no_object --ablation_protocol dexmachina', 'object environment'),
])
def test_rejected_settings(tokens, message):
    with pytest.raises(ValueError, match=message):
        lc.make_learner_config(parse(tokens))


def test_help_names_only_the_current_options():
    text = small_parser().format_help()
    assert '--per_hand_ppo' in text and '--value_mode' in text and '--ablation_protocol' in text
    for word in ('wrench', 'reference', 'legacy_'):
        assert word not in text.lower()


def test_stored_block_is_read_as_learner_cfg():
    cfg = lc.learner_cfg_from_legacy(released_block())
    assert cfg == PAPER_LEARNER and list(cfg) == list(PAPER_LEARNER)
    assert lc.learner_cfg_from_legacy(None) is None and lc.learner_cfg_from_legacy({}) is None
    # DexMachina's learner with zero weights (legacy protocol, no per-hand option): nothing to carry over
    assert lc.learner_cfg_from_legacy(dict(reward_weight=0.0, credit_weight=0.0, ablation_protocol='legacy')) is None
    # value fallbacks of runs that did not store them
    assert lc.learner_cfg_from_legacy(dict(reward_weight=0, credit_weight=0, ablation_protocol='v2')) == dict(
        ablation_protocol='v2', per_hand_ppo=False, value_mode='scalar', per_hand_value=False)
    block = released_block()
    del block['reward_weight']
    for bad in (released_block(reward_weight=0.5), released_block(credit_weight=0.3), block,
                released_block(reward_weight=math.nan), released_block(credit_weight=True)):
        with pytest.raises(ValueError, match='wrench-guidance rewards or credit'):
            lc.learner_cfg_from_legacy(bad)
    with pytest.raises(ValueError, match='legacy protocol'):
        lc.learner_cfg_from_legacy(released_block(ablation_protocol='legacy'))
    with pytest.raises(ValueError, match='Unknown ablation_protocol'):
        lc.learner_cfg_from_legacy(released_block(ablation_protocol='v3'))
    with pytest.raises(ValueError, match='dict'):
        lc.learner_cfg_from_legacy(['not', 'a', 'dict'])


def test_upgrade_legacy_env_kwargs():
    kw = {'env_cfg': {}, 'wrench_cfg': released_block()}
    assert lc.upgrade_legacy_env_kwargs(kw) is kw
    assert kw == {'env_cfg': {}, 'learner_cfg': PAPER_LEARNER}
    assert lc.upgrade_legacy_env_kwargs(kw) == {'env_cfg': {}, 'learner_cfg': PAPER_LEARNER}   # idempotent
    assert lc.upgrade_legacy_env_kwargs({'env_cfg': {}}) == {'env_cfg': {}}
    assert lc.upgrade_legacy_env_kwargs({'wrench_cfg': {}}) == {}
    with pytest.raises(ValueError, match='both'):
        lc.upgrade_legacy_env_kwargs({'wrench_cfg': released_block(), 'learner_cfg': PAPER_LEARNER})


def test_wrapper_value_size_follows_the_learner_cfg():
    pytest.importorskip('rl_games')
    from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper
    from rl_games.common.env_configurations import get_env_info
    for attributes, size in ((dict(per_hand_ppo=False), 1), (dict(per_hand_ppo=True), 2),
                             (dict(per_hand_ppo=True, per_hand_value=False), 1),
                             (dict(per_hand_ppo=False, per_hand_value=True), 2)):
        env = SimpleNamespace(device='cpu', obs_dim=19, num_actions=4, **attributes)
        wrapper = RlGamesVecEnvWrapper(env, 'cpu', 5., 1.)
        assert wrapper.value_size == size
        info = wrapper.get_env_info()
        if size == 2:
            assert info['value_size'] == 2
        else:
            assert 'value_size' not in info      # preserves the default info dict
        assert get_env_info(wrapper)['value_size'] == size


@pytest.fixture(scope='module')
def trainer():
    pytest.importorskip('genesis')
    pytest.importorskip('wandb')
    from dexmachina.rl import train_rl_games
    return train_rl_games


def golden_command(key='a/inspire_hand/Box-200/full'):
    return json.loads((ROOT / 'tests' / 'golden' / 'training_commands.json').read_text())[key].split(' ')[2:]


def test_trainer_parses_both_spellings_of_the_paper_command(trainer):
    from dexmachina.envs.constructors import get_common_argparser
    new = golden_command()
    text = ' '.join(new)
    assert NEW_TOKENS in text
    old = f'{text.replace(NEW_TOKENS, OLD_TOKENS)} {OLD_REFERENCE}'.split(' ')
    assert len(old) == len(new) + 10
    for parser in (trainer.build_parser(), get_common_argparser()):
        parsed_new = parser.parse_known_args(new)[0]
        parsed_old = parser.parse_known_args(old)[0]
        assert current(parsed_new) == current(parsed_old)
        assert lc.make_learner_config(parsed_new) == lc.make_learner_config(parsed_old) == PAPER_LEARNER
    assert trainer.build_parser().parse_args(old).legacy_wrench_reference == 'REF'
    for row in 'bcdfg':
        args = trainer.build_parser().parse_args(golden_command(f'{row}/inspire_hand/Box-200/full'))
        assert lc.make_learner_config(args) == PAPER_LEARNER
    assert lc.make_learner_config(trainer.build_parser().parse_args(golden_command('e/inspire_hand/Box-200/full'))) is None


def test_trainer_help_has_no_older_option_names(trainer):
    from dexmachina.envs.constructors import get_common_argparser
    for parser in (trainer.build_parser(), get_common_argparser()):
        text = parser.format_help()
        assert 'wrench' not in text.lower()
        assert '--per_hand_ppo' in text


def test_env_rejects_a_stored_block_with_guidance_before_building(trainer):
    from dexmachina.envs.base_env import BaseEnv
    with pytest.raises(ValueError, match='wrench-guidance rewards or credit'):
        BaseEnv({}, {}, {}, {}, {}, wrench_cfg={'reward_weight': 0.5, 'credit_weight': 0})
    with pytest.raises(ValueError, match='both'):
        BaseEnv({}, {}, {}, {}, {}, learner_cfg=PAPER_LEARNER, wrench_cfg=released_block())
