"""scripts/reproduce_paper.sh: the commands it would run (--dry-run, no GPU), the release layout it reads, DexMachina's
run list and the per-rollout MPPE it aggregates.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_reproduce_paper.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def dry(tmp_path, *args):
    env = dict(os.environ, DEXDUET_OUT=str(tmp_path / 'out'), DEXDUET_CHECKPOINTS=str(tmp_path / 'ckpt'),
               DEXDUET_PYTHON=sys.executable, CH_MANO_EXPORT=str(tmp_path / 'mano'), CUDA_VISIBLE_DEVICES='')
    result = subprocess.run(['bash', str(ROOT / 'scripts/reproduce_paper.sh'), '--dry-run', *args], cwd=ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [line[2:] for line in result.stdout.splitlines() if line.startswith('$ ')]


def test_main_dry_run(tmp_path):
    cmds = dry(tmp_path, 'main', '--task', 'Ketchup-300', '--hand', 'schunk_hand', '--seed', '24')
    assert 'scripts/train_coordinates_hand.py --task Ketchup-300 --hand schunk_hand --seed 24' in cmds[0]
    assert '--bc-weight 0.3' in cmds[0]                    # the paper's Schunk runs on the long clips
    assert '/resets/dexduet/schunk_hand/ketchup-300/evaluation_reset.json' in cmds[0]
    assert any('baselines.track2_metrics' in c for c in cmds) and any('dexmachina.eval.slip_auc' in c for c in cmds)
    assert '--bc-weight' not in dry(tmp_path, 'main', '--task', 'Box-200', '--hand', 'schunk_hand')[0]


def test_ablation_dry_run(tmp_path):
    cmds = dry(tmp_path, 'ablation', '--row', 'e', '--task', 'Mixer-170', '--seed', '113', '--smoke')
    train = next(c for c in cmds if 'train_rl_games.py' in c)
    assert '--coordinate_actor' not in train and '--per_hand_ppo' not in train      # unified policy
    assert '-exp abl-e-unified_mixer-170_seed113_smoke' in train and '-B 256' in train
    assert not any('wrench' in c for c in cmds)
    evaluate = next(c for c in cmds if 'eval_rl_games' in c)
    assert '--num_envs 100 --all_policy_envs --eval-seed 1042' in evaluate and '/resets/smoke/' in evaluate


def test_per_hand_ablation_dry_run(tmp_path):
    cmds = dry(tmp_path, 'ablation', '--row', 'b', '--task', 'Box-200', '--seed', '15')
    train = next(c for c in cmds if 'train_rl_games.py' in c)
    argv = train.split()
    assert '--per_hand_ppo' in argv and '--coordinate_actor' in argv
    assert argv[argv.index('--ablation_protocol') + 1] == 'dexmachina'
    assert argv[argv.index('--value_mode') + 1] == 'per_hand'
    assert argv[argv.index('--hand_graph_weight') + 1] == '0'
    assert not any(token.startswith('--wrench') for token in argv)
    assert not any('wrench' in c for c in cmds)


def test_release_layout(tmp_path, monkeypatch):
    monkeypatch.setenv('DEXDUET_CHECKPOINTS', str(tmp_path))
    spec = importlib.util.spec_from_file_location('reproduce_paper', ROOT / 'scripts/reproduce_paper.py')
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    runs = rp.release_runs('all', ['inspire_hand'], ['waffleiron-300'], [15], ['b'])
    assert [(r[0], r[1], r[2], r[3]) for r in runs] == [('DexDuet', 'inspire_hand', 'waffleiron-300', 15),
                                                       ('row_b', 'inspire_hand', 'waffleiron-300', 15)]
    assert runs[0][4] == tmp_path / 'dexduet/inspire_hand/waffleiron-300/seed15'
    assert runs[1][4] == tmp_path / 'ablations/row_b/inspire_hand/waffleiron-300/seed15'
    ablation = rp.release_runs('ablation', ['inspire_hand'], ['box-200'], None, ['e'])
    assert [(r[0], r[3]) for r in ablation] == [('DexDuet', 15), ('DexDuet', 42), ('DexDuet', 113),   # row (a)
                                                ('row_e', 15), ('row_e', 42), ('row_e', 113)]
    assert rp.SUITE_PACKAGES['main'] == ['dexduet_dexmachina_suite.zip']
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'RUN_INFO.json').write_text(json.dumps({'completed_updates': 6159}))
    assert rp.completed_updates(tmp_path / 'run') == 6159
    assert rp.manifest_path('dexmachina', 'xhand', 'Box-200').parts[-4:] == ('dexmachina', 'xhand', 'box-200', 'evaluation_reset.json')
    # DexMachina's main-results data: run_paths.json + runs/<name>/{config.json, logs/rl_games/<hand>/<run>/nn/<hand>.pth}
    archive = tmp_path / 'dm'
    name = 'xhand-x_box30-230-s01-u01_B12000_abc'
    run = archive / 'runs' / name / 'logs/rl_games/xhand' / name[:-4] / 'nn'
    run.mkdir(parents=True)
    (run / 'xhand.pth').write_bytes(b'')
    (archive / 'runs' / name / 'config.json').write_text(json.dumps({'params': {'seed': 66}}))
    (archive / 'run_paths.json').write_text(json.dumps({'xhand': {'box30-230-s01-u01': [f'stats/wb_data/{name}']}}))
    found = rp.dexmachina_runs(archive, ['xhand'], ['box-200'])
    assert [(h, t, s, f.name) for h, t, s, f in found] == [('xhand', 'box-200', 66, name[:-4])]
    assert rp.dexmachina_runs(archive, ['xhand'], ['box-200'], [15]) == []
    log = tmp_path / 'train.log'
    log.write_text('epoch: 10\nfoo\nepoch: 4440 frames\n')
    assert rp.completed_updates(tmp_path / 'none', log) == 4440


def test_summary_carries_the_released_values(tmp_path, monkeypatch):
    """write_summary copies a released run's table_metrics (RUN_INFO.json) into its summary, and nothing else from
    RUN_INFO.json; paper_table.py's check compares the reproduced values with them."""
    def load(name):
        spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    rp, table = load('reproduce_paper'), load('paper_table')
    monkeypatch.setattr(rp, 'OUT', tmp_path / 'out')
    run = tmp_path / 'run'
    (run / 'nn').mkdir(parents=True)
    (run / 'nn' / 'inspire_hand.pth').write_bytes(b'checkpoint')
    evaluation = run / 'inspire_hand_eval_all_policy'
    evaluation.mkdir()
    (evaluation / 'add_stats.json').write_text(json.dumps({'auc': {'overall': 0.9}}))
    (evaluation / 'track2_metrics.json').write_text(json.dumps({'sp_sr': 1.0, 'mp_sr': 1.0, 'mppe_cm': 1.0,
                                                                'per_rollout_mppe_cm': [1.0] * 100}))
    (evaluation / 'eval_protocol.json').write_text(json.dumps({'first_observation': 'corrected'}))
    (run / 'slip').mkdir()
    (run / 'slip' / 'slip_auc.json').write_text(json.dumps({'slip_auc': 0.8}))
    info = {'ablation_row': 'c', 'hand_key': 'inspire_hand', 'task': 'Box-200', 'training_seed': 15,
            'checkpoint': {'file': 'inspire_hand.pth', 'epoch': 4100, 'selection': 'native_best_training_reward',
                           'completed_updates': 4770},
            'table_metrics': {'add_auc_pct': 90.0, 'contact_auc_pct': 78.0, 'sp_sr_pct': 100.0, 'mp_sr_pct': 100.0,
                              'mppe_cm': 1.0, 'updates': 4770}}
    (run / 'RUN_INFO.json').write_text(json.dumps(info))
    assert rp.completed_updates(run) == 4770
    rp.write_summary(run, 'box-200', 'inspire_hand', 'row_c', 15, 'dexduet')
    summary = json.loads(rp.summary_path('row_c', 'inspire_hand', 'box-200', 15).read_text())
    assert summary['release_table_metrics'] == info['table_metrics'] and summary['updates'] == 4770
    assert not [k for k in summary if k.startswith('release_') and k != 'release_table_metrics']
    [r] = table.load(tmp_path / 'out' / 'summaries')
    assert [x for x in table.release_check([r]) if x.startswith('- ')] == [
        '- row_c inspire_hand box-200 seed 15: Contact-AUC 80.00 (reported 78.00)']
    (run / 'RUN_INFO.json').unlink()                              # a run trained here: no released values
    rp.write_summary(run, 'box-200', 'inspire_hand', 'row_c', 15, 'dexduet', updates=4770)
    [r] = table.load(tmp_path / 'out' / 'summaries')
    assert r['released'] is None and table.release_check([r]) == []


def test_per_rollout_mppe():
    from baselines.track2_metrics import track2_scores
    rng = np.random.default_rng(0)
    reference = np.zeros((20, 1, 2, 7))
    reference[..., :3] = rng.normal(size=(20, 1, 2, 3))
    reference[..., 6] = 1.0
    achieved = np.repeat(reference, 4, axis=1)
    achieved[..., :3] += rng.normal(scale=0.01, size=achieved[..., :3].shape)
    scores = track2_scores(achieved, reference)
    assert np.isclose(np.mean(scores['per_rollout_mppe_cm']), scores['mppe_cm'])
    achieved[3:, 2] = np.nan
    scores = track2_scores(achieved, reference)
    assert scores['per_rollout_mppe_cm'][2] is None and scores['nan_rollouts'] == 1


def test_dexmachina_runs_keep_the_zero_pose_last_lane(tmp_path):
    base = ['evaluate', '--run', str(tmp_path / 'run'), '--hand', 'allegro_hand', '--task', 'box-200', '--seed', '15']
    dm = dry(tmp_path, *base, '--label', 'DexMachina', '--group', 'dexmachina')
    assert all('--zero-pose-last-lane' in c for c in dm if 'eval_rl_games' in c or 'slip_record' in c)
    assert any('/resets/dexmachina/allegro_hand/box-200/' in c for c in dm)
    dd = dry(tmp_path, *base)
    assert not any('--zero-pose-last-lane' in c for c in dd)


def test_hold_test_dry_run(tmp_path):
    base = ['hold-test', '--run', str(tmp_path / 'run'), '--hand', 'allegro_hand', '--task', 'ketchup-300', '--seed', '15']
    dm = dry(tmp_path, *base, '--label', 'DexMachina', '--group', 'dexmachina')
    run = next(c for c in dm if 'dexmachina.eval.hold_test.run' in c)
    assert '--hold-steps 180' in run and '--zero-pose-last-lane' in run and '--method DexMachina' in run
    assert '/resets/dexmachina/allegro_hand/ketchup-300/' in run
    assert any('dexmachina.eval.hold_test.add' in c for c in dm) and any('dexmachina.eval.hold_test.metrics' in c for c in dm)
    dd = dry(tmp_path, *base)
    assert not any('--zero-pose-last-lane' in c for c in dd)
