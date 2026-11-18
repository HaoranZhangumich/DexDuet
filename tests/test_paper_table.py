"""scripts/paper_table.py: a cell is the mean over runs +- the sd over training-seed means, and each hand has one DexMachina
and one DexDuet row.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_paper_table.py
"""
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def table():
    spec = importlib.util.spec_from_file_location('paper_table', ROOT / 'scripts' / 'paper_table.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(root, label, hand, task, seed, add, updates=None):
    path = root / label / hand / task / f'seed{seed}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(label=label, hand=hand, task=task, seed=seed, add_auc=add, sp_sr=1.0, mp_sr=0.5,
                                    mppe_cm=2.0, mppe_cm_lanes0_98=1.5, contact_auc=0.8, updates=updates)))


def test_cell_is_mean_over_runs_and_sd_over_seed_means(table, tmp_path):
    values = {(t, s): 0.5 + 0.01 * i + 0.003 * j for i, t in enumerate(table.TASKS) for j, s in enumerate(table.MAIN_SEEDS)}
    for (t, s), v in values.items():
        write(tmp_path, 'DexDuet', 'xhand', t, s, v)
    runs = table.load(tmp_path)
    c = table.cell([r for r in runs if r['label'] == 'DexDuet'], 'add')
    seed_means = [statistics.mean(100 * v for (t, s), v in values.items() if s == seed) for seed in table.MAIN_SEEDS]
    assert c['n'] == 35 and c['seeds'] == 5
    assert c['mean'] == pytest.approx(100 * statistics.mean(values.values()))
    assert c['sd'] == pytest.approx(statistics.stdev(seed_means))
    assert table.cell(runs, 'mppe')['mean'] == pytest.approx(1.5)          # MPPE over lanes 0-98


def test_main_rows(table, tmp_path):
    for t in table.TASKS:
        for s in table.MAIN_SEEDS:
            write(tmp_path, 'DexDuet', 'inspire_hand', t, s, 0.8)
            write(tmp_path, 'DexMachina', 'inspire_hand', t, s, 0.5)
    rows = dict(table.main_rows(table.load(tmp_path), 'inspire_hand'))
    assert list(rows) == ['DexMachina', 'DexDuet'] and len(rows['DexDuet']) == 35
    assert table.cell(rows['DexDuet'], 'add')['mean'] == pytest.approx(80)


def test_ablation_row_a_is_dexduet_with_ablation_seeds(table, tmp_path):
    for t in table.TASKS:
        for s in table.MAIN_SEEDS:
            write(tmp_path, 'DexDuet', 'inspire_hand', t, s, 0.9, updates=5000 + s)
        for s in table.ABLATION_SEEDS:
            write(tmp_path, 'row_b', 'inspire_hand', t, s, 0.7, updates=6000)
    lines, rows = table.ablation_table(table.load(tmp_path))
    a, b = rows[0], rows[1]
    assert a['label'] == 'DexDuet' and a['runs'] == 21 and a['updates_mean'] == pytest.approx(5000 + statistics.mean((15, 42, 113)))
    assert b['label'] == 'row_b' and b['add_mean'] == pytest.approx(70.0)


def test_release_check(table, tmp_path):
    def put(label, hand, seed, contact, mppe_key='mppe_cm_lanes0_98'):
        path = tmp_path / label / hand / 'box-200' / f'seed{seed}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        released = {'add_auc_pct': 96.0, 'sp_sr_pct': 100.0, 'mp_sr_pct': 50.0, mppe_key: 1.5, 'contact_auc_pct': 80.5}
        path.write_text(json.dumps(dict(label=label, hand=hand, task='box-200', seed=seed, add_auc=0.96, sp_sr=1.0,
                                        mp_sr=0.5, mppe_cm=2.0, mppe_cm_lanes0_98=1.5, contact_auc=contact,
                                        release_table_metrics=released)))
    put('DexDuet', 'inspire_hand', 15, 0.805)
    put('DexDuet', 'xhand', 42, 0.805)
    put('row_b', 'inspire_hand', 15, 0.805, mppe_key='mppe_cm')        # ablation runs report mppe_cm
    put('row_b', 'inspire_hand', 42, 0.786, mppe_key='mppe_cm')
    put('row_e', 'inspire_hand', 42, 0.784)
    write(tmp_path, 'DexDuet', 'allegro_hand', 'box-200', 15, 0.9)       # trained here: no released values, not checked
    lines = table.release_check(table.load(tmp_path))
    assert '| released runs | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.1000 | 2 |' in lines
    assert any('not bit-reproducible across machines' in x for x in lines)
    assert [x for x in lines if x.startswith('- ')] == [
        '- row_b inspire_hand box-200 seed 42: Contact-AUC 78.60 (reported 80.50)',
        '- row_e inspire_hand box-200 seed 42: Contact-AUC 78.40 (reported 80.50)']
    assert len([x for x in lines if x.startswith('| ') and not x.startswith('| Runs')]) == 1     # one row for all runs


def test_release_check_lists_every_differing_value(table, tmp_path):
    path = tmp_path / 'DexDuet' / 'xhand' / 'box-200' / 'seed15.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(label='DexDuet', hand='xhand', task='box-200', seed=15, add_auc=0.90, sp_sr=1.0, mp_sr=0.5,
                                    mppe_cm=2.0, contact_auc=0.8,
                                    release_table_metrics={'add_auc_pct': 96.0, 'sp_sr_pct': 90.0, 'contact_auc_pct': 80.0})))
    lines = table.release_check(table.load(tmp_path))
    assert '| released runs | 1 | 6.0000 | 10.0000 | - | - | 0.0000 | 1 |' in lines
    assert [x for x in lines if x.startswith('- ')] == [
        '- DexDuet xhand box-200 seed 15: ADD-AUC 90.00 (reported 96.00), SP-SR 100.00 (reported 90.00)']
    assert table.release_check([]) == []
