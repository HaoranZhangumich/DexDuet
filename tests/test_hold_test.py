"""dexmachina/eval/hold_test: the hold test keeps the paper's constants and definitions (README.md there): 180 frozen
hold steps, Hold-ADD-AUC on 1-9 cm thresholds, SR@5, the drop rule, the replay checks and the seed-mean aggregation.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_hold_test.py
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from dexmachina.eval.hold_test import analyze as A
from dexmachina.eval.hold_test import metrics as M

ROOT = Path(__file__).resolve().parents[1]


def test_constants():
    assert np.allclose(M.TAU, np.arange(1, 10) / 100) and M.HOLD_T == (60, 120, 180)
    assert {t for t, c in M.TASK_CLASS.items() if c == 'resting'} == {'box-200', 'notebook-300'}
    assert A.B == 10000 and A.METHODS == ['DexDuet', 'DexMachina']
    tree = ast.parse((ROOT / 'dexmachina/eval/hold_test/run.py').read_text())
    hold = [k for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument'
            and n.args and getattr(n.args[0], 'value', None) == '--hold-steps' for k in n.keywords if k.arg == 'default']
    assert [ast.literal_eval(k.value) for k in hold] == [180]


def lanes(add, z, n=4, L=5, H=180):
    """Synthetic run: constant object ADD `add` (m) on every hold frame, object root height z at the end."""
    addB = np.full((n, H + 1), add)
    phys = np.zeros((L + H, n, 8))
    phys[:, :, 3] = 1.0
    phys[:, :, 2] = z
    g = np.array([0, 0, 1.0, 1, 0, 0, 0, 0])
    contact = np.zeros((L + H, n, 2, 4), bool)
    masks = dict(any=np.ones(4, bool))
    return M.lane_metrics(addB, addB, phys, g, contact, masks, np.arange(n), L, H)


def test_lane_metrics():
    held = lanes(0.0, 1.0)
    assert held['hold_add_auc'] == pytest.approx(100.0) and held['hold_sr5'] == 100.0 and held['drop_rate'] == 0.0
    near = lanes(0.045, 1.0)          # 4.5 cm: within 5 cm, not 3 cm; AUC counts thresholds 5..9 cm
    assert near['hold_sr5'] == 100.0 and near['hold_sr3'] == 0.0
    assert near['hold_add_auc'] == pytest.approx(100 * np.trapz([0, 0, 0, 0, 1, 1, 1, 1, 1], x=np.linspace(0, 1, 9)))
    dropped = lanes(0.5, 0.5)         # 50 cm below the reference and below the table
    assert dropped['drop_rate'] == 100.0 and dropped['hold_add_auc'] == 0.0


def test_aggregation_is_mean_over_runs_and_sd_over_seed_means():
    rows = [dict(hand='XHand', method=m, task=t, seed=s, flagged=False, key=f'{m}{t}{s}', u_hold_add_auc=v)
            for m, base in (('DexDuet', 40.0), ('DexMachina', 20.0)) for i, t in enumerate(A.TASKS)
            for j, s in enumerate((15, 24, 42, 66, 113)) for v in [base + i + 0.5 * j]]
    aggs, groups = A.aggregate(rows)
    a = next(x for x in aggs if x['variant'] == 'primary' and x['hands'] == 'XHand' and x['subset'] == 'all'
             and x['metric'] == 'u_hold_add_auc')
    seed_means = [40 + 3 + 0.5 * j for j in range(5)]
    assert a['DexDuet_mean'] == pytest.approx(40 + 3 + 1.0) and a['DexDuet_sd'] == pytest.approx(np.std(seed_means, ddof=1))
    assert a['diff'] == pytest.approx(20.0) and a['lo'] <= 20.0 <= a['hi']
    again, _ = A.aggregate(rows)
    assert [x.get('lo') for x in again] == [x.get('lo') for x in aggs]      # fixed bootstrap seed


def test_replay_checks_flag_a_diverging_replay():
    L, H, S = 10, 180, 190
    z = dict(phys_state=np.zeros((S, 3, 8)), nan=np.zeros((S, 3), bool), dones=np.zeros((S, 3), bool))
    meta = dict(L=L, study_protocol='legacy', reset_verify=dict(verification='exact_arrays_equal_before_first_policy_action'),
                **{'prefix_0_to_L-2_vs_scored': dict(bitwise_equal=True)}, ebuf_env_used_during_patch=[9],
                ebuf_robot_used_during_patch=[9], ebuf_obj_used_during_patch=[9], robot_t_at_action_during_hold=[9],
                eval_demo_index_lane0_during_hold=['9'], demo_state_rows_during_hold_equal_final_ref=True,
                residual_target_hold_equals_last_frame=True, residual_target_last_clip_step_equals_last_frame=True,
                phase_obs_values_during_patch=[0.8], phase_obs_last_clip_observation_acted_on=0.8,
                state_diff_goal_is_last_frame_maxabs=0.0, state_diff_obs_matches_buffer_maxabs=0.0,
                reward_pos_dist_vs_last_frame_maxabs=0.0, resets_in_patch_steps_non_nan_lanes=0, hold_patch_steps=[9, 189],
                overflow=[dict(pair_overflow_total=0, contact_overflow_total=0)])
    assert all(M.replay_checks(meta, z).values())
    meta['prefix_0_to_L-2_vs_scored'] = dict(bitwise_equal=False)
    assert not M.replay_checks(meta, z)['prefix']
