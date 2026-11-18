"""Hold test, step 4 (CPU): aggregation, DexDuet - DexMachina comparison and tables (the paper's hold tables).

The aggregation and bootstrap are those of the paper's analysis, unchanged:
  * per task: mean +- sd (ddof 1) over seeds; overall / in-hand / resting: mean over runs +- sd (ddof 1) over the
    training-seed means (a seed mean is the mean over that seed's runs in the subset); 'All hands' pools every hand;
  * difference DexDuet - DexMachina of the means over runs, 95% percentile bootstrap: 10,000 resamples, a fresh
    numpy default_rng(0) per comparison, runs resampled with replacement within each (hand, task, method) cell, cells drawn
    in the order hand (Inspire, Allegro, XHand, Schunk) > task > method (DexDuet before DexMachina), runs in seed order;
  * primary: every run; sensitivity ('excl_flagged'): without the runs whose replay is not identical (metrics.replay_checks).
The primary endpoint of the analysis is the per-hand difference of the unconditional Hold-ADD-AUC over all runs. On-track
values (runs with an on-track lane at the clip end) and pooled values are descriptive.

  python -m dexmachina.eval.hold_test.analyze <folder of hold test outputs> [--out <dir>]
  python -m dexmachina.eval.hold_test.analyze --per-run per_run.csv [--out <dir>]      # from per-run metrics
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from os.path import join

import numpy as np

B = 10000
HAND_ORDER = ['Inspire', 'Allegro', 'XHand', 'Schunk']
METHODS = ['DexDuet', 'DexMachina']
TASKS = ['box-200', 'ketchup-100', 'ketchup-300', 'mixer-170', 'mixer-300', 'notebook-300', 'waffleiron-300']
BASE = ['hold_add_auc', 'hold_sr3', 'hold_sr5', 'drop_rate', 'mean_hold_add_cm', 'contact_any', 'contact_finger',
        'contact_left', 'contact_right'] + [f'{q}_{t}s_{s}' for q in ('pos', 'rot', 'hinge', 'add', 'disp') for t in (1, 2, 3)
                                            for s in ('med', 'mean')]
METRICS = [f'{c}_{k}' for c in ('u', 'c') for k in BASE] + [
    'c_n_lanes', 'n_ontrack', 'ontrack_frac', 'l98_u_hold_add_auc', 'l98_u_hold_sr3', 'l98_u_hold_sr5', 'l98_u_drop_rate',
    'l98_c_hold_add_auc', 'l98_n_ontrack']
BOOT = ['u_hold_add_auc', 'u_hold_sr3', 'u_hold_sr5', 'u_drop_rate', 'u_pos_3s_med', 'u_rot_3s_med', 'u_contact_any',
        'c_hold_add_auc', 'c_hold_sr3', 'c_hold_sr5', 'c_drop_rate', 'c_pos_3s_med', 'c_rot_3s_med', 'c_contact_any',
        'l98_u_hold_add_auc', 'l98_c_hold_add_auc']


def seed_sd(rows, k):
    sm = defaultdict(list)
    for x in rows:
        if np.isfinite(x[k]):
            sm[x['seed']].append(x[k])
    means = [np.mean(v) for v in sm.values()]
    return (float(np.std(means, ddof=1)) if len(means) > 1 else np.nan), len(means)


def agg(rows, k):
    v = np.array([x[k] for x in rows], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan)
    sd, ns = seed_sd(rows, k)
    return dict(n=int(len(v)), mean=float(v.mean()), sd=sd, n_seeds=ns)


def agg_task(rows, k):
    v = np.array([x[k] for x in rows], float)
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), mean=float(v.mean()) if len(v) else np.nan,
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan)


def boot(rows, k, hands):
    """DexDuet - DexMachina difference of the means over runs, 95% percentile bootstrap (see the module docstring)."""
    rng = np.random.default_rng(0)
    sums = {m: np.zeros(B) for m in METHODS}
    ns = {m: 0 for m in METHODS}
    obs = {m: [] for m in METHODS}
    srt = sorted(rows, key=lambda x: x['order'])
    for h in hands:
        for t in TASKS:
            for m in METHODS:
                v = np.array([x[k] for x in srt if x['hand'] == h and x['task'] == t and x['method'] == m and np.isfinite(x[k])], float)
                if len(v) == 0:
                    continue
                idx = rng.integers(0, len(v), size=(B, len(v)))
                sums[m] += v[idx].sum(1)
                ns[m] += len(v)
                obs[m] += list(v)
    if ns['DexDuet'] == 0 or ns['DexMachina'] == 0:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan)
    d = sums['DexDuet'] / ns['DexDuet'] - sums['DexMachina'] / ns['DexMachina']
    return dict(diff=float(np.mean(obs['DexDuet']) - np.mean(obs['DexMachina'])), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)))


def load_rows(results=None, per_run=None):
    if per_run:
        rows = []
        for r in csv.DictReader(open(per_run)):
            row = dict(r)
            for k, v in r.items():
                if k not in ('key', 'hand', 'method', 'task', 'task_class', 'flagged', 'failed_checks', 'study_protocol',
                             'nan_lanes', 'set', 'run_id', 'out'):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = np.nan
            row['seed'] = int(float(r['seed']))
            row['flagged'] = str(r.get('flagged')).strip().lower() == 'true'
            rows.append(row)
        return rows
    from dexmachina.eval.hold_test.metrics import run_metrics
    rows = []
    for name in sorted(os.listdir(results)):
        folder = join(results, name)
        if os.path.exists(join(folder, 'hold_add.npz')) and os.path.exists(join(folder, 'hold_eval.json')):
            rows.append(run_metrics(folder))
    return rows


def aggregate(rows):
    from dexmachina.eval.hold_test.metrics import TASK_CLASS
    hands = [h for h in HAND_ORDER if any(x['hand'] == h for x in rows)] + sorted({x['hand'] for x in rows} - set(HAND_ORDER))
    # within-cell draw order of the bootstrap: seed order
    for i, x in enumerate(sorted(rows, key=lambda x: (hands.index(x['hand']), TASKS.index(x['task']), x['method'], x['seed']))):
        x['order'] = i
    subsets = dict(all=TASKS, **{'in-hand': [t for t in TASKS if TASK_CLASS[t] == 'in-hand'],
                                 'resting': [t for t in TASKS if TASK_CLASS[t] == 'resting']}, **{t: [t] for t in TASKS})
    groups = dict(**{h: [h] for h in hands})
    if len(hands) > 1:
        groups['All hands'] = hands
    metrics = [k for k in METRICS if k in rows[0]]
    out = []
    for var, keep in dict(primary=lambda x: True, excl_flagged=lambda x: not x['flagged']).items():
        for hg, hs in groups.items():
            for sub, tasks in subsets.items():
                sel = [x for x in rows if keep(x) and x['hand'] in hs and x['task'] in tasks]
                for k in metrics:
                    rec = dict(variant=var, hands=hg, subset=sub, metric=k)
                    for m in METHODS:
                        rm = [x for x in sel if x['method'] == m]
                        a = agg_task(rm, k) if (sub in TASKS and len(hs) == 1) else agg(rm, k)
                        rec.update({f'{m}_{q}': v for q, v in a.items()})
                    if k in BOOT:
                        rec.update(boot(sel, k, [h for h in hands if h in hs]))
                    out.append(rec)
    return out, groups


def tables(aggs, groups, rows):
    idx = {(a['variant'], a['hands'], a['subset'], a['metric']): a for a in aggs}
    f = lambda x, d=1: '-' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
    ms = lambda a, m: '-' if not np.isfinite(a[f'{m}_mean']) else (
        f"{a[f'{m}_mean']:.1f} ± {a[f'{m}_sd']:.1f}" if np.isfinite(a[f'{m}_sd']) else f"{a[f'{m}_mean']:.1f}")
    ci = lambda a: '-' if not np.isfinite(a.get('diff', np.nan)) else f"{a['diff']:+.1f} [{a['lo']:.1f}, {a['hi']:.1f}]"
    md = ['# Hold after the last frame', '',
          'Each policy keeps acting for 3 s with every reference frozen at the last frame; Hold-ADD-AUC (%) scores the object '
          'against its final pose. On track: runs within 3 cm of the reference at the last frame, with their number. '
          'Neither method was trained on a frozen reference, so the hold lies outside the training distribution of both.', '',
          '## Hold-ADD-AUC per hand', '',
          '| Hand | DexMachina | DexDuet | DexMachina on track | DexDuet on track |', '|---|---|---|---|---|']
    for hg in groups:
        u, c = idx[('primary', hg, 'all', 'u_hold_add_auc')], idx[('primary', hg, 'all', 'c_hold_add_auc')]
        md.append(f"| {hg} | {f(u['DexMachina_mean'])} | {f(u['DexDuet_mean'])} | {f(c['DexMachina_mean'])} ({c['DexMachina_n']}) | "
                  f"{f(c['DexDuet_mean'])} ({c['DexDuet_n']}) |")
    md += ['', '## Per hand: mean over runs ± sd over training-seed means (SR@5 and Drop are the hold rate and the drop rate '
           'of the paper\'s hold tables); difference DexDuet - DexMachina with its 95% bootstrap '
           'interval', '',
           '| Hand | Method | Hold-ADD-AUC | SR@5 | Drop | On track: Hold-ADD-AUC (runs) |', '|---|---|---|---|---|---|']
    for hg in groups:
        for m in METHODS:
            u, s5 = idx[('primary', hg, 'all', 'u_hold_add_auc')], idx[('primary', hg, 'all', 'u_hold_sr5')]
            dr, c = idx[('primary', hg, 'all', 'u_drop_rate')], idx[('primary', hg, 'all', 'c_hold_add_auc')]
            md.append(f"| {hg} | {m} | {ms(u, m)} | {ms(s5, m)} | {ms(dr, m)} | {ms(c, m)} ({c[f'{m}_n']}) |")
        md.append(f"| {hg} | Difference | {ci(idx[('primary', hg, 'all', 'u_hold_add_auc')])} | | | "
                  f"{ci(idx[('primary', hg, 'all', 'c_hold_add_auc')])} |")
    md += ['', '## Subsets: in the air (in-hand) and resting at the last frame', '',
           '| Hand | Subset | Method | Hold-ADD-AUC | SR@5 | Drop | On track: Hold-ADD-AUC (runs) |', '|---|---|---|---|---|---|---|']
    for hg in groups:
        for sub in ('in-hand', 'resting'):
            for m in METHODS:
                u, s5 = idx[('primary', hg, sub, 'u_hold_add_auc')], idx[('primary', hg, sub, 'u_hold_sr5')]
                dr, c = idx[('primary', hg, sub, 'u_drop_rate')], idx[('primary', hg, sub, 'c_hold_add_auc')]
                md.append(f"| {hg} | {sub} | {m} | {ms(u, m)} | {ms(s5, m)} | {ms(dr, m)} | {ms(c, m)} ({c[f'{m}_n']}) |")
    flagged = [x['key'] for x in rows if x['flagged']]
    md += ['', f"Runs: {len(rows)}; flagged (replay not identical, kept in these tables, left out in the 'excl_flagged' "
           f"rows of aggregates.csv): {', '.join(map(str, flagged)) or 'none'}.", '']
    return '\n'.join(md)


def wcsv(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('results', nargs='?', help='folder with one sub-folder per run (run.py + add.py outputs)')
    ap.add_argument('--per-run', default=None, help='per-run metrics CSV instead of run folders')
    ap.add_argument('--out', default=None, help='output folder (default: the results folder)')
    a = ap.parse_args()
    if not (a.results or a.per_run):
        ap.error('give a results folder or --per-run')
    rows = load_rows(a.results, a.per_run)
    if not rows:
        raise SystemExit('no runs found')
    aggs, groups = aggregate(rows)
    out = a.out or a.results
    os.makedirs(out, exist_ok=True)
    wcsv(join(out, 'per_run.csv'), rows)
    wcsv(join(out, 'aggregates.csv'), aggs)
    text = tables(aggs, groups, rows)
    open(join(out, 'TABLES.md'), 'w').write(text)
    print(text)
    json.dump(dict(n_runs=len(rows), hands=list(groups)), open(join(out, 'analysis.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
