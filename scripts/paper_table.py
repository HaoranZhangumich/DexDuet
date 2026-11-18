"""Aggregate per-run summaries (scripts/reproduce_paper.py) into the paper's DexMachina-suite tables.

Input: <summaries>/<label>/<hand>/<task>/seed<S>.json, one per evaluated run. Labels: DexDuet, DexMachina, row_b ... row_g.

Conventions (the paper's main and ablation tables):
  * ADD-AUC, SP-SR, MP-SR and Contact-AUC in %, over all 100 evaluation lanes; MPPE in cm over lanes 0-98 (DexMachina's
    evaluator starts lane 99 from a zero hand pose, which can diverge; DexDuet's MPPE is the same either way).
  * A cell is the mean over runs +- the sample standard deviation (ddof 1) over the training-seed means, where a seed mean is
    the mean over that seed's demonstrations. DexMachina's checkpoints are grouped by the training seed of their release
    config.
  * Main table: seeds 15, 24, 42, 66, 113 (7 demonstrations x 5 seeds = 35 runs per hand and method).
  * Ablation table: Inspire, seeds 15, 42, 113; row (a) is DexDuet with these seeds. Updates = mean number of PPO updates
    until training stopped.
A cell built from fewer runs than the paper's is marked with its run count.

  python scripts/paper_table.py paper_repro/summaries --out paper_repro/tables
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

HANDS = ('inspire_hand', 'allegro_hand', 'xhand', 'schunk_hand')
HAND_NAMES = {'inspire_hand': 'Inspire', 'allegro_hand': 'Allegro', 'xhand': 'XHand', 'schunk_hand': 'Schunk'}
TASKS = ('box-200', 'ketchup-100', 'ketchup-300', 'mixer-170', 'mixer-300', 'notebook-300', 'waffleiron-300')
MAIN_SEEDS = (15, 24, 42, 66, 113)
ABLATION_SEEDS = (15, 42, 113)
ROWS = ('a', 'b', 'c', 'd', 'e', 'f', 'g')
ROW_TEXT = {'a': 'DexDuet', 'b': 'without the hand-graph reward', 'c': 'without the hand-object contact reward',
            'd': 'without both interaction rewards', 'e': "DexMachina's unified policy", 'f': "(d) with DexMachina's curriculum",
            'g': "DexMachina's curriculum instead of A-RSI"}
METRICS = (('add', 'ADD-AUC', 1), ('sp', 'SP-SR', 1), ('mp', 'MP-SR', 1), ('mppe', 'MPPE (cm)', 2), ('contact', 'Contact-AUC', 1))


def load(root):
    runs = []
    for path in sorted(Path(root).glob('*/*/*/seed*.json')):
        s = json.loads(path.read_text())
        runs.append(dict(label=s['label'], hand=s['hand'], task=s['task'], seed=int(s['seed']), updates=s.get('updates'),
                         released=s.get('release_table_metrics'),
                         add=100 * s['add_auc'], sp=100 * s['sp_sr'], mp=100 * s['mp_sr'],
                         mppe=s['mppe_cm_lanes0_98'] if s.get('mppe_cm_lanes0_98') is not None else s['mppe_cm'],
                         contact=None if s.get('contact_auc') is None else 100 * s['contact_auc'], path=str(path)))
    return runs


def cell(runs, metric):
    """Mean over runs and sd over seed means (seed mean = mean over that seed's demonstrations)."""
    values = [r for r in runs if r[metric] is not None]
    if not values:
        return None
    mean = sum(r[metric] for r in values) / len(values)
    seeds = sorted({r['seed'] for r in values})
    seed_means = [statistics.mean(r[metric] for r in values if r['seed'] == s) for s in seeds]
    sd = statistics.stdev(seed_means) if len(seed_means) > 1 else None
    return dict(mean=mean, sd=sd, n=len(values), seeds=len(seeds))


def fmt(c, digits, expected):
    if c is None:
        return '-'
    text = f"{c['mean']:.{digits}f}" + (f" ± {c['sd']:.{digits}f}" if c['sd'] is not None else '')
    return text + (f" (n={c['n']})" if c['n'] != expected else '')


def main_rows(runs, hand):
    """(name, runs) per table row of one hand: DexMachina's released checkpoints and DexDuet's runs."""
    pick = lambda label: [r for r in runs if r['label'] == label and r['hand'] == hand and r['seed'] in MAIN_SEEDS]
    out = [('DexMachina', pick('DexMachina')), ('DexDuet', pick('DexDuet'))]
    return [(name, sel) for name, sel in out if sel]


def main_table(runs):
    lines = ['## DexMachina suite (main table)', '',
             '| Robot | Method | Runs | ADD-AUC ↑ | SP-SR ↑ | MP-SR ↑ | MPPE (cm) ↓ | Contact-AUC ↑ |', '|---|---|---|---|---|---|---|---|']
    rows = []
    for hand in HANDS:
        for label, sel in main_rows(runs, hand):
            expected = len(TASKS) * len(MAIN_SEEDS)
            cells = {m: cell(sel, m) for m, _name, _d in METRICS}
            lines.append(f"| {HAND_NAMES[hand]} | {label} | {len(sel)} of {expected} | "
                         + ' | '.join(fmt(cells[m], d, expected) for m, _name, d in METRICS) + ' |')
            rows.append(dict(table='main', hand=hand, label=label, runs=len(sel),
                             **{f'{m}_{k}': (cells[m] or {}).get(k) for m, _n, _d in METRICS for k in ('mean', 'sd', 'n')}))
    return lines, rows


def ablation_table(runs):
    lines = ['## Ablations (Inspire, seeds 15, 42, 113)', '',
             '| Row | Variant | Runs | ADD-AUC ↑ | Contact-AUC ↑ | Updates ↓ |', '|---|---|---|---|---|---|']
    rows = []
    for row in ROWS:
        label = 'DexDuet' if row == 'a' else f'row_{row}'
        sel = [r for r in runs if r['label'] == label and r['hand'] == 'inspire_hand' and r['seed'] in ABLATION_SEEDS]
        if not sel:
            continue
        expected = len(TASKS) * len(ABLATION_SEEDS)
        add, contact = cell(sel, 'add'), cell(sel, 'contact')
        updates = [r['updates'] for r in sel if r['updates'] is not None]
        mean_updates = sum(updates) / len(updates) if updates else None
        lines.append(f"| ({row}) | {ROW_TEXT[row]} | {len(sel)} of {expected} | {fmt(add, 1, expected)} | "
                     f"{fmt(contact, 1, expected)} | {'-' if mean_updates is None else f'{mean_updates:,.0f}'} |")
        rows.append(dict(table='ablation', hand='inspire_hand', label=label, runs=len(sel),
                         add_mean=(add or {}).get('mean'), add_sd=(add or {}).get('sd'),
                         contact_mean=(contact or {}).get('mean'), contact_sd=(contact or {}).get('sd'), updates_mean=mean_updates))
    return lines, rows


RELEASED = (('add', ('add_auc_pct',)), ('sp', ('sp_sr_pct',)), ('mp', ('mp_sr_pct',)),
            ('mppe', ('mppe_cm_lanes0_98', 'mppe_cm')), ('contact', ('contact_auc_pct',)))


def release_check(runs, tol=0.01):
    """Each released run's reproduced values against the values it was reported with (RUN_INFO.json table_metrics); the
    runs that differ by more than tol are listed with their differing values."""
    sel = [r for r in runs if r.get('released')]
    if not sel:
        return []
    lines = ['## Check against the released values', '',
             'Largest absolute difference between the reproduced values and the values each run was reported with '
             '(RUN_INFO.json, table_metrics). GPU evaluation is not bit-reproducible across machines, so a run evaluated on '
             'another machine can differ from its reported values (REPRODUCE.md, section 2.2).', '',
             '| Runs | Count | ADD-AUC | SP-SR | MP-SR | MPPE (cm) | Contact-AUC | Differing by more than ' + f'{tol} |',
             '|---|---|---|---|---|---|---|---|']
    names = {m: name for m, name, _d in METRICS}
    worst, off = {}, []
    for r in sel:
        differing = []
        for m, keys in RELEASED:
            key = next((k for k in keys if r['released'].get(k) is not None), None)
            if key is None or r[m] is None:
                continue
            reported = float(r['released'][key])
            d = abs(r[m] - reported)
            worst[m] = max(worst.get(m, 0.0), d)
            if d > tol:
                differing.append(f'{names[m]} {r[m]:.2f} (reported {reported:.2f})')
        if differing:
            off.append((r, differing))
    lines.append(f'| released runs | {len(sel)} | '
                 + ' | '.join(f'{worst[m]:.4f}' if m in worst else '-' for m, _k in RELEASED) + f' | {len(off)} |')
    lines.append('')
    for r, differing in sorted(off, key=lambda x: (x[0]['label'], x[0]['hand'], x[0]['task'], x[0]['seed'])):
        lines.append(f"- {r['label']} {r['hand']} {r['task']} seed {r['seed']}: " + ', '.join(differing))
    return lines + ([''] if off else [])


def per_task(runs):
    lines = ['## Per demonstration (mean ± sd over seeds)', '']
    for label in sorted({r['label'] for r in runs}):
        for hand in HANDS:
            sel = [r for r in runs if r['label'] == label and r['hand'] == hand]
            if not sel:
                continue
            lines += [f'### {label}, {HAND_NAMES[hand]}', '', '| Demonstration | Seeds | ADD-AUC | SP-SR | MP-SR | MPPE (cm) | Contact-AUC |',
                      '|---|---|---|---|---|---|---|']
            for task in TASKS:
                t = [r for r in sel if r['task'] == task]
                if not t:
                    continue
                parts = []
                for m, _name, d in METRICS:
                    v = [r[m] for r in t if r[m] is not None]
                    parts.append('-' if not v else f"{statistics.mean(v):.{d}f}" + (f" ± {statistics.stdev(v):.{d}f}" if len(v) > 1 else ''))
                lines.append(f"| {task} | {', '.join(str(r['seed']) for r in sorted(t, key=lambda r: r['seed']))} | " + ' | '.join(parts) + ' |')
            lines.append('')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('summaries', help='folder of per-run summaries (<label>/<hand>/<task>/seed<S>.json)')
    ap.add_argument('--out', default=None, help='write TABLES.md, tables.csv and per_run.csv here')
    a = ap.parse_args()
    runs = load(a.summaries)
    if not runs:
        raise SystemExit(f'no run summaries under {a.summaries}')
    main_lines, main_rows = main_table(runs)
    abl_lines, abl_rows = ablation_table(runs)
    text = '\n'.join(['# DexDuet paper tables (DexMachina suite)', '',
                      'Cells: mean over runs ± sd over training-seed means; MPPE over lanes 0-98. (n=k): fewer runs than in the paper.',
                      ''] + main_lines + [''] + abl_lines + [''] + release_check(runs) + per_task(runs)) + '\n'
    print(text)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'TABLES.md').write_text(text)
        rows = main_rows + abl_rows
        if rows:
            keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('table', 'hand', 'label', 'runs'), k))
            with open(out / 'tables.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        with open(out / 'per_run.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in runs[0] if k != 'released'], extrasaction='ignore')
            w.writeheader()
            w.writerows(runs)
        print(f'wrote {out / "TABLES.md"}, {out / "tables.csv"}, {out / "per_run.csv"}')


if __name__ == '__main__':
    main()
