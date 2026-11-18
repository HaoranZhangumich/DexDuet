"""Training arguments for the rows of the DexDuet ablation table (baselines/README.md).

| Row | Per-hand | A-RSI | r_hg | r_hoc |                                                     |
|-----|----------|-------|------|-------|-----------------------------------------------------|
| (a) | yes      | yes   | yes  | yes   | the full method (scripts/train_coordinates_hand.py) |
| (b) | yes      | yes   |      | yes   | hand-graph reward off                               |
| (c) | yes      | yes   | yes  |       | hand-object contact reward off                      |
| (d) | yes      | yes   |      |       | both rewards off ((b) and (c) at once)              |
| (e) |          | yes   | yes  | yes   | one shared actor: DexMachina's released learner     |
| (f) | yes      |       |      |       | (d) without A-RSI                                   |
| (g) | yes      |       | yes  | yes   | the full method without A-RSI: (a) without A-RSI    |

Every row starts from the exact flags of the paper's DexDuet runs (COMMON and TASKS in
scripts/train_coordinates_hand.py) and changes only what its row removes. (e) drops the per-hand policy: the factorized
actor and per-hand PPO (--coordinate_actor, --coord_ablation_id per-hand, --per_hand_ppo, --value_mode per_hand) and
with them the native protocol (--ablation_protocol dexmachina), which per-hand PPO requires. Run names carry the row
labels (nohg, nohoc, noir, unified, noir-noarsi, noarsi; noir = no interaction rewards). (d) sets both reward weights to
0, as (b) and (c) do one at a time. (f) also removes A-RSI: every --rsi_* flag goes and --curr_schedule goes from rsi
back to DexMachina's uniform schedule. (g) removes A-RSI from (a) exactly as (f) removes it from (d): the same --rsi_*
flags go and --curr_schedule becomes uniform, while both rewards keep the final method's weights; so (g) and (f) differ
only in --hand_graph_weight and --ho_contact_weight, as (a) and (d) do.

    python baselines/ablation_rows.py --row b --task box-200 --seed 15   # one argument per line
    python baselines/ablation_rows.py --diff                             # flag changes vs (a)
"""

import argparse
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROWS = {
    "b": "per-hand + A-RSI + r_hoc (no hand-graph reward)",
    "c": "per-hand + A-RSI + r_hg (no hand-object contact reward)",
    "d": "per-hand + A-RSI (no hand-graph, no hand-object contact reward)",
    "e": "unified policy (one shared actor) + A-RSI + r_hg + r_hoc",
    "f": "per-hand (no A-RSI, no hand-graph, no hand-object contact reward)",
    "g": "per-hand + r_hg + r_hoc (no A-RSI): the full method without A-RSI",
}
LABELS = {"b": "nohg", "c": "nohoc", "d": "noir", "e": "unified", "f": "noir-noarsi", "g": "noarsi"}
# Per-hand policy flags (value-taking ones list their value) that row (e) removes.
PER_HAND_FLAGS = {"--coordinate_actor": 0, "--coord_ablation_id": 1, "--per_hand_ppo": 0, "--ablation_protocol": 1,
                  "--value_mode": 1}
# A-RSI flags (with their number of values) that rows (f) and (g) remove, on top of --curr_schedule rsi -> uniform.
# Per task the final method has either --rsi_bin_frames or --rsi_bins (bin_frames 0 in TASKS).
A_RSI_FLAGS = {"--rsi_uniform_mix": 1, "--rsi_credit_gamma": 1, "--rsi_fail_alpha": 1, "--rsi_success_up": 1,
               "--rsi_success_down": 1, "--rsi_min_count": 1, "--rsi_min_gap": 1, "--rsi_min_horizon": 1,
               "--rsi_keyframes": 0, "--rsi_kf_capacity": 1, "--rsi_kf_prob": 1, "--rsi_kf_min_fill": 1,
               "--rsi_kf_min_task": 1, "--rsi_grace_bank": 1, "--rsi_kf_confirm": 1, "--rsi_kf_to_end": 0,
               "--rsi_frame0_share": 1, "--rsi_deadline": 1, "--rsi_zero_ramp": 1, "--rsi_bin_frames": 1, "--rsi_bins": 1}
NO_ARSI_SCHEDULE = "uniform"   # DexMachina's curriculum schedule, the trainer's default


def _launcher():
    spec = importlib.util.spec_from_file_location("train_coordinates_hand", REPO / "scripts" / "train_coordinates_hand.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_key(task):
    keys = {key.lower(): key for key in _launcher().TASKS}
    if task.lower() not in keys:
        raise SystemExit(f"unknown task {task}; one of {sorted(keys)}")
    return keys[task.lower()]


def experiment_name(row, task, seed):
    return f"abl-{row}-{LABELS[row]}_{task.lower()}_seed{seed}"


def full_method_args(task, seed, experiment):
    """Row (a): the launcher's own training command, without the interpreter and script path."""
    launcher = _launcher()
    command = launcher.training_command(task_key(task), seed, experiment, "coordinates-hand", "python")
    return command[2:]


def _set(args, flag, value):
    args = list(args)
    args[args.index(flag) + 1] = value
    return args


def _drop(args, flags):
    out, skip = [], 0
    for item in args:
        if skip:
            skip -= 1
            continue
        if item in flags:
            skip = flags[item]
            continue
        out.append(item)
    return out


def row_args(row, task, seed):
    if row not in ROWS:
        raise SystemExit(f"unknown row {row}; one of {sorted(ROWS)}")
    experiment = experiment_name(row, task, seed)
    args = full_method_args(task, seed, experiment)
    if row == "b":
        return _set(args, "--hand_graph_weight", "0")
    if row == "c":
        return _set(args, "--ho_contact_weight", "0")
    if row == "e":
        return _drop(args, PER_HAND_FLAGS)
    if row == "g":   # (a) without A-RSI, as (f) is (d) without A-RSI
        return _set(_drop(args, A_RSI_FLAGS), "--curr_schedule", NO_ARSI_SCHEDULE)
    args = _set(_set(args, "--hand_graph_weight", "0"), "--ho_contact_weight", "0")   # (d) and (f)
    if row == "f":
        args = _set(_drop(args, A_RSI_FLAGS), "--curr_schedule", NO_ARSI_SCHEDULE)
    return args


def diff():
    for task in sorted(_launcher().TASKS):
        full = full_method_args(task, 0, "x")
        for row in ROWS:
            args = row_args(row, task, 0)
            removed = [f for f in full if f.startswith("--") and f not in args]
            changed = [f for f in ("--hand_graph_weight", "--ho_contact_weight")
                       if f in args and args[args.index(f) + 1] != full[full.index(f) + 1]]
            other = [f"{f} {full[full.index(f) + 1]} -> {args[args.index(f) + 1]}" for f in ("--curr_schedule",)
                     if f in args and args[args.index(f) + 1] != full[full.index(f) + 1]]
            print(f"{task:15s} ({row}) removed {removed or '-'} set to 0 {changed or '-'}"
                  + (f" changed {other}" if other else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--row", choices=sorted(ROWS))
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference", help=argparse.SUPPRESS)   # older option, no longer needed; ignored
    parser.add_argument("--name", action="store_true", help="print the run name only")
    parser.add_argument("--diff", action="store_true", help="show each row's changes against the full method")
    args = parser.parse_args()
    if args.diff:
        diff()
        return
    if args.name:
        print(experiment_name(args.row, args.task, args.seed))
        return
    print("\n".join(row_args(args.row, args.task, args.seed)))


if __name__ == "__main__":
    main()
