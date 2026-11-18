"""Training-end records (final_checkpoint.json) and the checks evaluation applies to them.

Runs with --ablation_protocol dexmachina keep rl_games' native stopping rule; when training returns, the trainer saves the
terminal snapshot nn/terminal.pth and records the native best-training-reward checkpoint nn/<hand>.pth, its epoch, the
completed updates, the environment transitions and the stop reason. The evaluator accepts only that recorded checkpoint.
"""

import hashlib
import json
import os
from pathlib import Path

DEFAULT_HAND = "inspire_hand"
TERMINAL_NAME = "terminal"
LEGACY_TERMINAL_FILE = "wrench_final.pth"   # terminal snapshot of run folders written before the rename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_checkpoint_selection(run_dir, hand, completed_updates, transitions, stop_reason):
    """Select the native best-reward checkpoint, as DexMachina's launcher does."""
    import torch
    checkpoint = Path(run_dir) / 'nn' / f'{hand}.pth'
    if not checkpoint.is_file():
        raise RuntimeError(f'Native best-reward checkpoint missing: {checkpoint}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch = int(state['epoch'])
    if not 0 < epoch <= int(completed_updates):
        raise RuntimeError('Selected checkpoint epoch is outside the completed training range')
    return dict(checkpoint=str(checkpoint), selection='native_best_training_reward',
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                epoch=epoch, completed_updates=int(completed_updates),
                environment_transitions=int(transitions), stop_reason=stop_reason,
                terminal_checkpoint=str(Path(run_dir) / 'nn' / f'{TERMINAL_NAME}.pth'))


def record_training_end(algo, run_dir, hand, protocol, transitions, *, max_epochs, num_envs, horizon_length,
                        marl_baseline=None):
    """Save the terminal snapshot and write final_checkpoint.json for a v2 or dexmachina run; returns the record."""
    if protocol not in ('v2', 'dexmachina'):
        raise ValueError(f'final_checkpoint.json is written for --ablation_protocol v2|dexmachina, not {protocol!r}')
    if protocol == 'v2':
        expected_transitions = int(max_epochs) * num_envs * int(horizon_length)
        if int(algo.epoch_num) != int(max_epochs):
            raise RuntimeError('Ablation stopped before the declared update endpoint')
        if transitions != expected_transitions:
            raise RuntimeError('Ablation transition count does not match the fixed budget')
    # Preserve the terminal snapshot; native protocol evaluates the baseline's best checkpoint.
    final_path = os.path.join(str(run_dir), 'nn', TERMINAL_NAME)
    algo.save(final_path)
    if protocol == 'dexmachina':
        stop_reason = getattr(algo, 'training_stop_reason', 'native_training_loop_returned')
        if marl_baseline is not None:
            from dexmachina.rl.marl_baselines import native_stop_reason
            stop_reason = native_stop_reason(algo)
        selection = native_checkpoint_selection(run_dir, hand, algo.epoch_num, transitions, stop_reason)
    else:
        selection = dict(checkpoint=final_path + '.pth', selection='training_endpoint',
                         epoch=int(algo.epoch_num), environment_transitions=transitions)
    with open(os.path.join(str(run_dir), 'final_checkpoint.json'), 'x') as stream:
        json.dump(selection, stream, indent=2)
    return selection


def validate_training_checkpoint(run_dir: Path, expected: dict, protocol: str,
                                 accept_selections=("native_best_training_reward",)):
    """Validate trusted local native-best data without imposing a fixed end epoch.
    The native best checkpoint is rl_games' nn/<agent name>.pth, and train_rl_games.py names the agent after the hand: the
    saved agent config's name (as eval_rl_games.py passes it), else DEFAULT_HAND. accept_selections lists the selection
    rules final_checkpoint.json may declare (eval_rl_games.py --accept-selection adds one, e.g.
    best_zero_help_training_return for two released Inspire checkpoints)."""
    name = expected.get("name") or DEFAULT_HAND
    if protocol == "dexmachina":
        checkpoint = run_dir / "nn" / f"{name}.pth"
    else:
        checkpoint = run_dir / "nn" / f"{TERMINAL_NAME}.pth"
        if not checkpoint.is_file() and (run_dir / "nn" / LEGACY_TERMINAL_FILE).is_file():
            checkpoint = run_dir / "nn" / LEGACY_TERMINAL_FILE
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing checkpoint {checkpoint}")
    if protocol != "dexmachina":
        return checkpoint, None
    endpoint = json.loads((run_dir / "final_checkpoint.json").read_text())
    recorded = Path(endpoint.get("checkpoint", ""))
    # a run folder copied to another machine or path keeps its record: same <run>/nn/<file> and the same file content
    same = recorded.resolve() == checkpoint.resolve() or (
        recorded.parts[-3:] == checkpoint.resolve().parts[-3:]
        and endpoint.get("checkpoint_sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if endpoint.get("selection") not in accept_selections or not same:
        raise RuntimeError("DexMachina primary checkpoint must be the declared native best training reward"
                           + (f" (or one of {list(accept_selections)})" if len(accept_selections) > 1 else ""))
    selected, completed = endpoint.get("epoch"), endpoint.get("completed_updates")
    if (type(selected) is not int or type(completed) is not int
            or not 0 < selected <= completed <= expected["max_epochs"]):
        raise RuntimeError("DexMachina selected/completed epoch is invalid for the configured cap")
    transitions = completed * expected["num_actors"] * expected["horizon_length"]
    if endpoint.get("environment_transitions") != transitions:
        raise RuntimeError("DexMachina completed update count disagrees with environment transitions")
    # These files are written by this repository's trainer, as are the env.pkl files
    # already trusted by evaluation. Load on CPU without constructing an agent.
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != selected:
        raise RuntimeError("DexMachina checkpoint's actual saved epoch differs from final_checkpoint.json")
    if endpoint.get("checkpoint_sha256", sha256_file(checkpoint)) != sha256_file(checkpoint):
        raise RuntimeError("DexMachina selected checkpoint hash differs from final_checkpoint.json")
    return checkpoint, endpoint
