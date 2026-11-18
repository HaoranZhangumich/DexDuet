"""Explicit evaluation control counts; legacy behavior remains the default."""

import hashlib
import json
import os
from pathlib import Path


def policy_controlled_env_indices(num_envs, *, is_eval, eval_reference_env=True):
    if int(num_envs) != num_envs or num_envs < 1:
        raise ValueError("Evaluation/training requires a positive integer environment count")
    indices = list(range(int(num_envs)))
    if is_eval and eval_reference_env and num_envs > 1:
        return indices[:-1]
    return indices


def validate_evaluation_options(num_envs, eval_episodes, *, all_policy_envs, show_reference):
    policy_controlled_env_indices(num_envs, is_eval=True)
    if int(eval_episodes) != eval_episodes or eval_episodes < 1:
        raise ValueError("eval_episodes must be a positive integer")
    if all_policy_envs and show_reference:
        raise ValueError("--all_policy_envs cannot be combined with --show_reference")


def evaluation_folder_name(checkpoint_name, *, all_policy_envs=False):
    suffix = "_eval_all_policy" if all_policy_envs else "_eval"
    return Path(checkpoint_name).stem + suffix


def prepare_evaluation_directory(path, *, all_policy_envs=False):
    """New protocol refuses to overwrite even a partial earlier evaluation."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=not all_policy_envs)
    return path


def build_evaluation_metadata(checkpoint, *, num_envs, eval_episodes,
                              all_policy_envs, show_reference=False):
    validate_evaluation_options(num_envs, eval_episodes,
                                all_policy_envs=all_policy_envs,
                                show_reference=show_reference)
    checkpoint = Path(checkpoint).resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    controlled = policy_controlled_env_indices(
        num_envs, is_eval=True, eval_reference_env=not all_policy_envs,
    )
    return {
        "schema_version": 1,
        "protocol": "all_policy_controlled_v1" if all_policy_envs else "legacy_reserved_last_env_v1",
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest.hexdigest(),
        "num_envs": int(num_envs), "eval_episodes": int(eval_episodes),
        "num_trials": int(num_envs * eval_episodes),
        "policy_controlled_envs": len(controlled),
        "policy_controlled_trials": len(controlled) * int(eval_episodes),
        "policy_controlled_env_ids": controlled,
        "uncontrolled_env_ids": sorted(set(range(num_envs)).difference(controlled)),
        "show_reference": bool(show_reference),
        "deterministic_policy": True, "object_actuation": False,
        "early_reset_threshold": 0.0,
        "status": "running", "completed_episodes": 0,
    }


def verify_common_reset_state(manifest_path, arrays, *, seed, num_envs):
    """Save the first actual reset, then require byte-identical reset state.

    The current evaluation uses a deterministic reference start, repeated over
    environments. These are controlled trials, not sampled independent starts.
    A filesystem lock makes concurrent first evaluations safe. Numerical or
    configuration differences fail before a policy takes its first action.
    """
    import fcntl
    import numpy as np

    if not arrays or seed < 0 or num_envs < 1:
        raise ValueError("Common evaluation reset requires arrays, a seed and environments")
    arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Evaluation reset arrays must be finite and non-object")
    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = manifest_path.with_suffix(".npz")
    description = {
        "schema_version": 1, "seed": int(seed), "num_envs": int(num_envs),
        "initial_state_distribution": "fixed_reference_start_repeated_across_environments",
        "arrays": {
            key: {"shape": list(value.shape), "dtype": value.dtype.str,
                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for key, value in sorted(arrays.items())
        },
    }
    with manifest_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            expected = {key: saved[key] for key in description}
            if expected != description:
                raise RuntimeError("Evaluation reset differs from the frozen common initial states")
            if not state_path.is_file() or hashlib.sha256(state_path.read_bytes()).hexdigest() != saved["state_file_sha256"]:
                raise RuntimeError("Frozen evaluation reset state file is missing or changed")
            with np.load(state_path, allow_pickle=False) as frozen:
                if set(frozen.files) != set(arrays) or any(
                    not np.array_equal(frozen[key], arrays[key]) for key in arrays
                ):
                    raise RuntimeError("Evaluation reset arrays do not match the saved states")
        else:
            if state_path.exists():
                raise RuntimeError("Orphaned evaluation state file; use a fresh evaluation manifest")
            with state_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            saved = dict(description, state_file=str(state_path),
                         state_file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())
            temporary = manifest_path.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
            temporary.replace(manifest_path)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "state_file": str(state_path), "state_file_sha256": saved["state_file_sha256"],
        "seed": int(seed), "initial_state_distribution": description["initial_state_distribution"],
        "verification": "exact_arrays_equal_before_first_policy_action",
    }
