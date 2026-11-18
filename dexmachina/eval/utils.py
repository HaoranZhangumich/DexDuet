import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def match_dict(flat_dict: Dict[str, Any], filter_conditions: Dict[str, Any]) -> bool:
    for key_path, condition in filter_conditions.items():
        if key_path.startswith("_"):
            continue
        try:
            value = flat_dict.get(key_path, None)
            if value is None:
                raise KeyError(key_path)
            if isinstance(condition, list):
                if value not in condition:
                    return False
            elif isinstance(condition, str) and ("*" in condition or "?" in condition):
                if not fnmatch.fnmatch(str(value), condition):
                    return False
            elif isinstance(condition, dict) and "_range" in condition:
                range_min, range_max = condition["_range"]
                if not (range_min <= value <= range_max):
                    return False
            else:
                if value != condition:
                    return False
        except (KeyError, AttributeError):
            return False
    return True


def find_config_path(start_path: str) -> Optional[str]:
    path = Path(start_path).resolve()
    if path.is_file():
        path = path.parent
    for parent in [path] + list(path.parents):
        cfg = parent / "config.yaml"
        if cfg.exists():
            return str(cfg)
    return None


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_CLIP_RE = re.compile(r"[a-zA-Z]+\d+-\d+-s\d+-u\d+")


def find_params_dir(start_path: str) -> Optional[Path]:
    """rl_games run dirs (produced by train_rl_games.py) always have a
    params/ subdir with env.yaml + agent.yaml, alongside nn/ and summaries/."""
    path = Path(start_path).resolve()
    if path.is_file():
        path = path.parent
    for parent in [path] + list(path.parents):
        params_dir = parent / "params"
        if (params_dir / "env.yaml").exists() and (params_dir / "agent.yaml").exists():
            return params_dir
    return None


def synthesize_config_from_params(params_dir: Path) -> Dict[str, Any]:
    """Build a config dict shaped like the wandb-style config.yaml that
    parse_config_for_flat()/extract_clip() expect, from a run's own
    params/env.yaml + params/agent.yaml (which are always saved locally,
    unlike wandb's config.yaml which may not be reachable from eval_path)."""
    # env.yaml/agent.yaml contain non-standard yaml tags (python/tuple, PosixPath)
    with open(params_dir / "env.yaml", "r") as f:
        env_kwargs = yaml.unsafe_load(f)
    with open(params_dir / "agent.yaml", "r") as f:
        agent_cfg = yaml.unsafe_load(f)
    params = agent_cfg.get("params", agent_cfg) if isinstance(agent_cfg, dict) else {}

    # clip isn't stored as a field anywhere in env.yaml/agent.yaml, but the
    # run directory name always embeds it, e.g.
    # ".../inspire-reproduce_ketchup30-130-s01-u01_B12000_.../params"
    clip = None
    match = _CLIP_RE.search(str(params_dir.parent.name))
    if match:
        clip = match.group(0)

    return {"env_kwargs": env_kwargs, "params": params, "clip": clip}


def resolve_run_config(eval_path: str) -> Optional[Dict[str, Any]]:
    """Find a run's config, preferring a wandb-style config.yaml (if
    reachable by walking up from eval_path) and falling back to
    synthesizing one from the run's own params/env.yaml + params/agent.yaml."""
    config, _ = resolve_run_config_with_keyword_path(eval_path)
    return config


def resolve_run_config_with_keyword_path(
    eval_path: str,
) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
    """Like resolve_run_config, but also returns a path string suitable for
    the "keywords" substring-matching heuristic in group_results.py's
    condition_sets. That heuristic only makes sense against a real wandb
    config.yaml path (author-specific historical run-naming fragments like
    "fast1-420") -- when the config is synthesized from params/env.yaml
    instead, there's no equivalent path, so callers should treat keyword
    conditions as vacuously satisfied (skip them) rather than matching
    against an unrelated path that will never contain those fragments."""
    cfg_path = find_config_path(eval_path)
    if cfg_path:
        return load_config(cfg_path), cfg_path
    params_dir = find_params_dir(eval_path)
    if params_dir:
        return synthesize_config_from_params(params_dir), None
    return None, None


def extract_clip(config: Dict[str, Any]) -> Optional[str]:
    clip = config.get("clip", None)
    if isinstance(clip, dict):
        clip = clip.get("value", None)
    if isinstance(clip, str):
        return clip
    params = config.get("params", {})
    if isinstance(params, dict):
        params_val = params.get("value", params)
        exp = (
            params_val.get("config", {})
            .get("full_experiment_name", None)
        )
        if isinstance(exp, str):
            for snippet in exp.split("_"):
                if re.search(r"[a-zA-Z]+\\d", snippet):
                    return snippet
    return None


def infer_object_name_from_clip(clip: str) -> Optional[str]:
    if not clip:
        return None
    match = re.match(r"([a-zA-Z]+)", clip)
    return match.group(1) if match else None


def list_eval_files(input_path: str, pattern: str) -> List[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(p) for p in path.glob(pattern)]
    return []


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
