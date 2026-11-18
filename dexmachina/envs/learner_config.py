"""Options of DexDuet's learner (per-hand PPO, value mode, training protocol) and the environment's learner_cfg built from
them; without these options the trainer keeps DexMachina's learner and defaults.

Run folders written before two options were renamed store this setting under `wrench_cfg`, and commands written then use
the older option names; learner_cfg_from_legacy() and the hidden aliases read both. Standard library only, so that this
module loads without the simulator.
"""

import argparse
import math

PROTOCOLS = ("legacy", "v2", "dexmachina")
MODULE_PROTOCOLS = ("v2", "dexmachina")
VALUE_MODES = ("scalar", "per_hand")
# Errors for older options that the paper's commands did not use in this way (the paper passed --wrench_guidance team
# with --wrench_reward_weight 0 --wrench_credit_weight 0 and --ablation_protocol dexmachina).
LEGACY_WEIGHT_ERROR = ("a --wrench_reward_weight or --wrench_credit_weight other than 0 turns on wrench guidance, which is "
                       "not part of DexDuet's code; drop the --wrench_* options (the paper's commands passed both weights "
                       "as 0)")
LEGACY_UNUSED_WEIGHT_ERROR = ("--wrench_reward_weight and --wrench_credit_weight are no longer accepted (without "
                              "--wrench_guidance hand or team they had no effect); drop the --wrench_* options")
LEGACY_DEFAULT_WEIGHT_ERROR = ("--wrench_guidance without --wrench_reward_weight 0 turned wrench guidance on (the older "
                               "default weight was 1.0), which is not part of DexDuet's code; drop the --wrench_* options, "
                               "or pass --wrench_reward_weight 0 as the paper's commands did")
LEGACY_PROTOCOL_ERROR = ("--wrench_guidance with --ablation_protocol legacy is no longer supported; drop the --wrench_* "
                         "options to train with DexMachina's learner (DexDuet's per-hand commands use --ablation_protocol "
                         "dexmachina)")
LEGACY_RUN_ERROR = ("this run folder was trained with wrench-guidance rewards or credit, which DexDuet's code no longer "
                    "contains")


def add_learner_arguments(parser):
    group = parser.add_argument_group("Per-hand PPO and training protocol")
    group.add_argument(
        "--ablation_protocol", choices=PROTOCOLS, default="legacy",
        help="dexmachina: DexMachina's native PPO schedule and stopping rule through the learner that supports per-hand "
             "PPO, with the best-training-reward checkpoint recorded in final_checkpoint.json; needed by --per_hand_ppo")
    group.add_argument("--per_hand_ppo", action="store_true",
                       help="clip the PPO probability ratio of each hand separately (needs --ablation_protocol dexmachina)")
    group.add_argument("--value_mode", choices=VALUE_MODES, default=None,
                       help="critic outputs: one value per hand or one scalar (default: per_hand with --per_hand_ppo, "
                            "else scalar)")
    # Older spellings, kept so that stored commands still parse (docs/METHOD.md, "Names in the code").
    group.add_argument("--wrench_per_hand_ppo", dest="per_hand_ppo", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--wrench_value_mode", dest="value_mode", choices=VALUE_MODES, default=None, help=argparse.SUPPRESS)
    group.add_argument("--wrench_guidance", dest="legacy_wrench_guidance", choices=("none", "hand", "team"), default=None,
                       help=argparse.SUPPRESS)
    group.add_argument("--wrench_reward_weight", dest="legacy_wrench_reward_weight", type=float, default=None,
                       help=argparse.SUPPRESS)
    group.add_argument("--wrench_credit_weight", dest="legacy_wrench_credit_weight", type=float, default=None,
                       help=argparse.SUPPRESS)
    group.add_argument("--wrench_credit_mode", dest="legacy_wrench_credit_mode", choices=("individual", "shared"),
                       default=None, help=argparse.SUPPRESS)
    group.add_argument("--wrench_reference", dest="legacy_wrench_reference", default=None, help=argparse.SUPPRESS)
    return parser


def resolve_value_mode(args):
    """The value mode a run uses: the given one, else per_hand with --per_hand_ppo and scalar without."""
    return getattr(args, "value_mode", None) or ("per_hand" if getattr(args, "per_hand_ppo", False) else "scalar")


def check_learner_args(args):
    guided = getattr(args, "legacy_wrench_guidance", None) in ("hand", "team")
    for name in ("legacy_wrench_reward_weight", "legacy_wrench_credit_weight"):
        weight = getattr(args, name, None)
        if weight is not None and (not math.isfinite(weight) or weight != 0):
            raise ValueError(LEGACY_WEIGHT_ERROR if guided else LEGACY_UNUSED_WEIGHT_ERROR)
    protocol = getattr(args, "ablation_protocol", "legacy")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown --ablation_protocol {protocol!r}; one of {PROTOCOLS}")
    if guided:
        if getattr(args, "legacy_wrench_reward_weight", None) is None:
            raise ValueError(LEGACY_DEFAULT_WEIGHT_ERROR)
        if protocol not in MODULE_PROTOCOLS:
            raise ValueError(LEGACY_PROTOCOL_ERROR)
    per_hand = bool(getattr(args, "per_hand_ppo", False))
    value_mode = resolve_value_mode(args)
    if value_mode not in VALUE_MODES:
        raise ValueError(f"Unknown --value_mode {value_mode!r}; one of {VALUE_MODES}")
    if protocol == "legacy" and (per_hand or value_mode == "per_hand"):
        raise ValueError("--per_hand_ppo and --value_mode per_hand need --ablation_protocol dexmachina (or v2)")
    if protocol in MODULE_PROTOCOLS:
        if not getattr(args, "coordinate_actor", False):
            raise ValueError(f"--ablation_protocol {protocol} needs the per-hand actor (--coordinate_actor)")
        if getattr(args, "no_object", False) or getattr(args, "maniptrans", False):
            raise ValueError(f"--ablation_protocol {protocol} needs the ordinary DexMachina object environment")


def make_learner_config(args):
    """The environment's learner_cfg for these options, or None for DexMachina's own learner (the legacy protocol)."""
    check_learner_args(args)
    protocol = getattr(args, "ablation_protocol", "legacy")
    if protocol == "legacy":
        return None
    per_hand = bool(getattr(args, "per_hand_ppo", False))
    value_mode = resolve_value_mode(args)
    return dict(ablation_protocol=protocol, per_hand_ppo=per_hand, value_mode=value_mode,
                per_hand_value=value_mode == "per_hand")


def learner_cfg_from_legacy(old):
    """learner_cfg of a run folder that stored its setting under `wrench_cfg`, or None when it used DexMachina's learner."""
    if old is None:
        return None
    if not isinstance(old, dict):
        raise ValueError(f"wrench_cfg must be a dict, got {type(old).__name__}")
    if not old:
        return None
    weights = [old.get("reward_weight"), old.get("credit_weight")]
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w != 0 for w in weights):
        raise ValueError(LEGACY_RUN_ERROR)
    protocol = old.get("ablation_protocol", "legacy")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown ablation_protocol {protocol!r} in wrench_cfg")
    per_hand = bool(old.get("per_hand_ppo", False))
    value_mode = old.get("value_mode") or ("per_hand" if per_hand else "scalar")
    if value_mode not in VALUE_MODES:
        raise ValueError(f"Unknown value_mode {value_mode!r} in wrench_cfg")
    per_hand_value = bool(old.get("per_hand_value", per_hand))
    if protocol == "legacy":
        if per_hand or value_mode == "per_hand" or per_hand_value:
            raise ValueError("this run folder used a per-hand learner of the legacy protocol, which DexDuet's code no "
                             "longer contains")
        return None
    return dict(ablation_protocol=protocol, per_hand_ppo=per_hand, value_mode=value_mode, per_hand_value=per_hand_value)


def upgrade_legacy_env_kwargs(kw):
    """Replace a stored `wrench_cfg` by its learner_cfg, in place; returns kw."""
    if "wrench_cfg" not in kw:
        return kw
    if "learner_cfg" in kw:
        raise ValueError("environment configuration has both wrench_cfg and learner_cfg")
    learner_cfg = learner_cfg_from_legacy(kw.pop("wrench_cfg"))
    if learner_cfg is not None:
        kw["learner_cfg"] = learner_cfg
    return kw
