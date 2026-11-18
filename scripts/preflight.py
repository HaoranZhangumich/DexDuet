#!/usr/bin/env python3
"""Fail-fast checks for each DexMachina reproduction stage."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "dexmachina" / "assets"
CLIPS = (
    ("s01", "ketchup", "01"),
    ("s01", "box", "01"),
    ("s01", "mixer", "01"),
    ("s01", "ketchup", "02"),
    ("s02", "notebook", "02"),
    ("s01", "waffleiron", "01"),
)
EXPECTED_REVISIONS = {
    ROOT / "arctic": "49f3eb4b7f01663172576357bb3d95695d7a7689",
    # MandiZhao/Genesis e9df2d8 plus the broad-phase overflow fix (environment/patches/genesis/, applied by
    # scripts/setup_dexmachina_env.sh)
    ROOT / "Genesis": "fca4168f562b0762130932cda68ba09d011336ae",
    ROOT / "Genesis" / "rl_games": "2c1a7718a3eed71e7c230b6bb8fb2897b0b1d0c4",
}


class Checks:
    def __init__(self) -> None:
        self.failures = 0

    def ok(self, message: str) -> None:
        print(f"[OK]      {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[MISSING] {message}")

    def require_file(self, path: Path, description: str) -> bool:
        if path.is_file():
            self.ok(f"{description}: {path.relative_to(ROOT)}")
            return True
        self.fail(f"{description}: {path.relative_to(ROOT)}")
        return False


def git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def check_sources(checks: Checks) -> None:
    for path, expected in EXPECTED_REVISIONS.items():
        actual = git_revision(path)
        label = path.relative_to(ROOT)
        if actual is None:
            checks.fail(f"source checkout missing: {label}")
        elif actual != expected:
            checks.fail(f"{label} revision {actual}; expected {expected}")
        else:
            checks.ok(f"{label} revision {actual[:12]}")

    for distribution in ("torch", "genesis-world", "rl-games", "dexmachina"):
        try:
            version = importlib.metadata.version(distribution)
            checks.ok(f"Python package {distribution}=={version}")
        except importlib.metadata.PackageNotFoundError:
            checks.fail(f"Python package not installed: {distribution}")
    check_genesis_overflow_fix(checks)


def check_genesis_overflow_fix(checks: Checks) -> None:
    """The Genesis that Python imports must carry the broad-phase overflow fix (Collider.check_overflow)."""
    try:
        module = importlib.import_module("genesis.engine.solvers.rigid.collider_decomp")
    except Exception as exc:  # a broken install should be reported, not crash the other checks
        checks.fail(f"cannot import the Genesis collider: {exc}")
        return
    if hasattr(getattr(module, "Collider", None), "check_overflow"):
        checks.ok(f"Genesis broad-phase overflow fix: {module.__file__}")
    else:
        checks.fail(f"Genesis broad-phase overflow fix missing in {module.__file__}; "
                    "apply environment/patches/genesis/ (scripts/setup_dexmachina_env.sh)")


def require_cuda(checks: Checks) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            try:
                probe = torch.ones(1, device="cuda") + 1
                torch.cuda.synchronize()
                assert probe.item() == 2
                checks.ok(f"CUDA executable through PyTorch {torch.__version__}: {torch.cuda.get_device_name(0)}")
            except Exception as exc:
                checks.fail(f"CUDA is visible but cannot execute a PyTorch kernel: {exc}")
        else:
            checks.fail(f"CUDA is unavailable through PyTorch {torch.__version__}")
    except ImportError:
        checks.fail("PyTorch is not installed")


def check_arctic(checks: Checks, clips=CLIPS) -> None:
    expected_packages = {
        "torch": "1.13.0",
        "torchvision": "0.14.0",
        "pytorch3d": "0.7.3",
        "pytorch-lightning": "2.0.0",
        "aitviewer": "1.8.1",
        "numpy": "1.22.4",
    }
    for distribution, expected in expected_packages.items():
        try:
            actual = importlib.metadata.version(distribution)
            if actual == expected:
                checks.ok(f"ARCTIC package {distribution}=={actual}")
            else:
                checks.fail(f"ARCTIC package {distribution}=={actual}; expected {expected}")
        except importlib.metadata.PackageNotFoundError:
            checks.fail(f"ARCTIC package not installed: {distribution}")

    data_root = ROOT / "arctic" / "data"
    checks.require_file(
        data_root / "arctic_data" / "data" / "meta" / "misc.json",
        "ARCTIC metadata",
    )
    checks.require_file(
        data_root / "body_models" / "mano" / "MANO_RIGHT.pkl",
        "MANO model",
    )
    raw_root = data_root / "arctic_data" / "data" / "raw_seqs"
    for subject, obj, use in clips:
        checks.require_file(
            raw_root / subject / f"{obj}_use_{use}.mano.npy",
            "ARCTIC raw sequence",
        )


def check_processed(checks: Checks, clips=CLIPS) -> None:
    for subject, obj, use in clips:
        path = ASSETS / "arctic" / "processed" / subject / f"{obj}_use_{use}.npy"
        if not checks.require_file(path, "processed ARCTIC clip"):
            continue
        try:
            data = np.load(path, allow_pickle=True).item()
            frames = len(data["params"]["obj_trans"])
            if frames >= 600:
                checks.ok(f"{path.name} has {frames} frames")
            else:
                checks.fail(f"{path.name} has {frames} frames; at least 600 required")
        except Exception as exc:  # corrupt/incompatible data should fail before GPU work
            checks.fail(f"cannot read {path.relative_to(ROOT)}: {exc}")


def check_retargeted(checks: Checks, hands: list[str], include_contacts: bool, clips=CLIPS) -> None:
    try:
        import torch
    except ImportError:
        checks.fail("PyTorch is required to inspect retargeted files")
        return

    for hand in hands:
        for subject, obj, use in clips:
            path = ASSETS / "retargeted" / hand / subject / f"{obj}_use_{use}_vector_para.pt"
            if checks.require_file(path, "retargeted trajectory"):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                    tensors = [
                        data["retarget_data"][side][key]
                        for side in ("left", "right")
                        for key in ("kpt_pos", "wrist_pose")
                    ]
                    frames = tensors[0].shape[0]
                    finite = all(torch.isfinite(value).all() for value in tensors)
                    if frames >= 600 and finite:
                        checks.ok(f"{hand}/{path.name} has {frames} frames")
                    elif not finite:
                        checks.fail(f"{hand}/{path.name} contains non-finite values")
                    else:
                        checks.fail(f"{hand}/{path.name} has only {frames} frames")
                except Exception as exc:
                    checks.fail(f"cannot read {path.relative_to(ROOT)}: {exc}")

            if not include_contacts:
                continue
            contact = ASSETS / "contact_retarget" / hand / subject / f"{obj}_use_{use}.npy"
            if checks.require_file(contact, "retargeted contacts"):
                try:
                    data = np.load(contact, allow_pickle=True).item()
                    frames = data["left"]["dexlink_contacts"].shape[0]
                    if frames >= 600:
                        checks.ok(f"{hand}/{contact.name} has {frames} frames")
                    else:
                        checks.fail(f"{hand}/{contact.name} has only {frames} frames")
                except Exception as exc:
                    checks.fail(f"cannot read {contact.relative_to(ROOT)}: {exc}")


def check_checkpoint(checks: Checks, checkpoint: Path | None) -> None:
    if checkpoint is None:
        checks.fail("--checkpoint is required for the eval stage")
        return
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        checks.fail(f"checkpoint: {checkpoint}")
        return
    checks.ok(f"checkpoint: {checkpoint}")
    # Expected layout: <run>/nn/<file>.pth, with <run>/params/env.pkl.
    env_pickle = checkpoint.parent.parent / "params" / "env.pkl"
    if env_pickle.is_file():
        checks.ok(f"saved evaluation environment: {env_pickle}")
    else:
        checks.fail(f"saved evaluation environment: {env_pickle}")


def parse_clip(value: str) -> tuple[str, str, str]:
    """Select a benchmark source sequence without preslicing its stored assets."""
    match = re.fullmatch(r"([a-zA-Z0-9_]+)-(\d+)-(\d+)-(s\d{2})-u(\d{2})", value)
    if match is None:
        raise argparse.ArgumentTypeError("expected OBJECT-START-END-sNN-uNN")
    obj, start, end, subject, use = match.groups()
    if int(start) >= int(end) or int(end) > 600:
        raise argparse.ArgumentTypeError("benchmark clip requires 0 <= START < END <= 600")
    selection = (subject, obj, use)
    if selection not in CLIPS:
        raise argparse.ArgumentTypeError(f"not a configured benchmark source sequence: {value}")
    return selection


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("sources", "arctic", "retarget", "train", "eval"),
        default="train",
    )
    parser.add_argument("--hands", nargs="+", default=["inspire_hand"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--clip", action="append", type=parse_clip,
        help="Check only this benchmark sequence (repeatable); stored assets must still span >=600 frames.",
    )
    args = parser.parse_args(argv)
    args.clips = tuple(dict.fromkeys(args.clip)) if args.clip else CLIPS
    if args.clip and args.stage in ("sources", "eval"):
        parser.error("--clip applies only to arctic, retarget, and train stages")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    checks = Checks()
    if args.stage == "sources":
        check_sources(checks)
    elif args.stage == "arctic":
        check_arctic(checks, args.clips)
        if not args.allow_cpu:
            require_cuda(checks)
    elif args.stage == "retarget":
        check_processed(checks, args.clips)
        if not args.allow_cpu:
            require_cuda(checks)
    elif args.stage == "train":
        check_processed(checks, args.clips)
        check_retargeted(checks, args.hands, include_contacts=True, clips=args.clips)
        if not args.allow_cpu:
            require_cuda(checks)
    else:
        check_checkpoint(checks, args.checkpoint)
        if not args.allow_cpu:
            require_cuda(checks)

    if checks.failures:
        print(f"\nPreflight failed with {checks.failures} missing or invalid requirement(s).")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
