"""CPU checks of scripts/preflight.py: clip filters and the asset checks of a fresh machine (Notebook-300 as the example)."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from scripts import preflight


ROOT = Path(__file__).resolve().parents[1]


NOTEBOOK = (("s02", "notebook", "02"),)


class PreflightSelectionTests(unittest.TestCase):
    def test_no_filter_keeps_entire_benchmark(self):
        self.assertEqual(preflight.parse_args([]).clips, preflight.CLIPS)

    def test_task_ranges_select_same_full_source_once(self):
        args = preflight.parse_args([
            "--clip", "notebook-40-340-s02-u02", "--clip", "notebook-0-600-s02-u02",
        ])
        self.assertEqual(args.clips, NOTEBOOK)

    def test_repeatable_filter_selects_multiple_sources(self):
        args = preflight.parse_args([
            "--clip", "notebook-40-340-s02-u02", "--clip", "ketchup-0-600-s01-u01",
        ])
        self.assertEqual(args.clips, (*NOTEBOOK, ("s01", "ketchup", "01")))

    def test_invalid_clip_or_irrelevant_stage_rejected(self):
        for extra in (
            ["--clip", "../notebook"], ["--clip", "notebook-340-40-s02-u02"],
            ["--clip", "notebook-0-601-s02-u02"], ["--clip", "notebook-0-600-s01-u01"],
            ["--stage", "eval", "--clip", "notebook-40-340-s02-u02"],
            ["--stage", "sources", "--clip", "notebook-40-340-s02-u02"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    preflight.parse_args(extra)

    def test_only_requested_clip_reaches_training_checks(self):
        with mock.patch.object(preflight, "check_processed") as processed, \
             mock.patch.object(preflight, "check_retargeted") as retargeted, \
             mock.patch.object(preflight, "require_cuda") as cuda, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(preflight.main([
                "--stage", "train", "--clip", "notebook-40-340-s02-u02", "--allow-cpu",
            ]), 0)
        self.assertEqual(processed.call_args.args[1], NOTEBOOK)
        self.assertEqual(retargeted.call_args.kwargs["clips"], NOTEBOOK)
        self.assertTrue(retargeted.call_args.kwargs["include_contacts"])
        cuda.assert_not_called()

    def test_single_task_still_requires_600_frame_processed_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            sequence = assets / "arctic/processed/s02/notebook_use_02.npy"
            sequence.parent.mkdir(parents=True)
            for frames, expected_failures in ((300, 1), (600, 0)):
                with self.subTest(frames=frames):
                    np.save(sequence, {"params": {"obj_trans": np.zeros((frames, 3))}})
                    checks = preflight.Checks()
                    with mock.patch.object(preflight, "ROOT", root), \
                         mock.patch.object(preflight, "ASSETS", assets), \
                         contextlib.redirect_stdout(io.StringIO()):
                        preflight.check_processed(checks, NOTEBOOK)
                    self.assertEqual(checks.failures, expected_failures)

    def test_single_task_still_requires_600_frame_retarget_and_contacts(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            trajectory = assets / "retargeted/inspire_hand/s02/notebook_use_02_vector_para.pt"
            contacts = assets / "contact_retarget/inspire_hand/s02/notebook_use_02.npy"
            trajectory.parent.mkdir(parents=True)
            contacts.parent.mkdir(parents=True)
            for frames, expected_failures in ((300, 2), (600, 0)):
                with self.subTest(frames=frames):
                    torch.save({"retarget_data": {
                        side: {"kpt_pos": torch.zeros((frames, 1, 3)),
                               "wrist_pose": torch.zeros((frames, 7))}
                        for side in ("left", "right")
                    }}, trajectory)
                    np.save(contacts, {"left": {"dexlink_contacts": np.zeros((frames, 1))}})
                    checks = preflight.Checks()
                    with mock.patch.object(preflight, "ROOT", root), \
                         mock.patch.object(preflight, "ASSETS", assets), \
                         contextlib.redirect_stdout(io.StringIO()):
                        preflight.check_retargeted(checks, ["inspire_hand"], True, NOTEBOOK)
                    self.assertEqual(checks.failures, expected_failures)


class GenesisPinTests(unittest.TestCase):
    COLLIDER = "genesis.engine.solvers.rigid.collider_decomp"

    def overflow_check_failures(self, module):
        checks = preflight.Checks()
        with mock.patch.dict(sys.modules, {self.COLLIDER: module}), contextlib.redirect_stdout(io.StringIO()):
            preflight.check_genesis_overflow_fix(checks)
        return checks.failures

    def test_overflow_fix_required_in_imported_genesis(self):
        fixed, unpatched = types.ModuleType(self.COLLIDER), types.ModuleType(self.COLLIDER)
        fixed.__file__ = unpatched.__file__ = "genesis/engine/solvers/rigid/collider_decomp.py"
        fixed.Collider = type("Collider", (), {"check_overflow": lambda self: None})
        unpatched.Collider = type("Collider", (), {})
        self.assertEqual(self.overflow_check_failures(fixed), 0)
        self.assertEqual(self.overflow_check_failures(unpatched), 1)
        self.assertEqual(self.overflow_check_failures(None), 1)  # import fails

    def test_genesis_pins_agree(self):
        revisions = dict(
            line.split("=", 1) for line in (ROOT / "environment/source-revisions.txt").read_text().splitlines()
            if "=" in line and not line.startswith("#")
        )
        fixed = revisions["GENESIS_FIXED_REV"]
        self.assertTrue((ROOT / revisions["GENESIS_PATCH"]).is_file())
        self.assertEqual(preflight.EXPECTED_REVISIONS[preflight.ROOT / "Genesis"], fixed)
        setup = (ROOT / "scripts/setup_dexmachina_env.sh").read_text()
        self.assertIn(f'GENESIS_REV="${{GENESIS_REV:-{revisions["GENESIS_REV"]}}}"', setup)
        self.assertIn(f'GENESIS_FIXED_REV="${{GENESIS_FIXED_REV:-{fixed}}}"', setup)
        self.assertIn(f'$REPO_ROOT/{revisions["GENESIS_PATCH"]}"', setup)


if __name__ == "__main__":
    unittest.main()
