"""Check baselines/track2_metrics.py against the V2D kit's own Track 2 scoring functions.

    KIT=/path/to/v2d_submission_kit python baselines/test_track2_metrics.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track2_metrics import track2_scores  # noqa: E402

if not os.environ.get("KIT"):
    sys.exit("set KIT to the V2D submission kit folder (it holds metric_code/track_2)")
KIT = Path(os.environ["KIT"]) / "metric_code" / "track_2"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), KIT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_poses(rng, frames, rollouts, bodies, noise):
    reference = np.zeros((frames, 1, bodies, 7))
    reference[..., :3] = rng.normal(0.0, 0.3, (frames, 1, bodies, 3))
    quat = rng.normal(size=(frames, 1, bodies, 4))
    reference[..., 3:] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    achieved = np.repeat(reference, rollouts, axis=1).copy()
    scale = np.linspace(0.0, noise, rollouts)[None, :, None, None]
    achieved[..., :3] += rng.normal(size=achieved[..., :3].shape) * scale
    achieved[..., 3:] += rng.normal(size=achieved[..., 3:].shape) * scale * 2.0
    achieved[..., 3:] /= np.linalg.norm(achieved[..., 3:], axis=-1, keepdims=True)
    return achieved, reference


def main():
    kit = load("SP-SR")
    rng = np.random.default_rng(0)
    for noise in (0.0, 0.02, 0.08, 0.3):
        achieved, reference = random_poses(rng, frames=40, rollouts=16, bodies=2, noise=noise)
        ours = track2_scores(achieved, reference)
        ids = np.arange(2, dtype=np.int64)  # each part is one object slot, as in the kit's score()
        expected = {
            "sp_sr": float(np.mean(kit._spider_success(achieved, reference))),
            "mp_sr": float(np.mean(kit._maniptrans_success(achieved, reference, ids))),
            "mppe_cm": kit._mean_per_point_error_cm(achieved, reference),
        }
        for key, value in expected.items():
            assert np.isclose(ours[key], value, rtol=0, atol=1e-9), (noise, key, ours[key], value)
        print(f"noise {noise}: SP-SR {ours['sp_sr']:.3f} MP-SR {ours['mp_sr']:.3f} MPPE {ours['mppe_cm']:.2f} cm match the kit")
    perfect = track2_scores(*random_poses(rng, 10, 3, 2, 0.0))
    assert perfect["sp_sr"] == perfect["mp_sr"] == 1.0 and perfect["mppe_cm"] < 1e-9
    # A diverged rollout fails both success rates and is excluded from MPPE.
    achieved, reference = random_poses(rng, 10, 4, 2, 0.0)
    achieved[5:, 2] = np.nan
    scores = track2_scores(achieved, reference)
    assert scores["nan_rollouts"] == 1 and scores["sp_sr"] == scores["mp_sr"] == 0.75, scores
    assert scores["per_rollout_sp_success"] == [1, 1, 0, 1] and scores["mppe_cm"] < 1e-9, scores
    print("all checks passed")


if __name__ == "__main__":
    main()
