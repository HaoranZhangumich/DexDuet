"""V2D Track 2 object-tracking metrics (SP-SR, MP-SR, MPPE) for a DexMachina evaluation rollout.

The definitions are those of the V2D submission kit (``metric_code/track_2``, tier 1, without initial-pose
alignment), applied to the poses of the object's parts (its bottom and top links, one articulated object):

- SP-SR (``spider_sr``): fraction of rollouts whose clip-mean position error, averaged over the parts, is at
  most 0.1 m and whose clip-mean orientation error is at most 0.5 rad.
- MP-SR (``maniptrans_sr``): fraction of rollouts in which every part has a clip-mean position error below 3 cm
  and a clip-mean orientation error below 30 degrees (each part is one object slot, as in the kit's ``score()``).
- MPPE (``mppe_cm``): mean distance, in cm, of six keypoints placed 5 cm from each part frame along +-x, +-y
  and +-z, between the rollout and the demonstration, over every frame, rollout and part.

The part poses come from the scene and object states that ``dexmachina.eval.compute_add`` uses for ADD-AUC,
so all metrics of a run describe the same rollouts, with the same frame pairing (the recorded state after step t
against demonstration frame t). A rollout with a non-finite pose fails SP-SR and MP-SR and is left out of MPPE;
``nan_rollouts`` counts them. Writes ``track2_metrics.json`` next to the eval file.

    python -m baselines.track2_metrics --input <run>/inspire_hand_eval_all_policy/eval_ep0.npy --obj_name box
"""

import argparse
import json
import os

import numpy as np

SPIDER_POSITION_THRESHOLD_M = 0.1
SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
MANIPTRANS_POSITION_THRESHOLD_M = 0.03
MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
MPPE_KEYPOINT_RADIUS_M = 0.05
MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)
PARTS = ("bottom", "top")


def rotation_matrix(quat_xyzw):
    x, y, z, w = (quat_xyzw[..., index] for index in range(4))
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
            2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)


def geodesic_angle(a_xyzw, b_xyzw):
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def normalise(pose):
    pose = np.array(pose, dtype=np.float64, copy=True)
    norms = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("pose contains a degenerate (near-zero-norm) quaternion")
    pose[..., 3:] /= norms[..., None]
    return pose


def track2_scores(achieved, reference, object_ids=None):
    """Kit scores for one episode.

    achieved: (frames, rollouts, bodies, 7) poses, position then xyzw quaternion.
    reference: (frames, 1 or rollouts, bodies, 7) demonstration poses.
    object_ids: (bodies,) object slot of each body (each body its own slot by default, so MP-SR tests every part).
    """
    achieved, reference = np.asarray(achieved, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    rollouts = achieved.shape[1]
    if object_ids is None:
        object_ids = np.arange(achieved.shape[2], dtype=np.int64)
    # A rollout that diverged (non-finite pose in any frame) fails both success rates and is left out of MPPE.
    finite = np.isfinite(achieved).all(axis=(0, 2, 3))
    spider = np.zeros(rollouts, dtype=bool)
    maniptrans = np.zeros(rollouts, dtype=bool)
    mppe_cm = position_m = orientation_deg = float("nan")
    per_rollout_mppe = np.full(rollouts, np.nan)
    if finite.any():
        a = normalise(achieved[:, finite])
        r = normalise(reference if reference.shape[1] == 1 else reference[:, finite])
        position = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1).mean(axis=0)  # (rollouts, bodies)
        orientation = geodesic_angle(a[..., 3:], r[..., 3:]).mean(axis=0)
        spider[finite] = (position.mean(axis=-1) <= SPIDER_POSITION_THRESHOLD_M) & (
            orientation.mean(axis=-1) <= SPIDER_ORIENTATION_THRESHOLD_RAD
        )
        success = np.ones(int(finite.sum()), dtype=bool)
        for object_id in np.unique(object_ids):
            members = object_ids == object_id
            success &= (position[:, members].mean(axis=-1) < MANIPTRANS_POSITION_THRESHOLD_M) & (
                np.degrees(orientation[:, members]).mean(axis=-1) < MANIPTRANS_ORIENTATION_THRESHOLD_DEG
            )
        maniptrans[finite] = success
        offsets = MPPE_KEYPOINT_DIRECTIONS * MPPE_KEYPOINT_RADIUS_M
        keypoints_a = a[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_matrix(a[..., 3:]), offsets)
        keypoints_r = r[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_matrix(r[..., 3:]), offsets)
        distances = np.linalg.norm(keypoints_a - keypoints_r, axis=-1)  # (frames, rollouts, bodies, keypoints)
        mppe_cm = float(distances.mean() * 100.0)
        per_rollout_mppe[finite] = distances.mean(axis=(0, 2, 3)) * 100.0
        position_m, orientation_deg = float(position.mean()), float(np.degrees(orientation).mean())
    return {
        "sp_sr": float(np.mean(spider)),
        "mp_sr": float(np.mean(maniptrans)),
        "mppe_cm": mppe_cm,
        "nan_rollouts": int((~finite).sum()),
        "mean_position_error_m": position_m,
        "mean_orientation_error_deg": orientation_deg,
        "per_rollout_sp_success": spider.astype(int).tolist(),
        "per_rollout_mp_success": maniptrans.astype(int).tolist(),
        # MPPE of each rollout (None for a non-finite one); every rollout has the same frames, so MPPE is their mean.
        "per_rollout_mppe_cm": [None if not np.isfinite(v) else float(v) for v in per_rollout_mppe],
    }


def wxyz_to_xyzw(pose):
    return np.concatenate([pose[..., :3], pose[..., 4:7], pose[..., 3:4]], axis=-1)


def part_poses(obj, states):
    """(frames, parts, 7) part poses (position, wxyz) for (frames, 8) object states, one frame per scene env."""
    obj.reset()
    obj.set_object_state(root_pos=states[:, :3], root_quat=states[:, 3:7], joint_qpos=states[:, 7:])
    return np.stack([obj.get_part_pose(part).cpu().numpy() for part in PARTS], axis=1)


def compute_for_eval(eval_path, obj_name=None, out_name="track2_metrics.json"):
    import torch

    from dexmachina.eval.compute_add import create_scene, infer_object_name

    eval_data = np.load(eval_path, allow_pickle=True).item()
    demo_states, obj_states = eval_data["demo_state"], eval_data["obj_state"]
    if obj_states.ndim == 2:
        obj_states = obj_states[:, None, :]
    if demo_states.ndim == 3 and demo_states.shape[1] == 1:
        demo_states = demo_states[:, 0, :]
    obj_name = infer_object_name(eval_path, obj_name)
    frames, rollouts = obj_states.shape[0], obj_states.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene, obj = create_scene(obj_name, device, num_envs=frames)

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    reference = wxyz_to_xyzw(part_poses(obj, tensor(demo_states)))[:, None]  # (frames, 1, parts, 7)
    achieved = np.stack(
        [wxyz_to_xyzw(part_poses(obj, tensor(obj_states[:, index]))) for index in range(rollouts)], axis=1
    )
    # Reported pairing: the state after step t against demonstration frame t, as compute_add pairs them for
    # ADD-AUC (the evaluator reads the demo index before stepping). Training targets frame t+1, so the
    # next-frame pairing is stored too.
    scores = track2_scores(achieved, reference)
    shifted = track2_scores(achieved[:-1], reference[1:])
    scores["frame_pairing"] = "rollout state after step t vs demo frame t (same as dexmachina.eval.compute_add)"
    scores["next_frame"] = {key: shifted[key] for key in ("sp_sr", "mp_sr", "mppe_cm", "nan_rollouts")}
    scores.update(
        eval_path=os.path.abspath(eval_path), object=obj_name, parts=list(PARTS), frames=int(frames),
        rollouts=int(rollouts),
        definitions="V2D kit metric_code/track_2 (tier 1, no initial-pose alignment) on the bottom/top part poses",
    )
    out_path = os.path.join(os.path.dirname(eval_path), out_name)
    with open(out_path, "w") as handle:
        json.dump(scores, handle, indent=2)
    print(f"SP-SR {scores['sp_sr']:.3f} MP-SR {scores['mp_sr']:.3f} MPPE {scores['mppe_cm']:.2f} cm -> {out_path}")
    del scene, obj
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="eval_ep*.npy written by dexmachina.rl.eval_rl_games")
    parser.add_argument("--obj_name", default=None)
    parser.add_argument("--out_name", default="track2_metrics.json")
    args = parser.parse_args()
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    compute_for_eval(args.input, args.obj_name, args.out_name)


if __name__ == "__main__":
    main()
