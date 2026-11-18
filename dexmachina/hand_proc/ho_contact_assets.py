"""Build the assets of the hand-object contact reward (--ho_contact_weight, dexmachina/envs/ho_contact.py):

  grids : per object part, the distance to the part surface and the nearest surface point on a regular grid in the part frame
  human : per ARCTIC sequence and frame, each human anchor's distance to each part and its nearest surface spot (part frame)

  python dexmachina/hand_proc/ho_contact_assets.py grids --assets <assets dir> box mixer ketchup notebook waffleiron
  python dexmachina/hand_proc/ho_contact_assets.py human --assets <assets dir> --mano <export dir> s01/ketchup_use_02 ...

The MANO meshes come from export_mano_vertices.py (ARCTIC environment). Object poses are the processed ARCTIC parameters the
environment replays (arctic/processed); the top part turns about the bottom part's -z axis by the articulation angle, as in the
object URDFs (joint origin at the part frame, axis 0 0 -1).
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dexmachina.envs.ho_contact import (ANCHORS, MANO_ANCHOR_JOINTS, PARTS, grid_path, human_target_path,  # noqa: E402
                                        object_mesh_path)


def surface_samples(mesh, count, seed=0):
    return np.concatenate([trimesh.sample.sample_surface(mesh, count, seed=seed)[0], mesh.vertices], 0)


def build_grid(mesh, spacing=0.003, margin=0.06, samples=400000, seed=0):
    """Distance to the surface and the nearest surface point on a grid covering the mesh bounds plus margin."""
    lo = mesh.bounds[0] - margin
    size = np.ceil((mesh.bounds[1] + margin - lo) / spacing).astype(int) + 1
    axes = [lo[i] + spacing * np.arange(size[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    surf = surface_samples(mesh, samples, seed)
    dist, idx = cKDTree(surf).query(pts, workers=-1)
    return dict(origin=lo.astype(np.float32), spacing=np.float32(spacing), margin=np.float32(margin),
                dist=dist.reshape(*size).astype(np.float32), nearest=surf[idx].reshape(*size, 3).astype(np.float32))


def part_rotations(obj_quat, obj_arti):
    """World rotation of each part per frame: bottom = object pose; top = bottom * Rz(-arti). obj_quat is wxyz."""
    bottom = Rotation.from_quat(np.roll(obj_quat, -1, axis=-1))
    top = bottom * Rotation.from_rotvec(np.stack([np.zeros_like(obj_arti), np.zeros_like(obj_arti), -obj_arti], -1))
    return [bottom.as_matrix(), top.as_matrix()]


def human_targets(verts, vertex_joint, obj_trans, obj_quat, obj_arti, meshes, samples=200000, seed=0):
    """verts {side: (T, 778, 3)}, vertex_joint {side: (778,)} -> dist (T, 2, A, parts), spot (T, 2, A, parts, 3) in part frames."""
    frames, anchors = obj_trans.shape[0], ANCHORS
    dist = np.zeros((frames, 2, len(anchors), len(PARTS)), np.float32)
    spot = np.zeros((frames, 2, len(anchors), len(PARTS), 3), np.float32)
    rots = part_rotations(obj_quat, obj_arti)
    for p, mesh in enumerate(meshes):
        surf = surface_samples(mesh, samples, seed)
        tree = cKDTree(surf)
        for h, side in enumerate(('left', 'right')):
            local = np.einsum('tvi,tij->tvj', verts[side] - obj_trans[:, None], rots[p])       # world -> part frame
            d, idx = tree.query(local.reshape(-1, 3), workers=-1)
            d, idx = d.reshape(frames, -1), idx.reshape(frames, -1)
            for a, anchor in enumerate(anchors):
                members = np.where(np.isin(vertex_joint[side], MANO_ANCHOR_JOINTS[anchor]))[0]
                if len(members) == 0:
                    raise ValueError(f'{side} hand: no MANO vertices for anchor {anchor}')
                k = members[d[:, members].argmin(1)]
                dist[:, h, a, p] = d[np.arange(frames), k]
                spot[:, h, a, p] = surf[idx[np.arange(frames), k]]
    return dist, spot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['grids', 'human'])
    parser.add_argument('--assets', required=True, help='dexmachina assets directory (reads arctic/, writes ho_contact/)')
    parser.add_argument('--mano', help='human: directory written by export_mano_vertices.py')
    parser.add_argument('--spacing', type=float, default=0.003)
    parser.add_argument('--margin', type=float, default=0.06)
    parser.add_argument('items', nargs='+', help='grids: object names; human: <subject>/<object>_use_<clip>')
    args = parser.parse_args()
    if args.mode == 'grids':
        for obj in args.items:
            for part in PARTS:
                mesh = trimesh.load(object_mesh_path(args.assets, obj, part), force='mesh')
                grid = build_grid(mesh, args.spacing, args.margin)
                out = grid_path(args.assets, obj, part)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.savez_compressed(out, **grid)
                print(f'{obj} {part}: grid {grid["dist"].shape}, mesh extents {np.round(mesh.extents * 100, 1)} cm -> {out}')
        return
    if not args.mano:
        parser.error('human needs --mano')
    for item in args.items:
        subject, seq = item.split('/')
        obj, clip = seq.split('_use_')
        params = np.load(os.path.join(args.assets, 'arctic', 'processed', subject, f'{seq}.npy'), allow_pickle=True).item()['params']
        obj_trans = np.asarray(params['obj_trans'], np.float64)
        obj_quat = np.asarray(params['obj_quat'], np.float64)
        obj_arti = np.asarray(params['obj_arti'], np.float64).reshape(len(obj_trans), -1)[:, 0]
        hand = np.load(os.path.join(args.mano, subject, f'{seq}.npz'))
        verts = {s: hand[f'verts_{s}'].astype(np.float64) for s in ('left', 'right')}
        if verts['left'].shape[0] != obj_trans.shape[0]:
            raise ValueError(f'{item}: {verts["left"].shape[0]} MANO frames vs {obj_trans.shape[0]} object frames')
        meshes = [trimesh.load(object_mesh_path(args.assets, obj, part), force='mesh') for part in PARTS]
        anchors = ANCHORS
        dist, spot = human_targets(verts, {s: hand[f'vertex_joint_{s}'] for s in ('left', 'right')}, obj_trans, obj_quat, obj_arti, meshes)
        out = human_target_path(args.assets, subject, obj, clip)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, dist=dist, spot=spot, anchors=np.array(anchors), parts=np.array(PARTS))
        touch = (dist.min(-1) < 0.01).mean(0)
        print(f'{item}: {dist.shape[0]} frames -> {out}; share of frames touching (< 1 cm) left/right per anchor {anchors}:')
        print('   left ', np.round(touch[0], 2), ' right', np.round(touch[1], 2))


if __name__ == '__main__':
    main()
